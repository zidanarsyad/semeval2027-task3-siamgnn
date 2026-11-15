"""Siamese-GNN usage-graph model ("SiamGNN") for diachronic word sense induction.

For one target word, every usage is a node whose input feature is its frozen XL-LEXEME embedding
(a Transformer Siamese WiC encoder).  The model

  1. projects nodes with a shared (Siamese) MLP,                        h0 = MLP(x)
  2. refines them with L residual message-passing layers over the kNN usage graph
     (GraphSAGE-style mean aggregation, edges weighted by XL-LEXEME similarity)
                                                                        h_{l+1} = LN(h_l + GELU(W [h_l ; A h_l]))
  3. scores every pair with a Siamese similarity head
       "cos": p_ij = sigmoid( s * cos(h_i, h_j) + a * cos(x_i, x_j) + b )
       "mlp": p_ij = sigmoid( MLP([h_i * h_j , |h_i - h_j|]) + a * cos(x_i, x_j) + b )

It is trained across words (episodes = random usage subsets of one word) with plain, unweighted
binary cross-entropy on "do usages i and j share a gold sense?" (``balance=True`` switches to a
class-balanced loss), so p_ij stays calibrated to the empirical same-sense rate; at inference the
usage graph is clustered with correlation clustering at theta (0.5 in the submission).
``use_gnn=False`` gives the plain Siamese-head ablation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def knn_adj(X: torch.Tensor, k: int) -> torch.Tensor:
    """Row-normalised symmetric kNN adjacency (weights = clipped cosine), no self loops."""
    S = X @ X.T
    n = S.shape[0]
    k = max(1, min(k, n - 1))
    vals, idx = (S - 2 * torch.eye(n)).topk(k, dim=1)
    A = torch.zeros_like(S).scatter_(1, idx, vals.clamp(min=0))
    A = torch.maximum(A, A.T)
    return A / A.sum(1, keepdim=True).clamp(min=1e-9)


class SiamGNN(nn.Module):
    def __init__(self, d_in: int = 1024, d: int = 256, layers: int = 2, knn: int = 10, dropout: float = 0.1,
                 use_gnn: bool = True, head: str = "cos"):
        super().__init__()
        self.knn, self.use_gnn, self.head = knn, use_gnn, head
        self.proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_in, d), nn.GELU(), nn.Linear(d, d))
        n_layers = layers if use_gnn else 0
        self.mp = nn.ModuleList([nn.Linear(2 * d, d) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])
        if head == "mlp":
            self.edge = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.s = nn.Parameter(torch.tensor(5.0))
        self.a = nn.Parameter(torch.tensor(5.0))
        self.b = nn.Parameter(torch.tensor(-6.0))
        self.drop = nn.Dropout(dropout)

    def node_states(self, X: torch.Tensor) -> torch.Tensor:
        h = self.proj(X)
        if self.use_gnn:
            A = knn_adj(X, self.knn)
            for lin, ln in zip(self.mp, self.norms):
                h = ln(h + self.drop(F.gelu(lin(torch.cat([h, A @ h], dim=-1)))))
        return h

    def pair_logits(self, X: torch.Tensor) -> torch.Tensor:
        """Full n x n matrix of same-sense logits (symmetric by construction)."""
        h = self.node_states(X)
        if self.head == "mlp":
            n, d = h.shape
            hi, hj = h.unsqueeze(1).expand(n, n, d), h.unsqueeze(0).expand(n, n, d)
            z = self.edge(torch.cat([hi * hj, (hi - hj).abs()], dim=-1)).squeeze(-1)
        else:
            hn = F.normalize(h, dim=-1)
            z = self.s * (hn @ hn.T)
        return z + self.a * (X @ X.T) + self.b


@dataclass
class WordGraph:
    word: str
    X: np.ndarray            # n x 1024 normalised XL-LEXEME embeddings
    labels: list[set[int]]   # gold senses per usage (may be several)

    def targets(self) -> np.ndarray:
        n = len(self.labels)
        Y = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            li = self.labels[i]
            for j in range(i, n):
                if li & self.labels[j]:
                    Y[i, j] = Y[j, i] = 1.0
        return Y


def train_siamgnn(graphs: list[WordGraph], steps: int = 2000, lr: float = 3e-4, sub: int = 128,
                  seed: int = 0, verbose: bool = True, init: SiamGNN | None = None,
                  weights: list[float] | None = None, balance: bool = False, **model_kw) -> SiamGNN:
    """Episodic training: each step samples one word and a random subset of its usages."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = init if init is not None else SiamGNN(**model_kw)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    data = [(torch.tensor(g.X, dtype=torch.float32), torch.tensor(g.targets())) for g in graphs]
    p = None if weights is None else np.asarray(weights, dtype=float) / np.sum(weights)
    run = 0.0
    for step in range(steps):
        model.train()
        X, Y = data[rng.choice(len(data), p=p)]
        n = X.shape[0]
        idx = torch.tensor(rng.choice(n, size=min(sub, n), replace=False))
        Xs, Ys = X[idx], Y[idx][:, idx]
        logits = model.pair_logits(Xs)
        mask = ~torch.eye(len(idx), dtype=torch.bool)
        y = Ys[mask]
        if balance:  # class-balanced BCE (p=0.5 then means "balanced prior")
            pos = y.mean().clamp(0.02, 0.98)
            w = torch.where(y > 0.5, 0.5 / pos, 0.5 / (1 - pos))
        else:        # plain BCE: probabilities calibrated to the empirical same-sense prior
            w = None
        loss = F.binary_cross_entropy_with_logits(logits[mask], y, weight=w)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step else loss.item()
        if verbose and (step % 500 == 0 or step == steps - 1):
            print(f"  step {step:5d} loss {run:.4f}", flush=True)
    return model


@torch.no_grad()
def predict_pairs(model: SiamGNN, X: np.ndarray) -> np.ndarray:
    model.eval()
    return torch.sigmoid(model.pair_logits(torch.tensor(X, dtype=torch.float32))).numpy()


def save_model(model: SiamGNN, path, **meta) -> None:
    kw = dict(d_in=model.proj[1].in_features, d=model.proj[1].out_features, layers=len(model.mp) or 2,
              knn=model.knn, use_gnn=model.use_gnn, head=model.head)
    torch.save({"state": model.state_dict(), "kw": kw, "meta": meta}, path)


def load_model(path) -> SiamGNN:
    ck = torch.load(path, map_location="cpu")
    m = SiamGNN(**ck["kw"])
    m.load_state_dict(ck["state"])
    m.eval()
    return m
