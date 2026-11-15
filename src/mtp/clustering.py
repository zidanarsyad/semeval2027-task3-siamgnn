"""Word sense induction over all periods of one target word.

Existing methods (baselines from the LSC literature):
  * agglomerative clustering, average linkage on cosine distance (e.g. Deep-change @ AXOLOTL'24)
  * Affinity Propagation (Martinc et al., 2020; Periti et al., 2022)
  * K-means with silhouette model selection (Giulianelli et al., 2020)

Graph methods (this project):
  * correlation clustering of the Siamese usage graph (the procedure used to build DWUG gold
    clusters, Schlechtweg et al., 2021), solved with pivot initialisation + local search
  * DMoN - Deep Modularity Networks (Tsitsulin et al., 2023): a GCN encoder over the usage
    graph trained, per word and without labels, to maximise modularity; empty clusters are
    allowed so the number of senses is inferred
  * graph smoothing (SGC-style propagation of embeddings over the kNN usage graph) that can be
    put in front of any clustering algorithm
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import AffinityPropagation, KMeans
from sklearn.metrics import silhouette_score


# ----------------------------------------------------------------------------- helpers

def relabel(labels: np.ndarray) -> np.ndarray:
    """Map arbitrary cluster ids to 0..k-1 ordered by decreasing cluster size."""
    ids, counts = np.unique(labels, return_counts=True)
    order = ids[np.argsort(-counts, kind="stable")]
    remap = {c: i for i, c in enumerate(order)}
    return np.array([remap[c] for c in labels])


def merge_small(E: np.ndarray, labels: np.ndarray, min_size: int) -> np.ndarray:
    """Re-assign members of clusters smaller than ``min_size`` to the most similar large cluster."""
    labels = relabel(labels)
    sizes = np.bincount(labels)
    big = [c for c in range(len(sizes)) if sizes[c] >= min_size]
    if not big:
        return np.zeros_like(labels)
    cents = np.stack([E[labels == c].mean(0) for c in big])
    cents /= np.linalg.norm(cents, axis=1, keepdims=True)
    out = labels.copy()
    for i in np.where(sizes[labels] < min_size)[0]:
        out[i] = big[int(np.argmax(cents @ E[i]))]
    return relabel(out)


def knn_graph(S: np.ndarray, k: int = 10, mutual: bool = False) -> np.ndarray:
    """Symmetric weighted kNN adjacency from a similarity matrix (no self loops)."""
    n = len(S)
    k = min(k, n - 1)
    A = np.zeros_like(S)
    nn = np.argsort(-S + np.eye(n) * 1e9, axis=1)[:, :k]
    rows = np.repeat(np.arange(n), k)
    A[rows, nn.ravel()] = np.clip(S[rows, nn.ravel()], 0, None)
    A = np.minimum(A, A.T) if mutual else np.maximum(A, A.T)
    np.fill_diagonal(A, 0)
    return A


def smooth(E: np.ndarray, k: int = 10, steps: int = 2, alpha: float = 0.5) -> np.ndarray:
    """SGC/APPNP-style propagation: X <- (1-a) X0 + a * D^-1/2 (A+I) D^-1/2 X."""
    S = E @ E.T
    A = knn_graph(S, k) + np.eye(len(E))
    d = A.sum(1)
    An = A / np.sqrt(np.outer(d, d))
    X = E.copy()
    for _ in range(steps):
        X = (1 - alpha) * E + alpha * An @ X
    return X / np.linalg.norm(X, axis=1, keepdims=True)


# ----------------------------------------------------------------------------- existing methods

def agglomerative_sim(S: np.ndarray, threshold: float = 0.5, method: str = "average") -> np.ndarray:
    """Agglomerative clustering on a similarity matrix; merging stops at distance 1 - S > threshold."""
    if len(S) < 2:
        return np.zeros(len(S), dtype=int)
    D = np.clip(1 - (S + S.T) / 2, 0, 2)
    np.fill_diagonal(D, 0)
    Z = linkage(squareform(D, checks=False), method=method)
    return relabel(fcluster(Z, t=threshold, criterion="distance") - 1)


def agglomerative(E: np.ndarray, threshold: float = 0.5, method: str = "average") -> np.ndarray:
    return agglomerative_sim(E @ E.T, threshold, method)


def affinity_propagation(E: np.ndarray, preference: float | None = None, damping: float = 0.9,
                         seed: int = 0) -> np.ndarray:
    S = E @ E.T
    ap = AffinityPropagation(affinity="precomputed", damping=damping, preference=preference,
                             random_state=seed, max_iter=1000, convergence_iter=50)
    lab = ap.fit_predict(S)
    if (lab < 0).any():  # did not converge
        lab = np.zeros(len(E), dtype=int)
    return relabel(lab)


def kmeans_silhouette(E: np.ndarray, kmax: int = 8, seed: int = 0) -> np.ndarray:
    best, best_s = np.zeros(len(E), dtype=int), -1.0
    for k in range(2, min(kmax, len(E) - 1) + 1):
        lab = KMeans(k, n_init=10, random_state=seed).fit_predict(E)
        s = silhouette_score(E, lab, metric="cosine")
        if s > best_s:
            best, best_s = lab, s
    return relabel(best)


# ----------------------------------------------------------------------------- correlation clustering

def correlation_clustering(S: np.ndarray, threshold: float = 0.5, n_restarts: int = 20,
                           max_sweeps: int = 50, seed: int = 0) -> np.ndarray:
    """Maximise sum_{i<j same cluster} (S_ij - threshold): positive edges attract, negative repel.

    Pivot (KwikCluster) initialisation followed by greedy single-node moves; best of
    ``n_restarts`` random orders.  This is the objective used for DWUG gold sense clusters.
    """
    n = len(S)
    if n < 2:
        return np.zeros(n, dtype=int)
    W = (S + S.T) / 2 - threshold
    np.fill_diagonal(W, 0)
    rng = np.random.default_rng(seed)
    best_lab, best_obj = None, -np.inf
    for _ in range(n_restarts):
        # pivot initialisation (KwikCluster)
        lab = -np.ones(n, dtype=int)
        c = 0
        for p in rng.permutation(n):
            if lab[p] >= 0:
                continue
            members = np.where((lab < 0) & (W[p] > 0))[0]
            lab[p] = c
            lab[members] = c
            c += 1
        # local search with incremental node-to-cluster weights M[i, q] = sum_{j in q} W[i, j]
        K = n + 1  # enough slots for all singletons
        M = np.zeros((n, K))
        np.add.at(M.T, lab, W)  # M[:, q] += W[:, j] for every j with lab[j] = q
        sizes = np.bincount(lab, minlength=K)
        for _ in range(max_sweeps):
            moved = 0
            for i in rng.permutation(n):
                a = lab[i]
                gains = M[i].copy()
                empty = np.where(sizes == 0)[0]
                if len(empty):
                    gains[empty] = -np.inf
                    gains[empty[0]] = 0.0  # move to a new singleton cluster
                b = int(np.argmax(gains))
                if b != a and gains[b] > M[i, a] + 1e-12:
                    M[:, a] -= W[:, i]
                    M[:, b] += W[:, i]
                    sizes[a] -= 1
                    sizes[b] += 1
                    lab[i] = b
                    moved += 1
            if not moved:
                break
        obj = sum(M[lab == q, q].sum() for q in np.unique(lab)) / 2
        if obj > best_obj:
            best_obj, best_lab = obj, lab.copy()
    return relabel(best_lab)


# ----------------------------------------------------------------------------- DMoN (GNN)

class _GCN(torch.nn.Module):
    def __init__(self, d_in: int, d_hid: int, k: int, dropout: float):
        super().__init__()
        self.lin1 = torch.nn.Linear(d_in, d_hid)
        self.lin2 = torch.nn.Linear(d_hid, k)
        self.drop = torch.nn.Dropout(dropout)

    def forward(self, X, An):
        h = torch.selu(An @ self.lin1(self.drop(X)))
        return torch.softmax(An @ self.lin2(self.drop(h)), dim=-1)


def dmon(E: np.ndarray, k_max: int = 8, knn: int = 10, hidden: int = 64, epochs: int = 300,
         lr: float = 1e-3, collapse: float = 1.0, dropout: float = 0.2, seed: int = 0,
         A: np.ndarray | None = None) -> np.ndarray:
    """Deep Modularity Networks: unsupervised GCN graph clustering (Tsitsulin et al., JMLR 2023)."""
    torch.manual_seed(seed)
    n = len(E)
    if n < 3:
        return np.zeros(n, dtype=int)
    if A is None:
        A = knn_graph(E @ E.T, knn)
    A_t = torch.tensor(A, dtype=torch.float32)
    Ah = A_t + torch.eye(n)
    d = Ah.sum(1)
    An = Ah / torch.sqrt(torch.outer(d, d))
    deg = A_t.sum(1)
    m2 = deg.sum()  # = 2m
    X = torch.tensor(E, dtype=torch.float32)
    net = _GCN(X.shape[1], hidden, k_max, dropout)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(epochs):
        net.train()
        C = net(X, An)
        # modularity: Tr(C^T B C) / 2m with B = A - d d^T / 2m
        CA = C.T @ A_t @ C
        Cd = C.T @ deg
        Q = (torch.trace(CA) - (Cd @ Cd) / m2) / m2
        reg = torch.linalg.norm(C.sum(0)) / n * np.sqrt(k_max) - 1
        loss = -Q + collapse * reg
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        lab = net(X, An).argmax(1).numpy()
    return relabel(lab)


# ----------------------------------------------------------------------------- dispatcher

def cluster(E: np.ndarray, method: str, **kw) -> np.ndarray:
    pre = kw.pop("smooth", None)
    min_size = kw.pop("min_size", 0)
    X = smooth(E, **pre) if pre else E
    if method == "agglomerative":
        lab = agglomerative(X, **kw)
    elif method == "ap":
        lab = affinity_propagation(X, **kw)
    elif method == "kmeans":
        lab = kmeans_silhouette(X, **kw)
    elif method == "cc":
        lab = correlation_clustering(X @ X.T, **kw)
    elif method == "dmon":
        lab = dmon(X, **kw)
    elif method == "single":
        lab = np.zeros(len(E), dtype=int)
    else:
        raise ValueError(method)
    if min_size:
        lab = merge_small(E, lab, min_size)
    return lab
