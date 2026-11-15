"""Check the correlation-clustering solver against brute force on small random usage graphs.

For each of 20 random graphs with 8 nodes (4,140 set partitions), compare the objective
sum_{i<j, same cluster} (S_ij - threshold) reached by mtp.clustering.correlation_clustering
(10 restarts, as in the experiments) with the exact optimum found by enumerating all partitions.

    python scripts/test_cc_solver.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mtp.clustering import correlation_clustering  # noqa: E402


def partitions(n):
    """All set partitions of range(n) as restricted-growth label arrays."""
    labels = [0] * n

    def rec(i, k):
        if i == n:
            yield list(labels)
            return
        for c in range(k + 1):
            labels[i] = c
            yield from rec(i + 1, max(k, c + 1))
    yield from rec(1, 1) if n > 1 else iter([[0]])


def objective(W, lab):
    lab = np.asarray(lab)
    same = lab[:, None] == lab[None, :]
    return float(W[np.triu(same, 1)].sum())


def main(n=8, graphs=20, threshold=0.5, seed=0):
    rng = np.random.default_rng(seed)
    parts = list(partitions(n))
    hits = 0
    for g in range(graphs):
        # similarities from noisy points around 2-4 latent senses, mapped to [0, 1]
        k = rng.integers(2, 5)
        centres = rng.normal(size=(k, 8))
        X = centres[rng.integers(0, k, n)] + rng.normal(scale=0.9, size=(n, 8))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        S = (X @ X.T + 1) / 2
        W = S - threshold
        np.fill_diagonal(W, 0)
        best = max(objective(W, p) for p in parts)
        got = objective(W, correlation_clustering(S, threshold=threshold, n_restarts=10, seed=g))
        ok = got >= best - 1e-9
        hits += ok
        print(f"graph {g:2d}: solver {got:.4f}  optimum {best:.4f}  {'OK' if ok else 'SUBOPTIMAL'}")
    print(f"optimum reached on {hits}/{graphs} graphs ({len(parts)} partitions each)")
    return hits == graphs


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
