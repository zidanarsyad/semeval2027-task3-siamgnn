"""Subtask 2 - hypothesis-driven change detection as target-sense verification (TSV).

Given one definition per word, decide for every usage whether it expresses that sense.

Existing method (baseline): cosine similarity between the XL-LEXEME embedding of the usage and of
the definition turned into a pseudo-usage ("word: definition"), thresholded globally.

Few-shot Siamese + graph improvements (this project):
  * prototype refinement - the definition is a 1-shot support example; the most similar usages are
    added to the support set (transductive few-shot / prototypical network), which turns the
    gloss-usage comparison into usage-usage comparisons the WiC-trained Siamese encoder is good at
  * APPNP label propagation of the scores over the kNN usage graph
  * cluster vote - Subtask 1 sense clusters are labelled as a whole
"""
from __future__ import annotations

import numpy as np

from .clustering import knn_graph


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-9)


def propagate(E: np.ndarray, s0: np.ndarray, k: int = 10, alpha: float = 0.5, steps: int = 10) -> np.ndarray:
    """APPNP / personalised PageRank: s <- (1-a) s0 + a * D^-1/2 A D^-1/2 s (Gasteiger et al., 2019)."""
    A = knn_graph(E @ E.T, k) + np.eye(len(E))
    d = A.sum(1)
    An = A / np.sqrt(np.outer(d, d))
    s = s0.copy()
    for _ in range(steps):
        s = (1 - alpha) * s0 + alpha * An @ s
    return s


def prototype_scores(E: np.ndarray, g: np.ndarray, top: float = 0.2, n_iter: int = 2,
                     gloss_weight: float = 1.0, min_support: int = 3) -> np.ndarray:
    """Transductive prototypical refinement starting from the gloss embedding ``g``.

    Iteratively: support = gloss + the ``top`` fraction of usages most similar to the current
    prototype; prototype = normalised (weighted) mean of the support.  Returns cosine scores
    of all usages to the final prototype.
    """
    p = g / np.linalg.norm(g)
    n = len(E)
    m = max(min_support, int(round(top * n)))
    for _ in range(n_iter):
        s = E @ p
        sup = np.argsort(-s)[:m]
        p = gloss_weight * g / np.linalg.norm(g) + E[sup].sum(0)
        p /= np.linalg.norm(p)
    return E @ p


def siamgnn_support_scores(P: np.ndarray, s0: np.ndarray, top: float = 0.05, min_support: int = 3) -> np.ndarray:
    """Few-shot Siamese decision: mean learned same-sense probability to the pseudo-support set.

    ``P`` is the SiamGNN pair-probability matrix of the word's usages and ``s0`` the gloss
    similarities; the support set is the ``top`` fraction of usages closest to the definition.
    """
    m = max(min_support, int(round(top * len(s0))))
    sup = np.argsort(-s0)[:m]
    return P[:, sup].mean(1)


def word_features(X: np.ndarray, g: np.ndarray, P: np.ndarray | None = None) -> np.ndarray:
    """Per-usage features for the few-shot calibrator (all computed without labels)."""
    s = X @ g
    feats = [s, zscore(s), np.argsort(np.argsort(s)) / max(1, len(s) - 1),
             prototype_scores(X, g, top=0.05, n_iter=3), propagate(X, s, k=5, alpha=0.5)]
    if P is not None:
        feats.append(siamgnn_support_scores(P, s))
    return np.stack(feats, 1)


def otsu_threshold(s: np.ndarray, bins: int = 64) -> float:
    """Threshold maximising between-class variance of a 1-D score distribution."""
    hist, edges = np.histogram(s, bins=bins)
    mids = (edges[:-1] + edges[1:]) / 2
    w = hist / hist.sum()
    best_t, best_v = mids[len(mids) // 2], -1.0
    for i in range(1, bins):
        w0, w1 = w[:i].sum(), w[i:].sum()
        if w0 == 0 or w1 == 0:
            continue
        m0, m1 = (w[:i] * mids[:i]).sum() / w0, (w[i:] * mids[i:]).sum() / w1
        v = w0 * w1 * (m0 - m1) ** 2
        if v > best_v:
            best_v, best_t = v, edges[i]
    return float(best_t)


def cluster_vote(cluster_labels: np.ndarray, s: np.ndarray, tau: float, mode: str = "mean") -> np.ndarray:
    """Label each Subtask 1 cluster as a whole: positive iff its mean (or median) score > tau."""
    out = np.zeros(len(s), dtype=int)
    for c in np.unique(cluster_labels):
        idx = cluster_labels == c
        stat = np.median(s[idx]) if mode == "median" else s[idx].mean()
        out[idx] = int(stat > tau)
    return out
