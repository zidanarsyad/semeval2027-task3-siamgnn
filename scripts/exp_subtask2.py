"""Subtask 2 experiments on the Swedish dev set (official macro-F1).

Compares the existing method (gloss cosine + global threshold) against few-shot Siamese prototype
refinement, APPNP propagation over the usage graph and Subtask 1 cluster votes.  Because only 10
dev words exist, every tuned threshold is also evaluated with leave-one-word-out (LOWO): the
threshold is picked on 9 words and applied to the held-out one.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mtp.clustering import cluster  # noqa: E402
from mtp.data import load_language  # noqa: E402
from mtp.encoders import XLLexemeEncoder, get_embeddings, gloss_item  # noqa: E402
from mtp.scorer import score_subtask2  # noqa: E402
from mtp.subtask2 import cluster_vote, otsu_threshold, propagate, prototype_scores  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
ld = load_language(ROOT / "data/raw/traindev_phase/extracted/SV", "SV")
E_all = get_embeddings([(u.text, u.start, u.end) for u in ld.usages], tag="usages_SV")
index = {(u.word, u.sentence_id): i for i, u in enumerate(ld.usages)}
groups = ld.by_word()
words = list(groups)
gold = {w: np.array([ld.gold_s2[(u.word, u.sentence_id)] for u in groups[w]]) for w in words}
Ew = {w: E_all[[index[(u.word, u.sentence_id)] for u in groups[w]]] for w in words}

TEMPLATES = {"w: d": "{w}: {d}", "w (d)": "{w} ({d})", "w – d": "{w} – {d}"}
enc = None
G = {}
for name, tpl in TEMPLATES.items():
    items = [gloss_item(w, ld.definitions[w], tpl) for w in words]
    try:
        G[name] = get_embeddings(items, tag=f"defs_SV_{name.replace(' ', '').replace(':', 'c')}", encoder=enc)
    except Exception:
        enc = enc or XLLexemeEncoder()
        G[name] = get_embeddings(items, tag=f"defs_SV_{name.replace(' ', '').replace(':', 'c')}", encoder=enc)
# definition alone, the whole definition marked as target
items = [(ld.definitions[w], 0, len(ld.definitions[w])) for w in words]
G["d"] = get_embeddings(items, tag="defs_SV_donly", encoder=enc)


def f1w(y, p):
    return f1_score(y, p, average="macro", zero_division=0)


def evaluate(score_fn, name, taus=np.round(np.arange(0.1, 0.95, 0.025), 3), rows=None):
    """Global threshold (oracle-tuned and LOWO) for a per-word score function."""
    S = {w: score_fn(w) for w in words}
    table = np.array([[f1w(gold[w], (S[w] > t).astype(int)) for t in taus] for w in words])
    best = table.mean(0).argmax()
    lowo = []
    for i, w in enumerate(words):
        others = np.delete(table, i, axis=0).mean(0)
        lowo.append(table[i, others.argmax()])
    otsu = np.mean([f1w(gold[w], (S[w] > otsu_threshold(S[w])).astype(int)) for w in words])
    print(f"{name:52s} tuned={table.mean(0)[best]:.3f} (tau={taus[best]:.3f})  LOWO={np.mean(lowo):.3f}  otsu={otsu:.3f}")
    if rows is not None:
        rows.append({"system": name, "tuned": float(table.mean(0)[best]), "tau": float(taus[best]),
                     "lowo": float(np.mean(lowo)), "otsu": float(otsu)})
    return S, taus[best]


rows = []
print("== existing method: gloss cosine")
for name in G:
    gi = {w: G[name][i] for i, w in enumerate(words)}
    evaluate(lambda w: Ew[w] @ gi[w], f"cos(usage, gloss[{name}])", rows=rows)

best_tpl = max((r for r in rows), key=lambda r: r["lowo"])["system"].split("[")[1].rstrip("])")
gi = {w: G[best_tpl][i] for i, w in enumerate(words)}
print(f"\n(best template: {best_tpl})\n== few-shot prototype refinement")
for top in [0.05, 0.1, 0.2, 0.3]:
    for it in [1, 2, 3]:
        evaluate(lambda w: prototype_scores(Ew[w], gi[w], top=top, n_iter=it),
                 f"proto(top={top}, iter={it})", rows=rows)

print("\n== APPNP propagation of gloss scores")
for k in [5, 10, 20]:
    for a in [0.5, 0.8, 0.9]:
        evaluate(lambda w: propagate(Ew[w], Ew[w] @ gi[w], k=k, alpha=a), f"appnp(k={k}, a={a})", rows=rows)

print("\n== prototype + APPNP")
for top in [0.1, 0.2]:
    for a in [0.5, 0.8]:
        evaluate(lambda w: propagate(Ew[w], prototype_scores(Ew[w], gi[w], top=top, n_iter=2), k=10, alpha=a),
                 f"proto(top={top})+appnp(a={a})", rows=rows)

print("\n== cluster vote (clusters from Subtask 1, agglomerative t=0.5)")
for thr in [0.4, 0.5, 0.6]:
    cl = {w: cluster(Ew[w], "agglomerative", threshold=thr) for w in words}
    def vote_score(w, thr=thr):
        s = Ew[w] @ gi[w]
        c = cl[w]
        return np.array([s[c == c[i]].mean() for i in range(len(s))])
    evaluate(vote_score, f"cluster-mean gloss score (agglo t={thr})", rows=rows)

out = ROOT / "outputs" / "results"
(out / "exp_subtask2_dev_sv.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
