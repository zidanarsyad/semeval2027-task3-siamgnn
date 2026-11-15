"""Final Subtask 2 comparison on the Swedish dev set (official per-word macro-F1, LOWO protocol).

Systems
  1. existing: cos(usage, gloss) + global threshold (threshold picked on the other 9 words)
  2. few-shot prototype refinement (+ LOWO threshold)
  3. APPNP propagation over the usage graph (+ LOWO threshold)
  4. SiamGNN support probability (learned same-sense probability to the gloss' nearest usages), p > 0.5
  5. few-shot calibrator: logistic regression over the label-free features of 1-4, trained on 9 words
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mtp.data import load_language  # noqa: E402
from mtp.encoders import get_embeddings, gloss_item, sentence_embeddings  # noqa: E402
from mtp.scorer import score_subtask2  # noqa: E402
from mtp.siamgnn import load_model, predict_pairs  # noqa: E402
from mtp.subtask2 import propagate, prototype_scores, siamgnn_support_scores, word_features  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
ld = load_language(ROOT / "data/raw/traindev_phase/extracted/SV", "SV")
E = get_embeddings([(u.text, u.start, u.end) for u in ld.usages], tag="usages_SV")
idx = {(u.word, u.sentence_id): i for i, u in enumerate(ld.usages)}
groups = ld.by_word()
words = list(groups)
G = get_embeddings([gloss_item(w, ld.definitions[w], "{w}: {d}") for w in words], tag="defs_SV_wcd")
X = {w: E[[idx[(u.word, u.sentence_id)] for u in groups[w]]] for w in words}
g = {w: G[i] for i, w in enumerate(words)}
y = {w: np.array([ld.gold_s2[(u.word, u.sentence_id)] for u in groups[w]]) for w in words}

# sentence-level Transformer (multilingual-E5): whole usage sentence vs. definition text
SE = sentence_embeddings([u.text for u in ld.usages], tag="SV_usages")
SG = sentence_embeddings([f"{w}: {ld.definitions[w]}" for w in words], tag="SV_defs")
XS = {w: SE[[idx[(u.word, u.sentence_id)] for u in groups[w]]] for w in words}
gs = {w: SG[i] for i, w in enumerate(words)}

models = sorted((ROOT / "models").glob("siamgnn_synEN_seed*.pt"))
P = {}
if models:
    m = load_model(models[0])
    P = {w: predict_pairs(m, X[w]) for w in words}
    print("using", models[0].name)

TAUS = np.round(np.arange(0.0, 1.0, 0.0125), 4)


def lowo_global(score_fn):
    S = {w: score_fn(w) for w in words}
    table = {w: np.array([f1_score(y[w], (S[w] > t).astype(int), average="macro", zero_division=0) for t in TAUS])
             for w in words}
    labels, picked = {}, []
    for w in words:
        t = TAUS[np.mean([table[o] for o in words if o != w], axis=0).argmax()]
        picked.append(t)
        for u, p in zip(groups[w], (S[w] > t).astype(int)):
            labels[(u.word, u.sentence_id)] = int(p)
    return labels, S, float(np.median(picked))


def fixed(score_fn, t):
    labels = {}
    for w in words:
        for u, p in zip(groups[w], (score_fn(w) > t).astype(int)):
            labels[(u.word, u.sentence_id)] = int(p)
    return labels


rows = []


def report(name, labels, S=None, note=""):
    r = score_subtask2(ld, labels)
    auc = np.mean([roc_auc_score(y[w], S[w]) for w in words]) if S is not None else float("nan")
    rows.append({"system": name, "macro_f1": r["macro_f1"], "mean_auc": float(auc),
                 "per_word": {w: v["F1"] for w, v in r["per_word"].items()}, "note": note})
    print(f"{name:52s} macroF1={r['macro_f1']:.3f}  meanAUC={auc:.3f} {note}", flush=True)


lab, S, t = lowo_global(lambda w: X[w] @ g[w])
report("existing: gloss cosine + threshold", lab, S, f"(median tau={t:.3f})")
lab, S, t = lowo_global(lambda w: prototype_scores(X[w], g[w], top=0.05, n_iter=3))
report("few-shot prototype refinement", lab, S, f"(median tau={t:.3f})")
lab, S, t = lowo_global(lambda w: propagate(X[w], X[w] @ g[w], k=5, alpha=0.5))
report("APPNP propagation on usage graph", lab, S, f"(median tau={t:.3f})")
lab, S, t = lowo_global(lambda w: XS[w] @ gs[w])
report("sentence Transformer (E5) cosine + threshold", lab, S, f"(median tau={t:.3f})")
lab, S, t = lowo_global(lambda w: 0.5 * (X[w] @ g[w]) + 0.5 * (XS[w] @ gs[w]))
report("XL-LEXEME + E5 cosine average", lab, S, f"(median tau={t:.3f})")
if P:
    sfn = lambda w: siamgnn_support_scores(P[w], X[w] @ g[w], top=0.05)  # noqa: E731
    report("SiamGNN support probability, p>0.5 (no tuning)", fixed(sfn, 0.5), {w: sfn(w) for w in words})
    lab, S, t = lowo_global(sfn)
    report("SiamGNN support probability + LOWO threshold", lab, S, f"(median tau={t:.3f})")

# few-shot calibrator (logistic regression, LOWO)
F = {w: np.concatenate([word_features(X[w], g[w], P.get(w)),
                        np.stack([XS[w] @ gs[w], (XS[w] @ gs[w] - (XS[w] @ gs[w]).mean()) / ((XS[w] @ gs[w]).std() + 1e-9)], 1)], 1)
     for w in words}
labels, S = {}, {}
for w in words:
    Xtr = np.concatenate([F[o] for o in words if o != w])
    ytr = np.concatenate([y[o] for o in words if o != w])
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(Xtr, ytr)
    S[w] = clf.predict_proba(F[w])[:, 1]
    for u, p in zip(groups[w], (S[w] > 0.5).astype(int)):
        labels[(u.word, u.sentence_id)] = int(p)
report("few-shot calibrator (LogReg on label-free features)", labels, S)
cal = dict(S)
lab, S2, t = lowo_global(lambda w: cal[w])
report("few-shot calibrator + LOWO threshold", lab, S2, f"(median tau={t:.3f})")

(ROOT / "outputs/results/exp_s2_final.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
