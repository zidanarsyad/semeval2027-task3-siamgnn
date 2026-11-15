"""Parameter-free graph convolution (SGC / APPNP-style smoothing) in front of the existing clustering.

Same protocols as exp_main.py: threshold tuned on the synthetic-English TRAIN words ("synEN-tune")
and leave-one-word-out on Swedish ("SV-LOWO").  Results are appended to outputs/results/exp_graph.json.
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mtp.clustering import agglomerative, smooth  # noqa: E402
from mtp.data import load_language  # noqa: E402
from mtp.encoders import get_embeddings  # noqa: E402
from mtp.scorer import score_subtask1  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def load(folder, lang, tag):
    ld = load_language(folder, lang)
    E = get_embeddings([(u.text, u.start, u.end) for u in ld.usages], tag=tag)
    idx = {(u.word, u.sentence_id): i for i, u in enumerate(ld.usages)}
    return ld, {w: (us, E[[idx[(u.word, u.sentence_id)] for u in us]]) for w, us in ld.by_word().items()}


tr_ld, tr_g = load(ROOT / "data/synthetic_en/TRAIN_SUB/EN", "EN", "usages_synEN_TRAIN")
en_ld, en_g = load(ROOT / "data/synthetic_en/DEV20_SUB/EN", "EN", "usages_synEN_DEV")
sv_ld, sv_g = load(ROOT / "data/raw/traindev_phase/extracted/SV", "SV", "usages_SV")


def labels_of(groups, fn):
    return {(u.word, u.sentence_id): [int(l)] for w, (us, X) in groups.items() for u, l in zip(us, fn(X))}


def score(ld, lab):
    r = score_subtask1(ld, lab)
    return r["bcubed_f"], r["spearman"], float(np.mean([v["k_pred"] for v in r["per_word"].values()]))


GRID = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
rows = []
for name, sm in [("XL-LEXEME + agglomerative", None),
                 ("SGC smoothing (k=5, 2 hops) + agglomerative", dict(k=5, steps=2, alpha=0.5)),
                 ("SGC smoothing (k=10, 2 hops) + agglomerative", dict(k=10, steps=2, alpha=0.5))]:
    def fn(X, t, sm=sm):
        return agglomerative(smooth(X, **sm) if sm else X, threshold=t)
    # synEN-tune
    best_t = max(GRID, key=lambda t: sum(score(tr_ld, labels_of(tr_g, lambda X: fn(X, t)))[:2]))
    en = score(en_ld, labels_of(en_g, lambda X: fn(X, best_t)))
    sv = score(sv_ld, labels_of(sv_g, lambda X: fn(X, best_t)))
    rows.append({"system": name, "protocol": "synEN-tune", "t": best_t, "synEN_b3": en[0], "synEN_rho": en[1],
                 "synEN_k": en[2], "SV_b3": sv[0], "SV_rho": sv[1], "SV_k": sv[2]})
    print(f"{name:46s} [synEN-tune t={best_t}] synEN B3={en[0]:.3f} rho={en[1]:.3f} | SV B3={sv[0]:.3f} rho={sv[1]:.3f} k={sv[2]:.1f}", flush=True)
    # SV-LOWO
    cache = {t: {w: fn(X, t) for w, (us, X) in sv_g.items()} for t in GRID}
    lab = {}
    for w in sv_g:
        o = copy.copy(sv_ld)
        o.usages = [u for u in sv_ld.usages if u.word != w]
        o.gold_s1 = {k: v for k, v in sv_ld.gold_s1.items() if k[0] != w}
        others = {k: v for k, v in sv_g.items() if k != w}
        t = max(GRID, key=lambda t: sum(score(o, {(u.word, u.sentence_id): [int(c)] for ww, (us, X) in others.items()
                                                  for u, c in zip(us, cache[t][ww])})[:2]))
        lab.update({(u.word, u.sentence_id): [int(c)] for u, c in zip(sv_g[w][0], cache[t][w])})
    sv = score(sv_ld, lab)
    rows.append({"system": name, "protocol": "SV-LOWO", "SV_b3": sv[0], "SV_rho": sv[1], "SV_k": sv[2]})
    print(f"{name:46s} [SV-LOWO] SV B3={sv[0]:.3f} rho={sv[1]:.3f} k={sv[2]:.1f}", flush=True)

(ROOT / "outputs/results/exp_graph.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
