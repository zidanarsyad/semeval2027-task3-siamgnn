"""Seed ensemble of the trained SiamGNN models: average p(same sense) over seeds, then correlation clustering.

The clustering threshold theta is re-selected for the ensemble on the same 12 held-out synthetic-English
training words used in exp_main.py (never on evaluation words).
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mtp.clustering import correlation_clustering  # noqa: E402
from mtp.data import load_language  # noqa: E402
from mtp.encoders import get_embeddings  # noqa: E402
from mtp.scorer import score_subtask1  # noqa: E402
from mtp.siamgnn import load_model, predict_pairs  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def load(folder, lang, tag):
    ld = load_language(folder, lang)
    E = get_embeddings([(u.text, u.start, u.end) for u in ld.usages], tag=tag)
    idx = {(u.word, u.sentence_id): i for i, u in enumerate(ld.usages)}
    return ld, {w: (us, E[[idx[(u.word, u.sentence_id)] for u in us]]) for w, us in ld.by_word().items()}


tr_ld, tr_g = load(ROOT / "data/synthetic_en/TRAIN_SUB/EN", "EN", "usages_synEN_TRAIN")
en_ld, en_g = load(ROOT / "data/synthetic_en/DEV20_SUB/EN", "EN", "usages_synEN_DEV")
sv_ld, sv_g = load(ROOT / "data/raw/traindev_phase/extracted/SV", "SV", "usages_SV")
perm = np.random.default_rng(123).permutation(len(tr_g))
val_words = {list(tr_g)[i] for i in perm[:12]}
val_ld = copy.copy(tr_ld)
val_ld.usages = [u for u in tr_ld.usages if u.word in val_words]
val_ld.gold_s1 = {k: v for k, v in tr_ld.gold_s1.items() if k[0] in val_words}
val_g = {w: v for w, v in tr_g.items() if w in val_words}

out = {}
for kind in ["siamese", "siamgnn"]:
    models = [load_model(p) for p in sorted((ROOT / "models").glob(f"{kind}_synEN_seed*.pt"))]
    if not models:
        continue
    def P_of(X):
        return np.mean([predict_pairs(m, X) for m in models], axis=0)
    Pc = {name: {w: P_of(X) for w, (us, X) in g.items()} for name, g in [("val", val_g), ("en", en_g), ("sv", sv_g)]}

    def run(ld, groups, Pd, th):
        lab = {(u.word, u.sentence_id): [int(l)] for w, (us, X) in groups.items()
               for u, l in zip(us, correlation_clustering(Pd[w], threshold=th, n_restarts=10))}
        r = score_subtask1(ld, lab)
        return r["bcubed_f"], r["spearman"], float(np.mean([v["k_pred"] for v in r["per_word"].values()]))

    ths = [0.2, 0.3, 0.4, 0.5, 0.6]
    th = max(ths, key=lambda t: sum(run(val_ld, val_g, Pc["val"], t)[:2]))
    en, sv = run(en_ld, en_g, Pc["en"], th), run(sv_ld, sv_g, Pc["sv"], th)
    out[kind] = {"theta": th, "n_models": len(models), "synEN": en, "SV": sv}
    print(f"{kind} ensemble of {len(models)} (theta={th}): synEN B3={en[0]:.3f} rho={en[1]:.3f} k={en[2]:.1f} | "
          f"SV B3={sv[0]:.3f} rho={sv[1]:.3f} k={sv[2]:.1f}", flush=True)
(ROOT / "outputs/results/exp_ensemble.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
