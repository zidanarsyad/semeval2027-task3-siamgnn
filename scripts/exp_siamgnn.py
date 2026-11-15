"""Train the Siamese-GNN usage-graph model and compare it with the existing clustering methods.

Training words: synthetic English TRAIN split (disjoint from every evaluation word).
Evaluation: synthetic English DEV split (40 words) and the official Swedish dev set (10 words),
both scored with the official SemEval-2027 Task 3 scorer.

  python scripts/exp_siamgnn.py --epochs 60
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mtp.clustering import agglomerative, agglomerative_sim, correlation_clustering  # noqa: E402
from mtp.data import load_language  # noqa: E402
from mtp.encoders import get_embeddings  # noqa: E402
from mtp.scorer import score_subtask1  # noqa: E402
from mtp.siamgnn import WordGraph, predict_pairs, train_siamgnn  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--knn", type=int, default=10)
ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--seeds", type=int, nargs="*", default=[0])
ap.add_argument("--save", type=Path, default=ROOT / "models" / "siamgnn.pt")
args = ap.parse_args()
torch.set_num_threads(4)


def load(folder, lang, tag):
    ld = load_language(folder, lang)
    E = get_embeddings([(u.text, u.start, u.end) for u in ld.usages], tag=tag)
    idx = {(u.word, u.sentence_id): i for i, u in enumerate(ld.usages)}
    groups = {w: (us, E[[idx[(u.word, u.sentence_id)] for u in us]]) for w, us in ld.by_word().items()}
    return ld, groups


train_ld, train_g = load(ROOT / "data/synthetic_en/TRAIN/EN", "EN", "usages_synEN_TRAIN")
dev_en, dev_en_g = load(ROOT / "data/synthetic_en/DEV/EN", "EN", "usages_synEN_DEV")
dev_sv, dev_sv_g = load(ROOT / "data/raw/traindev_phase/extracted/SV", "SV", "usages_SV")
graphs = [WordGraph(w, X, [set(train_ld.gold_s1[(u.word, u.sentence_id)]) for u in us])
          for w, (us, X) in train_g.items()]


def labels_from(groups, fn):
    out = {}
    for w, (us, X) in groups.items():
        lab = fn(X)
        for u, l in zip(us, lab):
            out[(u.word, u.sentence_id)] = [int(l)]
    return out


def report(name, fn, rows):
    res = {}
    for set_name, ld, groups in [("synEN-dev", dev_en, dev_en_g), ("SV-dev", dev_sv, dev_sv_g)]:
        r = score_subtask1(ld, labels_from(groups, fn))
        res[set_name] = (r["bcubed_f"], r["spearman"], np.mean([v["k_pred"] for v in r["per_word"].values()]))
    print(f"{name:46s} " + "  ".join(f"{k}: B3={v[0]:.3f} rho={v[1]:.3f} k={v[2]:.1f}" for k, v in res.items()), flush=True)
    rows.append({"system": name, **{f"{k}_{m}": float(v[i]) for k, v in res.items()
                                     for i, m in enumerate(["b3", "rho", "k"])}})


rows = []
# ---- existing methods, thresholds chosen on the TRAIN words (never on evaluation words)
best = {}
for method in ["agglomerative", "cc"]:
    scores = {}
    for t in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
        fn = (lambda X, t=t: agglomerative(X, threshold=t)) if method == "agglomerative" else \
             (lambda X, t=t: correlation_clustering(X @ X.T, threshold=t, n_restarts=5))
        r = score_subtask1(train_ld, labels_from(train_g, fn))
        scores[t] = (r["bcubed_f"] + r["spearman"]) / 2
    best[method] = max(scores, key=scores.get)
    print(f"{method}: threshold picked on TRAIN = {best[method]}  (train mean={scores[best[method]]:.3f})")
report("1C1W (one cluster)", lambda X: np.zeros(len(X), dtype=int), rows)
report(f"XL-LEXEME + agglomerative (t={best['agglomerative']})",
       lambda X: agglomerative(X, threshold=best["agglomerative"]), rows)
report(f"XL-LEXEME + correlation clustering (t={best['cc']})",
       lambda X: correlation_clustering(X @ X.T, threshold=best["cc"], n_restarts=5), rows)

# ---- trained models
for use_gnn in [False, True]:
    for seed in args.seeds:
        t0 = time.time()
        name = ("SiamGNN" if use_gnn else "Siamese head (no GNN)") + f" seed={seed}"
        print(f"training {name} ...", flush=True)
        model = train_siamgnn(graphs, epochs=args.epochs, seed=seed, knn=args.knn, layers=args.layers,
                              use_gnn=use_gnn, verbose=False)
        print(f"  trained in {time.time() - t0:.0f}s", flush=True)
        report(name + " + CC@0.5",
               lambda X, m=model: correlation_clustering(predict_pairs(m, X), threshold=0.5, n_restarts=5), rows)
        report(name + " + agglo@0.5",
               lambda X, m=model: agglomerative_sim(predict_pairs(m, X), threshold=0.5), rows)
        if use_gnn and seed == args.seeds[0]:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state": model.state_dict(), "kw": dict(knn=args.knn, layers=args.layers)}, args.save)

(ROOT / "outputs/results/exp_siamgnn.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
