"""Grid of word-sense-induction systems on the Swedish dev set (official scorer).

Usage:  python scripts/exp_subtask1.py [--quick]
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mtp.clustering import cluster  # noqa: E402
from mtp.data import load_language  # noqa: E402
from mtp.encoders import get_embeddings  # noqa: E402
from mtp.scorer import score_subtask1  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
args = ap.parse_args()

ld = load_language(ROOT / "data/raw/traindev_phase/extracted/SV", "SV")
E_all = get_embeddings([(u.text, u.start, u.end) for u in ld.usages], tag="usages_SV")
index = {(u.word, u.sentence_id): i for i, u in enumerate(ld.usages)}
groups = ld.by_word()


def run(method, **kw):
    labels = {}
    for w, us in groups.items():
        idx = [index[(u.word, u.sentence_id)] for u in us]
        lab = cluster(E_all[idx], method, **dict(kw))
        for u, l in zip(us, lab):
            labels[(u.word, u.sentence_id)] = [int(l)]
    return labels


grid = []
for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
    for ms in [0, 3, 5]:
        grid.append(("agglomerative", dict(threshold=t, min_size=ms)))
for p in [None, -5, -10, -20, -40]:
    grid.append(("ap", dict(preference=p, min_size=0)))
    grid.append(("ap", dict(preference=p, min_size=5)))
grid.append(("kmeans", dict(kmax=8)))
grid.append(("kmeans", dict(kmax=8, min_size=5)))
for t in [0.4, 0.5, 0.6, 0.7]:
    for ms in [0, 3, 5]:
        grid.append(("cc", dict(threshold=t, min_size=ms)))
for col in [0.5, 1.0, 2.0]:
    for kn in [10, 20]:
        grid.append(("dmon", dict(k_max=8, knn=kn, collapse=col, min_size=0)))
        grid.append(("dmon", dict(k_max=8, knn=kn, collapse=col, min_size=5)))
if args.quick:
    grid = grid[::4]

rows = []
for method, kw in grid:
    t = time.time()
    r = score_subtask1(ld, run(method, **kw))
    k_mean = np.mean([v["k_pred"] for v in r["per_word"].values()])
    rows.append({"method": method, "params": kw, "bcubed_f": r["bcubed_f"], "spearman": r["spearman"],
                 "k_mean": k_mean, "avg": (r["bcubed_f"] + r["spearman"]) / 2})
    print(f"{method:14s} {json.dumps(kw):55s} B3={r['bcubed_f']:.3f} rho={r['spearman']:.3f} "
          f"k={k_mean:.1f}  ({time.time()-t:.1f}s)", flush=True)

out = ROOT / "outputs" / "results"
out.mkdir(parents=True, exist_ok=True)
(out / "exp_subtask1_dev_sv.json").write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
best = sorted(rows, key=lambda r: -r["avg"])[:8]
print("\nTop by mean(B3, rho):")
for r in best:
    print(f"  {r['method']:14s} {json.dumps(r['params']):55s} B3={r['bcubed_f']:.3f} rho={r['spearman']:.3f}")
