"""Trivial baselines + oracle sanity checks on the Swedish dev set, scored with the official scorer."""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mtp.data import load_language, period_start  # noqa: E402
from mtp.scorer import score_subtask1, score_subtask2  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
ld = load_language(ROOT / "data/raw/traindev_phase/extracted/SV", "SV")
keys = [(u.word, u.sentence_id) for u in ld.usages]
rows = []


def s1(name, lab):
    r = score_subtask1(ld, lab)
    rows.append({"system": name, "task": "S1", "bcubed_f": r["bcubed_f"], "spearman": r["spearman"]})
    print(f"{name:38s} B3-F={r['bcubed_f']:.3f} (P={r['bcubed_p']:.3f} R={r['bcubed_r']:.3f})  rho={r['spearman']:.3f}")


def s2(name, lab):
    r = score_subtask2(ld, lab)
    rows.append({"system": name, "task": "S2", "macro_f1": r["macro_f1"]})
    print(f"{name:38s} macroF1={r['macro_f1']:.3f}")


# --- Subtask 1
s1("oracle (gold labels)", dict(ld.gold_s1))
s1("one cluster per word (1C1W)", {k: [0] for k in keys})
s1("one cluster per usage (1C1I)", {k: [i] for i, k in enumerate(keys)})
era = {(u.word, u.sentence_id): [0 if period_start(u.period) < 1950 else 1] for u in ld.usages}
s1("cluster = era (pre/post 1950)", era)
per = {(u.word, u.sentence_id): [period_start(u.period)] for u in ld.usages}
s1("cluster = period", per)
random.seed(0)
for k in (2, 3, 5):
    s1(f"random {k} clusters", {key: [random.randrange(k)] for key in keys})
# dominant gold sense only (upper bound for single-label systems)
s1("oracle single-label (first gold sense)", {k: [v[0]] for k, v in ld.gold_s1.items()})

# --- Subtask 2
s2("oracle (gold labels)", dict(ld.gold_s2))
s2("all 1", {k: 1 for k in keys})
s2("all 0", {k: 0 for k in keys})
random.seed(0)
s2("random 50/50", {k: random.randrange(2) for k in keys})

out = ROOT / "outputs" / "results"
out.mkdir(parents=True, exist_ok=True)
(out / "baselines_dev_sv.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
