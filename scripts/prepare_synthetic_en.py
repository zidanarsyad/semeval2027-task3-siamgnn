"""Convert the synthetic English multi-period LSC dataset (Saponara, Tahmasebi & Cassotti, CLiC-it 2026;
Zenodo 10.5281/zenodo.21703081) into the SemEval-2027 Task 3 folder format (Subtask 1 gold only).

Each lemma has 10 periods (1810-1990, 20-year bins) x 30 generated sentences whose sense ids come
from the OED; sense ids are mapped to integers.  Lemmas are split into disjoint TRAIN / DEV lists so
that any trained component (few-shot Siamese head) is never evaluated on words it saw.
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--src", type=Path, default=ROOT / "data/raw/synthetic_en/extracted/generated_sentences")
ap.add_argument("--out", type=Path, default=ROOT / "data/synthetic_en")
ap.add_argument("--n-dev", type=int, default=40)
ap.add_argument("--n-train", type=int, default=60)
ap.add_argument("--max-senses", type=int, default=12)
ap.add_argument("--seed", type=int, default=13)
args = ap.parse_args()

files = sorted(p for p in args.src.glob("*_senses/*.jsonl") if int(p.parent.name.split("_")[0]) <= args.max_senses)
random.Random(args.seed).shuffle(files)


def convert(path: Path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if 0 <= r["start"] < r["end"] <= len(r["text"])]
    sid = {s: i for i, s in enumerate(sorted({r["sense_id"] for r in rows}))}
    usages, gold = [], []
    for i, r in enumerate(rows):
        y = int(r["year"])
        period = f"{y}-{y + 19}"
        w = r["lemma"]
        s_id = f"{w}_{i}"
        usages.append({"word": w, "sentence_id": s_id, "text": r["text"], "start": r["start"], "end": r["end"],
                       "year": str(y), "period_label": period, "pos": r.get("pos")})
        gold.append({"word": w, "sentence_id": s_id, "period_label": period, "label": [sid[r["sense_id"]]]})
    return usages, gold


def keep(usages, gold):
    """Need >= 2 periods and >= 2 senses actually used, else the word is uninformative."""
    if len({u["period_label"] for u in usages}) < 2:
        return False
    c = Counter(g["label"][0] for g in gold)
    return len(c) >= 2 and len(usages) >= 100


splits = {"DEV": [], "TRAIN": []}
for f in files:
    u, g = convert(f)
    if not keep(u, g):
        continue
    target = "DEV" if len(splits["DEV"]) < args.n_dev else "TRAIN"
    if target == "TRAIN" and len(splits["TRAIN"]) >= args.n_train:
        break
    splits[target].append((u, g))

for name, items in splits.items():
    d = args.out / name / "EN"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "usages.jsonl", "w", encoding="utf-8") as fu, open(d / "subtask1.jsonl", "w", encoding="utf-8") as fg:
        for u, g in items:
            for x in u:
                fu.write(json.dumps(x, ensure_ascii=False) + "\n")
            for x in g:
                fg.write(json.dumps(x, ensure_ascii=False) + "\n")
    n = sum(len(u) for u, _ in items)
    ks = [len({x["label"][0] for x in g}) for _, g in items]
    print(f"{name}: {len(items)} words, {n} usages, senses/word mean={sum(ks)/len(ks):.1f}")
