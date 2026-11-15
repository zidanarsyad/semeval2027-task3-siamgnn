"""Make the synthetic-English splits used in the experiments from the output of prepare_synthetic_en.py.

  TRAIN_SUB  all 60 TRAIN words, 15 random usages per (word, period)            (random.Random(0))
  DEV20      the first 20 of the 40 DEV words (disjoint from TRAIN)
  DEV20_SUB  DEV20 with 15 random usages per (word, period)                       (random.Random(1))

    python scripts/subsample_synthetic.py [--out data/synthetic_en]
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(folder):
    U = [json.loads(line) for line in open(folder / "usages.jsonl", encoding="utf-8")]
    G = [json.loads(line) for line in open(folder / "subtask1.jsonl", encoding="utf-8")]
    return U, G


def write(folder, usages, gold):
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "usages.jsonl", "w", encoding="utf-8") as fu, \
            open(folder / "subtask1.jsonl", "w", encoding="utf-8") as fg:
        for u, g in zip(usages, gold):
            fu.write(json.dumps(u, ensure_ascii=False) + "\n")
            fg.write(json.dumps(g, ensure_ascii=False) + "\n")


def per_period_sample(U, G, n, seed):
    gold = {g["sentence_id"]: g for g in G}
    rng = random.Random(seed)
    by = defaultdict(lambda: defaultdict(list))
    for u in U:
        by[u["word"]][u["period_label"]].append(u)
    keep = []
    for per in by.values():
        for us in per.values():
            rng.shuffle(us)
            keep += us[:n]
    return keep, [gold[u["sentence_id"]] for u in keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ROOT / "data/synthetic_en")
    ap.add_argument("--out", type=Path, default=ROOT / "data/synthetic_en")
    ap.add_argument("--per-period", type=int, default=15)
    args = ap.parse_args()

    U, G = read(args.src / "TRAIN" / "EN")
    write(args.out / "TRAIN_SUB" / "EN", *per_period_sample(U, G, args.per_period, seed=0))

    U, G = read(args.src / "DEV" / "EN")
    words = set(list(dict.fromkeys(u["word"] for u in U))[:20])
    U20 = [u for u in U if u["word"] in words]
    G20 = [g for g in G if g["word"] in words]
    # DEV20 keeps the original file order of usages and gold lines
    (args.out / "DEV20" / "EN").mkdir(parents=True, exist_ok=True)
    with open(args.out / "DEV20" / "EN" / "usages.jsonl", "w", encoding="utf-8") as f:
        for u in U20:
            f.write(json.dumps(u, ensure_ascii=False) + "\n")
    with open(args.out / "DEV20" / "EN" / "subtask1.jsonl", "w", encoding="utf-8") as f:
        for g in G20:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    write(args.out / "DEV20_SUB" / "EN", *per_period_sample(U20, G20, args.per_period, seed=1))
    for split in ["TRAIN_SUB", "DEV20", "DEV20_SUB"]:
        n = sum(1 for _ in open(args.out / split / "EN" / "usages.jsonl", encoding="utf-8"))
        print(f"{split}: {n} usages")


if __name__ == "__main__":
    main()
