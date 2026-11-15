"""End-to-end system: task folder (any languages) -> Codabench submission zip.

    python scripts/run_pipeline.py --data data/raw/traindev_phase/extracted --out outputs/submissions/dev_sv
    python scripts/run_pipeline.py --data <eval_phase1_folder> --out outputs/submissions/eval1 --subtasks 1
    python scripts/run_pipeline.py --data <eval_phase2_folder> --out outputs/submissions/eval2 --subtasks 2

Subtask 1: XL-LEXEME embeddings -> SiamGNN pair probabilities -> correlation clustering at p=0.5
           (``--s1-method baseline`` uses the existing XL-LEXEME + agglomerative method instead).
Subtask 2: definition as a one-shot prototype -> transductive prototype refinement -> global threshold
           (optional APPNP propagation over the usage graph; disabled in configs/final.json, s2_alpha = 0)
           (``--s2-method baseline`` uses plain gloss cosine + threshold).
If gold files are present in the data folder, the official scorer is run on the produced zip.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mtp.clustering import agglomerative, correlation_clustering  # noqa: E402
from mtp.data import load_task_root, write_subtask1, write_subtask2, zip_submission  # noqa: E402
from mtp.encoders import XLLexemeEncoder, encode_cached, gloss_item  # noqa: E402
from mtp.scorer import official_eval_zip  # noqa: E402
from mtp.siamgnn import load_model, predict_pairs  # noqa: E402
from mtp.subtask2 import propagate, prototype_scores  # noqa: E402

CONFIG = json.loads((ROOT / "configs" / "final.json").read_text(encoding="utf-8")) \
    if (ROOT / "configs" / "final.json").exists() else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--langs", nargs="*", default=None)
    ap.add_argument("--subtasks", nargs="*", default=["1", "2"])
    ap.add_argument("--s1-method", choices=["siamgnn", "baseline"], default="siamgnn")
    ap.add_argument("--s2-method", choices=["fewshot", "baseline"], default="fewshot")
    ap.add_argument("--model", type=Path, default=ROOT / "models" / "siamgnn_final.pt")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    cfg = {"s1_baseline_threshold": 0.4, "s1_siamgnn_theta": 0.5, "s1_cc_restarts": 10, "s2_tau": 0.8625,
           "s2_top": 0.05, "s2_iter": 3,
           "s2_alpha": 0.0, "s2_knn": 5, "s2_baseline_tau": 0.762,
           **{k: v for k, v in CONFIG.items() if not k.startswith("_")}}
    data = load_task_root(args.data, args.langs)
    if not data:
        sys.exit(f"no language folders with usages.jsonl under {args.data}")
    enc = XLLexemeEncoder()
    gnn = None
    if "1" in args.subtasks and args.s1_method == "siamgnn":
        gnn = load_model(args.model)

    args.out.mkdir(parents=True, exist_ok=True)
    for lang, ld in data.items():
        E = encode_cached(enc, [(u.text, u.start, u.end) for u in ld.usages], tag=f"usages_{lang}",
                          batch_size=args.batch_size)
        index = {(u.word, u.sentence_id): i for i, u in enumerate(ld.usages)}
        groups = ld.by_word()
        if "1" in args.subtasks:
            labels = {}
            for w, us in groups.items():
                X = E[[index[(u.word, u.sentence_id)] for u in us]]
                if args.s1_method == "siamgnn":
                    lab = correlation_clustering(predict_pairs(gnn, X), threshold=cfg["s1_siamgnn_theta"],
                                                 n_restarts=cfg["s1_cc_restarts"])
                else:
                    lab = agglomerative(X, threshold=cfg["s1_baseline_threshold"])
                for u, l in zip(us, lab):
                    labels[(u.word, u.sentence_id)] = [int(l)]
            write_subtask1(args.out / f"{lang}_subtask1.jsonl", ld.usages, labels)
            print(f"{lang}: subtask 1 written ({len(groups)} words)", flush=True)
        if "2" in args.subtasks and ld.definitions:
            words = list(groups)
            G = encode_cached(enc, [gloss_item(w, ld.definitions[w], "{w}: {d}") for w in words],
                              tag=f"defs_{lang}_wcd", batch_size=args.batch_size)
            labels2 = {}
            for gi, w in zip(G, words):
                us = groups[w]
                X = E[[index[(u.word, u.sentence_id)] for u in us]]
                if args.s2_method == "fewshot":
                    s = prototype_scores(X, gi, top=cfg["s2_top"], n_iter=cfg["s2_iter"])
                    s = propagate(X, s, k=cfg["s2_knn"], alpha=cfg["s2_alpha"])
                    pred = (s > cfg["s2_tau"]).astype(int)
                else:
                    pred = (X @ gi > cfg["s2_baseline_tau"]).astype(int)
                for u, p in zip(us, pred):
                    labels2[(u.word, u.sentence_id)] = int(p)
            write_subtask2(args.out / f"{lang}_subtask2.jsonl", ld.usages, labels2)
            print(f"{lang}: subtask 2 written", flush=True)

    z = zip_submission(args.out, args.out.with_suffix(".zip"))
    print("submission:", z)
    if any((args.data / lang / "subtask1.jsonl").exists() or (args.data / lang / "subtask2.jsonl").exists()
           for lang in data):
        res = official_eval_zip(z, args.data)
        print(json.dumps(res["results"], indent=2))
        if res["errors"]:
            print("ERRORS:", res["errors"])


if __name__ == "__main__":
    main()
