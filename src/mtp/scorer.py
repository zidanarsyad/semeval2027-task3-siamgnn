"""Scoring helpers built directly on the official SemEval-2027 Task 3 scorer.

The official ``evaluation.py`` / ``evaluation_utils.py`` (GitHub ChangeIsKey/semeval2027-task3,
commit 0e69238, "final" scorer announced 12 Nov 2026) are imported unchanged, so every number
reported by this project is computed with exactly the metric used on Codabench:

* Subtask 1a - extended BCubed F1 (overlapping clusters), per word, macro-averaged.
* Subtask 1b - Spearman rho between gold and predicted Jensen-Shannon *distances* of sense
  distributions of consecutive periods (pairs where both gold periods have >= 10 usages).
* Subtask 2  - macro-F1 (both classes) per word, averaged over words.
"""
from __future__ import annotations

import importlib
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import bcubed
import numpy as np
from sklearn.metrics import f1_score

from .data import LanguageData, write_subtask1, write_subtask2

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_DIR = ROOT / "external" / "semeval2027-task3"


def _official():
    if str(OFFICIAL_DIR) not in sys.path:
        sys.path.insert(0, str(OFFICIAL_DIR))
    return importlib.import_module("evaluation"), importlib.import_module("evaluation_utils")


def _gold_s1_file(ld: LanguageData, tmp: Path) -> Path:
    p = tmp / "gold_s1.jsonl"
    period = {(u.word, u.sentence_id): u.period for u in ld.usages}
    with open(p, "w", encoding="utf-8") as f:
        for (w, sid), lab in ld.gold_s1.items():
            f.write(json.dumps({"word": w, "sentence_id": sid, "period_label": period[(w, sid)],
                                "label": lab}, ensure_ascii=False) + "\n")
    return p


def _gold_s2_file(ld: LanguageData, tmp: Path) -> Path:
    p = tmp / "gold_s2.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for (w, sid), lab in ld.gold_s2.items():
            f.write(json.dumps({"word": w, "sentence_id": sid, "label": lab}, ensure_ascii=False) + "\n")
    return p


def score_subtask1(ld: LanguageData, labels: dict[tuple[str, str], list[int]]) -> dict:
    """Official 1a (BCubed) and 1b (Spearman) plus per-word BCubed diagnostics."""
    ev, _ = _official()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        pred = td / "pred_s1.jsonl"
        write_subtask1(pred, ld.usages, labels)
        b3, rho = ev.evaluate_subtask1(_gold_s1_file(ld, td), pred)
    per_word = {}
    for w, us in ld.by_word().items():
        g = {i: set(ld.gold_s1[(u.word, u.sentence_id)]) for i, u in enumerate(us)}
        p = {i: set(labels[(u.word, u.sentence_id)]) for i, u in enumerate(us)}
        P, R = bcubed.precision(p, g), bcubed.recall(p, g)
        per_word[w] = {"P": P, "R": R, "F": bcubed.fscore(P, R),
                       "k_pred": len({x for s in p.values() for x in s}),
                       "k_gold": len({x for s in g.values() for x in s})}
    return {"bcubed_f": b3["fcubed"], "bcubed_p": b3["precision"], "bcubed_r": b3["recall"],
            "spearman": rho, "per_word": per_word}


def score_subtask2(ld: LanguageData, labels: dict[tuple[str, str], int]) -> dict:
    ev, _ = _official()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        pred = td / "pred_s2.jsonl"
        write_subtask2(pred, ld.usages, labels)
        f1 = ev.evaluate_subtask2(_gold_s2_file(ld, td), pred)
    per_word = {}
    by = defaultdict(lambda: ([], []))
    for (w, sid), g in ld.gold_s2.items():
        by[w][0].append(g)
        by[w][1].append(int(labels[(w, sid)]))
    for w, (g, p) in by.items():
        per_word[w] = {"F1": float(f1_score(g, p, average="macro")), "pos_gold": int(np.sum(g)),
                       "pos_pred": int(np.sum(p)), "n": len(g)}
    return {"macro_f1": f1, "per_word": per_word}


def official_eval_zip(zip_path: Path, gold_root: Path, out_json: Path | None = None) -> dict:
    """Run the unmodified official CLI scorer on a submission zip (end-to-end check)."""
    zip_path, gold_root = Path(zip_path).resolve(), Path(gold_root).resolve()
    out_json = Path(out_json or zip_path.with_suffix(".eval.json")).resolve()
    proc = subprocess.run([sys.executable, str(OFFICIAL_DIR / "evaluation.py"), str(zip_path),
                           str(gold_root), "-o", str(out_json)], capture_output=True, text=True,
                          cwd=str(OFFICIAL_DIR))
    if proc.returncode != 0:
        raise RuntimeError(f"official scorer failed:\n{proc.stderr[-2000:]}")
    return json.loads(out_json.read_text(encoding="utf-8"))
