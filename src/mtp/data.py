"""Loading task data and writing submission files.

Task data layout (one folder per language, e.g. ``SV/``)::

    usages.jsonl       one usage per line: word, sentence_id, text, start, end, year, period_label, pos
    definitions.jsonl  one target-sense definition per word (Subtask 2 only)
    subtask1.jsonl     gold sense labels (dev only): word, sentence_id, period_label, label=[int,...]
    subtask2.jsonl     gold binary labels (dev only): word, sentence_id, label=0/1
"""
from __future__ import annotations

import json
import re
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

LANGUAGES = ["EN", "SV", "IT", "ES", "NL", "RU"]


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def period_start(period_label: str) -> int:
    """First four-digit year in a period label (same rule as the official scorer)."""
    m = re.search(r"\d{4}", period_label)
    return int(m.group()) if m else 10**9


@dataclass
class Usage:
    word: str
    sentence_id: str
    text: str
    start: int
    end: int
    year: int
    period: str
    pos: str | None = None

    @property
    def target(self) -> str:
        return self.text[self.start:self.end]


@dataclass
class LanguageData:
    lang: str
    usages: list[Usage]
    definitions: dict[str, str] = field(default_factory=dict)
    gold_s1: dict[tuple[str, str], list[int]] | None = None
    gold_s2: dict[tuple[str, str], int] | None = None

    @property
    def words(self) -> list[str]:
        seen: dict[str, None] = {}
        for u in self.usages:
            seen.setdefault(u.word, None)
        return list(seen)

    def by_word(self) -> dict[str, list[Usage]]:
        out: dict[str, list[Usage]] = defaultdict(list)
        for u in self.usages:
            out[u.word].append(u)
        return dict(out)

    def periods(self, word: str | None = None) -> list[str]:
        ps = {u.period for u in self.usages if word is None or u.word == word}
        return sorted(ps, key=lambda p: (period_start(p), p))


def _fix_offsets(u: dict) -> tuple[int, int]:
    """Return (start, end); keep given offsets but clamp them to the sentence."""
    text = u["text"]
    s, e = int(u["start"]), int(u["end"])
    s = max(0, min(s, len(text)))
    e = max(s, min(e, len(text)))
    if e == s:  # empty span: fall back to searching the lemma prefix
        m = re.search(re.escape(u["word"][:4]), text, flags=re.I)
        if m:
            s = m.start()
            e = s + len(re.match(r"\w+", text[s:]).group()) if re.match(r"\w+", text[s:]) else m.end()
    return s, e


def load_language(folder: Path, lang: str | None = None) -> LanguageData:
    folder = Path(folder)
    lang = lang or folder.name
    usages = []
    for u in read_jsonl(folder / "usages.jsonl"):
        s, e = _fix_offsets(u)
        try:
            year = int(str(u.get("year", "0"))[:4])
        except ValueError:
            year = 0
        usages.append(Usage(u["word"], str(u["sentence_id"]), u["text"], s, e, year,
                            u["period_label"], u.get("pos")))
    defs = {}
    if (folder / "definitions.jsonl").exists():
        defs = {d["word"]: d["definition"] for d in read_jsonl(folder / "definitions.jsonl")}
    g1 = g2 = None
    if (folder / "subtask1.jsonl").exists():
        g1 = {(d["word"], str(d["sentence_id"])): list(d["label"]) for d in read_jsonl(folder / "subtask1.jsonl")}
    if (folder / "subtask2.jsonl").exists():
        g2 = {(d["word"], str(d["sentence_id"])): int(d["label"]) for d in read_jsonl(folder / "subtask2.jsonl")}
    return LanguageData(lang, usages, defs, g1, g2)


def load_task_root(root: Path, languages: list[str] | None = None) -> dict[str, LanguageData]:
    root = Path(root)
    out = {}
    for lang in languages or LANGUAGES:
        if (root / lang / "usages.jsonl").exists():
            out[lang] = load_language(root / lang, lang)
    return out


# ----------------------------------------------------------------------------- submissions

def write_subtask1(path: Path, usages: list[Usage], labels: dict[tuple[str, str], list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for u in usages:
            lab = labels[(u.word, u.sentence_id)]
            lab = sorted({int(x) for x in lab})
            f.write(json.dumps({"word": u.word, "sentence_id": u.sentence_id, "label": lab},
                               ensure_ascii=False) + "\n")


def write_subtask2(path: Path, usages: list[Usage], labels: dict[tuple[str, str], int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for u in usages:
            f.write(json.dumps({"word": u.word, "sentence_id": u.sentence_id,
                                "label": int(labels[(u.word, u.sentence_id)])},
                               ensure_ascii=False) + "\n")


def zip_submission(folder: Path, zip_path: Path) -> Path:
    """Zip all ``*_subtask{1,2}.jsonl`` files of ``folder`` at the archive root."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(Path(folder).glob("*_subtask*.jsonl")):
            z.write(p, arcname=p.name)
    return zip_path
