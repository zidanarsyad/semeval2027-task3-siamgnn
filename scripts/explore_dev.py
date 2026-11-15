"""Quick exploratory analysis of the SemEval-2027 Task 3 Swedish dev set."""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
root = Path(sys.argv[1])
use = [json.loads(l) for l in open(root / "usages.jsonl", encoding="utf-8")]
s1 = {(d["word"], d["sentence_id"]): d for d in map(json.loads, open(root / "subtask1.jsonl", encoding="utf-8"))}
s2 = {(d["word"], d["sentence_id"]): d for d in map(json.loads, open(root / "subtask2.jsonl", encoding="utf-8"))}
defs = {d["word"]: d["definition"] for d in map(json.loads, open(root / "definitions.jsonl", encoding="utf-8"))}

print("keys in usages:", sorted(use[0].keys()))
print("POS:", Counter(u.get("pos") for u in use))
byw = defaultdict(list)
for u in use:
    byw[u["word"]].append(u)
bad_off = 0
for u in use:
    span = u["text"][u["start"]:u["end"]]
    if not span or span.lower()[:3] != u["word"].lower()[:3]:
        bad_off += 1
print("offset spans not starting with lemma prefix:", bad_off, "of", len(use))

for w, us in byw.items():
    labs = [tuple(s1[(w, u["sentence_id"])]["label"]) for u in us]
    multi = sum(len(l) > 1 for l in labs)
    sense_c = Counter(x for l in labs for x in l)
    y2 = [s2[(w, u["sentence_id"])]["label"] for u in us]
    # which subtask1 sense(s) co-occur with subtask2 positives?
    pos_senses = Counter(x for l, y in zip(labs, y2) if y == 1 for x in l)
    neg_senses = Counter(x for l, y in zip(labs, y2) if y == 0 for x in l)
    periods = Counter(u["period_label"] for u in us)
    print(f"\n=== {w}  n={len(us)}  periods={len(periods)}  multi-label={multi}  pos(s2)={sum(y2)} ({sum(y2)/len(y2):.2f})")
    print("  senses:", dict(sorted(sense_c.items(), key=lambda kv: -kv[1])))
    print("  s2=1 senses:", dict(pos_senses), " | s2=0 senses:", dict(neg_senses))
    # per-period sense distribution
    for p in sorted(periods):
        c = Counter(x for u, l in zip(us, labs) if u["period_label"] == p for x in l)
        print(f"   {p}: n={periods[p]:3d} ", dict(sorted(c.items())))
    print("  def:", defs[w][:150])
    # show 2 examples of span
    for u in us[:2]:
        print("   ex:", u["text"][max(0,u["start"]-60):u["end"]+60].replace("\n"," "), "| span=", u["text"][u["start"]:u["end"]], "| year", u["year"])
