"""Encode the synthetic-English DEV20 and TRAIN_SUB splits in one process (checkpointed)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mtp.data import load_language
from mtp.encoders import XLLexemeEncoder, encode_cached
enc = XLLexemeEncoder(num_threads=4)
for split, tag in [("TRAIN_SUB", "usages_synEN_TRAIN"), ("DEV20_SUB", "usages_synEN_DEV")]:
    ld = load_language(ROOT / f"data/synthetic_en/{split}/EN", "EN")
    E = encode_cached(enc, [(u.text, u.start, u.end) for u in ld.usages], tag=tag, batch_size=32)
    print(split, E.shape, flush=True)
