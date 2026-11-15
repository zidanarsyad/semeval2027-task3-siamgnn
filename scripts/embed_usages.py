"""Pre-compute XL-LEXEME embeddings for every usage (and definition) of a task folder."""
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mtp.data import load_task_root
from mtp.encoders import XLLexemeEncoder, encode_cached, gloss_item

ap = argparse.ArgumentParser()
ap.add_argument("root", type=Path, help="folder with one sub-folder per language")
ap.add_argument("--langs", nargs="*", default=None)
ap.add_argument("--batch-size", type=int, default=16)
ap.add_argument("--tag-prefix", default="usages_", help="cache tag prefix, e.g. usages_synEN_TRAIN")
args = ap.parse_args()

enc = XLLexemeEncoder()
for lang, ld in load_task_root(args.root, args.langs).items():
    t = time.time()
    items = [(u.text, u.start, u.end) for u in ld.usages]
    tag = f"{args.tag_prefix}{lang}" if args.tag_prefix == "usages_" else args.tag_prefix
    E = encode_cached(enc, items, tag=tag, batch_size=args.batch_size)
    print(f"{lang}: {E.shape} usages in {time.time()-t:.0f}s", flush=True)
    if ld.definitions:  # gloss pseudo-usages for Subtask 2 (stable cache tag)
        g = [gloss_item(w, ld.definitions[w], "{w}: {d}") for w in ld.words]
        encode_cached(enc, g, tag=f"defs_{lang}_wcd", batch_size=args.batch_size)
        print(f"{lang}: definitions encoded", flush=True)
