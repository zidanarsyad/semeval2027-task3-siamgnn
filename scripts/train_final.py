"""Final SiamGNN for the evaluation phase: all 60 synthetic-English words, then few-shot adaptation
on all 10 labelled Swedish dev words (50% Swedish episodes), saved to models/siamgnn_final.pt."""
import sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mtp.data import load_language
from mtp.encoders import get_embeddings
from mtp.siamgnn import WordGraph, save_model, train_siamgnn
torch.set_num_threads(3)
def graphs(folder, lang, tag):
    ld = load_language(folder, lang)
    E = get_embeddings([(u.text, u.start, u.end) for u in ld.usages], tag=tag)
    idx = {(u.word, u.sentence_id): i for i, u in enumerate(ld.usages)}
    return [WordGraph(w, E[[idx[(u.word, u.sentence_id)] for u in us]], [set(ld.gold_s1[(u.word, u.sentence_id)]) for u in us])
            for w, us in ld.by_word().items()]
en = graphs(ROOT / "data/synthetic_en/TRAIN_SUB/EN", "EN", "usages_synEN_TRAIN")
sv = graphs(ROOT / "data/raw/traindev_phase/extracted/SV", "SV", "usages_SV")
m = train_siamgnn(en, steps=2000, seed=0, knn=10, use_gnn=True, verbose=True)
data = sv + en
w = [len(en) / len(sv)] * len(sv) + [1.0] * len(en)
m = train_siamgnn(data, steps=300, lr=1e-4, seed=0, init=m, weights=w, verbose=True)
save_model(m, ROOT / "models" / "siamgnn_final.pt", steps="2000+300", theta=0.5, train="synEN TRAIN_SUB (60 words) + SV dev (10 words)")
print("saved models/siamgnn_final.pt")
