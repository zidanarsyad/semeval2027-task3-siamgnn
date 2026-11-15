"""Pure few-shot SiamGNN on Swedish: train on 9 dev words, cluster the held-out word (LOWO)."""
import sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mtp.clustering import correlation_clustering, agglomerative_sim
from mtp.data import load_language
from mtp.encoders import get_embeddings
from mtp.scorer import score_subtask1
from mtp.siamgnn import WordGraph, predict_pairs, train_siamgnn
sys.stdout.reconfigure(encoding="utf-8")
torch.set_num_threads(2)
ld = load_language(ROOT / "data/raw/traindev_phase/extracted/SV", "SV")
E = get_embeddings([(u.text, u.start, u.end) for u in ld.usages], tag="usages_SV")
idx = {(u.word, u.sentence_id): i for i, u in enumerate(ld.usages)}
groups = {w: (us, E[[idx[(u.word, u.sentence_id)] for u in us]]) for w, us in ld.by_word().items()}
graphs = {w: WordGraph(w, X, [set(ld.gold_s1[(u.word, u.sentence_id)]) for u in us]) for w, (us, X) in groups.items()}
for use_gnn in [False, True]:
    for steps, balance in [(600, False)]:
        t = time.time()
        labels_cc, labels_ag = {}, {}
        for w in groups:
            m = train_siamgnn([g for k, g in graphs.items() if k != w], steps=steps, seed=0, use_gnn=use_gnn, balance=balance, verbose=False)
            us, X = groups[w]
            P = predict_pairs(m, X)
            for u, a, b in zip(us, correlation_clustering(P, 0.5, n_restarts=10), agglomerative_sim(P, 0.5)):
                labels_cc[(u.word, u.sentence_id)] = [int(a)]
                labels_ag[(u.word, u.sentence_id)] = [int(b)]
        for name, lab in [("CC@0.5", labels_cc), ("agglo@0.5", labels_ag)]:
            r = score_subtask1(ld, lab)
            k = np.mean([v["k_pred"] for v in r["per_word"].values()])
            print(f"{'SiamGNN' if use_gnn else 'Siamese head'} steps={steps} bal={balance} {name:10s} B3={r['bcubed_f']:.3f} rho={r['spearman']:.3f} k={k:.1f}  ({time.time()-t:.0f}s)", flush=True)
