"""Clustering algorithm x similarity source on top of the trained SiamGNN (thresholds from synEN-val only)."""
import copy, json, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mtp.clustering import agglomerative_sim, correlation_clustering
from mtp.data import load_language
from mtp.encoders import get_embeddings
from mtp.scorer import score_subtask1
from mtp.siamgnn import WordGraph, load_model, predict_pairs
sys.stdout.reconfigure(encoding="utf-8")
def load(folder, lang, tag):
    ld = load_language(folder, lang)
    E = get_embeddings([(u.text, u.start, u.end) for u in ld.usages], tag=tag)
    idx = {(u.word, u.sentence_id): i for i, u in enumerate(ld.usages)}
    return ld, {w: (us, E[[idx[(u.word, u.sentence_id)] for u in us]]) for w, us in ld.by_word().items()}
tr_ld, tr_g = load(ROOT / "data/synthetic_en/TRAIN_SUB/EN", "EN", "usages_synEN_TRAIN")
en_ld, en_g = load(ROOT / "data/synthetic_en/DEV20_SUB/EN", "EN", "usages_synEN_DEV")
sv_ld, sv_g = load(ROOT / "data/raw/traindev_phase/extracted/SV", "SV", "usages_SV")
perm = np.random.default_rng(123).permutation(len(tr_g))
val_words = {list(tr_g)[i] for i in perm[:12]}
val_ld = copy.copy(tr_ld); val_ld.usages = [u for u in tr_ld.usages if u.word in val_words]
val_ld.gold_s1 = {k: v for k, v in tr_ld.gold_s1.items() if k[0] in val_words}
val_g = {w: v for w, v in tr_g.items() if w in val_words}
fit_g = {w: v for w, v in tr_g.items() if w not in val_words}
# calibrated cosine: logistic regression cos -> p(same sense) on the 48 fit words
xs, ys = [], []
for w, (us, X) in fit_g.items():
    Y = WordGraph(w, X, [set(tr_ld.gold_s1[(u.word, u.sentence_id)]) for u in us]).targets()
    iu = np.triu_indices(len(X), 1)
    xs.append((X @ X.T)[iu]); ys.append(Y[iu])
lr = LogisticRegression().fit(np.concatenate(xs)[:, None], np.concatenate(ys))
def pcos(X): return lr.predict_proba((X @ X.T).reshape(-1, 1))[:, 1].reshape(len(X), len(X))
models = [load_model(p) for p in sorted((ROOT / "models").glob("siamgnn_synEN_seed*.pt"))]
def pgnn(X): return np.mean([predict_pairs(m, X) for m in models], axis=0)
SIMS = {"calibrated cosine": pcos, "SiamGNN": pgnn, "hybrid (mean)": lambda X: 0.5 * (pcos(X) + pgnn(X))}
ALGS = {"CC": lambda P, t: correlation_clustering(P, threshold=t, n_restarts=10),
        "agglo-avg": lambda P, t: agglomerative_sim(P, threshold=1 - t)}
def run(ld, groups, cache, alg, t):
    lab = {(u.word, u.sentence_id): [int(l)] for w, (us, X) in groups.items() for u, l in zip(us, ALGS[alg](cache[w], t))}
    r = score_subtask1(ld, lab)
    return r["bcubed_f"], r["spearman"], float(np.mean([v["k_pred"] for v in r["per_word"].values()]))
rows = []
for sname, f in SIMS.items():
    cache = {name: {w: f(X) for w, (us, X) in g.items()} for name, g in [("val", val_g), ("en", en_g), ("sv", sv_g)]}
    for alg in ALGS:
        ths = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        t = max(ths, key=lambda t: sum(run(val_ld, val_g, cache["val"], alg, t)[:2]))
        en, sv = run(en_ld, en_g, cache["en"], alg, t), run(sv_ld, sv_g, cache["sv"], alg, t)
        rows.append({"similarity": sname, "algorithm": alg, "theta": t, "synEN": en, "SV": sv})
        print(f"{sname:18s} {alg:9s} theta={t}: synEN B3={en[0]:.3f} rho={en[1]:.3f} k={en[2]:.1f} | SV B3={sv[0]:.3f} rho={sv[1]:.3f} k={sv[2]:.1f}", flush=True)
(ROOT / "outputs/results/exp_hybrid.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
