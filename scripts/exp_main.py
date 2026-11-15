"""Main Subtask 1 comparison: existing methods vs. Siamese / Siamese-GNN usage-graph models.

Protocol (no evaluation word is ever used for training or for picking a threshold):
  * "tuned on synEN-train": thresholds / models fit on the synthetic-English TRAIN words only.
  * "LOWO": for each Swedish dev word, the threshold (baselines) or few-shot adaptation (SiamGNN)
    uses only the other 9 Swedish words; predictions of all folds are then scored together.
All numbers come from the official SemEval-2027 Task 3 scorer.

    python scripts/exp_main.py --seeds 0 1 2
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mtp.clustering import affinity_propagation, agglomerative, correlation_clustering, dmon  # noqa: E402
from mtp.data import load_language  # noqa: E402
from mtp.encoders import get_embeddings  # noqa: E402
from mtp.scorer import score_subtask1  # noqa: E402
from mtp.siamgnn import WordGraph, predict_pairs, save_model, train_siamgnn  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--adapt-steps", type=int, default=300)
ap.add_argument("--knn", type=int, default=10)
ap.add_argument("--skip-slow", action="store_true", help="skip AP / DMoN baselines")
ap.add_argument("--balance", action="store_true", help="class-balanced BCE (default: plain BCE)")
args = ap.parse_args()
torch.set_num_threads(4)
OUT = ROOT / "outputs" / "results"
OUT.mkdir(parents=True, exist_ok=True)


def load(folder, lang, tag):
    ld = load_language(folder, lang)
    E = get_embeddings([(u.text, u.start, u.end) for u in ld.usages], tag=tag)
    idx = {(u.word, u.sentence_id): i for i, u in enumerate(ld.usages)}
    groups = {w: (us, E[[idx[(u.word, u.sentence_id)] for u in us]]) for w, us in ld.by_word().items()}
    return ld, groups


def graphs_of(ld, groups):
    return [WordGraph(w, X, [set(ld.gold_s1[(u.word, u.sentence_id)]) for u in us]) for w, (us, X) in groups.items()]


tr_ld, tr_g = load(ROOT / "data/synthetic_en/TRAIN_SUB/EN", "EN", "usages_synEN_TRAIN")
en_ld, en_g = load(ROOT / "data/synthetic_en/DEV20_SUB/EN", "EN", "usages_synEN_DEV")
sv_ld, sv_g = load(ROOT / "data/raw/traindev_phase/extracted/SV", "SV", "usages_SV")
train_graphs, sv_graphs = graphs_of(tr_ld, tr_g), graphs_of(sv_ld, sv_g)
SV_WORDS = list(sv_g)


def to_labels(groups, per_word_fn):
    out = {}
    for w, (us, X) in groups.items():
        for u, l in zip(us, per_word_fn(w, X)):
            out[(u.word, u.sentence_id)] = [int(l)]
    return out


def score(ld, labels):
    r = score_subtask1(ld, labels)
    return r["bcubed_f"], r["spearman"], float(np.mean([v["k_pred"] for v in r["per_word"].values()]))


rows = []


def log(system, protocol, res_en, res_sv, seed=None, **extra):
    row = {"system": system, "protocol": protocol, "seed": seed, **extra}
    for name, res in [("synEN", res_en), ("SV", res_sv)]:
        if res is not None:
            row.update({f"{name}_b3": res[0], f"{name}_rho": res[1], f"{name}_k": res[2]})
    rows.append(row)
    s = "  ".join(f"{n}: B3={r[0]:.3f} rho={r[1]:.3f} k={r[2]:.1f}" for n, r in [("synEN", res_en), ("SV", res_sv)] if r)
    print(f"{system:44s} [{protocol:10s}] {s}", flush=True)
    (OUT / "exp_main.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def tune_threshold(ld, groups, fn, grid):
    best, best_v = None, -1
    for t in grid:
        b3, rho, _ = score(ld, to_labels(groups, lambda w, X: fn(X, t)))
        if (b3 + rho) / 2 > best_v:
            best, best_v = t, (b3 + rho) / 2
    return best


def lowo_threshold(fn, grid):
    """For each SV word pick the threshold on the other 9 words (mean of B3 and rho)."""
    cache = {t: {w: fn(X, t) for w, (us, X) in sv_g.items()} for t in grid}
    labels = {}
    for w in SV_WORDS:
        others = {k: v for k, v in sv_g.items() if k != w}
        ld_o = copy.copy(sv_ld)
        ld_o.usages = [u for u in sv_ld.usages if u.word != w]
        ld_o.gold_s1 = {k: v for k, v in sv_ld.gold_s1.items() if k[0] != w}
        best, best_v = None, -1
        for t in grid:
            b3, rho, _ = score(ld_o, to_labels(others, lambda ww, X: cache[t][ww]))
            if (b3 + rho) / 2 > best_v:
                best, best_v = t, (b3 + rho) / 2
        us, _ = sv_g[w]
        for u, l in zip(us, cache[best][w]):
            labels[(u.word, u.sentence_id)] = [int(l)]
    return labels


GRID = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
agg = lambda X, t: agglomerative(X, threshold=t)  # noqa: E731
cc = lambda X, t: correlation_clustering(X @ X.T, threshold=t, n_restarts=10)  # noqa: E731

# ------------------------------------------------------------------ existing / unsupervised methods
one = lambda w, X: np.zeros(len(X), dtype=int)  # noqa: E731
log("1C1W (one sense per word)", "-", score(en_ld, to_labels(en_g, one)), score(sv_ld, to_labels(sv_g, one)))
for name, fn in [("XL-LEXEME + agglomerative", agg), ("XL-LEXEME + correlation clustering", cc)]:
    t = tune_threshold(tr_ld, tr_g, fn, GRID)
    log(f"{name} (t={t})", "synEN-tune",
        score(en_ld, to_labels(en_g, lambda w, X: fn(X, t))), score(sv_ld, to_labels(sv_g, lambda w, X: fn(X, t))))
    log(name, "SV-LOWO", None, score(sv_ld, lowo_threshold(fn, GRID)))
if not args.skip_slow:
    log("XL-LEXEME + Affinity Propagation", "default",
        score(en_ld, to_labels(en_g, lambda w, X: affinity_propagation(X))),
        score(sv_ld, to_labels(sv_g, lambda w, X: affinity_propagation(X))))
    log("XL-LEXEME + DMoN (unsupervised GNN)", "default",
        score(en_ld, to_labels(en_g, lambda w, X: dmon(X, k_max=8, knn=10))),
        score(sv_ld, to_labels(sv_g, lambda w, X: dmon(X, k_max=8, knn=10))))

# ------------------------------------------------------------------ trained Siamese / Siamese-GNN
models_dir = ROOT / "models"
models_dir.mkdir(exist_ok=True)
# 48 training words / 12 validation words (validation picks the clustering threshold theta)
rng_split = np.random.default_rng(123)
perm = rng_split.permutation(len(train_graphs))
val_ids = set(perm[:12].tolist())
fit_graphs = [g for i, g in enumerate(train_graphs) if i not in val_ids]
val_words = {train_graphs[i].word for i in val_ids}
val_ld = copy.copy(tr_ld)
val_ld.usages = [u for u in tr_ld.usages if u.word in val_words]
val_ld.gold_s1 = {k: v for k, v in tr_ld.gold_s1.items() if k[0] in val_words}
val_g = {w: v for w, v in tr_g.items() if w in val_words}
THETAS = [0.2, 0.3, 0.4, 0.5, 0.6]
for use_gnn in [False, True]:
    tag = "SiamGNN" if use_gnn else "Siamese head"
    for seed in args.seeds:
        t0 = time.time()
        m = train_siamgnn(fit_graphs, steps=args.steps, seed=seed, knn=args.knn, use_gnn=use_gnn,
                          balance=args.balance, verbose=False)
        def fn_theta(X, th, m=m):
            return correlation_clustering(predict_pairs(m, X), threshold=th, n_restarts=10)
        theta = tune_threshold(val_ld, val_g, fn_theta, THETAS)
        fn = lambda w, X, m=m, th=theta: correlation_clustering(predict_pairs(m, X), threshold=th, n_restarts=10)  # noqa: E731
        log(f"{tag} + CC (theta tuned on synEN-val)", "synEN-train", score(en_ld, to_labels(en_g, fn)),
            score(sv_ld, to_labels(sv_g, fn)), seed, theta=theta)
        fn5 = lambda w, X, m=m: correlation_clustering(predict_pairs(m, X), threshold=0.5, n_restarts=10)  # noqa: E731
        log(f"{tag} + CC@0.5", "synEN-train", score(en_ld, to_labels(en_g, fn5)),
            score(sv_ld, to_labels(sv_g, fn5)), seed)
        print(f"   (train+eval {time.time() - t0:.0f}s)", flush=True)
        save_model(m, models_dir / f"{'siamgnn' if use_gnn else 'siamese'}_synEN_seed{seed}.pt",
                   steps=args.steps, theta=theta, balance=args.balance)
        # few-shot adaptation to Swedish: LOWO fine-tuning on the other 9 dev words (+ replay of EN words)
        labels = {}
        for w in SV_WORDS:
            fold = [g for g in sv_graphs if g.word != w]
            data = fold + train_graphs
            wts = [len(train_graphs) / len(fold)] * len(fold) + [1.0] * len(train_graphs)  # 50% SV episodes
            ma = train_siamgnn(data, steps=args.adapt_steps, lr=1e-4, seed=seed, init=copy.deepcopy(m),
                               weights=wts, balance=args.balance, verbose=False)
            us, X = sv_g[w]
            for u, l in zip(us, correlation_clustering(predict_pairs(ma, X), threshold=theta, n_restarts=10)):
                labels[(u.word, u.sentence_id)] = [int(l)]
        log(f"{tag} + few-shot SV adaptation + CC", "SV-LOWO", None, score(sv_ld, labels), seed, theta=theta)

# ------------------------------------------------------------------ summary (mean +- std over seeds)
print("\nSummary (mean over seeds):")
by = {}
for r in rows:
    by.setdefault((r["system"], r["protocol"]), []).append(r)
for (sys_, prot), rs in by.items():
    parts = []
    for k in ["synEN_b3", "synEN_rho", "SV_b3", "SV_rho"]:
        v = [r[k] for r in rs if k in r]
        if v:
            parts.append(f"{k}={np.mean(v):.3f}±{np.std(v):.3f}" if len(v) > 1 else f"{k}={v[0]:.3f}")
    print(f"  {sys_:48s} [{prot:10s}] " + "  ".join(parts))
