"""Corpus statistics and per-word analysis for the system-description paper (report/).

Writes outputs/results/report_stats.json with
  data      - statistics of the Swedish dev set and of the synthetic-English splits
  per_word  - Swedish dev, per target word: gold senses, clusters and BCubed-F1 of the existing
              method (XL-LEXEME + agglomerative, t = 0.4) and of SiamGNN (zero-shot seed ensemble +
              correlation clustering, theta = 0.5), Subtask 2 F1/AUC, and mean XL-LEXEME cosine of
              same-sense pairs within one era, same-sense pairs across eras and different-sense pairs
  pairs     - the same three cosine statistics pooled over all Swedish words

    python scripts/report_stats.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mtp.clustering import agglomerative, correlation_clustering  # noqa: E402
from mtp.data import load_language  # noqa: E402
from mtp.encoders import get_embeddings, gloss_item  # noqa: E402
from mtp.scorer import score_subtask1  # noqa: E402
from mtp.siamgnn import load_model, predict_pairs  # noqa: E402
from mtp.subtask2 import prototype_scores  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
RES = ROOT / "outputs" / "results"


def data_stats(ld):
    byw = ld.by_word()
    per_word = [len(v) for v in byw.values()]
    senses = [len({s for u in us for s in ld.gold_s1[(u.word, u.sentence_id)]}) for us in byw.values()]
    pairs = 0
    for w, us in byw.items():
        cnt = Counter(u.period for u in us)
        ps = ld.periods(w)
        pairs += sum(cnt[a] >= 10 and cnt[b] >= 10 for a, b in zip(ps, ps[1:]))
    years = [u.year for u in ld.usages if u.year]
    out = {"words": len(byw), "usages": len(ld.usages), "periods": len(ld.periods()),
           "period_labels": ld.periods(), "years": [min(years), max(years)],
           "usages_per_word": [int(np.min(per_word)), float(np.mean(per_word)), int(np.max(per_word))],
           "gold_senses_per_word": [int(np.min(senses)), float(np.mean(senses)), int(np.max(senses))],
           "multi_sense_usage_rate": float(np.mean([len(v) > 1 for v in ld.gold_s1.values()])),
           "scored_1b_pairs": int(pairs)}
    if ld.gold_s2:
        out["s2_positive_rate"] = float(np.mean(list(ld.gold_s2.values())))
    return out


def load(folder, lang, tag):
    ld = load_language(folder, lang)
    E = get_embeddings([(u.text, u.start, u.end) for u in ld.usages], tag=tag)
    idx = {(u.word, u.sentence_id): i for i, u in enumerate(ld.usages)}
    return ld, {w: (us, E[[idx[(u.word, u.sentence_id)] for u in us]]) for w, us in ld.by_word().items()}


def labels_of(groups, fn):
    return {(u.word, u.sentence_id): [int(l)] for w, (us, X) in groups.items() for u, l in zip(us, fn(X))}


def era(u):
    return "early" if u.year < 1950 else "late"


def jsd_pairs(ld, labels):
    """Official JSD distances of consecutive periods (scored pairs only), keyed by (word, p1, p2)."""
    sys.path.insert(0, str(ROOT / "external" / "semeval2027-task3"))
    from evaluation_utils import calculate_JSD_distances  # noqa: E402
    from scipy.stats import spearmanr  # noqa: F401,E402

    def counts(lab):
        c = {}
        for u in ld.usages:
            ls = lab[(u.word, u.sentence_id)]
            for x in ls:
                c.setdefault(u.word, {}).setdefault(u.period, Counter())[x] += 1 / len(ls)
        return c
    gc, pc = counts(ld.gold_s1), counts(labels)
    gold, pred = calculate_JSD_distances(gc), calculate_JSD_distances(pc)
    out = {}
    for key, g in gold.items():
        w, p1, p2 = key.rsplit("_", 2)
        if sum(gc[w][p1].values()) >= 10 and sum(gc[w][p2].values()) >= 10:
            out[(w, p1, p2)] = (g, pred[key])
    return out


def rho_breakdown(ld, labels):
    """Spearman rho over all scored pairs, and without the cross-era pair (1912-1915 -> 2006-2007)."""
    from scipy.stats import spearmanr
    pairs = jsd_pairs(ld, labels)
    cross = {k for k in pairs if int(k[1][:4]) < 1950 <= int(k[2][:4])}
    allv = np.array(list(pairs.values()))
    within = np.array([v for k, v in pairs.items() if k not in cross])
    crossv = np.array([v for k, v in pairs.items() if k in cross])
    gold_rank_cross = [int((allv[:, 0] > g).sum()) + 1 for g in crossv[:, 0]] if len(crossv) else []
    return {"n_pairs": len(pairs), "n_cross": len(cross),
            "rho_all": float(spearmanr(allv[:, 0], allv[:, 1])[0]),
            "rho_within_era": float(spearmanr(within[:, 0], within[:, 1])[0]),
            "gold_rank_of_cross_pairs": gold_rank_cross,
            "mean_gold_jsd": {"cross": float(crossv[:, 0].mean()) if len(crossv) else None,
                              "within": float(within[:, 0].mean())}}


def main():
    stats = {}
    for name, folder, lang in [("SV_dev", "data/raw/traindev_phase/extracted/SV", "SV"),
                               ("synEN_TRAIN_SUB", "data/synthetic_en/TRAIN_SUB/EN", "EN"),
                               ("synEN_DEV20_SUB", "data/synthetic_en/DEV20_SUB/EN", "EN")]:
        stats[name] = data_stats(load_language(ROOT / folder, lang))
        print(name, {k: v for k, v in stats[name].items() if k != "period_labels"})

    ld, groups = load(ROOT / "data/raw/traindev_phase/extracted/SV", "SV", "usages_SV")
    words = list(groups)

    # Subtask 1: existing method vs. SiamGNN (zero-shot seed ensemble; theta from exp_ensemble.json)
    ens = json.loads((RES / "exp_ensemble.json").read_text(encoding="utf-8"))["siamgnn"]
    models = [load_model(p) for p in sorted((ROOT / "models").glob("siamgnn_synEN_seed*.pt"))]
    P = {w: np.mean([predict_pairs(m, X) for m in models], axis=0) for w, (us, X) in groups.items()}
    base = score_subtask1(ld, labels_of(groups, lambda X: agglomerative(X, threshold=0.4)))
    gnn_lab = {(u.word, u.sentence_id): [int(l)] for w, (us, X) in groups.items()
               for u, l in zip(us, correlation_clustering(P[w], threshold=ens["theta"], n_restarts=10))}
    gnn = score_subtask1(ld, gnn_lab)
    print(f"check: existing B3={base['bcubed_f']:.3f} rho={base['spearman']:.3f} | "
          f"SiamGNN ens B3={gnn['bcubed_f']:.3f} rho={gnn['spearman']:.3f}")
    base_lab = labels_of(groups, lambda X: agglomerative(X, threshold=0.4))
    era_lab = {(u.word, u.sentence_id): [0 if u.year < 1950 else 1] for u in ld.usages}
    breakdown = {"existing": rho_breakdown(ld, base_lab), "siamgnn": rho_breakdown(ld, gnn_lab),
                 "era_baseline": rho_breakdown(ld, era_lab)}
    for k, v in breakdown.items():
        print("rho breakdown", k, v)

    # thresholds the existing method picks in each leave-one-word-out fold (same grid as exp_main.py)
    grid = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    cache = {t: {w: agglomerative(X, threshold=t) for w, (us, X) in groups.items()} for t in grid}
    lowo = {}
    for w in words:
        ld_o = type(ld)(ld.lang, [u for u in ld.usages if u.word != w], ld.definitions,
                        {k: v for k, v in ld.gold_s1.items() if k[0] != w}, ld.gold_s2)
        others = {k: v for k, v in groups.items() if k != w}

        def mean_score(t):
            r = score_subtask1(ld_o, {(u.word, u.sentence_id): [int(l)] for ww, (us, X) in others.items()
                                      for u, l in zip(us, cache[t][ww])})
            return (r["bcubed_f"] + r["spearman"]) / 2
        lowo[w] = max(grid, key=mean_score)
    print("LOWO thresholds (existing method):", lowo)

    # Subtask 2 (per-word F1 from the LOWO experiment; AUC recomputed here)
    s2 = {r["system"]: r for r in json.loads((RES / "exp_s2_final.json").read_text(encoding="utf-8"))}
    G = get_embeddings([gloss_item(w, ld.definitions[w], "{w}: {d}") for w in words], tag="defs_SV_wcd")

    per_word, pooled = {}, {"same_within": [], "same_across": [], "diff": []}
    for gi, w in zip(G, words):
        us, X = groups[w]
        y = np.array([ld.gold_s2[(u.word, u.sentence_id)] for u in us])
        S = X @ X.T
        lab = [set(ld.gold_s1[(u.word, u.sentence_id)]) for u in us]
        eras = [era(u) for u in us]
        cos = {"same_within": [], "same_across": [], "diff": []}
        iu, ju = np.triu_indices(len(us), 1)
        for i, j in zip(iu, ju):
            key = ("same_within" if eras[i] == eras[j] else "same_across") if lab[i] & lab[j] else "diff"
            cos[key].append(S[i, j])
        for k in cos:
            pooled[k] += cos[k]
        per_word[w] = {
            "n": len(us),
            "periods": len({u.period for u in us}),
            "k_gold": base["per_word"][w]["k_gold"],
            "existing": {"k": base["per_word"][w]["k_pred"], "b3": base["per_word"][w]["F"]},
            "siamgnn": {"k": gnn["per_word"][w]["k_pred"], "b3": gnn["per_word"][w]["F"]},
            "s2_pos_rate": float(y.mean()),
            "s2_f1_existing": s2["existing: gloss cosine + threshold"]["per_word"][w],
            "s2_f1_prototype": s2["few-shot prototype refinement"]["per_word"][w],
            "s2_auc_existing": float(roc_auc_score(y, X @ gi)),
            "s2_auc_prototype": float(roc_auc_score(y, prototype_scores(X, gi, top=0.05, n_iter=3))),
            "cos": {k: (float(np.mean(v)) if v else None) for k, v in cos.items()},
            "definition": ld.definitions[w],
        }
        pw = per_word[w]
        print(f"{w:10s} n={pw['n']:3d} kgold={pw['k_gold']} k_ex={pw['existing']['k']} k_gnn={pw['siamgnn']['k']} "
              f"B3 {pw['existing']['b3']:.3f}/{pw['siamgnn']['b3']:.3f}  S2pos={pw['s2_pos_rate']:.2f} "
              f"AUC {pw['s2_auc_existing']:.2f}/{pw['s2_auc_prototype']:.2f}  cos "
              + " ".join(f"{k}={v:.3f}" if v is not None else f"{k}=NA" for k, v in pw["cos"].items()))
    pooled = {k: float(np.mean(v)) for k, v in pooled.items()}
    print("pooled cosines:", pooled)
    out = {"data": stats, "per_word": per_word, "pairs": pooled, "rho_breakdown": breakdown,
           "lowo_thresholds_existing": lowo,
           "check": {"existing": [base["bcubed_f"], base["spearman"]], "siamgnn_ens": [gnn["bcubed_f"], gnn["spearman"]]}}
    (RES / "report_stats.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("written", RES / "report_stats.json")


if __name__ == "__main__":
    main()
