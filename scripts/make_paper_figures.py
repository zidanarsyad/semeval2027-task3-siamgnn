"""Figures for the system-description paper (report/figures/*.pdf), built from saved results.

  threshold.pdf     existing method (XL-LEXEME + agglomerative): BCubed-F1, Spearman rho and clusters per
                    word as a function of the distance threshold, on the Swedish dev set and on the 20
                    held-out synthetic-English words (curve cached in outputs/results/threshold_curve.json)
  trajectories.pdf  sense distributions over the ten periods, gold senses vs. SiamGNN clusters
                    (zero-shot seed ensemble, correlation clustering at theta = 0.5) for herre and fru

    python scripts/make_paper_figures.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mtp.clustering import agglomerative, correlation_clustering  # noqa: E402
from mtp.data import load_language  # noqa: E402
from mtp.encoders import get_embeddings  # noqa: E402
from mtp.scorer import score_subtask1  # noqa: E402
from mtp.siamgnn import load_model, predict_pairs  # noqa: E402

FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RES = ROOT / "outputs" / "results"

# reference data-viz palette (light mode, validated: blue/orange pass every check; 8-slot order for stacks)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
COL_W, TEXT_W = 3.03, 6.30  # ACL column / text width in inches

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.labelcolor": INK, "ytick.labelcolor": INK, "axes.labelcolor": INK, "text.color": INK,
})


def load(folder, lang, tag):
    ld = load_language(ROOT / folder, lang)
    E = get_embeddings([(u.text, u.start, u.end) for u in ld.usages], tag=tag)
    idx = {(u.word, u.sentence_id): i for i, u in enumerate(ld.usages)}
    return ld, {w: (us, E[[idx[(u.word, u.sentence_id)] for u in us]]) for w, us in ld.by_word().items()}


# ---------------------------------------------------------------- threshold sensitivity
def threshold_curve():
    path = RES / "threshold_curve.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    grid = np.round(np.arange(0.25, 0.7501, 0.025), 3)
    out = {}
    for name, folder, lang, tag in [("SV", "data/raw/traindev_phase/extracted/SV", "SV", "usages_SV"),
                                    ("synEN", "data/synthetic_en/DEV20_SUB/EN", "EN", "usages_synEN_DEV")]:
        ld, groups = load(folder, lang, tag)
        rows = []
        for t in grid:
            lab = {(u.word, u.sentence_id): [int(l)] for w, (us, X) in groups.items()
                   for u, l in zip(us, agglomerative(X, threshold=float(t)))}
            r = score_subtask1(ld, lab)
            rows.append({"t": float(t), "b3": r["bcubed_f"], "rho": r["spearman"],
                         "k": float(np.mean([v["k_pred"] for v in r["per_word"].values()])),
                         "k_gold": float(np.mean([v["k_gold"] for v in r["per_word"].values()]))})
            print(f"{name} t={t:.3f} B3={rows[-1]['b3']:.3f} rho={rows[-1]['rho']:.3f} k={rows[-1]['k']:.2f}", flush=True)
        out[name] = rows
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def fig_threshold():
    cur = threshold_curve()
    fig, (ax, axk) = plt.subplots(2, 1, figsize=(COL_W, 2.65), sharex=True,
                                  gridspec_kw={"height_ratios": [1.65, 1], "hspace": 0.12})
    styles = {"SV": dict(ls="-", marker="o", ms=2.8, mfc="white", mew=0.9),
              "synEN": dict(ls=(0, (3.2, 1.6)), marker=None)}
    for ds in ["synEN", "SV"]:
        t = [r["t"] for r in cur[ds]]
        ax.plot(t, [r["b3"] for r in cur[ds]], color=BLUE, lw=1.4, **styles[ds])
        ax.plot(t, [r["rho"] for r in cur[ds]], color=ORANGE, lw=1.4, **styles[ds])
        axk.plot(t, [r["k"] for r in cur[ds]], color=INK, lw=1.2, **styles[ds])
        kg = cur[ds][0]["k_gold"]
        axk.axhline(kg, color=MUTED, lw=0.7, ls=styles[ds]["ls"])
        axk.text(0.752, kg + 0.25, f"gold senses {kg:.2f}", ha="right", va="bottom", fontsize=6.5, color=MUTED)
    for a in (ax, axk):
        a.axvline(0.4, color=MUTED, lw=0.7, ls=":")
        a.grid(axis="y", color=GRID, lw=0.5)
        a.set_axisbelow(True)
    axk.text(0.408, 9.9, "$t$ = 0.40 (selected on\nsynthetic-EN training words)", fontsize=6.5, color=MUTED, va="top")
    ax.set_ylim(-0.02, 0.92)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_ylabel("score")
    axk.set_ylim(0, 10.5)
    axk.set_yticks([0, 5, 10])
    axk.set_ylabel("clusters / word")
    axk.set_xlabel("distance threshold $t$ (agglomerative, average linkage)")
    axk.set_xlim(0.245, 0.755)
    handles = [Line2D([], [], color=BLUE, lw=1.4, label="1a B$^3$-F1"),
               Line2D([], [], color=ORANGE, lw=1.4, label="1b $\\rho$"),
               Line2D([], [], color=MUTED, lw=1.2, marker="o", ms=2.8, mfc="white", label="Swedish dev"),
               Line2D([], [], color=MUTED, lw=1.2, ls=(0, (3.2, 1.6)), label="synth. EN")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.56, 1.005),
               handlelength=1.7, columnspacing=0.8, handletextpad=0.35)
    fig.subplots_adjust(left=0.15, right=0.98, top=0.9, bottom=0.15)
    fig.savefig(FIG / "threshold.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- sense trajectories (gold vs. SiamGNN)
def distributions(us, lab_lists, periods):
    M = []
    for p in periods:
        c = Counter()
        for u, labs in zip(us, lab_lists):
            if u.period == p:
                for l in labs:
                    c[l] += 1 / len(labs)
        tot = sum(c.values())
        M.append({k: v / tot for k, v in c.items()} if tot else {})
    keys = sorted({k for m in M for k in m}, key=lambda k: -sum(m.get(k, 0) for m in M))
    return np.array([[m.get(k, 0) for k in keys] for m in M]).T  # senses x periods, rank-ordered


def fig_trajectories(words=("herre", "fru")):
    ld, groups = load("data/raw/traindev_phase/extracted/SV", "SV", "usages_SV")
    ens = json.loads((RES / "exp_ensemble.json").read_text(encoding="utf-8"))["siamgnn"]
    models = [load_model(p) for p in sorted((ROOT / "models").glob("siamgnn_synEN_seed*.pt"))]
    fig, axes = plt.subplots(2, len(words), figsize=(TEXT_W, 2.75), sharey=True)
    n_max = 0
    for j, w in enumerate(words):
        us, X = groups[w]
        P = np.mean([predict_pairs(m, X) for m in models], axis=0)
        pred = correlation_clustering(P, threshold=ens["theta"], n_restarts=10)
        periods = ld.periods(w)
        x = np.array([i if int(p[:4]) < 1950 else i + 0.8 for i, p in enumerate(periods)])  # visual era gap
        rows = [(distributions(us, [ld.gold_s1[(u.word, u.sentence_id)] for u in us], periods), "gold senses"),
                (distributions(us, [[int(l)] for l in pred], periods), "SiamGNN clusters")]
        for i, (M, name) in enumerate(rows):
            ax = axes[i, j]
            bottom = np.zeros(len(periods))
            for s in range(min(len(M), len(SERIES))):
                ax.bar(x, M[s], 0.78, bottom=bottom, color=SERIES[s], edgecolor="white", linewidth=0.6)
                bottom += M[s]
            if len(M) > len(SERIES):  # fold the tail into one grey "other" segment
                ax.bar(x, M[len(SERIES):].sum(0), 0.78, bottom=bottom, color="#b9b8b3", edgecolor="white", linewidth=0.6)
            n_max = max(n_max, min(len(M), len(SERIES)))
            ax.set_ylim(0, 1)
            ax.set_yticks([0, 0.5, 1])
            ax.set_title(f"{w}: {name} ({len(M)})", loc="left", fontsize=8)
            ax.set_xticks(x)
            ax.set_xticklabels([p[:4] if i == 1 else "" for p in periods], rotation=90, fontsize=6.5)
            ax.tick_params(axis="x", length=0)
    axes[0, 0].set_ylabel("share of usages")
    axes[1, 0].set_ylabel("share of usages")
    handles = [Patch(color=SERIES[s], label=f"#{s + 1}") for s in range(n_max)]
    fig.legend(handles=handles, loc="center right", frameon=False, title="sense / cluster\nby frequency",
               title_fontsize=7, handlelength=1.0, handleheight=0.9, borderaxespad=0.2)
    fig.subplots_adjust(left=0.07, right=0.88, top=0.92, bottom=0.17, hspace=0.42, wspace=0.08)
    fig.savefig(FIG / "trajectories.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_threshold()
    fig_trajectories()
    print("figures written to", FIG)
