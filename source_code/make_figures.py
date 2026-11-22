"""Step 24: generate the paper's figures from cached results."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd

from . import config

# Compact figures with large fonts so a single IEEEtran column reads without
# zooming. Saved as PDF (vector) so LaTeX rescaling never blurs text.
plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.constrained_layout.use": False,
    "axes.edgecolor": "#4a4a4a",
    "axes.grid": True,
    "grid.color": "#dfe6ec",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

# House palette: a blue -> light-yellow scale. Blues carry the baseline /
# comparison models, gold highlights our proposed model (SP-CGA-GAT) so the
# paper's headline result reads instantly even in grayscale print. Blue-gold
# is also colorblind-safe, unlike blue-red.
BLUE_DARK = "#154360"
BLUE_MED = "#2E86AB"
BLUE_LIGHT = "#8FBFE0"
GOLD = "#E8A33D"
GOLD_LIGHT = "#F2C14E"
GRAY_FLOOR = "#8C97A0"  # non-learned floor/persistence baselines

_BLUE_TO_GOLD = LinearSegmentedColormap.from_list(
    "blue_to_gold", [BLUE_DARK, BLUE_MED, BLUE_LIGHT, GOLD_LIGHT]
)


def _gradient(n):
    """n evenly spaced colors along the blue->gold house palette."""
    if n == 1:
        return [BLUE_MED]
    return [_BLUE_TO_GOLD(i / (n - 1)) for i in range(n)]


# Canonical display labels, matching the tables in the paper.
DISPLAY_NAMES = {
    "centrality-ranking": "Persistence",
    "persistence-floor": "Persistence",
    "trend-momentum": "Trend/momentum",
    "mlp": "MLP",
    "xgboost": "XGBoost",
    "gcn": "GCN",
    "sage": "GraphSAGE",
    "gat": "GAT",
    "cga-gat": "CGA-GAT",
    "gat-res": "GAT-Res",
    "cga-gat-res": "CGA-GAT-Res",
    "spcga-gat": "SP-CGA-GAT",
}


# Endpoint kept mid-saturation (not pale) so the lightest line still reads
# clearly against the grid when several baselines cluster near the top.
_BLUES = LinearSegmentedColormap.from_list("blues", [BLUE_DARK, "#5DA8D8"])
_MARKERS = ["o", "^", "D", "v", "P", "X", "*"]


def _blues(n):
    """n evenly spaced blue shades (dark -> light) for baseline lines."""
    if n == 1:
        return [BLUE_MED]
    return [_BLUES(i / (n - 1)) for i in range(n)]


def _line_styles(names):
    """Gray dashed for the non-learned floor, gold solid for our best
    model, graded blue + distinct markers for everything else -- keeps the
    eye on what matters and keeps adjacent blue shades tellable apart when
    lines cross or cluster."""
    others = [n for n in names if n not in ("centrality-ranking", "persistence-floor", "spcga-gat")]
    blues = dict(zip(others, _blues(len(others))))
    markers = dict(zip(others, _MARKERS))
    styles = {}
    for name in names:
        if name in ("centrality-ranking", "persistence-floor"):
            styles[name] = dict(color=GRAY_FLOOR, linestyle="--", linewidth=1.4, marker="s", markersize=5)
        elif name == "spcga-gat":
            styles[name] = dict(color=GOLD, linestyle="-", linewidth=2.8, marker="o", markersize=7, zorder=5)
        else:
            styles[name] = dict(color=blues[name], linestyle="-", linewidth=1.4,
                                 marker=markers[name], markersize=5.5, alpha=0.95)
    return styles


def fig_main_comparison(horizon=1):
    with open(config.RESULTS / f"main_comparison_h{horizon}.json") as f:
        results = json.load(f)
    order = ["centrality-ranking", "mlp", "xgboost", "gcn", "sage", "gat",
              "cga-gat", "gat-res", "cga-gat-res", "spcga-gat"]
    names = [n for n in order if n in results]
    labels = [DISPLAY_NAMES[n] for n in names]
    f1s = [results[n]["overall"]["f1"] for n in names]
    aucs = [results[n]["overall"]["auc"] for n in names]

    x = range(len(names))
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    width = 0.35
    bars_f1 = ax.bar([i - width / 2 for i in x], f1s, width, label="F1", color=BLUE_DARK)
    bars_auc = ax.bar([i + width / 2 for i in x], aucs, width, label="ROC-AUC", color=GOLD)
    for bar, name in zip(bars_f1, names):
        if name == "spcga-gat":
            bar.set_edgecolor(BLUE_DARK)
            bar.set_linewidth(1.6)
            bar.set_color(BLUE_MED)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Main model comparison (t+{horizon})")
    ax.legend(loc="lower right", framealpha=0.95, ncol=2)
    fig.tight_layout(pad=0.4)
    fig.savefig(config.FIGURES / f"fig2_main_comparison_h{horizon}.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_horizon():
    with open(config.RESULTS / "horizon_experiment.json") as f:
        results = json.load(f)
    order = ["centrality-ranking", "mlp", "sage", "gat", "cga-gat", "cga-gat-res", "spcga-gat"]
    names = [n for n in order if n in results]
    styles = _line_styles(names)
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for name in names:
        per_h = results[name]
        hs = sorted(int(h) for h in per_h.keys())
        aucs = [per_h[str(h)]["auc"] for h in hs]
        ax.plot(hs, aucs, label=DISPLAY_NAMES[name], **styles[name])
    ax.set_xlabel("Forecast horizon (days)")
    ax.set_ylabel("AUC")
    ax.set_xticks(sorted(int(h) for h in next(iter(results.values())).keys()))
    ax.set_title("AUC vs. forecast horizon")
    ax.legend(fontsize=8.5, loc="best", framealpha=0.95, ncol=2)
    fig.tight_layout(pad=0.4)
    fig.savefig(config.FIGURES / "fig3_horizon.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_ablation(tag="full", horizon=1):
    with open(config.RESULTS / f"centrality_ablation_{tag}_h{horizon}.json") as f:
        results = json.load(f)
    order = ["gat", "gat_nocent", "cga-gat", "cga-gat_attnonly"]
    names = [n for n in order if n in results]
    f1s = [results[n]["f1_mean"] for n in names]
    f1_err = [results[n]["f1_std"] for n in names]
    aucs = [results[n]["auc_mean"] for n in names]
    auc_err = [results[n]["auc_std"] for n in names]

    x = range(len(names))
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    width = 0.35
    ax.bar([i - width / 2 for i in x], f1s, width, yerr=f1_err, capsize=4, label="F1")
    ax.bar([i + width / 2 for i in x], aucs, width, yerr=auc_err, capsize=4, label="ROC-AUC")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Deconfounded ablation (3 seeds)")
    ax.legend(fontsize=10, loc="lower right", framealpha=0.9)
    fig.tight_layout(pad=0.4)
    fig.savefig(config.FIGURES / f"fig5_ablation_{tag}_h{horizon}.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_emergence(horizon=1):
    with open(config.RESULTS / f"emergence_comparison_h{horizon}.json") as f:
        results = json.load(f)
    order = ["persistence-floor", "trend-momentum", "mlp", "xgboost", "gcn", "sage",
              "gat", "cga-gat", "gat-res", "cga-gat-res", "spcga-gat"]
    names = [n for n in order if n in results]
    labels_x = [DISPLAY_NAMES[n] for n in names]

    def _get(n, key):
        r = results[n]
        if f"{key}_mean" in r:
            return r[f"{key}_mean"], r[f"{key}_std"]
        return r["overall"][key], 0.0

    metrics = ["f1", "auc", "pr_auc", "precision_at_k"]
    metric_labels = ["F1", "AUC", "PR-AUC", "P@k"]
    colors = _gradient(len(metrics))
    x = range(len(names))
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    width = 0.8 / len(metrics)
    for mi, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        vals = [_get(n, metric)[0] for n in names]
        errs = [_get(n, metric)[1] for n in names]
        offsets = [i + (mi - (len(metrics) - 1) / 2) * width for i in x]
        ax.bar(offsets, vals, width, yerr=errs, capsize=3, label=label, color=color)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels_x, rotation=25, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(f"Emergence task comparison (t+{horizon})")
    ax.legend(fontsize=10, ncol=2, loc="upper left", framealpha=0.95)
    fig.tight_layout(pad=0.4)
    fig.savefig(config.FIGURES / f"fig6_emergence_h{horizon}.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_emerging_case_study():
    df = pd.read_csv(config.RESULTS / "emerging_videos.csv")
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for c in pred_cols:
        ax.hist(df[c], bins=20, alpha=0.5, label=c.replace("pred_", ""))
    ax.set_xlabel("Pred. prob. of future top-20%")
    ax.set_ylabel("Count")
    ax.set_title("Emerging-video score distributions")
    ax.legend(fontsize=9, framealpha=0.9)
    fig.tight_layout(pad=0.4)
    fig.savefig(config.FIGURES / "fig4_emerging_videos.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_cross_dataset(horizon=1):
    with open(config.RESULTS / f"cross_dataset_h{horizon}.json") as f:
        results = json.load(f)
    order = ["centrality-ranking", "mlp", "gcn", "gat", "cga-gat", "gat-res", "cga-gat-res", "spcga-gat"]
    names = [n for n in order if n in results]

    def _get(n, key):
        r = results[n]["overall"]
        if f"{key}_mean" in r:
            return r[f"{key}_mean"], r[f"{key}_std"]
        return r[key], 0.0

    metrics = ["f1", "auc", "pr_auc"]
    labels = ["F1", "AUC", "PR-AUC"]
    x = range(len(names))
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    width = 0.8 / len(metrics)
    for mi, (metric, label) in enumerate(zip(metrics, labels)):
        vals = [_get(n, metric)[0] for n in names]
        errs = [_get(n, metric)[1] for n in names]
        offsets = [i + (mi - (len(metrics) - 1) / 2) * width for i in x]
        ax.bar(offsets, vals, width, yerr=errs, capsize=3, label=label)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(f"Cross-dataset generalization (t+{horizon})")
    ax.legend(fontsize=10, loc="lower right", framealpha=0.9)
    fig.tight_layout(pad=0.4)
    fig.savefig(config.FIGURES / f"fig7_cross_dataset_h{horizon}.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_cross_era():
    df = pd.read_csv(config.RESULTS / "cross_era_summary_h1.csv").set_index("model")
    order = ["centrality-ranking", "mlp", "gcn", "gat", "cga-gat", "gat-res", "spcga-gat"]
    names = [m for m in order if m in df.index]
    df = df.loc[names]
    styles = _line_styles(names)

    eras = ["2018", "2020_22", "2026"]
    x = [0, 1, 2]
    xlabels = ["2018\n(pre-training)", "2020-22\n(training)", "2026\n(post-training)"]

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for name, row in df.iterrows():
        ys = [row.get(f"auc_{e}") for e in eras]
        ax.plot(x, ys, label=DISPLAY_NAMES[name], **styles[name])
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("AUC")
    ax.set_title("AUC across three collection eras")
    ax.legend(fontsize=8.5, ncol=2, loc="lower left", framealpha=0.95)
    fig.tight_layout(pad=0.4)
    fig.savefig(config.FIGURES / "fig8_cross_era.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_main_comparison(horizon=1)
    if (config.RESULTS / "horizon_experiment.json").exists():
        fig_horizon()
    if (config.RESULTS / "centrality_ablation_full_h1.json").exists():
        fig_ablation(tag="full", horizon=1)
    if (config.RESULTS / "emergence_comparison_h1.json").exists():
        fig_emergence(horizon=1)
    if (config.RESULTS / "emerging_videos.csv").exists():
        fig_emerging_case_study()
    if (config.RESULTS / "cross_dataset_h1.json").exists():
        fig_cross_dataset(horizon=1)
    if (config.RESULTS / "cross_era_summary_h1.csv").exists():
        fig_cross_era()
    print("figures written to", config.FIGURES)
