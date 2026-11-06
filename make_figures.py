from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter
import numpy as np
import pandas as pd

from src.data import DATASETS, TITLES

OUT = Path(__file__).resolve().parent / "results"
TAB, FIG = OUT / "tables", OUT / "figures"

BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"
MAGENTA = "#e87ba4"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e6e5e1"
EXPLANATIONS = {
    "rule_consequent": ("ANFIS rules, THEN-part", BLUE, "o"),
    "rule_antecedent": ("ANFIS rules, IF-part", ORANGE, "s"),
    "rule_full": ("ANFIS rules, IF + THEN", AQUA, "^"),
    "shap_lgbm": ("SHAP on LightGBM", YELLOW, "D"),
}

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False, "savefig.dpi": 200,
})


def panels(n=3, height=3.6):
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, height), constrained_layout=True)
    return fig, axes


def save(fig, name):
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def line_with_band(ax, stats, label, color, marker):
    x = stats.index.to_numpy()
    ax.plot(x, stats["mean"], color=color, marker=marker, markersize=6, linewidth=2, label=label)
    ax.fill_between(x, stats["mean"] - stats["std"], stats["mean"] + stats["std"],
                    color=color, alpha=0.15, linewidth=0)


def fig_performance(perf):
    fig, axes = panels()
    for ax, name in zip(axes, DATASETS):
        p = perf[perf.dataset == name]
        a = p[p.model == "ANFIS"].groupby("n_rules").roc_auc.agg(["mean", "std"])
        line_with_band(ax, a, "ANFIS", BLUE, "o")
        for model, color, style in [("LightGBM", ORANGE, "--"), ("LogisticRegression", AQUA, ":")]:
            ax.axhline(p[p.model == model].roc_auc.mean(), color=color, linestyle=style,
                       linewidth=2, label=model)
        ax.set_title(f"{TITLES[name]} ({int(p.n_attributes.iloc[0])} attributes)")
        ax.set_xlabel("Number of fuzzy rules")
        ax.set_xticks(a.index)
    axes[0].set_ylabel("Test ROC-AUC")
    axes[0].legend(loc="lower right")
    save(fig, "fig1_accuracy_vs_rules.png")


def fig_consistency(cons, study, xcol, xlabel, metric, ylabel, fname):
    fig, axes = panels()
    for ax, name in zip(axes, DATASETS):
        c = cons[(cons.dataset == name) & (cons.study == study)]
        for key, (label, color, marker) in EXPLANATIONS.items():
            stats = c[c.explanation == key].groupby(xcol)[metric].agg(["mean", "std"])
            line_with_band(ax, stats, label, color, marker)
        ax.set_title(TITLES[name])
        ax.set_xlabel(xlabel)
        ax.set_xticks(sorted(c[xcol].unique()))
        ax.set_ylim(-0.1, 1.05)
    axes[0].set_ylabel(ylabel)
    axes[0].legend(loc="lower left", fontsize=8.5)
    save(fig, fname)


def fig_global_importance():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
    series = [("rule_full", "ANFIS rules (IF + THEN)", AQUA), ("shap_anfis", "SHAP on ANFIS", VIOLET),
              ("shap_lgbm", "SHAP on LightGBM", YELLOW)]
    rows = []
    for ax, name in zip(axes, DATASETS):
        b = joblib.load(OUT / "models" / f"{name}_seed0.joblib")
        attrs = b["anfis"][4]["attributions"]
        imp = {k: np.abs(attrs[k]).mean(axis=0) for k, _, _ in series}
        imp = {k: v / v.sum() for k, v in imp.items()}
        order = np.argsort(-imp["shap_anfis"])[:8][::-1]
        y = np.arange(len(order))
        for i, (k, label, color) in enumerate(series):
            ax.barh(y + (1 - i) * 0.27, imp[k][order], height=0.24, color=color, label=label)
        ax.set_yticks(y, [b["features"][j] for j in order])
        ax.tick_params(axis="y", colors=INK)
        ax.grid(axis="y", visible=False)
        ax.set_title(TITLES[name])
        ax.set_xlabel("Share of total importance")
        for j, f in enumerate(b["features"]):
            rows.append({"dataset": name, "attribute": f, **{k: imp[k][j] for k, _, _ in series}})
    axes[0].legend(loc="lower right", fontsize=8.5)
    save(fig, "fig4_global_importance.png")
    pd.DataFrame(rows).round(4).to_csv(TAB / "global_importance_seed0_rules4.csv", index=False)


def fig_shap_cost(timing):
    fig, axes = panels()
    t = timing[timing.study == "attributes"]
    for ax, name in zip(axes, DATASETS):
        d = t[t.dataset == name]
        for col, label, color, marker in [
                ("blackbox_shap_sec_per_patient", "Black-box SHAP on ANFIS", VIOLET, "o"),
                ("tree_shap_sec_per_patient", "TreeSHAP on LightGBM", YELLOW, "D")]:
            stats = (d.groupby("n_attributes")[col].agg(["mean", "std"]) * 1000)
            ax.plot(stats.index, stats["mean"], color=color, marker=marker, markersize=6,
                    linewidth=2, label=label)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(TITLES[name])
        ax.set_xlabel("Number of attributes")
        ax.set_xticks(sorted(d.n_attributes.unique()))
    axes[0].set_ylabel("Milliseconds per explained patient (log scale)")
    axes[0].legend(loc="upper left", fontsize=8.5)
    save(fig, "fig5_shap_cost.png")


def summary_tables(perf, cons):
    fmt = lambda g: g.agg(["mean", "std"]).round(4)
    p = perf.assign(model=np.where(perf.model == "ANFIS",
                                   "ANFIS-" + perf.n_rules.fillna(0).astype(int).astype(str),
                                   perf.model))
    fmt(p.groupby(["dataset", "model"], sort=False)[["accuracy", "balanced_accuracy", "f1", "roc_auc"]]
        ).to_csv(TAB / "summary_performance.csv")
    metrics = ["spearman_local", "kendall_local", "top1_match", "top3_overlap",
               "sign_agreement", "spearman_global"] + (["weighted_kendall"] if "weighted_kendall" in cons else [])
    for study, key in [("rules", "n_rules"), ("attributes", "n_attributes")]:
        c = cons[cons.study == study]
        fmt(c.groupby(["dataset", key, "explanation"], sort=False)[metrics]
            ).to_csv(TAB / f"summary_consistency_vs_{study}.csv")
    print("wrote summary tables")


def fig_deletion():
    d = pd.read_csv(TAB / "deletion.csv")
    d = d[(d.study == "rules") & (d.n_rules == 4) & (d.removal == "background")]
    series = [("shap_anfis", "SHAP on ANFIS", VIOLET, "-"), ("lime_anfis", "LIME on ANFIS", MAGENTA, "-"),
              ("rule_consequent", "ANFIS rules, THEN-part", BLUE, "-"), ("rule_full", "ANFIS rules, IF + THEN", AQUA, "-"),
              ("random", "Random order (control)", MUTED, "--")]
    fig, axes = panels()
    for ax, name in zip(axes, DATASETS):
        x = d[d.dataset == name]
        for key, label, color, style in series:
            stats = x[x.explanation == key].groupby("k").mean_abs_change.agg(["mean", "std"])
            ax.plot(stats.index, stats["mean"], color=color, linestyle=style, marker="o", markersize=6,
                    linewidth=2, label=label)
        ax.set_title(TITLES[name])
        ax.set_xlabel("Attributes removed (most important first)")
        ax.set_xticks([1, 2, 3])
    axes[0].set_ylabel("Mean change in model output (log-odds)")
    axes[0].legend(loc="lower right", fontsize=8.5)
    save(fig, "fig7_deletion_test.png")


def fig_variants(cons, perf):
    c = cons[(cons.study == "variants")]
    v = perf[perf.model.str.startswith("ANFIS (")]
    labels = [("gauss", 1, "Gaussian,\nfirst order"), ("bell", 1, "Bell,\nfirst order"),
              ("gauss", 0, "Gaussian,\nzero order"), ("bell", 0, "Bell,\nzero order")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.9), constrained_layout=True)
    width = 0.26
    for i, name in enumerate(DATASETS):
        auc_vals, agr_vals = [], []
        for mf, order, _ in labels:
            auc_vals.append(v[(v.dataset == name) & (v.mf == mf) & (v.order == order)].roc_auc.mean())
            key = "rule_consequent" if order == 1 else "rule_full"
            agr_vals.append(c[(c.dataset == name) & (c.mf == mf) & (c.order == order)
                              & (c.explanation == key)].spearman_local.mean())
        xs = np.arange(len(labels)) + (i - 1) * width
        color = [BLUE, ORANGE, AQUA][i]
        axes[0].bar(xs, auc_vals, width * 0.92, color=color, label={"pima": "Pima", "early": "Early Stage", "cdc": "CDC"}[name])
        axes[1].bar(xs, agr_vals, width * 0.92, color=color, label=TITLES[name])
    for ax, title in [(axes[0], "Test ROC-AUC"), (axes[1], "Rule explanation vs SHAP (Spearman)")]:
        ax.set_xticks(np.arange(len(labels)), [l for _, _, l in labels])
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="x", visible=False)
    for ax in axes:
        ax.set_ylim(0, 1.22)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axes[0].legend(loc="upper left", ncols=3, fontsize=8.5)
    save(fig, "fig8_variants.png")


def fig_lime(cons_all):
    c = cons_all[(cons_all.study == "rules") & (cons_all.n_rules == 4)]
    bars = [("rule_consequent", "shap_anfis", "Rules THEN-part vs SHAP", BLUE),
            ("rule_consequent", "lime_anfis", "Rules THEN-part vs LIME", AQUA),
            ("lime_anfis", "shap_anfis", "LIME vs SHAP", MAGENTA)]
    fig, ax = plt.subplots(figsize=(8.5, 3.8), constrained_layout=True)
    width = 0.26
    for j, (ex, ref, label, color) in enumerate(bars):
        vals = [c[(c.dataset == n) & (c.explanation == ex) & (c.reference == ref)].spearman_local.mean() for n in DATASETS]
        xs = np.arange(3) + (j - 1) * width
        ax.bar(xs, vals, width * 0.92, color=color, label=label)
        for x_, v_ in zip(xs, vals):
            ax.text(x_, v_ + 0.02, f"{v_:.2f}", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(range(3), [TITLES[n] for n in DATASETS])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean per-patient Spearman")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", fontsize=8.5, ncols=3)
    save(fig, "fig9_lime_check.png")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    perf = pd.read_csv(TAB / "performance.csv")
    cons_all = pd.read_csv(TAB / "consistency.csv")
    cons = cons_all[cons_all.reference == "shap_anfis"] if "reference" in cons_all else cons_all
    timing = pd.read_csv(TAB / "shap_timing.csv")
    fig_performance(perf)
    fig_consistency(cons, "rules", "n_rules", "Number of fuzzy rules", "spearman_local",
                    "Rank agreement with SHAP on ANFIS\n(mean per-patient Spearman)",
                    "fig2_agreement_vs_rules.png")
    fig_consistency(cons, "attributes", "n_attributes", "Number of attributes used",
                    "spearman_local",
                    "Rank agreement with SHAP on ANFIS\n(mean per-patient Spearman)",
                    "fig3_agreement_vs_attributes.png")
    fig_global_importance()
    fig_shap_cost(timing)
    fig_consistency(cons, "rules", "n_rules", "Number of fuzzy rules", "top3_overlap",
                    "Top-3 attribute overlap with SHAP on ANFIS", "fig6_top3_overlap_vs_rules.png")
    summary_tables(perf, cons)
    if (TAB / "deletion.csv").exists():
        fig_deletion()
        fig_variants(cons, perf)
        fig_lime(cons_all)


if __name__ == "__main__":
    main()
