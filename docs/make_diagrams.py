from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle

FIG = Path(__file__).resolve().parent.parent / "results" / "figures"
NAVY, INK, ORANGE, SLATE, TINT = "#1E2761", "#1B1F3B", "#E8734A", "#5B6B8C", "#EEF2FB"
COLS = {"data": "#DCE6F7", "prep": "#E3F1EA", "model": "#FDEBDD", "explain": "#ECE6FA", "eval": "#FFF4D6", "out": "#E6EEF5"}


def box(ax, x, y, w, h, title, lines=(), fill="white", edge=NAVY, title_color=NAVY, size=9.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=fill, edgecolor=edge, linewidth=1.2))
    ax.text(x + w / 2, y + h - 0.17, title, ha="center", va="top", fontsize=size, fontweight="bold", color=title_color)
    for i, line in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.47 - i * 0.24, line, ha="center", va="top", fontsize=size - 1.6, color=INK)


def arrow(ax, x0, y0, x1, y1, color=ORANGE, lw=2.0, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=14, color=color, linewidth=lw))


def system():
    fig, ax = plt.subplots(figsize=(17, 8.6))
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 8.6)
    ax.axis("off")
    layers = [("1  DATA", "data", 0.2), ("2  PREPARATION", "prep", 2.95), ("3  MODELS", "model", 5.7),
              ("4  EXPLANATIONS", "explain", 8.45), ("5  EVALUATION", "eval", 11.2), ("6  OUTPUTS", "out", 13.95)]
    W, top, bottom = 2.55, 7.55, 0.95
    for name, key, x in layers:
        ax.add_patch(FancyBboxPatch((x, bottom), W, top - bottom, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    facecolor=COLS[key], edgecolor="none"))
        ax.text(x + W / 2, top - 0.22, name, ha="center", va="top", fontsize=11, fontweight="bold", color=NAVY)
    bw, bh = W - 0.3, 1.15
    content = {
        0.2: [("Pima Indians Diabetes", ["768 patients, 8 attributes", "lab measurements"]),
              ("Early Stage Diabetes", ["520 patients, 16 attributes", "yes/no symptoms"]),
              ("CDC Health Indicators", ["253,680 people, 21 attrib.", "BRFSS 2015 survey"])],
        2.95: [("Cleaning", ["impossible zeros -> missing", "yes/no -> 1/0, drop ID"]),
               ("Stratified split", ["60% train / 20% val.", "20% test, 5 seeds"]),
               ("Imputation + scaling", ["median and mean/std", "from TRAINING data only"])],
        5.7: [("ANFIS (from scratch)", ["k-means rules, LSE + Adam", "Gauss/bell, order 1/0, R=2..16"]),
              ("LightGBM", ["gradient-boosted trees", "strong baseline"]),
              ("Logistic regression", ["linear baseline", "same split"])],
        8.45: [("Rule-based attribution", ["THEN-part, IF-part,", "IF + THEN (from rules)"]),
               ("SHAP + LIME on ANFIS", ["exact / permutation SHAP", "LIME local surrogate"]),
               ("TreeSHAP on LightGBM", ["exact, fast", "same 100 background"])],
        11.2: [("Agreement metrics", ["Spearman, Kendall, wtd. tau", "top-1, top-3, sign, global"]),
               ("Deletion test", ["remove top-k attributes", "measure output change"]),
               ("Statistics", ["Wilcoxon, Friedman", "Holm correction"])],
        13.95: [("Tables + figures", ["results/tables", "results/figures"]),
                ("Live dashboard", ["Streamlit, localhost:8501", "explains one patient"]),
                ("Report + article", ["generated from the", "same result files"])],
    }
    ys = [5.75, 3.85, 1.95]
    for x, items in content.items():
        for (title, lines), y in zip(items, ys):
            box(ax, x + 0.15, y, bw, bh, title, lines)
    for (_, _, x0), (_, _, x1) in zip(layers[:-1], layers[1:]):
        arrow(ax, x0 + W + 0.02, 4.2, x1 - 0.02, 4.2, lw=2.6)
    files = {0.2: "data/raw/", 2.95: "src/data.py", 5.7: "src/anfis.py", 8.45: "src/attribution.py",
             11.2: "consistency.py, analyze.py", 13.95: "make_figures.py, app.py"}
    for x, f in files.items():
        ax.text(x + W / 2, 1.13, f, ha="center", va="center", fontsize=7.8, family="monospace", color=SLATE)
    ax.add_patch(FancyBboxPatch((0.2, 0.12), 16.3, 0.55, boxstyle="round,pad=0.02,rounding_size=0.1",
                                facecolor=NAVY, edgecolor="none"))
    ax.text(8.35, 0.395, "run_experiments.py  ·  Study A: rule count  ·  B: attribute count  ·  C: SHAP cost  ·  "
            "D: membership / TSK order  ·  5 seeds, run in parallel", ha="center", va="center", fontsize=10.5,
            color="white", fontweight="bold")
    ax.text(8.5, 8.3, "System architecture of the ANFIS interpretability audit", ha="center", fontsize=14,
            fontweight="bold", color=NAVY)
    fig.savefig(FIG / "arch_system.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def anfis():
    fig, ax = plt.subplots(figsize=(15, 7.2))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    xs = [0.9, 3.4, 6.2, 8.4, 10.6, 13.0]
    heads = ["Inputs\n(attributes)", "Layer 1\nFuzzification", "Layer 2\nRule firing", "Layer 3\nNormalisation",
             "Layer 4\nConsequent", "Layer 5\nOutput"]
    forms = ["x₁ … x_d", "μ_rj(x_j)\nGaussian or bell", "w_r = (Π_j μ_rj)^(1/d)", "w̄_r = w_r / Σ_k w_k",
             "w̄_r · f_r,  f_r = a_r·x + b_r", "z = Σ_r w̄_r f_r\nP = sigmoid(z)"]
    for x, h, f in zip(xs, heads, forms):
        ax.text(x, 6.95, h, ha="center", va="top", fontsize=10.5, fontweight="bold", color=NAVY)
        ax.text(x, 0.55, f, ha="center", va="center", fontsize=9, color=SLATE)
    inputs = [5.2, 3.6, 2.0]
    in_names = ["x₁ (e.g. Glucose)", "x₂ (e.g. BMI)", "x_d (e.g. Age)"]
    rules = [5.0, 3.6, 2.2]
    mf_y = [[5.75, 5.25, 4.75], [4.25, 3.6, 2.95], [2.45, 1.95, 1.45]]
    for y, nm in zip(inputs, in_names):
        ax.add_patch(Circle((xs[0], y), 0.28, facecolor="white", edgecolor=NAVY, linewidth=1.4))
        ax.text(xs[0] - 0.4, y, nm, ha="right", va="center", fontsize=9, color=INK)
    ax.text(xs[0], 2.8, "⋮", ha="center", va="center", fontsize=16, color=SLATE)
    for r, ys in enumerate(mf_y):
        for j, y in enumerate(ys):
            ax.add_patch(FancyBboxPatch((xs[1] - 0.42, y - 0.17), 0.84, 0.34, boxstyle="round,pad=0.02,rounding_size=0.08",
                                        facecolor="#DCE6F7", edgecolor=NAVY, linewidth=1))
            ax.text(xs[1], y, f"μ_{r + 1},{['1', '2', 'd'][j]}", ha="center", va="center", fontsize=8.5)
            ax.plot([xs[0] + 0.28, xs[1] - 0.42], [inputs[j], y], color="#9AA5C8", linewidth=0.8)
            ax.plot([xs[1] + 0.42, xs[2] - 0.33], [y, rules[r]], color="#9AA5C8", linewidth=0.8)
    for r, y in enumerate(rules):
        for x, lab, fill in [(xs[2], "Π", "#FDEBDD"), (xs[3], "N", "#ECE6FA"), (xs[4], f"f{r + 1}", "#FFF4D6")]:
            ax.add_patch(Circle((x, y), 0.33, facecolor=fill, edgecolor=NAVY, linewidth=1.4))
            ax.text(x, y, lab, ha="center", va="center", fontsize=11, fontweight="bold", color=NAVY)
        ax.text(xs[2], y - 0.5, f"rule {r + 1}", ha="center", fontsize=8, color=SLATE)
        for y2 in rules:
            ax.plot([xs[2] + 0.33, xs[3] - 0.33], [y, y2], color="#9AA5C8", linewidth=0.7 if y2 != y else 1.3)
        ax.plot([xs[3] + 0.33, xs[4] - 0.33], [y, y], color=NAVY, linewidth=1.3)
        ax.plot([xs[4] + 0.33, xs[5] - 0.4], [y, 3.6], color=NAVY, linewidth=1.3)
    ax.text(xs[2], 1.45, "⋮  R rules", ha="center", fontsize=10, color=SLATE)
    ax.annotate("", xy=(xs[4], 6.0), xytext=(xs[0] + 0.3, 6.0),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, linestyle="--", linewidth=1.3))
    ax.text((xs[0] + xs[4]) / 2, 6.12, "inputs also feed the linear consequents (first order)",
            ha="center", fontsize=8.5, color=ORANGE)
    for x in [xs[4]]:
        ax.plot([x, x], [6.0, rules[0] + 0.33], color=ORANGE, linestyle="--", linewidth=1.2)
    ax.add_patch(Circle((xs[5], 3.6), 0.4, facecolor=NAVY, edgecolor=NAVY))
    ax.text(xs[5], 3.6, "Σ", ha="center", va="center", fontsize=15, color="white", fontweight="bold")
    ax.text(xs[5] + 0.6, 3.6, "P(diabetic)", ha="left", va="center", fontsize=10, color=INK, fontweight="bold")
    ax.text(xs[1], 6.35, "adaptive: centre c, width σ", ha="center", fontsize=8.3, color=ORANGE)
    ax.text(xs[4], 6.35, "adaptive: a_r, b_r", ha="center", fontsize=8.3, color=ORANGE)
    ax.text(xs[2], 6.35, "fixed", ha="center", fontsize=8.3, color=SLATE)
    ax.text(xs[3], 6.35, "fixed", ha="center", fontsize=8.3, color=SLATE)
    fig.savefig(FIG / "arch_anfis.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    system()
    anfis()
    print("wrote arch_system.png and arch_anfis.png")
