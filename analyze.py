from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, spearmanr, wilcoxon

from src.data import DATASETS, load_raw, make_split

ROOT = Path(__file__).resolve().parent
TAB = ROOT / "results" / "tables"


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    d = a - b
    nz = d[d != 0]
    ranks = pd.Series(np.abs(nz)).rank().to_numpy()
    rbc = (ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum() if len(nz) else 0.0
    p = wilcoxon(a, b, zero_method="wilcox").pvalue if len(nz) else 1.0
    return {"n_pairs": int(len(d)), "mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "median_difference": float(np.median(d)), "rank_biserial": float(rbc), "p_value": float(p)}


def pivot(pp, filt, column="explanation"):
    d = pp
    for k, v in filt.items():
        d = d[d[k] == v]
    return d.pivot_table(index=["seed", "patient"], columns=column, values="spearman")


def main():
    pp = pd.read_csv(TAB / "per_patient.csv.gz")
    dele = pd.read_csv(TAB / "deletion.csv")
    perf = pd.read_csv(TAB / "performance.csv")
    cons = pd.read_csv(TAB / "consistency.csv")
    tests, fried = [], []

    for ds in DATASETS:
        rules_pp = pp[(pp.dataset == ds) & (pp.study == "rules")]
        then = rules_pp[rules_pp.explanation == "rule_consequent"].pivot_table(
            index=["seed", "patient"], columns="n_rules", values="spearman").dropna()
        lo, hi = then.columns.min(), then.columns.max()
        tests.append({"dataset": ds, "test": f"THEN-part agreement, {lo} rules vs {hi} rules",
                      **paired(then[lo], then[hi])})
        stat = friedmanchisquare(*[then[c] for c in then.columns])
        fried.append({"dataset": ds, "n_patients": len(then), "rule_counts": list(map(int, then.columns)),
                      "chi_square": float(stat.statistic), "p_value": float(stat.pvalue)})

        at4 = pivot(rules_pp, {"n_rules": 4})
        tests.append({"dataset": ds, "test": "4 rules: IF + THEN vs THEN-part", **paired(at4["rule_full"], at4["rule_consequent"])})
        tests.append({"dataset": ds, "test": "4 rules: THEN-part vs IF-part", **paired(at4["rule_consequent"], at4["rule_antecedent"])})
        tests.append({"dataset": ds, "test": "4 rules: LIME vs THEN-part", **paired(at4["lime_anfis"], at4["rule_consequent"])})

        var = pp[(pp.dataset == ds) & (pp.study == "variants")]
        g1 = pivot(var, {"mf": "gauss", "order": 1})
        g0 = pivot(var, {"mf": "gauss", "order": 0})
        b1 = pivot(var, {"mf": "bell", "order": 1})
        idx = g1.index.intersection(g0.index).intersection(b1.index)
        tests.append({"dataset": ds, "test": "Zero-order IF-part vs first-order THEN-part (Gaussian)",
                      **paired(g0.loc[idx, "rule_full"], g1.loc[idx, "rule_consequent"])})
        tests.append({"dataset": ds, "test": "Bell vs Gaussian memberships, THEN-part (first order)",
                      **paired(b1.loc[idx, "rule_consequent"], g1.loc[idx, "rule_consequent"])})

        dd = dele[(dele.dataset == ds) & (dele.study == "rules") & (dele.k == 3) & (dele.removal == "background")]
        dp = dd.pivot_table(index=["seed", "n_rules"], columns="explanation", values="mean_abs_change")
        tests.append({"dataset": ds, "test": "Deletion (top 3): THEN-part vs random order",
                      **paired(dp["rule_consequent"], dp["random"])})
        tests.append({"dataset": ds, "test": "Deletion (top 3): SHAP vs THEN-part",
                      **paired(dp["shap_anfis"], dp["rule_consequent"])})

    sig = pd.DataFrame(tests)
    sig["p_holm"] = holm(sig.p_value)
    sig["significant_at_0.05"] = sig.p_holm < 0.05
    sig.to_csv(TAB / "significance.csv", index=False)
    pd.DataFrame(fried).to_csv(TAB / "friedman_rules.csv", index=False)

    d = dele[dele.study == "rules"]
    d.groupby(["dataset", "removal", "k", "explanation"]).mean_abs_change.agg(["mean", "std"]).round(4) \
        .to_csv(TAB / "deletion_summary.csv")

    v = perf[perf.model.str.startswith("ANFIS (")]
    acc = v.groupby(["dataset", "mf", "order"]).roc_auc.agg(["mean", "std"]).rename(columns=lambda c: f"auc_{c}")
    cv = cons[(cons.study == "variants") & (cons.reference == "shap_anfis")]
    agr = cv.pivot_table(index=["dataset", "mf", "order"], columns="explanation", values="spearman_local")
    acc.join(agr).round(4).to_csv(TAB / "variants_summary.csv")

    rows = []
    for ds in DATASETS:
        X, _ = load_raw(ds)
        sp = make_split(ds, 0)
        corr = np.abs(np.corrcoef(sp.X_train, rowvar=False))
        off = corr[~np.eye(len(corr), dtype=bool)]
        p4 = pp[(pp.dataset == ds) & (pp.study == "rules") & (pp.n_rules == 4) & (pp.explanation == "rule_consequent")]
        b = joblib.load(ROOT / "results" / "models" / f"{ds}_seed0.joblib")
        sv = np.abs(b["anfis"][4]["attributions"]["shap_anfis"])
        small = (sv < 0.1 * sv.max(axis=1, keepdims=True)).mean()
        median = p4.shap_top3_share.median()
        rows.append({
            "dataset": ds,
            "attributes": X.shape[1],
            "share_yes_no_attributes": float((X.nunique() <= 2).mean()),
            "mean_abs_correlation_between_attributes": float(off.mean()),
            "mean_shap_top3_share": float(p4.shap_top3_share.mean()),
            "share_attributes_below_10pct_of_top": float(small),
            "mean_max_firing": float(p4.max_firing.mean()),
            "spearman_vs_top3_share": float(spearmanr(p4.shap_top3_share, p4.spearman, nan_policy="omit").statistic),
            "spearman_vs_max_firing": float(spearmanr(p4.max_firing, p4.spearman, nan_policy="omit").statistic),
            "agreement_when_concentrated": float(p4[p4.shap_top3_share >= median].spearman.mean()),
            "agreement_when_spread": float(p4[p4.shap_top3_share < median].spearman.mean()),
        })
    pd.DataFrame(rows).round(4).to_csv(TAB / "early_stage_diagnostics.csv", index=False)
    print(sig[["dataset", "test", "mean_a", "mean_b", "rank_biserial", "p_holm"]].round(4).to_string())
    print(pd.DataFrame(fried).round(4).to_string())
    print(pd.DataFrame(rows).round(3).T.to_string())


if __name__ == "__main__":
    main()
