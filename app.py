from pathlib import Path
from types import SimpleNamespace

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.anfis import ANFIS
from src.attribution import lime_local, rule_attribution, shap_blackbox, shap_tree
from src.consistency import compare
from src.data import DATASETS, TITLES

ROOT = Path(__file__).resolve().parent
TAB, FIG, MODELS = ROOT / "results" / "tables", ROOT / "results" / "figures", ROOT / "results" / "models"
BLUE, AQUA, VIOLET, YELLOW, MAGENTA, MUTED = "#2a78d6", "#1baf7a", "#4a3aa7", "#eda100", "#e87ba4", "#898781"
VARIANT_NAMES = {"gauss-1": "Gaussian memberships, first-order TSK", "bell-1": "Bell memberships, first-order TSK",
                 "gauss-0": "Gaussian memberships, zero-order TSK", "bell-0": "Bell memberships, zero-order TSK"}

st.set_page_config(page_title="ANFIS vs SHAP audit", layout="wide")

if not (TAB / "performance.csv").exists():
    st.error("No results found. Run run_experiments.py and make_figures.py first.")
    st.stop()


@st.cache_resource
def load_bundle(name):
    return joblib.load(MODELS / f"{name}_seed0.joblib")


def build_anfis(bundle, n_rules, variant="gauss-1"):
    mf, order = variant.split("-")
    model = ANFIS(n_rules=n_rules, mf=mf, order=int(order))
    if variant == "gauss-1":
        model.set_params(bundle["anfis"][n_rules]["params"])
    else:
        model.set_params(bundle["variants"][variant])
    return model


def bar_panel(ax, values, names, color, title):
    order = np.argsort(np.abs(values))[-8:]
    ax.barh(range(len(order)), values[order], color=color, height=0.6)
    ax.set_yticks(range(len(order)), [names[j] for j in order], fontsize=8)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.tick_params(axis="x", labelsize=8, colors=MUTED)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


st.title("Are ANFIS's fuzzy rules really interpretable?")
st.caption("A SHAP-based consistency audit on three diabetes datasets with 8, 16 and 21 attributes")

qp = st.query_params
TABS = {"patient": "Explain a patient", "results": "Experiment results", "data": "Datasets and models"}
tab_patient, tab_results, tab_data = st.tabs(list(TABS.values()),
                                             default=TABS.get(qp.get("tab", "patient"), TABS["patient"]))

with tab_patient:
    left, right = st.columns([1, 3])
    with left:
        start = qp.get("dataset", "pima")
        name = st.selectbox("Dataset", DATASETS, format_func=lambda n: TITLES[n],
                            index=DATASETS.index(start) if start in DATASETS else 0)
        bundle = load_bundle(name)
        options = ["gauss-1"] + [v for v in VARIANT_NAMES if v != "gauss-1" and v in bundle.get("variants", {})]
        want = qp.get("variant", "gauss-1")
        variant = st.selectbox("ANFIS variant", options, format_func=VARIANT_NAMES.get,
                               index=options.index(want) if want in options else 0)
        if variant == "gauss-1":
            want_r = int(qp.get("rules", 4))
            n_rules = st.select_slider("Number of fuzzy rules", options=sorted(bundle["anfis"]),
                                       value=want_r if want_r in bundle["anfis"] else 4)
        else:
            n_rules = 4
            st.caption("The membership and TSK-order variants were trained with 4 rules.")
        idx = st.number_input("Test patient number", 0, len(bundle["y_test"]) - 1,
                              min(int(qp.get("patient", 0)), len(bundle["y_test"]) - 1))
    features = bundle["features"]
    x = bundle["X_test"][idx:idx + 1]
    anfis, lgbm = build_anfis(bundle, n_rules, variant), bundle["lgbm"]
    z, wbar, f = anfis.forward(x)
    p_anfis = float(anfis.predict_proba(x)[0])
    p_lgbm = float(lgbm.predict_proba(x)[0, 1])

    with left:
        st.metric("Actual label", "Diabetic" if bundle["y_test"][idx] == 1 else "Not diabetic")
        st.metric("ANFIS: probability of diabetes", f"{p_anfis:.1%}")
        st.metric("LightGBM: probability of diabetes", f"{p_lgbm:.1%}")
        original = x[0] * bundle["std"] + bundle["mean"]
        st.dataframe(pd.DataFrame({"attribute": features, "value": np.round(original, 2)}),
                     hide_index=True, height=300)

    with right:
        bg = bundle["background"]
        rule = rule_attribution(anfis, x)
        shap_a, _, method = shap_blackbox(anfis.decision_function, x, bg)
        shap_l, _ = shap_tree(lgbm, x, bg)
        lime_a, _ = lime_local(anfis.decision_function, x, bg)
        st.subheader("What pushed this prediction? (log-odds; right = towards diabetic)")
        first_order = anfis.order == 1
        panels = ([(rule["consequent"][0], BLUE, "ANFIS rules, THEN-part")] if first_order else []) + [
            (rule["full"][0], AQUA, "ANFIS rules, IF + THEN"),
            (shap_a[0], VIOLET, f"SHAP on ANFIS ({method})"),
            (lime_a[0], MAGENTA, "LIME on ANFIS"),
            (shap_l[0], YELLOW, "SHAP on LightGBM (TreeSHAP)")]
        fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.6), constrained_layout=True)
        for ax, (vals, color, title) in zip(axes, panels):
            bar_panel(ax, vals, features, color, title)
        st.pyplot(fig)
        plt.close(fig)
        if not first_order:
            st.caption("Zero-order TSK: every rule's THEN part is a constant, so the rules explain only through "
                       "which rules fire (the IF part).")

        explanations = ([("ANFIS rules, THEN-part", rule["consequent"])] if first_order else []) + [
            ("ANFIS rules, IF-part", rule["antecedent"]), ("ANFIS rules, IF + THEN", rule["full"]),
            ("LIME on ANFIS", lime_a), ("SHAP on LightGBM", shap_l)]
        rows = []
        for label, vals in explanations:
            m = compare(vals, shap_a)
            rows.append({"explanation": label, "Spearman": m["spearman_local"],
                         "Kendall": m["kendall_local"], "same top attribute": bool(m["top1_match"]),
                         "top-3 overlap": m["top3_overlap"], "sign agreement": m["sign_agreement"]})
        st.markdown("**Agreement with SHAP on ANFIS for this patient**")
        st.dataframe(pd.DataFrame(rows).round(3), hide_index=True)

        st.markdown("**Deletion check for this patient** (no reference explainer needed)")
        k = min(3, len(features))
        del_rows = []
        for label, vals in [(l, v) for l, v in explanations if not l.startswith("SHAP on LightGBM")] + \
                           [("SHAP on ANFIS", shap_a)]:
            top = np.argsort(-np.abs(vals[0]))[:k]
            Xb = np.repeat(x, len(bg), axis=0)
            Xb[:, top] = bg[:, top]
            new_p = float(sigmoid(anfis.decision_function(Xb)).mean())
            del_rows.append({"explanation": label, f"its top {k} attributes": ", ".join(features[j] for j in top),
                             "probability after removing them": f"{new_p:.1%}",
                             "change": f"{(new_p - p_anfis) * 100:+.1f} points"})
        st.dataframe(pd.DataFrame(del_rows), hide_index=True)
        st.caption(f"Each explanation's {k} most important attributes are replaced with values from the "
                   f"{len(bg)} background patients and ANFIS is run again. A faithful explanation picks the "
                   "attributes whose removal changes the prediction most.")

        st.subheader("Rules that fired")
        split_info = SimpleNamespace(features=features, mean=bundle["mean"], std=bundle["std"])
        originals = bundle["X_test"] * bundle["std"] + bundle["mean"]
        is_binary = [len(np.unique(np.round(col, 6))) <= 2 for col in originals.T]
        described = anfis.describe_rules(split_info, binary=is_binary)
        for r in np.argsort(-wbar[0]):
            d = described[r]
            coefs = d['then_top_coefficients'].replace('*', ' × ')
            then = f"{d['then_bias']:+.2f} {coefs} ..." if first_order else f"{d['then_bias']:+.2f} (constant)"
            st.markdown(f"**Rule {d['rule']}** &nbsp; firing strength **{wbar[0][r]:.3f}**, "
                        f"rule output {f[0][r]:+.2f} log-odds  \n"
                        f"IF {d['if']}  \n"
                        f"THEN score = {then}")
        st.caption("Rules are listed from strongest to weakest for this patient. Each IF part shows the rule's "
                   "four narrowest memberships (centres in original units; MEDIUM means near the dataset average). "
                   "Each THEN part shows its constant and four largest coefficients on standardised attributes. "
                   "The prediction is the firing-strength-weighted sum of the rule outputs.")

with tab_results:
    captions = {
        "fig1_accuracy_vs_rules.png": "Test ROC-AUC of ANFIS against the rule count, with LightGBM and logistic regression as baselines (mean and one standard deviation over seeds).",
        "fig2_agreement_vs_rules.png": "How closely each explanation ranks attributes like SHAP on ANFIS, as rules are added.",
        "fig6_top3_overlap_vs_rules.png": "Share of the three most important attributes per patient that match SHAP on ANFIS.",
        "fig3_agreement_vs_attributes.png": "The same agreement when the models are given only the k most informative attributes (4 rules).",
        "fig4_global_importance.png": "Dataset-level importance from three explanations (seed 0, 4 rules).",
        "fig5_shap_cost.png": "Measured time to explain one patient as the attribute count grows.",
        "fig7_deletion_test.png": "Deletion test: how much ANFIS's output changes when each explanation's top attributes are removed (4 rules). Higher = more faithful; the dashed line is a random choice.",
        "fig8_variants.png": "Membership shape and TSK order (4 rules): accuracy and agreement of the rule explanation with SHAP.",
        "fig9_lime_check.png": "The rules compared with a second explainer, LIME, as well as with SHAP (4 rules).",
    }
    for fname, cap in captions.items():
        if (FIG / fname).exists():
            st.image(str(FIG / fname), caption=cap)
    if (TAB / "significance.csv").exists():
        st.markdown("**Significance tests** (paired Wilcoxon signed-rank, Holm-corrected)")
        sig = pd.read_csv(TAB / "significance.csv")
        st.dataframe(sig[["dataset", "test", "n_pairs", "mean_a", "mean_b", "rank_biserial", "p_holm",
                          "significant_at_0.05"]].round(4), hide_index=True)
    if (TAB / "early_stage_diagnostics.csv").exists():
        st.markdown("**Why is Early Stage different?** (diagnostics per dataset, 4 rules)")
        st.dataframe(pd.read_csv(TAB / "early_stage_diagnostics.csv").set_index("dataset").T)
    for fname in ["summary_consistency_vs_rules.csv", "summary_consistency_vs_attributes.csv"]:
        if (TAB / fname).exists():
            st.markdown(f"**{fname}**")
            st.dataframe(pd.read_csv(TAB / fname, header=[0, 1], index_col=[0, 1, 2]))

with tab_data:
    st.markdown("**Datasets** (raw files in `data/raw/`)")
    st.dataframe(pd.read_csv(TAB / "datasets.csv"), hide_index=True)
    if (TAB / "summary_performance.csv").exists():
        st.markdown("**Test performance** (mean and standard deviation over seeds)")
        st.dataframe(pd.read_csv(TAB / "summary_performance.csv", header=[0, 1], index_col=[0, 1]))
    if (TAB / "variants_summary.csv").exists():
        st.markdown("**ANFIS variants** (4 rules): ROC-AUC and agreement of each rule reading with SHAP")
        st.dataframe(pd.read_csv(TAB / "variants_summary.csv"), hide_index=True)
    info = (TAB / "run_info.json")
    if info.exists():
        st.markdown("**Run information**")
        st.json(info.read_text())
