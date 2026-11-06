from deck_helpers import *

LIME_C = "E87BA4"
GREY = "9AA0AE"
SIG = {(r["dataset"], r["test"]): r for r in D.get("significance", [])}
FR, FV = D.get("firing_rules", {}), D.get("firing_variants", {})
VAR, LREF, DEL, DIAG = D.get("variants", {}), D.get("lime_ref", {}), D.get("deletion", {}), D.get("diagnostics", {})


def sig(ds, contains):
    return next(v for (d, t), v in SIG.items() if d == ds and contains in t)


def p_txt(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def why_card(s, x, y, w, h, body, title="Analysis"):
    card(s, x, y, w, h, NAVY)
    text(s, x + 0.22, y + 0.12, w - 0.44, 0.3, title, size=13, bold=True, color=ORANGE)
    text(s, x + 0.22, y + 0.45, w - 0.44, h - 0.55, body, size=12, color=WHITE, space=5)


def takeaway(s, msg, y=6.4):
    card(s, 0.6, y, 12.1, 0.5, TINT, radius=0.2)
    text(s, 0.85, y, 11.6, 0.5, [[("Inference: ", {"bold": True, "color": NAVY}), (msg, {})]], size=13, color=INK,
         anchor=MSO_ANCHOR.MIDDLE)


def fr(ds, k):
    return FR[ds][str(k)]


def results_slides(_g=None):
    lo, hi = rules[0], rules[-1]

    s = new_slide("", "Results: Classification Performance",
                  f"Test ROC-AUC, mean over {n_seeds} seeds; ANFIS with 4 rules",
                  notes="First we check that ANFIS is a good model. On all three datasets it reaches the same ROC-AUC as "
                        "LightGBM. Why? Even plain logistic regression is almost as good, so the signal in these data is "
                        "nearly additive. A few locally linear rules are enough, and extra rules have nothing more to learn.")
    series = [("Logistic regression", [auc(d, "LogisticRegression") for d in DS], SLATE),
              ("LightGBM", [auc(d, "LightGBM") for d in DS], YELLOW),
              ("ANFIS, 4 rules", [auc(d, "ANFIS-4") for d in DS], BLUE)]
    legend(s, 0.7, 1.9, [(n_, c) for n_, _, c in series])
    bar_chart(s, 0.5, 2.2, 7.6, 4.1, [f"{DS_NAME[d]} ({recs[d2]['attributes']} attr.)" for d, d2 in
                                       zip(DS, ["Pima Indians Diabetes", "Early Stage Diabetes Risk", "CDC Diabetes Health Indicators"])],
              series, vmin=0, vmax=1, fmt="0.000", ytitle="Test ROC-AUC", label_size=11)
    stat(s, 8.4, 1.9, 4.3, 1.4, f"{auc('cdc', 'ANFIS-4'):.3f} vs {auc('cdc', 'LightGBM'):.3f}",
         "ANFIS vs LightGBM on the 253,680-record CDC data", big_size=26)
    why_card(s, 8.4, 3.45, 4.3, 2.8,
             [f"Even linear logistic regression scores {auc('pima', 'LogisticRegression'):.3f} vs LightGBM's {auc('pima', 'LightGBM'):.3f} (Pima) and "
              f"{auc('cdc', 'LogisticRegression'):.3f} vs {auc('cdc', 'LightGBM'):.3f} (CDC): the signal is nearly additive.",
              "So a few locally linear rules suffice, and more rules have no interactions left to learn.",
              f"Low CDC F1 (~{PERF['cdc']['LightGBM']['f1'][0]:.2f}) is the data: 14% positives; LightGBM has the same."])
    takeaway(s, "We get the same accuracy with ANFIS as with LightGBM, so using ANFIS for its rules costs us no accuracy.")

    keys = [("rule_consequent", "Rules: THEN-part", BLUE), ("rule_antecedent", "Rules: IF-part", C_ORANGE),
            ("rule_full", "Rules: IF + THEN", AQUA), ("lime_anfis", "LIME on ANFIS", LIME_C), ("shap_lgbm", "SHAP on LightGBM", YELLOW)]
    s = new_slide("", "Results: Agreement of Rule Explanations with SHAP",
                  "Per-patient rank agreement (Spearman) with SHAP on ANFIS, 4 rules; 1.0 = identical ranking",
                  notes="This is the main result. The blue bars, the THEN-part, are what a person reads from the rules. They "
                        "agree well on Pima and CDC but only 0.54 on Early Stage. The direction of each effect is almost "
                        "always right. Why only partly? The THEN-part holds the rule weights fixed, but in the real model "
                        "changing an attribute also changes which rules fire, and SHAP sees that.")
    legend(s, 0.7, 1.9, [(lab, col) for _, lab, col in keys], gap=1.62)
    bar_chart(s, 0.5, 2.2, 8.0, 4.1, [DS_NAME[d] for d in DS], [(lab, [cons(d, k_, 4) for d in DS], col) for k_, lab, col in keys],
              vmin=0, vmax=1, fmt="0.00", ytitle="Spearman agreement", label_size=9.5)
    stat(s, 8.8, 1.9, 3.9, 1.3, f"≥ {min(sign_vals) * 100:.0f}%", "THEN-part agrees with SHAP on the direction of each effect", big_size=28)
    why_card(s, 8.8, 3.35, 3.9, 2.9,
             ["THEN-part = the fired rules' formulas with the rule weights held fixed.",
              "In the real model an attribute also changes which rules fire; SHAP sees that effect, the THEN-part does not.",
              "IF-part alone is weak: in a first-order model most of the output comes from the THEN-parts."])
    takeaway(s, "We found that the rules give the right direction of each effect, but the order of importance for one patient matches SHAP only partly.")

    s = new_slide("", "Results: Dataset-Level Feature Importance",
                  "Share of total importance of the six leading attributes (seed 0, 4 rules)",
                  notes="Averaged over all patients the explanations agree much better, and the result makes medical sense: "
                        "glucose, age and BMI on Pima; frequent urination and thirst on Early Stage; general health, age and "
                        "BMI on CDC. Patient-level disagreements partly cancel out when averaged.")
    imp_series = [("rule_full", "ANFIS rules (IF + THEN)", AQUA), ("shap_anfis", "SHAP on ANFIS", VIOLET), ("shap_lgbm", "SHAP on LightGBM", YELLOW)]
    legend(s, 0.7, 1.9, [(lab, col) for _, lab, col in imp_series], gap=3.0)
    for n_, ds in enumerate(DS):
        x = 0.45 + n_ * 4.2
        text(s, x + 0.2, 2.3, 3.9, 0.3, DS_FULL[ds], size=14, bold=True, color=NAVY)
        rows_ = IMP[ds]
        bar_chart(s, x, 2.6, 4.15, 3.6, [r_["attribute"] for r_ in rows_],
                  [(lab, [r_[key] for r_ in rows_], col) for key, lab, col in imp_series],
                  vmin=0, vmax=None, fmt="0.00", labels=False, horizontal=True)
    takeaway(s, f"When we average over all patients the rankings agree much more (THEN-part: Pima {r2(cons('pima', 'rule_consequent', 4, 'spearman_global'))}, Early Stage "
                f"{r2(cons('early', 'rule_consequent', 4, 'spearman_global'))}, CDC {r2(cons('cdc', 'rule_consequent', 4, 'spearman_global'))}).")

    s = new_slide("", "Results: Effect of the Number of Rules",
                  "Spearman agreement with SHAP on ANFIS against the number of fuzzy rules",
                  notes="As we add rules, the THEN-part agreement drops, significantly on all three datasets. Why? With more "
                        "rules the vote is shared: the strongest rule's share falls. Each patient's answer becomes a blend of "
                        "many rules, so more of the effect is in which rule fires. We see it directly: the IF-part's own "
                        "agreement rises with rules.")
    legend(s, 0.7, 1.9, [(lab, col) for _, lab, col in EXPL[:3]], gap=2.3)
    for n_, ds in enumerate(DS):
        x = 0.45 + n_ * 2.85
        text(s, x + 0.2, 2.25, 2.7, 0.3, DS_NAME[ds], size=14, bold=True, color=NAVY)
        line_chart(s, x, 2.5, 2.95, 2.75, rules, [(lab, [cons(ds, key, k) for k in rules], col) for key, lab, col in EXPL[:3]],
                   vmin=0, vmax=1, xtitle="Rules", ytitle="Spearman" if n_ == 0 else None)
    why_card(s, 9.1, 1.9, 3.6, 4.35,
             [f"Strongest rule's vote share falls: Pima {fr('pima', lo):.2f} → {fr('pima', hi):.2f}, CDC {fr('cdc', lo):.2f} → {fr('cdc', hi):.2f}.",
              "More rules blend per patient, so more of the effect is in rule switching, which the THEN-part ignores.",
              f"Evidence: the IF-part's own agreement rises (Pima {r2(cons('pima', 'rule_antecedent', lo))} → {r2(cons('pima', 'rule_antecedent', hi))}, "
              f"Early {r2(cons('early', 'rule_antecedent', lo))} → {r2(cons('early', 'rule_antecedent', hi))})."])
    rows_ = [f"{DS_NAME[d]}: {r2(cons(d, 'rule_consequent', lo))} → {r2(cons(d, 'rule_consequent', hi))} (r = {sig(d, 'rules vs')['rank_biserial']:+.2f})" for d in DS]
    text(s, 0.6, 5.45, 8.3, 0.9, [[("THEN-part, 2 → 16 rules:  ", {"bold": True, "color": NAVY}), ("   ·   ".join(rows_), {})],
                                  [(f"Wilcoxon, Holm-corrected, all {p_txt(max(sig(d, 'rules vs')['p_holm'] for d in DS))}; Friedman across all 7 rule counts, all p < 0.001", {"color": SLATE})]],
         size=12, color=INK)
    takeaway(s, "We found that adding rules makes the explanation less faithful without improving accuracy, so a small rule base is better.")

    s = new_slide("", "Results: Effect of the Number of Attributes",
                  "Spearman agreement with SHAP on ANFIS using only the top-k attributes (4 rules)",
                  notes="Here we vary the number of attributes given to the models. On Pima and CDC agreement rises with "
                        "more attributes, on Early Stage it falls. Our interpretation, which we have not tested separately: "
                        "the added attributes on Pima and CDC act mostly linearly, while on Early Stage each added yes/no "
                        "symptom adds more rule switching. With only 3 or 4 attributes the ranking is also coarse.")
    keys4 = EXPL[:1] + EXPL[2:3] + [("lime_anfis", "LIME on ANFIS", LIME_C)] + EXPL[3:]
    legend(s, 0.7, 1.9, [(lab, col) for _, lab, col in keys4], gap=2.3)
    for n_, ds in enumerate(DS):
        x = 0.45 + n_ * 2.85
        text(s, x + 0.2, 2.25, 2.7, 0.3, DS_NAME[ds], size=14, bold=True, color=NAVY)
        line_chart(s, x, 2.5, 2.95, 2.75, subs[ds], [(lab, [cons(ds, key, k, study=ATT) for k in subs[ds]], col) for key, lab, col in keys4],
                   vmin=0, vmax=1, xtitle="Attributes", ytitle="Spearman" if n_ == 0 else None)
    why_card(s, 9.1, 1.9, 3.6, 4.35,
             ["Our interpretation (not separately tested):",
              "Pima, CDC: added attributes act mostly linearly, which the THEN-part captures, so agreement rises.",
              "Early Stage: each added yes/no symptom adds rule switching, so agreement falls.",
              "With 3 or 4 attributes rank correlation is coarse (wide bands)."], title="Analysis")
    rows_ = [f"{DS_NAME[d]}: {r2(cons(d, 'rule_consequent', subs[d][0], study=ATT))} → {r2(cons(d, 'rule_consequent', subs[d][-1], study=ATT))} ({subs[d][0]} → {subs[d][-1]} attr.)" for d in DS]
    text(s, 0.6, 5.45, 8.3, 0.9, [[("THEN-part:  ", {"bold": True, "color": NAVY}), ("   ·   ".join(rows_), {})]], size=12, color=INK)
    takeaway(s, "We did not find a single trend: with more attributes agreement went up on Pima and CDC and down on Early Stage.")

    cdc_bb = TIM["cdc"]["21"]["blackbox_ms"]
    full_rows, full_sec = INFO["cdc_full_test_rows"], INFO["cdc_full_test_treeshap_seconds"]
    s = new_slide("", "Results: Computational Cost of the Explanations",
                  "Measured time to explain one patient on the CDC dataset",
                  notes="SHAP is costly on a large dataset. Black-box SHAP must try combinations "
                        "of attributes, so its cost grows fast with attributes. TreeSHAP uses the tree structure and stays "
                        "flat; it explained all fifty thousand CDC test people in under two minutes. LIME fits one small "
                        "regression, so it is cheap. The 101 minutes figure for black-box SHAP is an estimate.")
    legend(s, 0.7, 1.9, [("Black-box SHAP on ANFIS", VIOLET), ("TreeSHAP on LightGBM", YELLOW), ("LIME on ANFIS", LIME_C)], gap=3.0)
    ks = subs["cdc"]
    line_chart(s, 0.5, 2.2, 7.3, 3.9, ks, [("Black-box SHAP on ANFIS", [TIM["cdc"][str(k)]["blackbox_ms"] for k in ks], VIOLET),
                                          ("TreeSHAP on LightGBM", [TIM["cdc"][str(k)]["tree_ms"] for k in ks], YELLOW),
                                          ("LIME on ANFIS", [TIM["cdc"][str(k)]["lime_ms"] for k in ks], LIME_C)],
               vmin=0, vmax=None, fmt="0", xtitle="Number of attributes", ytitle="Milliseconds per patient")
    stat(s, 8.1, 1.9, 4.6, 1.3, f"{full_sec:.0f} s", f"TreeSHAP for all {full_rows:,} CDC test people (measured)", big_size=30)
    stat(s, 8.1, 3.3, 4.6, 1.3, f"≈ {full_rows * cdc_bb / 60000:.0f} min", f"black-box SHAP, same people, at {cdc_bb:.0f} ms each (estimate)", big_size=30)
    why_card(s, 8.1, 4.7, 4.6, 1.55, ["Exact SHAP needs 2^d coalitions (2²¹ = 2,097,152); TreeSHAP uses the tree structure; LIME fits one small model."])
    takeaway(s, "Because black-box SHAP is slow, we explained a sample of 200 patients per model and used TreeSHAP on LightGBM for the full test set.")

    labels = [("gauss-1", "Gauss, 1st"), ("bell-1", "Bell, 1st"), ("gauss-0", "Gauss, 0th"), ("bell-0", "Bell, 0th")]
    s = new_slide("", "Results: Effect of Membership Function and TSK Order",
                  "Rule explanation vs SHAP (Spearman), 4 rules. First order: THEN-part; zero order: IF-part (its THEN-part is a constant)",
                  notes="All four ANFIS variants are equally accurate, but their explanations differ. On Early Stage the bell "
                        "membership and the zero-order model both improve agreement a lot; on CDC they make it worse. Why? It "
                        "depends on how strongly one rule dominates. On Early Stage one rule takes 91 percent of the vote, and "
                        "softening that helps. On CDC the vote is already shared, and softening it more blends too many rules.")
    cols = [BLUE, C_ORANGE, AQUA]
    legend(s, 0.7, 1.9, [(DS_NAME[d], c) for d, c in zip(DS, cols)], gap=1.8)
    series = []
    for d, c in zip(DS, cols):
        vals = [VAR[d][v]["rule_consequent" if v.endswith("1") else "rule_full"]["spearman_local"] for v, _ in labels]
        series.append((DS_NAME[d], vals, c))
    bar_chart(s, 0.5, 2.2, 7.0, 3.3, [l for _, l in labels], series, vmin=0, vmax=1, fmt="0.00", label_size=10, ytitle="Spearman vs SHAP")
    aucs = "   ".join(f"{DS_NAME[d]} {min(VAR[d][v]['auc'][0] for v, _ in labels):.3f}–{max(VAR[d][v]['auc'][0] for v, _ in labels):.3f}" for d in DS)
    text(s, 0.6, 5.6, 7.0, 0.6, [[("ROC-AUC range over the 4 variants:  ", {"bold": True, "color": NAVY}), (aucs, {})]], size=12, color=INK)
    tbl = [[("Strongest rule's vote share", {"bold": True, "color": NAVY})]] + [
        [(f"{DS_NAME[d]}:  Gaussian {FV[d]['gauss-1']:.2f}  →  bell {FV[d]['bell-1']:.2f}", {})] for d in DS]
    card(s, 7.8, 1.9, 4.9, 1.6)
    text(s, 8.0, 2.0, 4.5, 1.45, tbl, size=12.5, color=INK, space=3)
    why_card(s, 7.8, 3.6, 4.9, 2.65,
             ["Bell memberships soften rule firing on every dataset.",
              f"Early Stage: Gaussian firing is near all-or-nothing ({FV['early']['gauss-1']:.2f}); softening it helps "
              f"({r2(VAR['early']['gauss-1']['rule_consequent']['spearman_local'])} → {r2(VAR['early']['bell-1']['rule_consequent']['spearman_local'])}).",
              "CDC: firing is already shared; softening further blends more rules, as adding rules does, so it hurts.",
              "Zero order on Early: output is decided by which rule fires, exactly what its IF-part explains."])
    takeaway(s, "We found the rule explanation works best when no single rule completely dominates the vote (our interpretation).")

    s = new_slide("", "Validation: LIME and the Deletion Test",
                  "Left: agreement (Spearman, 4 rules). Right: deletion test, change in output after removing each explanation's top 3 attributes",
                  notes="Two checks so that our conclusions do not depend on SHAP alone. Left: LIME agrees with SHAP more than "
                        "the rules do, because both watch the model's output on the same perturbed patients. Right: the "
                        "deletion test removes the top three attributes each explanation picks. The rules' picks change the "
                        "prediction far more than random picks, and almost as much as SHAP's. SHAP's small edge is partly by "
                        "construction, because this removal is exactly how SHAP itself measures importance.")
    leg = [("THEN vs SHAP", BLUE), ("THEN vs LIME", AQUA), ("LIME vs SHAP", LIME_C)]
    legend(s, 0.7, 1.95, leg, gap=1.95)
    bar_chart(s, 0.5, 2.25, 6.0, 3.0, [DS_NAME[d] for d in DS],
              [("THEN vs SHAP", [cons(d, "rule_consequent", 4) for d in DS], BLUE),
               ("THEN vs LIME", [LREF[d]["rule_consequent"]["4"]["spearman_local"] for d in DS], AQUA),
               ("LIME vs SHAP", [cons(d, "lime_anfis", 4) for d in DS], LIME_C)], vmin=0, vmax=1, fmt="0.00", label_size=9.5)
    dkeys = [("shap_anfis", "SHAP", VIOLET), ("lime_anfis", "LIME", LIME_C), ("rule_consequent", "Rules THEN", BLUE), ("random", "Random", GREY)]
    legend(s, 6.9, 1.95, [(l, c) for _, l, c in dkeys], gap=1.45)
    bar_chart(s, 6.8, 2.25, 6.0, 3.0, [DS_NAME[d] for d in DS],
              [(l, [DEL[d]["background"][k_]["3"] for d in DS], c) for k_, l, c in dkeys], vmin=0, vmax=None, fmt="0.0", label_size=9.5)
    why_card(s, 0.6, 5.35, 6.0, 0.95, ["LIME and SHAP both watch the model's output on the same perturbed patients, so they see rule switching."],
             title="Why LIME agrees more with SHAP")
    why_card(s, 6.8, 5.35, 5.9, 0.95, ["This removal is SHAP's own definition. With mean removal the rules tie SHAP on Pima and CDC."],
             title="Why SHAP scores highest in the deletion test")
    takeaway(s, f"The attributes picked by the rules changed the prediction {DEL['pima']['background']['rule_consequent']['3'] / DEL['pima']['background']['random']['3']:.1f}×–"
                f"{DEL['cdc']['background']['rule_consequent']['3'] / DEL['cdc']['background']['random']['3']:.1f} times more than random picks, close to SHAP's picks.")

    s = new_slide("", "Statistical Significance and Early Stage Analysis",
                  f"Wilcoxon signed-rank on matched patients, Holm-corrected across all {len(SIG)} tests; effect r from −1 to +1",
                  notes="On the left, the key statistical tests. Fewer rules, THEN over IF, LIME over rules, and rules over random "
                        "are all significant on every dataset. Adding the IF-part helps significantly only on CDC. On the "
                        "right, why Early Stage is hardest. We tested two ideas and both were rejected by the data. What is "
                        "different is that one rule takes 91 percent of the vote, and when we soften that, agreement improves.")
    tests_ = [("2 rules > 16 rules (THEN-part)", "rules vs"), ("THEN-part > IF-part", "THEN-part vs IF-part"),
              ("IF + THEN > THEN-part", "IF + THEN vs"), ("LIME > THEN-part (vs SHAP)", "LIME vs THEN"),
              ("Rules' picks > random (deletion)", "THEN-part vs random"), ("SHAP's picks > rules' (deletion)", "SHAP vs THEN")]
    card(s, 0.6, 1.9, 7.2, 0.42, NAVY, radius=0.15)
    for x, h_ in [(0.75, "Comparison"), (4.1, "Pima"), (5.35, "Early Stage"), (6.6, "CDC")]:
        text(s, x, 1.96, 1.8, 0.3, h_, size=12, bold=True, color=WHITE)
    for n_, (lab, key) in enumerate(tests_):
        y = 2.38 + n_ * 0.6
        if n_ % 2 == 0:
            card(s, 0.6, y, 7.2, 0.56, TINT, radius=0.1)
        text(s, 0.75, y + 0.03, 3.3, 0.5, lab, size=11.5, color=INK, bold=True, anchor=MSO_ANCHOR.MIDDLE)
        for x, d in zip([4.1, 5.35, 6.6], DS):
            t = sig(d, key)
            ok = t["significant_at_0.05"] in (True, "True")
            text(s, x, y + 0.03, 1.2, 0.5, [[(f"r {t['rank_biserial']:+.2f}", {"bold": True, "color": "0B7A45" if ok else SLATE})],
                                            [("significant" if ok else "not sig.", {"color": SLATE, "size": 9.5})]], size=11.5, space=0)
    e, p_ = DIAG["early"], DIAG["pima"]
    card(s, 8.1, 1.9, 4.6, 4.35)
    text(s, 8.3, 2.0, 4.2, 0.3, "Why Early Stage has the lowest agreement", size=13, bold=True, color=NAVY)
    text(s, 8.3, 2.35, 4.25, 3.85, [
        [("Rejected: ", {"bold": True, "color": C_ORANGE}), (f"flat SHAP profiles (agreement {e['agreement_when_concentrated']:.2f} vs {e['agreement_when_spread']:.2f} for concentrated vs spread patients)", {})],
        [("Rejected: ", {"bold": True, "color": C_ORANGE}), (f"attribute correlation (|r| {e['mean_abs_correlation_between_attributes']:.2f}, like Pima {p_['mean_abs_correlation_between_attributes']:.2f})", {})],
        [("Supported: ", {"bold": True, "color": "0B7A45"}), (f"one rule takes {e['mean_max_firing']:.0%} of the vote (others ~50%), and {e['share_yes_no_attributes']:.0%} of attributes are yes/no", {})],
        [("Evidence: ", {"bold": True, "color": NAVY}), ("softening the firing (bell) or using zero order raises agreement (slide 20)", {})]],
        size=11.5, color=INK, space=6)
    takeaway(s, "All main differences we report are statistically significant; the weak Early Stage result comes from almost on/off rule firing.")

    s = new_slide("", "Working Prototype: Interactive Dashboard",
                  "streamlit run app.py  →  localhost:8501; every explanation and the deletion check are computed live",
                  notes="This is our dashboard. You choose a dataset, the ANFIS variant, the number of rules and a real test "
                        "patient. It shows both models' predictions, five explanations side by side, their agreement with "
                        "SHAP, a live deletion check, and the rules that fired, in words. We can demonstrate it now.")
    picture(s, "docs/screens/patient_cdc.png", 0.6, 1.8, 6.6, crop=(40, 100, 1460, 1150), name="dash_top")
    picture(s, "docs/screens/patient_cdc.png", 7.8, 1.8, 4.9, crop=(420, 1190, 1460, 1720), name="dash_rules")
    calls = [("Choose", "dataset, ANFIS variant, rule count, test patient"),
             ("Explain", "THEN, IF + THEN, SHAP, LIME, TreeSHAP side by side"),
             ("Check", "agreement with SHAP and a live deletion test"),
             ("Read", "the fired rules in words, strongest first")]
    for n_, (a, b) in enumerate(calls):
        y = 4.45 + n_ * 0.6
        dot(s, 7.8, y + 0.06, 0.38, ORANGE, str(n_ + 1), size=12)
        text(s, 8.3, y, 4.4, 0.55, [[(a + ": ", {"bold": True, "color": NAVY}), (b, {})]], size=12, color=INK, anchor=MSO_ANCHOR.MIDDLE)

    s = new_slide("", "Conclusion and Future Work", dark=True,
                  notes="To conclude: partly. ANFIS is as accurate as LightGBM, its rules get the direction right and pick truly "
                        "important attributes, but the ranking for one patient is only partly faithful and gets worse with more "
                        "rules. So the rules should be verified, not assumed. The report and the article draft are ready. Thank you; we are happy to answer questions and "
                        "show the dashboard.")
    text(s, 0.6, 1.4, 12.1, 0.6, "Our answer is partly: the rules show the right direction and key attributes, but not always their order.",
         size=19, color=ORANGE, bold=True, font=HEAD)
    finds = [(f"{auc('cdc', 'ANFIS-4'):.2f}", "ROC-AUC on CDC, equal to LightGBM"),
             (f"≥ {min(sign_vals) * 100:.0f}%", "direction agreement with SHAP"),
             (f"{r2(cons('early', 'rule_consequent', 4))}–{r2(cons('cdc', 'rule_consequent', 4))}", "per-patient rank agreement"),
             (f"{DEL['pima']['background']['rule_consequent']['3'] / DEL['pima']['background']['random']['3']:.1f}×", "rules' picks vs random (Pima deletion)")]
    for n_, (a, b) in enumerate(finds):
        x = 0.6 + n_ * 3.07
        card(s, x, 2.2, 2.9, 1.45, "262B52")
        text(s, x + 0.2, 2.3, 2.5, 0.65, a, size=28, bold=True, color=WHITE, font=HEAD)
        text(s, x + 0.2, 3.0, 2.5, 0.6, b, size=12, color=ICE)
    blocks = [("Recommendations", ["Keep the rule base small (2–4 rules)", "Check rankings with SHAP or LIME when a decision depends on them", "If one rule dominates, prefer zero-order rules"]),
              ("Limitations", ["SHAP and LIME are references, not truth", "IF-part attribution is first-order", "200 patients per model; diabetes data only"]),
              ("Future work", ["Exact IF-part attribution", "Train rules to be faithful", "Other diseases; clinician user study"])]
    for n_, (h_, items) in enumerate(blocks):
        x = 0.6 + n_ * 4.1
        card(s, x, 3.85, 3.9, 2.25, "262B52")
        text(s, x + 0.2, 3.95, 3.5, 0.3, h_, size=14, bold=True, color=ORANGE)
        text(s, x + 0.2, 4.3, 3.55, 1.8, items, size=12, color=WHITE, bullets=True, space=4)
    text(s, 8.9, 6.2, 3.8, 0.6, "Thank you · Questions", size=24, bold=True, color=WHITE, font=HEAD, align=PP_ALIGN.RIGHT)
