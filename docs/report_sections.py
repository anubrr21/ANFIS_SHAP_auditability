import numpy as np
import pandas as pd

from report_common import DS, NAME, TAB, Results, cite


def _pp():
    if not hasattr(_pp, "df"):
        _pp.df = pd.read_csv(TAB / "per_patient.csv.gz")
    return _pp.df


def firing(ds, rules=None, mf=None, order=None):
    p = _pp()
    if mf is None:
        d = p[(p.dataset == ds) & (p.study == "rules") & (p.n_rules == rules)]
    else:
        d = p[(p.dataset == ds) & (p.study == "variants") & (p.mf == mf) & (p.order == order)]
    return float(d.max_firing.mean())


def fmt_p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def why(title, body):
    return f"<p>{body}</p>"


def facts(r):
    lo, hi = r.rules[0], r.rules[-1]
    f = {"lo": lo, "hi": hi}
    f["sign_min"] = min(r.c(d, "rule_consequent", "sign_agreement", rules=k) for d in DS for k in r.rules)
    f["if_max"] = max(r.c(d, "rule_antecedent", rules=k) for d in DS for k in r.rules)
    f["if_min"] = min(r.c(d, "rule_antecedent", rules=k) for d in DS for k in r.rules)
    for d in DS:
        f[f"then_{d}"] = r.c(d, "rule_consequent")
        f[f"lime_{d}"] = r.c(d, "lime_anfis")
        f[f"auc_anfis_{d}"] = r.auc(d, "ANFIS", 4).mean()
        f[f"auc_lgbm_{d}"] = r.auc(d, "LightGBM").mean()
        f[f"auc_lr_{d}"] = r.auc(d, "LogisticRegression").mean()
    return f


def abstract_findings(r):
    f = facts(r)
    t = lambda d, s: r.test(d, s)
    return (
        f"ANFIS reached the ROC-AUC of LightGBM on all three datasets ({f['auc_anfis_pima']:.3f}, {f['auc_anfis_early']:.3f} "
        f"and {f['auc_anfis_cdc']:.3f} against {f['auc_lgbm_pima']:.3f}, {f['auc_lgbm_early']:.3f} and {f['auc_lgbm_cdc']:.3f}). "
        f"The THEN-part rule explanation agreed with SHAP on the direction of each attribute's effect in at least "
        f"{f['sign_min'] * 100:.0f}% of cases, but its per-patient rank agreement was only partial (Spearman "
        f"{f['then_early']:.2f} to {f['then_cdc']:.2f} with four rules) and decreased significantly from {f['lo']} to "
        f"{f['hi']} rules on every dataset, while accuracy did not improve. The IF-part alone was a poor explanation "
        f"(Spearman at most {f['if_max']:.2f}). LIME agreed with SHAP better than the rules did, and in the deletion test "
        "the attributes chosen by the rules changed the output far more than random choices, though slightly less than "
        "those chosen by SHAP. The weakest agreement, on the Early Stage data, coincided with one rule taking about "
        f"{firing('early', rules=4) * 100:.0f}% of the vote; softening the firing with bell memberships, or using a "
        "zero-order model, raised agreement there. ANFIS rule explanations are therefore informative but should be "
        "verified rather than assumed, and compact rule bases are preferable.")


def results_chapter(r, T, F):
    f = facts(r)
    lo, hi = f["lo"], f["hi"]
    t = r.test
    subs = r.subs
    fried = r.fried.set_index("dataset")
    d_early = r.diag.loc["early"]
    S = ["<h1>8. Results and Analysis</h1>",
         f"<p>All values are means over {r.seeds} random splits on test patients that were not used for training. "
         "Agreement values are per-patient averages unless stated otherwise. Each result is followed by an analysis of "
         "why it occurs, based on additional quantities measured in the same runs.</p>"]

    S.append(f"""<h2>8.1 Predictive performance (RQ1)</h2>
{T(r.table_accuracy(), 'Test performance, mean ± standard deviation over seeds.')}
{F('fig1_accuracy_vs_rules.png', 'Test ROC-AUC of ANFIS against the number of rules, with LightGBM and logistic regression. How to read: each panel is one dataset; a flat ANFIS line means adding rules does not change accuracy; the horizontal lines are the baselines.')}
<p>ANFIS with four rules reached a ROC-AUC of {f['auc_anfis_pima']:.3f} on Pima, {f['auc_anfis_early']:.3f} on Early
Stage and {f['auc_anfis_cdc']:.3f} on CDC, against {f['auc_lgbm_pima']:.3f}, {f['auc_lgbm_early']:.3f} and
{f['auc_lgbm_cdc']:.3f} for LightGBM. Changing the rule count from {lo} to {hi} changed the ROC-AUC by at most a few
thousandths except on Early Stage ({r.auc('early', 'ANFIS', lo).mean():.3f} to {r.auc('early', 'ANFIS', hi).mean():.3f}).
The F1 on CDC is about {r.perf[(r.perf.dataset == 'cdc') & (r.perf.model == 'LightGBM')].f1.mean():.2f} for every model,
including LightGBM.</p>
{why('Why ANFIS matches LightGBM, and why more rules do not help',
     f"Logistic regression, a purely linear model, is already within {abs(f['auc_lgbm_pima'] - f['auc_lr_pima']):.3f} "
     f"(Pima) and {abs(f['auc_lgbm_cdc'] - f['auc_lr_cdc']):.3f} (CDC) ROC-AUC of LightGBM. The diagnosis signal in these "
     "datasets is therefore close to additive, with little interaction structure for trees to exploit, and a smooth blend "
     "of a few locally linear rules is sufficient. Extra rules can only model interactions that are not there. Early "
     f"Stage is the exception where the linear model is clearly weaker ({f['auc_lr_early']:.3f}), and there extra rules "
     "help a little. The low CDC F1 is a property of the data (14% positives and overlapping self-reported answers): "
     "the strongest baseline has the same F1.")}""")

    S.append(f"""<h2>8.2 Agreement of the rule explanation with SHAP (RQ2)</h2>
{T(r.table_agreement(), 'Agreement of each explanation with SHAP on ANFIS (4 rules). Weighted τ emphasises the top ranks.')}
{F('fig4_global_importance.png', 'Dataset-level importance from the rules (IF + THEN), SHAP on ANFIS and SHAP on LightGBM (seed 0, 4 rules).')}
<p>The THEN-part reading agreed with SHAP at Spearman {f['then_pima']:.2f} on Pima, {f['then_early']:.2f} on Early Stage
and {f['then_cdc']:.2f} on CDC. It identified the same most important attribute for
{r.c('pima', 'rule_consequent', 'top1_match') * 100:.0f}%, {r.c('early', 'rule_consequent', 'top1_match') * 100:.0f}% and
{r.c('cdc', 'rule_consequent', 'top1_match') * 100:.0f}% of patients. Its sign agreement was at least
{f['sign_min']:.2f} in every setting, and its global (dataset-level) agreement was higher than its per-patient agreement.
The IF-part on its own agreed poorly (Spearman {f['if_min']:.2f} to {f['if_max']:.2f}); the THEN-part was significantly
better than the IF-part on every dataset (rank-biserial r between {min(t(d, 'THEN-part vs IF-part').rank_biserial for d in DS):+.2f}
and {max(t(d, 'THEN-part vs IF-part').rank_biserial for d in DS):+.2f}, all {fmt_p(max(t(d, 'THEN-part vs IF-part').p_holm for d in DS))}).
Adding the IF-part to the THEN-part improved agreement significantly only on CDC
({t('cdc', 'IF + THEN vs THEN-part').mean_b:.3f} to {t('cdc', 'IF + THEN vs THEN-part').mean_a:.3f},
{fmt_p(t('cdc', 'IF + THEN vs THEN-part').p_holm)}); on Pima and Early Stage the difference was not significant after
correction. The dataset-level rankings are medically sensible: glucose, age and BMI on Pima; polyuria and polydipsia on
Early Stage; general health, age, BMI and high blood pressure on CDC.</p>
{why('Why the THEN-part explains well but not perfectly',
     "The THEN-part formula (Section 5.7) holds the firing strengths fixed and reports only what the fired rules' linear formulas do. SHAP, by "
     "contrast, observes the full model, in which changing an attribute also changes which rules fire. The THEN-part "
     "is therefore right about the direction of each effect (the coefficients of the dominant rules carry the sign) "
     "but misses the part of the effect that acts through rule switching, which reorders attributes for individual "
     "patients. Averaged over a dataset, these patient-level reorderings partly cancel, which is why the global "
     "agreement is higher than the local one.")}
{why('Why the IF-part alone is a poor explanation',
     "In a first-order model most of the output variation is carried by the consequents a<sub>r</sub>·x; the IF-part "
     "only captures how attributes shift the vote between rules, a second-order effect when the rules give similar "
     "answers. It is also a first-order approximation of a non-linear (softmax) dependence. Its contribution becomes "
     "important only when rules disagree strongly and the vote moves, which matches the CDC result where adding it "
     "helped.")}""")

    S.append(f"""<h2>8.3 Effect of the number of rules (RQ3)</h2>
{T(r.table_rules(), 'Per-patient Spearman agreement of the THEN-part with SHAP for each rule count.')}
{F('fig2_agreement_vs_rules.png', 'Agreement with SHAP on ANFIS against the number of rules (bands: one standard deviation over seeds). How to read: the y-axis is the per-patient Spearman correlation with SHAP (1 = same order of importance); a falling line means the explanation becomes less faithful as rules are added.')}
{F('fig6_top3_overlap_vs_rules.png', 'Top-3 attribute overlap with SHAP on ANFIS against the number of rules. How to read: 1.0 means the explanation and SHAP name the same three most important attributes for every patient.')}
<p>Agreement fell as rules were added: from {r.c('pima', 'rule_consequent', rules=lo):.2f} to
{r.c('pima', 'rule_consequent', rules=hi):.2f} on Pima, from {r.c('early', 'rule_consequent', rules=lo):.2f} to
{r.c('early', 'rule_consequent', rules=hi):.2f} on Early Stage, and from {r.c('cdc', 'rule_consequent', rules=lo):.2f} to
{r.c('cdc', 'rule_consequent', rules=hi):.2f} on CDC (where it peaked at four rules,
{r.c('cdc', 'rule_consequent', rules=4):.2f}). The decrease from {lo} to {hi} rules is significant on all datasets
(Wilcoxon, effect r = {t('pima', 'rules vs').rank_biserial:+.2f}, {t('early', 'rules vs').rank_biserial:+.2f} and
{t('cdc', 'rules vs').rank_biserial:+.2f}; all {fmt_p(max(t(d, 'rules vs').p_holm for d in DS))}), and the Friedman test
across all seven rule counts rejects equal agreement on every dataset (χ² = {fried.loc['pima', 'chi_square']:.0f},
{fried.loc['early', 'chi_square']:.0f} and {fried.loc['cdc', 'chi_square']:.0f}, all p &lt; 0.001). The effect on CDC is
small. Most of the change happens between 2 and 6 rules.</p>
{why('Why more rules make the rule explanation less faithful',
     f"With more rules the vote is shared more widely: the strongest rule's share falls from {firing('pima', rules=lo):.2f} "
     f"to {firing('pima', rules=hi):.2f} on Pima and from {firing('cdc', rules=lo):.2f} to {firing('cdc', rules=hi):.2f} on CDC. "
     "Each patient's output is then a blend of more rules, and the effect of moving between those rules, which the "
     "THEN-part ignores, grows. This is visible directly: the IF-part's own agreement with SHAP rises with the rule count "
     f"on Pima ({r.c('pima', 'rule_antecedent', rules=lo):.2f} to {r.c('pima', 'rule_antecedent', rules=hi):.2f}) and "
     f"Early Stage ({r.c('early', 'rule_antecedent', rules=lo):.2f} to {r.c('early', 'rule_antecedent', rules=hi):.2f}), "
     "meaning that more of what drives the model now sits in the switching. Since accuracy does not improve (Section "
     "8.1), the rule explosion problem costs interpretability without buying accuracy here.")}""")

    S.append(f"""<h2>8.4 Effect of the number of attributes (RQ4)</h2>
{T(r.table_attributes(), 'Agreement with SHAP on ANFIS when only the top-k attributes are used (4 rules).')}
{F('fig3_agreement_vs_attributes.png', 'Agreement with SHAP on ANFIS against the number of attributes used (4 rules). How to read: the x-axis is k, the number of most informative attributes the models were trained on.')}
<p>On Pima and CDC the THEN-part agreed better when more attributes were used
({r.c('pima', 'rule_consequent', attrs=subs['pima'][0]):.2f} with {subs['pima'][0]} to
{r.c('pima', 'rule_consequent', attrs=subs['pima'][-1]):.2f} with {subs['pima'][-1]}; CDC
{r.c('cdc', 'rule_consequent', attrs=subs['cdc'][0]):.2f} with {subs['cdc'][0]} to
{r.c('cdc', 'rule_consequent', attrs=subs['cdc'][-1]):.2f} with {subs['cdc'][-1]}); on Early Stage it agreed less
({r.c('early', 'rule_consequent', attrs=subs['early'][0]):.2f} with {subs['early'][0]} to
{r.c('early', 'rule_consequent', attrs=subs['early'][-1]):.2f} with {subs['early'][-1]}). There is therefore no single
effect of the number of attributes.</p>
{why('Why the trend differs between datasets',
     "Our interpretation, not separately tested: with very few attributes, the few that remain are the strongest predictors and dominate every rule's firing, so "
     "the rule switching effect that the THEN-part misses is concentrated on exactly the attributes being ranked. "
     "Adding weaker attributes to Pima and CDC adds attributes whose effect is mostly linear and is captured well by "
     "the consequents, which raises rank agreement. On Early Stage the added attributes are further yes/no symptoms, "
     "and the model's firing there is nearly all-or-nothing (Section 8.7), so each added symptom adds switching "
     "effects that the THEN-part misses. Rank correlation over 3 or 4 positions is also coarse, so the points with few "
     "attributes are the least reliable (wider bands in the figure).")}""")

    tim = r.time[r.time.study == "attributes"]
    S.append(f"""<h2>8.5 Cost of the explanations (RQ4)</h2>
{F('fig5_shap_cost.png', 'Measured time to explain one patient (logarithmic scale). How to read: each step up on the y-axis is ten times slower; the exact Pima points show the doubling of black-box SHAP cost with each extra attribute.')}
<p>Black-box SHAP on ANFIS needed {r.ms('pima', subs['pima'][0], 'blackbox_shap_sec_per_patient'):.1f} ms per patient
with {subs['pima'][0]} attributes and {r.ms('pima', subs['pima'][-1], 'blackbox_shap_sec_per_patient'):.1f} ms with
{subs['pima'][-1]} on Pima (exact), and {r.ms('cdc', 21, 'blackbox_shap_sec_per_patient'):.0f} ms with 21 attributes
on CDC (sampled). TreeSHAP on LightGBM needed about {r.ms('cdc', 21, 'tree_shap_sec_per_patient'):.0f} ms per patient
with the same background, and in its path-dependent mode explained the entire CDC test set of
{r.info['cdc_full_test_rows']:,} people in {r.info['cdc_full_test_treeshap_seconds']:.0f} s. At the measured rate,
black-box SHAP on ANFIS would need about
{r.info['cdc_full_test_rows'] * r.ms('cdc', 21, 'blackbox_shap_sec_per_patient') / 60000:.0f} minutes for the same people
(an estimate by multiplication). LIME needed {tim.lime_sec_per_patient.min() * 1000:.1f} to
{tim.lime_sec_per_patient.max() * 1000:.1f} ms per patient. Timings were recorded while several experiments ran in
parallel, so only their relative size is meaningful.</p>
{why('Why TreeSHAP and LIME are so much cheaper than black-box SHAP',
     "Exact Shapley values for a black box require the model output for every coalition of attributes, 2<sup>d</sup> of "
     "them, each averaged over the background; the cost doubles with every attribute, which is what the exact Pima points "
     "show. TreeSHAP exploits the tree structure to compute the same values in time polynomial in the tree size "
     f"{cite('treeshap')}. LIME fits one small regression on a fixed number of perturbations, so its cost hardly depends "
     "on d. This is why, on the 253,680-record dataset, SHAP on ANFIS is applied to a sample of patients.")}""")

    S.append(f"""<h2>8.6 Membership function and TSK order (RQ5)</h2>
{T(r.table_variants(), 'ANFIS variants with 4 rules: accuracy and agreement of the rule explanation with SHAP.')}
{F('fig8_variants.png', 'Variants with 4 rules. Left: ROC-AUC. Right: agreement of the rule explanation with SHAP (THEN-part for first order; IF-part for zero order, whose THEN-part is a constant).')}
<p>All four variants reached almost the same ROC-AUC. Their explanations did not behave the same. On Early Stage the
bell membership raised the THEN-part agreement from {r.c('early', 'rule_consequent', study='variants', mf='gauss', order=1):.2f}
to {r.c('early', 'rule_consequent', study='variants', mf='bell', order=1):.2f}
({fmt_p(t('early', 'Bell vs Gaussian').p_holm)}), and the zero-order model, which explains only through its IF-part,
reached {r.c('early', 'rule_full', study='variants', mf='gauss', order=0):.2f} ({fmt_p(t('early', 'Zero-order').p_holm)}).
On CDC both changes lowered agreement (bell {r.c('cdc', 'rule_consequent', study='variants', mf='bell', order=1):.2f},
zero order {r.c('cdc', 'rule_full', study='variants', mf='gauss', order=0):.2f}, against
{r.c('cdc', 'rule_consequent', study='variants', mf='gauss', order=1):.2f}); on Pima the bell membership lowered it and the
zero-order model made no significant difference.</p>
{why('Why the same change helps on one dataset and hurts on another',
     f"The common factor is how concentrated the rule firing is. With Gaussian memberships the strongest rule takes "
     f"{firing('early', mf='gauss', order=1):.2f} of the vote on Early Stage (almost all-or-nothing switching) but only "
     f"{firing('cdc', mf='gauss', order=1):.2f} on CDC. The bell membership, with its flatter top and heavier tails, "
     f"softens the firing on every dataset (Early Stage {firing('early', mf='bell', order=1):.2f}, CDC "
     f"{firing('cdc', mf='bell', order=1):.2f}). On Early Stage that moves the model away from hard switching, so less of "
     "the effect hides in the switching and the THEN-part becomes more faithful. On CDC the firing was already shared, "
     "and softening it further blends more rules per patient, the same mechanism that lowers agreement when rules are "
     "added (Section 8.3). Our interpretation is that the THEN-part is most faithful at a moderate degree of firing "
     "concentration. The zero-order model on Early Stage works for a related reason: when one rule dominates, a "
     "zero-order model's output is decided almost entirely by which rule fires, which is exactly what its IF-part "
     "explanation describes.")}""")

    S.append(f"""<h2>8.7 Why the Early Stage data is the hardest case</h2>
{T(r.table_diagnostics(), 'Diagnostics per dataset for the Gaussian first-order model with 4 rules.')}
<p>Two explanations proposed after Review 2 were tested and <b>not supported</b>. If the low agreement were caused by
many attributes having similarly small SHAP values (so that their order is arbitrary), agreement would be lower for
patients whose SHAP values are spread out; it is not (Spearman between concentration and agreement
{d_early['spearman_vs_top3_share']:+.2f}; agreement {d_early['agreement_when_concentrated']:.2f} for concentrated and
{d_early['agreement_when_spread']:.2f} for spread-out patients). Correlation between attributes is also not the cause:
the mean absolute correlation on Early Stage ({d_early['mean_abs_correlation_between_attributes']:.3f}) is similar to
Pima's ({r.diag.loc['pima', 'mean_abs_correlation_between_attributes']:.3f}). What does distinguish Early Stage is that
{d_early['share_yes_no_attributes'] * 100:.0f}% of its attributes are yes/no symptoms and that the strongest rule takes
{d_early['mean_max_firing']:.2f} of the vote, against about 0.5 on the other datasets. Section 8.6 shows that changing
this firing concentration changes the agreement in the expected direction, which supports rule switching as the main
cause. Within Early Stage the firing is uniformly high, so it cannot explain differences between its patients
(Spearman {d_early['spearman_vs_max_firing']:+.2f}).</p>""")

    S.append(f"""<h2>8.8 A second reference explainer: LIME (RQ5)</h2>
{T(r.table_lime(), 'The rule explanation compared with LIME as well as with SHAP (4 rules).')}
{F('fig9_lime_check.png', 'Agreement of the THEN-part with SHAP and with LIME, and of LIME with SHAP (4 rules).')}
<p>Measured against LIME instead of SHAP, the THEN-part agreed to a similar degree on Pima and CDC
({r.c('pima', 'rule_consequent', ref='lime_anfis'):.2f}, {r.c('cdc', 'rule_consequent', ref='lime_anfis'):.2f}) and less on
Early Stage ({r.c('early', 'rule_consequent', ref='lime_anfis'):.2f}), so the conclusions do not depend on choosing SHAP
as the reference. LIME agreed with SHAP better than the rules did on every dataset (Spearman {f['lime_pima']:.2f},
{f['lime_early']:.2f}, {f['lime_cdc']:.2f}; all {fmt_p(max(t(d, 'LIME vs THEN-part').p_holm for d in DS))}).</p>
{why('Why LIME agrees with SHAP better than the rules do',
     "Both LIME and SHAP are computed from the same thing: the model's output on perturbed patients drawn from the same "
     "background data. They therefore see the complete behaviour of the model, including the effect of rule switching. "
     "The rule explanation instead reads the model's parameters with the firing strengths held fixed. Closer agreement "
     "between LIME and SHAP shows that they measure similar things, not that either is the truth; that is why the "
     "deletion test of the next section is needed.")}""")

    dd = lambda d, e, rem="background", k=3: r.deletion(d, e, k=k, removal=rem)
    S.append(f"""<h2>8.9 Deletion test: which explanation picks the attributes that matter? (RQ5)</h2>
{T(r.table_deletion('background'), 'Mean change in ANFIS output (log-odds) after removing the top-3 attributes of each explanation (background replacement, 4 rules). Higher is better.')}
{T(r.table_deletion('mean'), 'The same with replacement by the training mean.')}
{F('fig7_deletion_test.png', 'Deletion test (background replacement, 4 rules). How to read: each bar is the average change in ANFIS output when the top-k attributes of that explanation are removed; higher is better; the dashed line is a random choice of attributes.')}
<p>Removing the three attributes chosen by the THEN-part changed the output far more than removing three random
attributes on every dataset ({dd('pima', 'rule_consequent'):.2f} against {dd('pima', 'random'):.2f} on Pima,
{dd('early', 'rule_consequent'):.2f} against {dd('early', 'random'):.2f} on Early Stage, {dd('cdc', 'rule_consequent'):.2f}
against {dd('cdc', 'random'):.2f} on CDC; effect r = +1.00 in every case). With background replacement SHAP's choice
changed the output slightly more than the rules' choice ({dd('pima', 'shap_anfis'):.2f}, {dd('early', 'shap_anfis'):.2f},
{dd('cdc', 'shap_anfis'):.2f}; all significant). With replacement by the training mean the gap almost disappears on
Pima and CDC, and for a single removed attribute the rules even do slightly better on Pima
({dd('pima', 'rule_consequent', 'mean', 1):.3f} against {dd('pima', 'shap_anfis', 'mean', 1):.3f}) and Early Stage
(IF + THEN {dd('early', 'rule_full', 'mean', 1):.2f} against {dd('early', 'shap_anfis', 'mean', 1):.2f}).</p>
{why('Why SHAP wins the deletion test, and why that is partly by construction',
     "SHAP's value function (Section 5.8) is defined by replacing absent attributes with background values, which is exactly the "
     "removal used in our main deletion test, so SHAP is optimising the quantity being measured. The THEN-part is "
     "defined relative to the training mean (Section 5.7), and under mean replacement it performs as well as SHAP. The "
     "fair conclusion is that the rules point to genuinely important attributes, far better than chance and close to "
     "the best post-hoc method, while being less exact about their order. The largest gap is again on Early Stage, "
     "where the rule switching effect is largest.")}""")

    S.append(f"""<h2>8.10 Summary of the statistical tests</h2>
{T(r.table_significance(), 'Paired Wilcoxon signed-rank tests on (seed, patient) units for agreement, and on (seed, rule count) units for deletion, with Holm correction. Effect r: matched-pairs rank-biserial correlation; positive means A is higher.')}
<p>Of {len(r.sig)} tests, {int(r.sig['significant_at_0.05'].sum())} are significant at 0.05 after Holm correction. The
non-significant ones are the IF + THEN against THEN-part comparisons on Pima and Early Stage and the zero-order against
first-order comparison on Pima.</p>""")
    return "\n".join(S)


def comparison_chapter(r, T):
    f = facts(r)
    rows = [
        ("ANFIS, 4 rules (ours)", f"{f['auc_anfis_pima']:.3f}", f"{f['auc_anfis_early']:.3f}", f"{f['auc_anfis_cdc']:.3f}", "Rules (audited here)"),
        ("LightGBM", f"{f['auc_lgbm_pima']:.3f}", f"{f['auc_lgbm_early']:.3f}", f"{f['auc_lgbm_cdc']:.3f}", "None (needs SHAP)"),
        ("Logistic regression", f"{f['auc_lr_pima']:.3f}", f"{f['auc_lr_early']:.3f}", f"{f['auc_lr_cdc']:.3f}", "Coefficients"),
    ]
    acc = pd.DataFrame(rows, columns=["Model", "Pima ROC-AUC", "Early Stage ROC-AUC", "CDC ROC-AUC", "Built-in explanation"])
    ex = pd.DataFrame([
        ("Rules: THEN-part (ANFIS's own)", f"{f['then_pima']:.2f} / {f['then_early']:.2f} / {f['then_cdc']:.2f}",
         " / ".join(f"{r.deletion(d, 'rule_consequent'):.2f}" for d in DS), "Free (from the parameters)", "Yes, as rules"),
        ("Rules: IF-part", " / ".join(f"{r.c(d, 'rule_antecedent'):.2f}" for d in DS),
         " / ".join(f"{r.deletion(d, 'rule_antecedent'):.2f}" for d in DS), "Free", "Yes, as rule activations"),
        ("LIME on ANFIS", f"{f['lime_pima']:.2f} / {f['lime_early']:.2f} / {f['lime_cdc']:.2f}",
         " / ".join(f"{r.deletion(d, 'lime_anfis'):.2f}" for d in DS), "About 1 to 4 ms per patient", "No"),
        ("SHAP on ANFIS", "reference", " / ".join(f"{r.deletion(d, 'shap_anfis'):.2f}" for d in DS),
         f"up to {r.ms('early', 16, 'blackbox_shap_sec_per_patient'):.0f} ms per patient", "No"),
        ("Random attribute choice", "n/a", " / ".join(f"{r.deletion(d, 'random'):.2f}" for d in DS), "Free", "No"),
    ], columns=["Explanation", "Spearman vs SHAP (Pima / Early / CDC)", "Deletion top-3 (Pima / Early / CDC)", "Cost", "Readable by a clinician"])
    return f"""
<h1>9. Comparison with Existing Methods</h1>
<h2>9.1 Prediction: ANFIS against standard models</h2>
{T(acc, 'ROC-AUC of our ANFIS against the baselines (mean over seeds).')}
<p>Our ANFIS is as accurate as gradient-boosted trees (LightGBM {cite('lgbm')}) and better than or equal to logistic
regression on all three datasets, while being the only one of the three whose reasoning can be read as rules. The
accuracy cost of choosing ANFIS for interpretability is therefore negligible on these data.</p>
<h2>9.2 Explanation: ANFIS's own rules against post-hoc methods</h2>
{T(ex, 'The rule explanation against existing explanation methods (4 rules).')}
<p>Compared with the established post-hoc methods, the rule explanation is free to compute and directly readable, and
the attributes it selects are nearly as important to the model as those selected by SHAP {cite('shap')} and LIME
{cite('lime')}. Its weakness is the order of the attributes for individual patients, where post-hoc methods, which
observe the whole model, are more faithful. A practical combination follows: show the rules, and use SHAP or LIME to
check the attribute ranking for the patients where a decision depends on it.</p>
<h2>9.3 Evaluation approach: ours against existing interpretability assessment</h2>
<table class="t"><thead><tr><th>Aspect</th><th>Existing ANFIS work</th><th>This project</th></tr></thead>
<tr><td>What is measured</td><td>Accuracy, rule count, rule length, set distinguishability {cite('gacto')}</td><td>Faithfulness of the explanation to the model, per patient</td></tr>
<tr><td>Explanation check</td><td>Rules reported as they are, or SHAP applied alone</td><td>Rules cross-checked against SHAP and LIME, plus a reference-free deletion test</td></tr>
<tr><td>Which part of the rule</td><td>Not separated</td><td>IF-part and THEN-part measured separately</td></tr>
<tr><td>Robustness</td><td>Single split, single configuration</td><td>{r.seeds} seeds, {len(r.rules)} rule counts, attribute subsets, 4 variants, corrected significance tests</td></tr>
<tr><td>High dimension</td><td>Grid partition, rule explosion</td><td>Clustering rules with HTSK firing {cite('cui')}, up to 21 attributes and 253,680 records</td></tr>
<tr><td>Deliverable</td><td>A trained model</td><td>A reusable audit pipeline, dashboard, report and article</td></tr></table>
<p class="cap">Comparison of the evaluation approach.</p>
<h2>9.4 Explanation cost: black-box SHAP against TreeSHAP</h2>
<p>For black-box SHAP the cost per patient grows exponentially with the number of attributes; TreeSHAP
{cite('treeshap')} for LightGBM stays nearly constant and explained {r.info['cdc_full_test_rows']:,} patients in
{r.info['cdc_full_test_treeshap_seconds']:.0f} s. The rule explanation of ANFIS costs nothing beyond one forward pass,
which is an advantage of intrinsic explanations when they are faithful enough.</p>"""


def conclusion_chapter(r):
    f = facts(r)
    return f"""
<h1>10. Conclusion</h1>
<p>This project tested whether the IF-THEN rules of ANFIS faithfully explain its predictions, using our own ANFIS
implementation, three diabetes datasets with 8, 16 and 21 attributes, two independent post-hoc explainers, a
reference-free deletion test and corrected significance tests. The answers to the research questions are:</p>
<ol>
<li><b>RQ1.</b> ANFIS is as accurate as LightGBM on all three datasets, so its explanations are worth auditing.</li>
<li><b>RQ2.</b> The explanation read from the rules' THEN-parts is faithful in direction (sign agreement at least
{f['sign_min']:.2f}) and points to genuinely important attributes, but agrees only partly with SHAP on their order for
individual patients (Spearman {f['then_early']:.2f} to {f['then_cdc']:.2f}). The IF-part alone is a poor explanation.</li>
<li><b>RQ3.</b> More rules make the explanation significantly less faithful without improving accuracy, because the
vote is shared by more rules and more of the effect moves into rule switching.</li>
<li><b>RQ4.</b> The number of attributes has no single effect; it depends on the dataset. Black-box SHAP becomes
expensive with many attributes, while TreeSHAP scales to the full 253,680-record dataset.</li>
<li><b>RQ5.</b> The conclusions hold with LIME as the reference and are supported by the deletion test. The membership
function and TSK order change the faithfulness of the explanation through the concentration of rule firing.</li>
</ol>
<p>ANFIS remains a competitive and useful model, but its rules should be treated as a hypothesis about the model's
reasoning that is verified, not as a guaranteed explanation. In practice: keep the rule base small, and check the
attribute ranking with a post-hoc method where an individual decision depends on it.</p>"""


def future_chapter(r):
    return """
<h1>11. Scope for Future Work</h1>
<ul>
<li><b>An exact IF-part attribution.</b> Replace the first-order approximation in the IF-part formula (Section 5.7) with an exact
decomposition, for example by integrating along a path from the reference patient, so that the full rule explanation
satisfies efficiency like SHAP.</li>
<li><b>Training for faithfulness.</b> Add a penalty on firing concentration, or on the difference between the rule
explanation and SHAP, to the loss, and measure the trade-off with accuracy.</li>
<li><b>Other neuro-fuzzy models.</b> Apply the pipeline, which only needs a new rule-attribution function, to other
TSK systems and to fuzzy classifiers with linguistic consequents.</li>
<li><b>Correlation-aware references.</b> Repeat the audit with SHAP variants that respect dependence between attributes,
since standard SHAP replaces attributes independently.</li>
<li><b>More data and diseases.</b> Extend to other clinical tabular datasets, to multi-class problems and to external
validation across hospitals.</li>
<li><b>User study.</b> Measure whether clinicians make better decisions with rule explanations, post-hoc explanations or
both, which is the final test of interpretability.</li>
<li><b>Publication.</b> Submit the research article to a soft computing or fuzzy systems venue.</li>
</ul>"""


def article_results(r, T, F):
    f = facts(r)
    t = r.test
    lo, hi = f["lo"], f["hi"]
    acc = r.table_accuracy()
    acc = acc[acc.Model.isin(["LightGBM", "Logistic regression", "ANFIS (4 rules)"])][["Dataset", "Model", "ROC-AUC", "F1"]]
    agree = r.table_agreement()[["Dataset", "Explanation vs SHAP on ANFIS", "Spearman", "Weighted τ", "Top-1", "Top-3", "Sign"]]
    agree.columns = ["Data", "Explanation", "ρ", "wτ", "Top-1", "Top-3", "Sign"]
    return f"""
<h2>V. Results</h2>
<h3>A. Accuracy</h3>
{T(acc, 'Test performance (mean ± sd over five seeds).')}
<p>ANFIS matches LightGBM (Table {['I','II','III','IV','V','VI'][0]}); logistic regression is close on Pima and CDC, so the
signal is nearly additive and few rules suffice. Changing R from {lo} to {hi} changes ROC-AUC by at most a few
thousandths except on Early Stage.</p>
<h3>B. Agreement with SHAP</h3>
{T(agree, 'Per-patient agreement with SHAP on ANFIS, R = 4.')}
<p>The consequent reading agrees in sign in at least {f['sign_min'] * 100:.0f}% of cases but only partly in rank
(ρ = {f['then_pima']:.2f}, {f['then_early']:.2f}, {f['then_cdc']:.2f}); the antecedent alone is weak
(ρ ≤ {f['if_max']:.2f}). It ignores the dependence of the firing strengths on x, which SHAP observes; patient-level
reorderings cancel partly at dataset level, so global agreement is higher.</p>
{F('fig2_agreement_vs_rules.png', 'Agreement with SHAP versus the number of rules. Blue: THEN-part; orange: IF-part; green: IF + THEN; yellow: SHAP on LightGBM.', wide=True)}
<h3>C. Rule count and attributes</h3>
<p>Agreement falls from R = {lo} to R = {hi} on all datasets (Wilcoxon, Holm-corrected {fmt_p(max(t(d, 'rules vs').p_holm for d in DS))};
r = {t('pima', 'rules vs').rank_biserial:+.2f}, {t('early', 'rules vs').rank_biserial:+.2f}, {t('cdc', 'rules vs').rank_biserial:+.2f}).
As R grows the strongest rule's vote share falls (Pima {firing('pima', rules=lo):.2f} → {firing('pima', rules=hi):.2f})
and the antecedent's own agreement with SHAP rises (Pima {r.c('pima', 'rule_antecedent', rules=lo):.2f} →
{r.c('pima', 'rule_antecedent', rules=hi):.2f}): more of the effect moves into rule switching, which the consequent
reading misses. Restricting attributes raised agreement with more attributes on Pima and CDC but lowered it on Early
Stage, so there is no single attribute-count effect.</p>
<h3>D. Variants, LIME and deletion</h3>
{F('fig8_variants.png', 'Variants (R = 4): ROC-AUC (left) and agreement of the rule explanation with SHAP (right).', wide=True)}
<p>Variants differ little in accuracy but differ in faithfulness, through firing concentration. On Early Stage the
Gaussian model's strongest rule takes {firing('early', mf='gauss', order=1):.2f} of the vote; bell memberships soften
this to {firing('early', mf='bell', order=1):.2f} and raise agreement from
{r.c('early', 'rule_consequent', study='variants', mf='gauss', order=1):.2f} to
{r.c('early', 'rule_consequent', study='variants', mf='bell', order=1):.2f}, and a zero-order model, explained purely
by which rule fires, reaches {r.c('early', 'rule_full', study='variants', mf='gauss', order=0):.2f}. On CDC, where firing
is already shared ({firing('cdc', mf='gauss', order=1):.2f}), both changes lower agreement. Two alternative
explanations, flat SHAP profiles and attribute correlation, were tested and not supported. LIME agrees with SHAP more
than the rules do (ρ = {f['lime_pima']:.2f}, {f['lime_early']:.2f}, {f['lime_cdc']:.2f}), as both observe the model on
the same perturbations. In the deletion test (Fig. {3}), the rules' top-3 attributes move the output
{r.deletion('pima', 'rule_consequent') / r.deletion('pima', 'random'):.1f}×, {r.deletion('early', 'rule_consequent') / r.deletion('early', 'random'):.1f}×
and {r.deletion('cdc', 'rule_consequent') / r.deletion('cdc', 'random'):.1f}× as much as random ones and slightly less
than SHAP's; on Pima and CDC the remaining gap largely vanishes under mean replacement, the convention of equation (6), since
SHAP's own value function uses background replacement; on Early Stage a gap remains.</p>
{F('fig7_deletion_test.png', 'Deletion test (background replacement, R = 4); dashed: random order.', wide=True)}
<h3>E. Cost</h3>
<p>Black-box SHAP takes {r.ms('pima', 8, 'blackbox_shap_sec_per_patient'):.0f} ms per patient on Pima (exact, d = 8) and
{r.ms('cdc', 21, 'blackbox_shap_sec_per_patient'):.0f} ms on CDC (sampled, d = 21); TreeSHAP explains all
{r.info['cdc_full_test_rows']:,} CDC test patients in {r.info['cdc_full_test_treeshap_seconds']:.0f} s.</p>"""


def article_discussion(r):
    return f"""
<h2>VI. Discussion</h2>
<p>Intrinsic and post-hoc explanations of ANFIS answer slightly different questions. The consequent reading describes
what the fired rules compute with their firing held fixed; SHAP and LIME describe how the output responds when inputs
change, including changes in which rules fire. The two coincide when firing is moderately concentrated and diverge
when it is nearly all-or-nothing or spread over many rules. This gives a practical design rule: prefer compact rule
bases, monitor firing concentration, and when one rule dominates, a zero-order formulation yields an explanation that
matches the model better.</p>
<p><i>Limitations.</i> SHAP and LIME are references, not ground truth; our deletion test mitigates this but depends on the
removal convention. The antecedent attribution is a first-order approximation. Above ten attributes SHAP is sampled,
and 200 patients per model are explained. All datasets concern diabetes, and SHAP replaces attributes independently.</p>
<h2>VII. Conclusion</h2>
<p>ANFIS is accurate and its rules are informative: they point to genuinely important attributes and get the direction
of their effect right. But the order of importance they imply for an individual patient is only partly faithful, and it
degrades with more rules at no gain in accuracy. ANFIS rule explanations should be verified, not assumed. Future work
includes an exact antecedent attribution, training objectives that favour faithful rules, and user studies with
clinicians.</p>"""
