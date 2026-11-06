import json
from pathlib import Path

from deck_helpers import *
from final_results_slides import results_slides

LIME_C = "E87BA4"
VAR, LREF, DEL = D.get("variants", {}), D.get("lime_ref", {}), D.get("deletion", {})
titles = {"pima": "Pima Indians Diabetes", "early": "Early Stage Diabetes Risk", "cdc": "CDC Diabetes Health Indicators"}

s = new_slide("", "Are ANFIS's Fuzzy Rules Really Interpretable?", dark=True,
              notes="Good morning. Our project checks whether the IF-THEN rules of ANFIS really explain its predictions. "
                    "We present the complete work: the system, the implementation, all experiments with statistical tests, "
                    "the analysis of the results, the dashboard, the report and the article.")
s.shapes.title.top, s.shapes.title.height = Inches(1.5), Inches(1.6)
s.shapes.title.text_frame.paragraphs[0].runs[0].font.size = Pt(42)
text(s, 0.6, 3.15, 11.8, 0.5, "A SHAP-based consistency audit of neuro-fuzzy rule explanations on diabetes diagnosis",
     size=20, color=ICE)
text(s, 0.6, 3.75, 11.8, 0.4, "Soft Computing  |  Final Review", size=16, color=ORANGE, bold=True)
team_file = Path(__file__).resolve().parent / "team.json"
team = json.loads(team_file.read_text(encoding="utf-8")) if team_file.exists() else []
if team:
    text(s, 0.6, 4.6, 5.0, 0.35, "Team members", size=13, color=ICE, bold=True)
for n_, (name, reg) in enumerate(team):
    text(s, 0.6, 5.0 + n_ * 0.42, 4.0, 0.4, name, size=15, color=WHITE)
    text(s, 4.2, 5.0 + n_ * 0.42, 2.0, 0.4, reg, size=15, color=ICE)
text(s, 0.6, 6.55, 11.8, 0.4, "School of Computer Science and Engineering, VIT-AP University", size=13, color=ICE)

s = new_slide("", "Project Progress",
              "Work completed at each review",
              notes="In Review 1 we proposed the idea. In Review 2 we showed ANFIS written from scratch, the comparison "
                    "with SHAP on three diabetes datasets, and LightGBM as a baseline. For the final review we added more "
                    "ANFIS variants, a second explainer, a test that does not depend on SHAP, statistical tests, the "
                    "analysis of why the results differ, the architecture diagrams, the report and the article.")
stages = [("Review 1", "Problem and proposal", ["Question: are ANFIS rules faithful?", "Literature survey and research gap", "Plan: compare ANFIS rules with SHAP on diabetes data"]),
          ("Review 2", "Design and first results", ["ANFIS implemented from scratch, 2 to 16 rules", "Rule explanations vs SHAP on 3 datasets", "LightGBM + TreeSHAP baseline", "Dashboard, results over 5 seeds"]),
          ("Final review", "Complete evaluation", ["Bell memberships and zero-order TSK", "LIME as a second explainer", "Deletion test (no reference needed)", "Wilcoxon / Friedman tests with Holm correction", "Analysis of why the results differ", "Report and research article"])]
for n_, (tag, head, items) in enumerate(stages):
    x = 0.6 + n_ * 4.1
    card(s, x, 1.85, 3.9, 3.0, NAVY if n_ == 2 else TINT)
    dark_ = n_ == 2
    text(s, x + 0.25, 1.97, 3.4, 0.3, tag, size=12, bold=True, color=ORANGE)
    text(s, x + 0.25, 2.28, 3.4, 0.4, head, size=17, bold=True, color=WHITE if dark_ else NAVY, font=HEAD)
    text(s, x + 0.25, 2.8, 3.45, 2.0, items, size=12.5, color=ICE if dark_ else INK, bullets=True, space=3)
    if n_ < 2:
        arrow(s, x + 3.92, 3.2, 0.17, 0.3)
card(s, 0.6, 5.05, 12.1, 1.75)
text(s, 0.85, 5.17, 11.6, 0.3, "Problem statement", size=14, bold=True, color=NAVY)
text(s, 0.85, 5.52, 11.6, 1.25, "ANFIS is preferred in clinical prediction because it explains its decisions with IF-THEN rules. "
     "We test whether the explanation read from those rules matches what actually drives the model's predictions, by "
     "comparing it with independent explanations (SHAP and LIME) and a deletion test, on three diabetes datasets with "
     "8, 16 and 21 attributes, and we analyse when and why the two disagree.", size=13.5, color=INK)

s = new_slide("", "Literature Review and Research Gap",
              "Existing work improves ANFIS rules or explains black-box models; the two are rarely compared",
              notes="Existing ANFIS work improves accuracy or rule structure. HTSK fixes ANFIS for many attributes and we "
                    "use it. SHAP, TreeSHAP and LIME explain any model. Removal tests evaluate explanations without "
                    "ground truth, and the disagreement problem shows explainers often disagree. Nobody measures whether "
                    "the ANFIS rule explanation agrees with independent explanations of the same model. That is our gap.")
lit = [("Jang 1993; Takagi and Sugeno 1985", "ANFIS and TSK fuzzy rules, hybrid learning", "Our base model and training"),
       ("ANFIS variants (Review 1 survey)", "Simpler, compact or more accurate rule bases", "Improve rules; never test faithfulness"),
       ("Cui, Wu and Xu 2021 (HTSK); Gacto et al. 2011", "High-dimensional TSK fix; structural interpretability measures", "HTSK adopted; measures readability, not faithfulness"),
       ("Lundberg and Lee 2017; Lundberg et al. 2020", "SHAP; exact fast TreeSHAP", "Independent reference; scalable SHAP for LightGBM"),
       ("Ribeiro et al. 2016 (LIME)", "Local linear surrogate explanations", "Second independent reference"),
       ("Samek et al. 2017; Hooker et al. 2019; Krishna et al. 2022", "Removal-based evaluation; explainers disagree", "Our deletion test; why we use two references")]
heads = [("Work", 0.6, 4.1), ("What it does", 4.8, 3.9), ("How it relates to us", 8.8, 3.9)]
card(s, 0.6, 1.85, 12.1, 0.42, NAVY, radius=0.15)
for h_, x, w in heads:
    text(s, x + 0.15, 1.92, w - 0.2, 0.3, h_, size=13, bold=True, color=WHITE)
for n_, row in enumerate(lit):
    y = 2.32 + n_ * 0.55
    if n_ % 2 == 0:
        card(s, 0.6, y, 12.1, 0.52, TINT, radius=0.1)
    for (h_, x, w), val in zip(heads, row):
        text(s, x + 0.15, y + 0.03, w - 0.25, 0.46, val, size=11.5, color=INK, bold=(h_ == "Work"), anchor=MSO_ANCHOR.MIDDLE)
card(s, 0.6, 5.7, 12.1, 1.15, "FDEFE9")
text(s, 0.9, 5.8, 11.5, 0.3, "Research gap", size=14, bold=True, color=ORANGE)
text(s, 0.9, 6.12, 11.5, 0.7, "No study measures, patient by patient, whether the explanation read from ANFIS's own rules "
     "agrees with independent explanations of the same model, how this changes with rules, attributes and rule form, "
     "and which part of the rule (IF or THEN) carries the faithful information.", size=13.5, color=INK)

s = new_slide("", "Objectives and Status",
              "All objectives set in Review 1 are implemented, evaluated and documented",
              notes="These are our five objectives. All are complete, and the file that implements each one is shown.")
objs = [("Train ANFIS across attribute counts, rule counts and rule forms", f"From scratch; {len(rules)} rule counts, Gaussian and bell, first and zero order", "src/anfis.py"),
        ("Extract a per-patient importance vector from the rules", "IF-part, THEN-part and combined attributions", "src/attribution.py"),
        ("Explain the same patients independently", "SHAP (exact / permutation), LIME, TreeSHAP for LightGBM", "src/attribution.py"),
        ("Quantify agreement and test it", "7 agreement metrics, deletion test, Wilcoxon, Friedman, Holm", "consistency.py, analyze.py"),
        ("Deliver pipeline, dashboard, report and article", "Reproducible pipeline, live dashboard, report, article draft", "app.py, docs/")]
for n_, (o, done, f) in enumerate(objs):
    y = 1.85 + n_ * 1.0
    card(s, 0.6, y, 12.1, 0.9)
    dot(s, 0.82, y + 0.18, 0.54, NAVY, str(n_ + 1), size=16)
    text(s, 1.6, y + 0.08, 5.3, 0.75, o, size=13.5, bold=True, color=NAVY, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 7.1, y + 0.06, 3.7, 0.55, done, size=12, color=INK, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 7.1, y + 0.6, 3.7, 0.25, f, size=10, color=SLATE, font=MONO)
    card(s, 10.95, y + 0.23, 1.55, 0.44, "DFF3E8", radius=0.4)
    text(s, 10.95, y + 0.23, 1.55, 0.44, "Completed", size=12, bold=True, color="0B7A45", align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

s = new_slide("", "System Architecture",
              "Block diagram: six layers from raw data to outputs, with the source file for each layer",
              notes="This is the block diagram of our system. Data flows left to right: the three datasets, preparation "
                    "without leakage, the models, the explanation methods, the evaluation, and the outputs. Under each "
                    "layer is the file that implements it. The bar at the bottom is the experiment controller that runs "
                    "all four studies for five seeds.")
picture(s, "results/figures/arch_system.png", 1.3, 1.75, 10.7, crop=(0, 95, 2675, 1364), name="arch_sys_crop")

s = new_slide("", "ANFIS Architecture",
              "Five-layer network as implemented in src/anfis.py; adaptive parameters in layers 1 and 4",
              notes="This is the ANFIS network. Layer one has a membership function for every rule and attribute, with a "
                    "learnable centre and width. Layer two combines them into the rule's firing strength using the "
                    "geometric mean, layer three normalises, layer four is each rule's formula, and layer five adds them "
                    "up and gives the probability. On the right are the design choices that make it work for 21 attributes.")
picture(s, "results/figures/arch_anfis.png", 0.6, 1.75, 8.0)
choices = [("Rules from k-means, not a grid", "A 2-set grid needs 2^d rules (2,097,152 for 21 attributes). One rule per cluster lets us choose 2 to 16 rules."),
           ("Geometric-mean AND (HTSK)", "The product of 21 memberships collapses to 0; the d-th root keeps every rule usable."),
           ("Hybrid learning", "k-means starts the rules, least squares the THEN-parts, Adam tunes all parameters with early stopping."),
           ("Variants", "Gaussian or bell (b = 2) memberships; first-order (a·x + b) or zero-order (constant) THEN-parts.")]
for n_, (a, b) in enumerate(choices):
    y = 1.8 + n_ * 1.27
    card(s, 9.05, y, 3.65, 1.15, NAVY if n_ < 3 else TINT)
    text(s, 9.25, y + 0.08, 3.3, 0.3, a, size=13, bold=True, color=WHITE if n_ < 3 else NAVY)
    text(s, 9.25, y + 0.42, 3.3, 0.72, b, size=10.5, color=ICE if n_ < 3 else INK)
card(s, 0.6, 5.55, 8.2, 1.35)
text(s, 0.8, 5.62, 7.9, 1.25, [
    [("Training objective:  ", {"bold": True, "color": NAVY}), ("L = −(1/n) Σᵢ [yᵢ log pᵢ + (1 − yᵢ) log(1 − pᵢ)] + (λ/2)‖a‖²", {"font": "Cambria Math"})],
    [("Rule initialisation (k-means):  ", {"bold": True, "color": NAVY}), ("min Σᵢ ‖xᵢ − c_k(i)‖²", {"font": "Cambria Math"})],
    [("Consequent initialisation (least squares):  ", {"bold": True, "color": NAVY}), ("θ = (ΦᵀΦ + I)⁻¹ Φᵀt", {"font": "Cambria Math"})],
    [(f"Parameters for 4 rules, 21 attributes: 2Rd + R(d+1) = {2 * 4 * 21 + 4 * 22}; gradients derived by hand and checked numerically.", {"color": SLATE, "size": 11})]],
     size=12, color=INK, space=2)

s = new_slide("", "Experimental Design",
              "Four studies; every setting repeated for 5 random seeds on the same 200 explained test patients",
              notes="We ran four studies. A changes the number of rules, B the number of attributes, C measures SHAP cost "
                    "and D changes the membership function and TSK order. Every ANFIS is explained by its rules, by SHAP "
                    "and by LIME, and all explanations go through the deletion test. Then we test differences statistically.")
studies = [("A", "Rule count", f"R = {', '.join(map(str, rules))}; all attributes"),
           ("B", "Attribute count", "Top-k attributes by mutual information; 4 rules"),
           ("C", "SHAP cost", "Time per patient: black-box SHAP, TreeSHAP, LIME"),
           ("D", "Model variants", "Gaussian / bell × first / zero order; 4 rules")]
for n_, (a, b, c) in enumerate(studies):
    x = 0.6 + n_ * 3.07
    card(s, x, 1.9, 2.9, 1.7)
    dot(s, x + 0.2, 2.05, 0.5, ORANGE, a, size=16)
    text(s, x + 0.85, 2.1, 1.95, 0.4, b, size=15, bold=True, color=NAVY, anchor=MSO_ANCHOR.MIDDLE)
    text(s, x + 0.2, 2.7, 2.55, 0.85, c, size=12, color=INK)
steps = [("Explanation", "Rules (THEN, IF, IF+THEN)\nSHAP and LIME on ANFIS\nTreeSHAP on LightGBM"),
         ("Agreement", "Spearman, Kendall, weighted τ\ntop-1, top-3, sign, global"),
         ("Faithfulness", "Deletion test: remove top-k\nattributes, measure output change"),
         ("Significance", "Wilcoxon signed-rank, Friedman\nHolm correction, effect size r"),
         ("Analysis", "Diagnostics: firing concentration,\nSHAP concentration, correlation")]
for n_, (a, b) in enumerate(steps):
    x = 0.6 + n_ * 2.47
    card(s, x, 3.9, 2.2, 2.05, NAVY)
    text(s, x + 0.18, 4.02, 1.9, 0.35, a, size=15, bold=True, color=ORANGE)
    text(s, x + 0.18, 4.45, 1.95, 1.45, b.split("\n"), size=11.5, color=WHITE, space=2)
    if n_ < 4:
        arrow(s, x + 2.23, 4.75, 0.2, 0.3)
text(s, 0.6, 6.2, 12.1, 0.6, f"Split 60/20/20 stratified · decision thresholds chosen on validation data · SHAP background: "
     f"{INFO['n_background']} training patients · explained: {INFO['n_explain']} test patients per model and seed.",
     size=12.5, color=SLATE)

s = new_slide("", "Datasets and Preprocessing",
              "Three public diabetes datasets with 8, 16 and 21 attributes; raw files kept unmodified in data/raw",
              notes="We use three public diabetes datasets with 8, 16 and 21 attributes, from 520 to over two hundred and "
                    "fifty thousand records. All imputation and scaling values are learned from training patients only.")
meta = {"pima": ("Lab measurements of women aged 21+, NIDDK", "Glucose, BMI, Age, Insulin ..."),
        "early": ("Symptom questionnaire, Sylhet Diabetes Hospital (UCI 529)", "Polyuria, Polydipsia, Age, Gender ..."),
        "cdc": ("US telephone health survey BRFSS 2015 (UCI 891)", "GenHlth, HighBP, BMI, Age ...")}
for n_, ds in enumerate(DS):
    x = 0.6 + n_ * 4.1
    d = recs[titles[ds]]
    card(s, x, 1.95, 3.9, 2.75)
    text(s, x + 0.25, 2.08, 3.4, 0.35, DS_FULL[ds], size=15, bold=True, color=NAVY)
    text(s, x + 0.25, 2.48, 1.6, 0.7, str(d["attributes"]), size=40, bold=True, color=ORANGE, font=HEAD)
    text(s, x + 0.25, 3.22, 1.6, 0.3, "attributes", size=12, color=SLATE)
    text(s, x + 1.75, 2.48, 2.0, 0.7, f"{d['records']:,}", size=30, bold=True, color=NAVY, font=HEAD, anchor=MSO_ANCHOR.BOTTOM)
    text(s, x + 1.75, 3.22, 2.0, 0.3, f"records, {d['positive_rate'] * 100:.0f}% diabetic", size=12, color=SLATE)
    text(s, x + 0.25, 3.62, 3.45, 1.0, [meta[ds][0], [(meta[ds][1], {"color": SLATE})]], size=12, color=INK)
prep = [("Fix invalid values", f"Pima: impossible zeros in 5 columns ({recs[titles['pima']]['missing_cells']} cells) treated as missing, filled with the training median"),
        ("Encode", "Early Stage: Yes/No and gender to 1/0. CDC: ID column dropped"),
        ("Split", "Stratified 60% train, 20% validation, 20% test, per seed"),
        ("Scale without leakage", "Mean 0, spread 1, using training statistics only")]
for n_, (a, b) in enumerate(prep):
    x = 0.6 + n_ * 3.07
    dot(s, x, 4.98, 0.45, NAVY, str(n_ + 1), size=14)
    text(s, x + 0.58, 4.95, 2.4, 0.35, a, size=14, bold=True, color=NAVY)
    text(s, x + 0.58, 5.32, 2.4, 1.45, b, size=12, color=INK)

s = new_slide("", "Justification of the Soft Computing Technique",
              notes="ANFIS combines fuzzy logic and neural learning, which suits medical values with no sharp borders. It "
                    "is also the model whose interpretability claim we test, so it must be our core model. What is new is "
                    "that we measure the faithfulness of its own explanation, split into IF and THEN parts, with two "
                    "independent explainers, a test that needs no explainer, and statistics.")
just = [("Fuzzy logic fits clinical data", "Glucose or BMI are not just normal or abnormal; membership degrees model 'fairly high' as a clinician does."),
        ("Neural learning tunes the rules", "Centres, widths and rule formulas are learned from data, so accuracy does not rely on hand-written rules."),
        ("It is the model under test", "ANFIS is chosen in healthcare for its interpretability claim; we measure exactly that claim."),
        ("It is competitive", f"ANFIS matches LightGBM's ROC-AUC on all three datasets (CDC {auc('cdc', 'ANFIS-4'):.3f} vs {auc('cdc', 'LightGBM'):.3f}).")]
for n_, (a, b) in enumerate(just):
    x, y = 0.6 + (n_ % 2) * 4.1, 1.75 + (n_ // 2) * 2.3
    card(s, x, y, 3.9, 2.1)
    dot(s, x + 0.25, y + 0.22, 0.5, ORANGE, str(n_ + 1), size=15)
    text(s, x + 0.9, y + 0.22, 2.85, 0.5, a, size=15, bold=True, color=NAVY, anchor=MSO_ANCHOR.MIDDLE)
    text(s, x + 0.25, y + 0.9, 3.4, 1.15, b, size=12.5, color=INK)
card(s, 8.85, 1.75, 3.85, 4.4, NAVY)
text(s, 9.1, 1.95, 3.35, 0.35, "Novelty of our work", size=13, bold=True, color=ORANGE)
nov = ["First per-patient audit of ANFIS rule explanations against independent explainers",
       "Rule explanation split into an exact THEN-part and a first-order IF-part",
       "Two references (SHAP, LIME) plus a reference-free deletion test",
       "Effects of rules, attributes, membership shape and TSK order, with statistics"]
text(s, 9.1, 2.4, 3.35, 3.7, nov, size=12.5, color=WHITE, bullets=True, space=9)

s = new_slide("", "Explanation Methods and Evaluation",
              "Every method gives one number per attribute per patient, in log-odds",
              notes="For each patient we get several explanations. From inside ANFIS: the THEN-part, what you read from the "
                    "rule formulas, and the IF-part, the hidden effect on which rules fire. From outside: SHAP and LIME. "
                    "We compare them with agreement scores, and the deletion test checks which explanation picks the "
                    "attributes that really matter.")
card(s, 0.6, 1.9, 3.9, 3.4)
text(s, 0.8, 2.02, 3.5, 0.3, "From the rules (intrinsic)", size=13, bold=True, color=BLUE)
text(s, 0.8, 2.38, 3.5, 0.3, "THEN-part", size=14, bold=True, color=NAVY)
text(s, 0.8, 2.7, 3.5, 0.4, "φⱼ = Σᵣ w̄ᵣ · aᵣⱼ · xⱼ", size=15, color=ORANGE, bold=True, font="Cambria Math")
text(s, 0.8, 3.15, 3.5, 0.3, "IF-part", size=14, bold=True, color=NAVY)
text(s, 0.8, 3.47, 3.5, 0.4, "φⱼ = Σᵣ w̄ᵣ (fᵣ − z) Δlog μᵣⱼ", size=15, color=ORANGE, bold=True, font="Cambria Math")
text(s, 0.8, 4.0, 3.5, 1.25, "THEN-part + fired constants = model output exactly (verified). Zero-order models have no THEN-part.",
     size=11.5, color=INK)
card(s, 4.7, 1.9, 3.9, 3.4)
text(s, 4.9, 2.02, 3.5, 0.3, "Post-hoc (independent)", size=13, bold=True, color=VIOLET)
text(s, 4.9, 2.38, 3.55, 2.9, [[("SHAP: ", {"bold": True, "color": NAVY}), ("exact up to 10 attributes, sampled above", {})],
                              [("φⱼ = Σ_S |S|!(d−|S|−1)!/d! · [v(S∪{j}) − v(S)]", {"font": "Cambria Math", "color": ORANGE, "bold": True, "size": 11})],
                              [("LIME (our code): ", {"bold": True, "color": NAVY}), ("1,000 perturbed patients, weighted ridge fit", {})],
                              [("min_β Σ π(x,z)(f(z) − β·z)² + ‖β‖²", {"font": "Cambria Math", "color": ORANGE, "bold": True, "size": 11})],
                              [("TreeSHAP: ", {"bold": True, "color": NAVY}), ("exact SHAP for LightGBM; same 100 background patients for all", {})]], size=11.5, color=INK, space=5)
card(s, 8.8, 1.9, 3.9, 3.4, NAVY)
text(s, 9.0, 2.02, 3.5, 0.3, "Deletion test (no reference)", size=13, bold=True, color=ORANGE)
text(s, 9.0, 2.4, 3.5, 2.85, ["Take each explanation's top-k attributes", "Replace them with values from background patients",
                              "Re-run ANFIS and measure the change in output", "Larger change = more faithful choice; random order is the control",
                              [("Δₖ = mean |z(x) − E[z(x with top-k replaced)]|", {"font": "Cambria Math", "size": 11})]],
     size=12, color=WHITE, bullets=True, space=5)
mets = [("Spearman / Kendall", "Same order of importance?"), ("Weighted τ", "Same order at the top?"),
        ("Top-1 / top-3", "Same most important attributes?"), ("Sign", "Same direction?")]
for n_, (a, b) in enumerate(mets):
    x = 0.6 + n_ * 3.07
    card(s, x, 5.55, 2.9, 1.25)
    text(s, x + 0.15, 5.65, 2.6, 0.35, a, size=13.5, bold=True, color=NAVY)
    text(s, x + 0.15, 6.02, 2.6, 0.7, b, size=12, color=INK)

s = new_slide("", "Implementation: ANFIS Forward Pass",
              "src/anfis.py; each line corresponds to one layer of the network",
              notes="This is the real forward-pass code. log_membership is layer one, for Gaussian or bell curves. The mean "
                    "over attributes is the geometric-mean AND of layer two. Dividing by the sum is layer three, X times "
                    "a plus b is layer four, and the weighted sum is layer five.")
code_box(s, 0.6, 1.85, 12.1, 3.2, "src/anfis.py", "    def forward", "    def decision_function", size=15)
maps = [("Layer 1", "log_membership()"), ("Layer 2", ".mean(axis=2)"), ("Layer 3", "w / w.sum(...)"),
        ("Layer 4", "X @ self.a.T + self.b"), ("Layer 5", "(wbar * f).sum(axis=1)")]
for n_, (a, b) in enumerate(maps):
    x = 0.6 + n_ * 2.47
    card(s, x, 5.35, 2.22, 1.0)
    text(s, x + 0.18, 5.45, 2.0, 0.35, a, size=14, bold=True, color=NAVY)
    text(s, x + 0.18, 5.85, 2.0, 0.4, b, size=11, color=INK, font=MONO)

s = new_slide("", "Implementation: Rule-Based Attribution",
              "src/attribution.py; computed only from the trained model's own parameters",
              notes="This function produces the explanation from the rules. 'consequent' is the THEN-part: vote share times "
                    "coefficient times the patient's value. 'antecedent' is the IF-part. It uses only the model's own "
                    "numbers. LIME and the deletion test are in the same file and are our own code too.")
code_box(s, 0.6, 1.85, 12.1, 3.3, "src/attribution.py", "def rule_attribution", "def shap_blackbox", size=12, skip_doc=True)
pts = [("consequent", "THEN-part: vote share × coefficient × patient's value, summed over rules"),
       ("antecedent", "IF-part: how each attribute shifted the rules' votes, to first order"),
       ("Also in this file", "shap_blackbox, shap_tree, lime_local (our code), deletion_test (our code)")]
for n_, (a, b) in enumerate(pts):
    x = 0.6 + n_ * 4.1
    card(s, x, 5.4, 3.9, 1.35)
    text(s, x + 0.2, 5.5, 3.5, 0.4, a, size=14, bold=True, color=NAVY)
    text(s, x + 0.2, 5.9, 3.5, 0.8, b, size=12.5, color=INK)

s = new_slide("", "Implementation Status and Testing",
              "Complete pipeline with automated correctness checks; all numbers in these slides come from the result files",
              notes="Everything on the left is implemented and runs end to end. On the right is the real output of our "
                    "correctness checks: gradients match a numerical estimate for all four ANFIS variants, the rule "
                    "explanation rebuilds the model output exactly, and SHAP adds up exactly.")
done = ["Leakage-free data pipeline for 3 datasets",
        "ANFIS from scratch: 5 layers, hybrid learning, 4 variants",
        "Rule attribution: IF-part, THEN-part, combined",
        "SHAP, TreeSHAP, LIME (own code), deletion test (own code)",
        "7 agreement metrics, Wilcoxon / Friedman / Holm",
        f"Studies A to D, {n_seeds} seeds ({INFO['total_runtime_seconds'] / 60:.0f} min of computing)",
        "9 result figures, dashboard, report, article",
        "Correctness tests for all of the above"]
for n_, d_ in enumerate(done):
    y = 1.95 + n_ * 0.58
    dot(s, 0.6, y + 0.03, 0.34, "0B7A45", "✓", size=13)
    text(s, 1.1, y, 5.6, 0.45, d_, size=14, color=INK, anchor=MSO_ANCHOR.MIDDLE)
card(s, 6.8, 1.95, 5.9, 4.0, "1E2430", radius=0.03)
text(s, 7.05, 2.05, 5.4, 0.3, "python tests/test_anfis.py", size=11, color="8F9BB3", font=MONO, bold=True)
text(s, 7.05, 2.45, 5.5, 3.45, [[(l, {})] for l in D["tests"]], size=10, color="A5D6A7", font=MONO, space=2)
card(s, 6.8, 6.1, 5.9, 0.75)
text(s, 7.05, 6.1, 5.4, 0.75, "Run order: tests → run_experiments.py → analyze.py → make_figures.py → app.py",
     size=12, color=INK, font=MONO, anchor=MSO_ANCHOR.MIDDLE)

results_slides(globals())

import sys
out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "Final_Review_ANFIS_SHAP_Audit.pptx"
prs.save(out)
print("wrote", out, f"({count[0]} slides)")
