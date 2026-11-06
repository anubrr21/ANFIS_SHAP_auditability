import subprocess
import sys
from datetime import date

import joblib
import numpy as np
import pandas as pd

import report_example as E
from report_common import CHROME, snap, team, FIG, ROOT, Results, cite, references_html
from report_sections import comparison_chapter, conclusion_chapter, firing, future_chapter, results_chapter

sys.path.insert(0, str(ROOT))
from src.data import load_raw

DOCS = ROOT / "docs"
SCR = DOCS / "screens"

CSS = """
@page { size: A4; margin: 18mm 18mm 17mm 20mm; }
body { font-family: 'Times New Roman', Times, serif; font-size: 11.5pt; line-height: 1.4; color: #111; margin: 0; text-align: justify; }
h1 { font-size: 15.5pt; text-align: center; margin: 0 0 5mm; text-transform: uppercase; page-break-before: always; }
h2 { font-size: 13pt; margin: 5.5mm 0 1.8mm; page-break-after: avoid; }
h3 { font-size: 11.8pt; margin: 4mm 0 1.2mm; font-style: italic; page-break-after: avoid; }
p { margin: 0 0 2.4mm; } ul, ol { margin: 0 0 2.4mm; padding-left: 7mm; } li { margin-bottom: 0.8mm; }
code { font-family: Consolas, monospace; font-size: 9.6pt; }
table.t { border-collapse: collapse; width: 100%; margin: 1.5mm 0 1mm; font-size: 9.6pt; page-break-inside: avoid; text-align: left; line-height: 1.3; }
table.t tbody th { white-space: nowrap; text-align: left; }
table.t thead th { text-align: center !important; vertical-align: middle; }
table.t th { border-top: 1.4px solid #111; border-bottom: 0.9px solid #111; padding: 1mm 1.6mm; font-weight: bold; vertical-align: bottom; }
table.t td { padding: 0.8mm 1.6mm; border-bottom: 0.5px solid #ccc; vertical-align: top; }
table.t tr:last-child td { border-bottom: 1.4px solid #111; }
.cap { font-size: 9.6pt; text-align: center; margin: 1mm 0 4mm; }
figure { margin: 2.5mm 0 2mm; text-align: center; page-break-inside: avoid; }
figure img { max-width: 100%; }
figure.half img { max-width: 62%; } figure.small img { max-width: 40%; }
figcaption { font-size: 9.6pt; margin-top: 1mm; text-align: justify; }
.eq { margin: 1.8mm 0 1.8mm 7mm; font-size: 11pt; position: relative; text-align: left; padding-right: 11mm; page-break-inside: avoid; }
.eqn { position: absolute; right: 0; }
.cover { text-align: center; padding-top: 16mm; }
.cover h1 { page-break-before: auto; font-size: 20pt; text-transform: none; margin-bottom: 10mm; }
.cover p { text-align: center; margin: 2mm 0; }
.toc { list-style: none; padding: 0; } .toc li { margin-bottom: 0.9mm; display: flex; }
.toc li span:first-child { width: 13mm; } .toc li.sub { padding-left: 13mm; font-size: 10.5pt; }
.refs li { margin-bottom: 1.4mm; font-size: 10.5pt; text-align: left; }
.abstract { margin: 0 6mm; }
.snap { margin: 2mm 0 3.5mm; border: 1px solid #999; page-break-inside: avoid; }
.snap .bar { background: #eee; font-size: 8.8pt; padding: 1mm 2mm; font-family: Consolas, monospace; }
.snap .bar i { display: none; } .snap .bar em { float: right; font-style: normal; color: #555; }
.snap pre { margin: 0; padding: 1.6mm 3mm; font-family: Consolas, monospace; font-size: 8.2pt; line-height: 1.28; white-space: pre-wrap; }
.snap .ln { color: #888; display: inline-block; width: 8mm; }
.kw { color: #7a2f00; font-weight: bold; } .st { color: #22663a; } .cm { color: #666; font-style: italic; } .nu { color: #1a4f9c; }
pre.out { font-family: Consolas, monospace; font-size: 8.6pt; border: 1px solid #999; padding: 2mm 3mm; white-space: pre-wrap; margin: 1mm 0 3mm; }
.note { border-left: 2.5px solid #555; padding: 0.5mm 0 0.5mm 3mm; margin: 1mm 0 3mm; }
"""

CHAPTERS = [
    ("", "Abstract", []),
    ("1", "Introduction", ["Background", "Terms used in this report", "Problem statement", "Objectives", "Contributions",
                           "Organisation of the report"]),
    ("2", "Literature Review", ["Neuro-fuzzy systems and their interpretability", "Post-hoc explanation methods",
                                "Evaluating explanations", "Summary of existing work", "Research gap"]),
    ("3", "Motivation", []),
    ("4", "Proposed Work", ["Research questions", "Proposed audit framework", "Why each component is needed",
                            "Expected deliverables"]),
    ("5", "Methodology", ["Datasets", "Data preparation", "Fuzzy sets and membership functions", "The ANFIS model, layer by layer",
                          "Why the geometric mean (HTSK) is needed", "How the rules are created and trained",
                          "Turning the rules into an explanation", "SHAP: the first independent explanation",
                          "LIME: the second independent explanation", "Deletion test", "Agreement metrics",
                          "Experimental design", "Statistical analysis", "Summary of all formulas"]),
    ("6", "Model / Architecture / Design", ["System architecture", "ANFIS network architecture", "Number of parameters",
                                            "Module design", "Flow of one experiment run", "Dashboard design"]),
    ("7", "Implementation", ["Environment", "Code organisation", "Key code", "Verification", "The dashboard in use"]),
    ("8", "Results and Analysis", ["Predictive performance", "Agreement of the rule explanation with SHAP",
                                   "Effect of the number of rules", "Effect of the number of attributes",
                                   "Cost of the explanations", "Membership function and TSK order",
                                   "Why the Early Stage data is the hardest case", "A second reference explainer: LIME",
                                   "Deletion test", "Summary of the statistical tests"]),
    ("9", "Comparison with Existing Methods", ["Prediction", "Explanation", "Evaluation approach", "Explanation cost"]),
    ("10", "Conclusion", []), ("11", "Scope for Future Work", []), ("", "References", []),
    ("", "Appendix", ["A. Reproducing the results", "B. Files produced by the pipeline", "C. Attributes of the datasets"]),
]


def _numeric(col):
    txt = col.astype(str).str.replace(r"<[^>]+>|[+−\-±%,<>×.\s]|yes|no|n/a|reference", "", regex=True)
    return bool(len(col)) and txt.str.fullmatch(r"\d*").all()


def table(df, caption=None, index=False):
    df = df.copy()
    num = [c for c in df.columns if _numeric(df[c])]
    pos = {c: i for i, c in enumerate(df.columns)}
    html = df.to_html(index=index, border=0, classes="t", escape=False)
    style = "".join(f"table.t.n{abs(hash(tuple(df.columns))) % 10**8} td:nth-of-type({pos[c] + 1})" + "{text-align:center}" for c in num)
    tag = f"n{abs(hash(tuple(df.columns))) % 10**8}"
    html = html.replace('class="dataframe t"', f'class="dataframe t {tag}"', 1)
    return (f"<style>{style}</style>" if style else "") + html + (f"<p class='cap'>{caption}</p>" if caption else "")


def eq(n, body):
    return f'<div class="eq"><span class="eqn">({n})</span>{body}</div>'


def f3(v):
    return f"{v:+.3f}"


def build():
    r = Results()
    info = r.info
    lo, hi = r.rules[0], r.rules[-1]
    n = {"t": 0, "f": 0}

    def T(df, caption, index=False):
        n["t"] += 1
        return table(df, f"Table {n['t']}. {caption}", index)

    def F(path, caption, cls=""):
        n["f"] += 1
        src = path if "/" in str(path) or "\\" in str(path) else FIG / path
        return (f"<figure class='{cls}'><img src='{src.as_uri() if hasattr(src, 'as_uri') else src}'>"
                f"<figcaption>Figure {n['f']}. {caption}</figcaption></figure>")

    crops = E.make_crops()
    e = E.worked_example(0)
    dis_t, dis_then, dis_full, dis_w = E.disagreement_example(25)
    h = E.htsk_demo()
    feats = e["F"]
    tests = subprocess.run([str(ROOT / ".venv" / "Scripts" / "python.exe"), "-W", "ignore",
                            str(ROOT / "tests" / "test_anfis.py")], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    raw = {d: load_raw(d)[0] for d in ["pima", "early", "cdc"]}
    bundles = {d: joblib.load(ROOT / "results" / "models" / f"{d}_seed0.joblib") for d in ["pima", "early", "cdc"]}
    mi_top = {d: ", ".join(np.array(b["features"])[b["mi_ranking"][:4]]) for d, b in bundles.items()}
    ratios = [r.deletion(d, 'rule_consequent') / r.deletion(d, 'random') for d in ['pima', 'early', 'cdc']]
    members = ("<p>Submitted by</p>" + "".join(f"<p><b>{nm}</b> ({reg})</p>" for nm, reg in team())) if team() else ""
    S = []
    add = S.append

    toc = []
    for num, title, subs in CHAPTERS:
        toc.append(f"<li><span>{num}</span><span><b>{title}</b></span></li>")
        toc += [f"<li class='sub'><span>{num + '.' + str(i + 1) if num else ''}</span><span>{s}</span></li>"
                for i, s in enumerate(subs)]

    add(f"""
<div class="cover">
<p style="font-size:13pt">SOFT COMPUTING &middot; PROJECT REPORT</p>
<h1>Are ANFIS's Fuzzy Rules Really Interpretable?<br>A SHAP-Based Consistency Audit on Diabetes Diagnosis</h1>
{members}
<p style="margin-top:14mm">School of Computer Science and Engineering<br>VIT-AP University</p>
<p style="margin-top:8mm">{date.today().strftime('%B %Y')}</p>
</div>
<h1>Abstract</h1>
<div class="abstract">
<p>An adaptive neuro-fuzzy inference system (ANFIS) explains each of its predictions with IF-THEN rules, and this is the
main reason it is chosen for medical decision support. Whether the explanation read from those rules actually matches
what drives the model's output is rarely checked. This project checks it. We implemented a first-order
Takagi-Sugeno-Kang ANFIS from scratch, with rules placed by k-means clustering, dimension-normalised rule firing (HTSK)
and hybrid learning, and trained it on three public diabetes datasets with 8, 16 and 21 attributes (768 to 253,680
records). For every explained patient we turned the rule base into a score for each attribute, separately for the
THEN part of the rules (what the rule formulas compute) and the IF part (how the attribute changes which rules fire),
and compared these scores with SHAP values of the same model, with LIME as a second independent explainer, and with a
deletion test that needs no reference explainer. LightGBM with TreeSHAP and logistic regression were used as baselines.
The study was repeated over {r.seeds} random splits, {len(r.rules)} rule counts ({lo} to {hi}), subsets of the attributes,
and Gaussian and bell membership functions in first- and zero-order form, and all differences were tested with
Holm-corrected Wilcoxon and Friedman tests.</p>
<p>ANFIS was as accurate as LightGBM on all three datasets (ROC-AUC {r.auc('pima', 'ANFIS', 4).mean():.3f},
{r.auc('early', 'ANFIS', 4).mean():.3f} and {r.auc('cdc', 'ANFIS', 4).mean():.3f}). The THEN-part explanation got the
direction of each attribute's effect right in at least
{min(r.c(d, 'rule_consequent', 'sign_agreement', rules=k) for d in ['pima', 'early', 'cdc'] for k in r.rules) * 100:.0f}% of
cases and pointed at attributes that really change the output ({min(ratios):.1f} to {max(ratios):.1f} times more than
random ones in the deletion test), but the order of importance it gave for an individual patient matched SHAP only partly (Spearman
{r.c('early', 'rule_consequent'):.2f} to {r.c('cdc', 'rule_consequent'):.2f} with four rules). This agreement fell
significantly as rules were added, while accuracy did not improve. The cause is rule switching: an attribute also
changes which rules fire, and the THEN-part reading does not see this. The weakest case, Early Stage, is the one where a
single rule takes about {firing('early', rules=4) * 100:.0f}% of the vote. ANFIS rule explanations are therefore useful but
should be checked rather than assumed, and small rule bases should be preferred.</p>
<p><b>Keywords:</b> ANFIS, neuro-fuzzy systems, interpretability, explainable AI, SHAP, LIME, faithfulness, diabetes.</p>
</div>
<h1>Table of Contents</h1>
<ul class="toc">{''.join(toc)}</ul>""")

    terms = pd.DataFrame([
        ("Attribute", "One input measurement of a patient, for example Glucose or BMI (also called a feature)."),
        ("Log-odds (score z)", "The model's raw output before it is turned into a probability: z = log(p / (1 − p)). z = 0 means 50%, positive z means more likely diabetic. Explanations are given in log-odds because there the contributions of attributes add up."),
        ("Fuzzy set, membership μ", "A set with soft edges. μ between 0 and 1 says how much a value belongs to it, for example how much a glucose of 140 is \"high\"."),
        ("Rule", "An IF-THEN statement inside ANFIS. The IF part says which patients the rule is about; the THEN part gives the rule's score."),
        ("Firing strength w̄", "How well a patient matches a rule's IF part, normalised so that the strengths of all rules add up to 1. It works like each rule's share of the vote."),
        ("Attribution φ", "An explanation written as one number per attribute: how much that attribute pushed this patient's score up (positive) or down (negative)."),
        ("THEN-part / IF-part attribution", "The attribution read from the rules' THEN formulas, and the attribution that comes from attributes changing which rules fire. Defined in Section 5.7."),
        ("Faithful explanation", "An explanation that describes what really drives the model's output. Readable is not the same as faithful."),
        ("SHAP", "A method from game theory that measures each attribute's contribution by testing the model with and without it. Used as the independent reference."),
        ("LIME", "A method that fits a simple straight-line model around one patient and uses its slopes as the explanation. Used as a second reference."),
        ("Deletion test", "A check that needs no reference: remove the attributes an explanation calls most important and measure how much the prediction changes."),
        ("Agreement", "How similar two explanations are, measured by rank correlation (Spearman, Kendall), top-k overlap and sign agreement (Section 5.11)."),
        ("Seed", "A number that fixes the random split of the data. Repeating with five seeds shows whether results are stable."),
    ], columns=["Term", "Meaning in this report"])

    add(f"""
<h1>1. Introduction</h1>
<h2>1.1 Background</h2>
<p>Machine-learning models are used to screen patients for chronic diseases such as diabetes. In a clinic a prediction
alone is not enough: before acting, a doctor needs to know <i>why</i> a patient was flagged. This has made interpretable
models attractive. Among soft computing techniques, the adaptive neuro-fuzzy inference system (ANFIS) {cite('jang')} is
widely chosen for this reason. It combines the IF-THEN rules of a Takagi-Sugeno-Kang (TSK) fuzzy system
{cite('takagi')} with neural-network-style learning, so that its decisions can be read as statements such as
<i>IF glucose is HIGH AND BMI is HIGH THEN the risk score is high</i>. Fuzzy logic handles values such as "fairly high
glucose" naturally, and the learning step tunes the rules from data instead of requiring an expert to write them.</p>
<h2>1.2 Terms used in this report</h2>
{T(terms, 'Terms used throughout the report.')}
<h2>1.3 Problem statement</h2>
<p>Readable rules do not by themselves guarantee a faithful explanation. A trained ANFIS mixes several partly fired
rules, each with its own formula, and how much each rule counts itself depends on the patient. Suppose the rules that
fire for a patient have a large coefficient on glucose: a reader concludes that glucose drove the prediction. But a change
in BMI may move the patient from one rule to another and change the output more than glucose does. The explanation read
from the rules would then be readable but wrong. The problem addressed in this project is:</p>
<p class="note"><b>For a trained ANFIS, how closely does the explanation obtained from its own rule base agree with
independent measurements of what drove each prediction, which part of the rule (IF or THEN) carries the faithful part,
and how does this depend on the number of rules, the number of attributes and the form of the rules?</b></p>
<h2>1.4 Objectives</h2>
<ol>
<li>Implement ANFIS from scratch and train it on diabetes datasets that differ in size and number of attributes, with a
controllable number of rules, two membership functions and two TSK orders.</li>
<li>Derive, for each patient, a score per attribute directly from the rule base (firing strengths, memberships and
THEN formulas), separately for the THEN part and the IF part.</li>
<li>Explain the same model and patients independently with SHAP and LIME, and test all explanations with a deletion
test that needs no reference.</li>
<li>Measure the agreement with rank correlation, top-k overlap and sign agreement, and test differences statistically.</li>
<li>Deliver a reproducible pipeline, an interactive dashboard, this report and a research article.</li>
</ol>
<h2>1.5 Contributions</h2>
<ul>
<li>A from-scratch ANFIS in NumPy with hand-derived gradients, checked numerically, for Gaussian and bell
memberships in first- and zero-order form.</li>
<li>A split of the rule explanation into a THEN part, which adds up exactly to the model output, and an IF part, which
captures the effect of rule switching; both are derived step by step in Section 5.7.</li>
<li>A systematic audit on three datasets with {r.seeds} repetitions, {len(r.rules)} rule counts, attribute subsets and
four model variants, using two independent explainers, a deletion test and corrected significance tests, together with
measurements that explain <i>why</i> each result occurs.</li>
<li>A reproducible pipeline and a live dashboard that explains any test patient.</li>
</ul>
<h2>1.6 Organisation of the report</h2>
<p>Chapter 2 reviews the literature and Chapter 3 gives the motivation. Chapter 4 states the research questions and the
proposed framework. Chapter 5 explains the method completely: the data, the ANFIS model and every formula, how the rules
are trained, how the explanations are computed and compared, and the statistics. Chapter 6 presents the architecture and
Chapter 7 the implementation, including the dashboard. Chapter 8 reports and analyses the results, Chapter 9 compares
them with existing methods, and Chapters 10 and 11 conclude and outline future work.</p>""")

    lit = r.table_literature()
    add(f"""
<h1>2. Literature Review</h1>
<h2>2.1 Neuro-fuzzy systems and their interpretability</h2>
<p>Takagi and Sugeno {cite('takagi')} introduced fuzzy models whose rule outputs are functions of the inputs. Jang
{cite('jang')} showed that such a system can be written as an adaptive network and trained by a hybrid of least squares
(for the THEN parts) and gradient descent (for the membership functions); this is ANFIS. A known weakness is the size of
the rule base: with grid partitioning, where each attribute is split into a few fuzzy sets and every combination becomes
a rule, the number of rules grows exponentially with the number of attributes. Cui, Wu and Xu {cite('cui')} showed that
TSK systems also fail numerically in high dimension, because the product of many membership values collapses towards
zero and one rule takes the whole vote, and proposed normalising the firing level by the input dimension (HTSK).
Interpretability of fuzzy rule bases has mostly been measured structurally, by the number of rules, the number of
conditions per rule and how distinct the fuzzy sets are {cite('gacto')}. Such measures say how <i>readable</i> a rule
base is, not whether it describes the model <i>faithfully</i>.</p>
<h2>2.2 Post-hoc explanation methods</h2>
<p>Post-hoc methods explain a trained model from the outside. SHAP {cite('shap')} gives each attribute its Shapley value
{cite('shapley')}, the only attribution that satisfies the efficiency, symmetry, dummy and additivity properties. For a
black-box model the exact computation needs a number of model evaluations that doubles with every attribute. For tree
ensembles, TreeSHAP {cite('treeshap')} computes exact values in polynomial time, which makes SHAP practical on large data
with models such as LightGBM {cite('lgbm')}. LIME {cite('lime')} instead fits a weighted linear model around the explained
case.</p>
<h2>2.3 Evaluating explanations</h2>
<p>Different explainers often disagree about the same prediction, which Krishna et al. {cite('krishna')} call the
disagreement problem. Because a true explanation is normally unknown, removal-based tests are used: attributes are
removed in the order an explanation ranks them, and the change in output is measured, as in the perturbation analysis of
Samek et al. {cite('samek')} and the benchmark of Hooker et al. {cite('hooker')}.</p>
<h2>2.4 Summary of existing work</h2>
{T(lit, 'Existing work and its relation to this project.')}
<h2>2.5 Research gap</h2>
<p>Work on ANFIS has concentrated on accuracy and on making rule bases smaller or better behaved in high dimension,
while work on post-hoc explanation treats models as black boxes. We found no study that measures, patient by patient,
whether the explanation read from an ANFIS rule base agrees with independent explanations of the same model, how that
agreement changes with the number of rules and attributes and with the form of the rules, or whether the IF part or the
THEN part of the rules carries the faithful part of the explanation. This project fills that gap.</p>""")

    add(f"""
<h1>3. Motivation</h1>
<p><b>Diabetes screening is common and important.</b> Diabetes is among the most common chronic diseases, and models that
flag patients for follow-up tests are already used. A wrong or unexplained flag directly affects patient care.</p>
<p><b>ANFIS is chosen precisely for its explanations.</b> In healthcare, ANFIS is often preferred over more accurate
black-box models because its rules can be shown to a doctor. If that explanation is not faithful, the main reason for
choosing ANFIS falls away, and a doctor may trust a reason that does not reflect the model.</p>
<p><b>The claim can now be checked.</b> SHAP, LIME and removal-based tests make it possible to measure what drives a
model's output independently of its rules. They are widely used on black boxes but rarely used to audit models that
already claim to be interpretable.</p>
<p><b>Real clinical data is varied, and SHAP is costly on large data.</b> Diabetes data mixes laboratory values, yes/no
symptoms and survey answers. Using three datasets with 8, 16 and 21 attributes, one of them with a quarter of a million
records, tests the claim under realistic conditions and also shows how the cost of SHAP grows with the number of
attributes and records.</p>
<p><b>The answer is useful either way.</b> If the rules are faithful, practitioners gain evidence for using them as they
are; if not, the audit shows when and why, and what to change (for example fewer rules or a different rule form).</p>""")

    comp = pd.DataFrame([
        ("ANFIS implemented by us", "The model under test", "Needed to read and differentiate the rules; a library would hide the internals."),
        ("Three datasets (8, 16, 21 attributes)", "Does the answer depend on the data?", "One dataset could give an accidental result; different attribute counts and sizes test generality and SHAP cost."),
        ("Rule attribution (THEN, IF, IF + THEN)", "What do the rules say drove this prediction?", "Puts the rule explanation into the same form as SHAP (one number per attribute) so the two can be compared."),
        ("SHAP on ANFIS", "What really drove the prediction, measured from outside?", "An independent, mathematically grounded reference for the same model and patients."),
        ("LIME on ANFIS", "Do the conclusions depend on choosing SHAP?", "A second, differently built reference; if both agree, the conclusion does not hinge on one method."),
        ("Deletion test", "Which explanation picks attributes that really matter?", "SHAP and LIME are not ground truth; this test needs no reference at all."),
        ("LightGBM + TreeSHAP", "Is ANFIS accurate enough to be worth auditing? Can SHAP scale?", "A strong standard model as an accuracy benchmark; TreeSHAP shows how SHAP can be applied to all 253,680 records."),
        ("Logistic regression", "How non-linear is the data?", "If a linear model is almost as good, few rules are needed; this explains several results."),
        ("Studies A to D", "What changes the faithfulness?", "Changing one factor at a time (rules, attributes, membership, order) isolates its effect."),
        ("Five seeds, Wilcoxon, Friedman, Holm", "Are the differences real?", "Rules out luck from one split and from running many tests."),
        ("Diagnostics (firing concentration, etc.)", "Why does a result occur?", "Turns observations into explanations that are tested, not guessed."),
    ], columns=["Component", "Question it answers", "Why it is needed"])
    add(f"""
<h1>4. Proposed Work</h1>
<h2>4.1 Research questions</h2>
<ol>
<li><b>RQ1.</b> Is the ANFIS trained here as accurate as standard models, so that its explanations are worth auditing?</li>
<li><b>RQ2.</b> How closely does the explanation read from the rules agree with SHAP for individual patients, and which
part of the rules (THEN or IF) carries that agreement?</li>
<li><b>RQ3.</b> How does the agreement change with the number of rules?</li>
<li><b>RQ4.</b> How does it change with the number of attributes, and what does SHAP cost on a large dataset?</li>
<li><b>RQ5.</b> Do the conclusions depend on the membership function, the TSK order or the choice of reference
explainer, and are they confirmed by a test that uses no reference explainer?</li>
</ol>
<h2>4.2 Proposed audit framework</h2>
<p>The idea is to explain the <b>same trained model</b> for the <b>same patients</b> twice: once from the inside, by
reading its rules, and once from the outside, by testing how its output reacts to its inputs. If the rules are
faithful, the two explanations should agree. The framework has six steps, repeated for every dataset, seed and setting:</p>
<ol>
<li>Prepare the data (clean, split 60/20/20, scale using training data only).</li>
<li>Train ANFIS and the baselines; measure accuracy on test patients.</li>
<li>Explain up to {info['n_explain']} test patients from the rules (THEN-part, IF-part, IF + THEN).</li>
<li>Explain the same patients with SHAP and LIME (and LightGBM with TreeSHAP).</li>
<li>Score the agreement between explanations, and run the deletion test on all of them.</li>
<li>Test the differences statistically and measure diagnostics that explain them.</li>
</ol>
<p>Only step 3 is specific to ANFIS, so the framework can audit any TSK-type neuro-fuzzy model.</p>
<h2>4.3 Why each component is needed</h2>
{T(comp, 'Every component of the project, the question it answers and why it is included.')}
<h2>4.4 Expected deliverables</h2>
<ul><li>A measured, statistically tested answer to whether ANFIS rule explanations are faithful on diabetes data.</li>
<li>Evidence on how faithfulness changes with rule count, attribute count and rule form, and why.</li>
<li>Practical guidance on when ANFIS explanations can be trusted.</li>
<li>A reusable pipeline, an interactive dashboard, this report and a research article.</li></ul>""")

    ds = r.table_datasets()
    prep = pd.DataFrame([
        ("1. Mark impossible zeros as missing (Pima)", f"A value of 0 for Glucose, BloodPressure, SkinThickness, Insulin or BMI is set to missing ({int(ds['Missing cells'].iloc[0])} cells).", "A living patient cannot have zero glucose or BMI; treating 0 as a real value would teach the model nonsense."),
        ("2. Encode categories", "Early Stage: Yes/No → 1/0 and gender → Male = 1/0. CDC: the ID column is removed.", "Models need numbers; the ID carries no medical information and would only add noise."),
        ("3. Stratified split", "60% training, 20% validation, 20% test, keeping the share of diabetic patients equal in each part; a new split for every seed.", "Training data fits the model, validation data chooses the stopping point and threshold, and test data is touched only for the final scores."),
        ("4. Fill missing values", "Each missing value is replaced by the median of that attribute in the training part.", "The median is robust to outliers; using training data only prevents test information leaking in."),
        ("5. Standardise", "x<sub>std</sub> = (x − mean) / standard deviation, with mean and standard deviation from the training part.", "Puts all attributes on one scale so that one learning rate and one width range work for all, and makes 0 the average patient (the reference point for the explanations)."),
    ], columns=["Step", "What is done", "Why"])

    j = 1
    cz, sg = e["cz"][0, j], e["sigma"][0, j]
    u = (e["x"][j] - cz) / sg
    add(f"""
<h1>5. Methodology</h1>
<h2>5.1 Datasets</h2>
{T(ds, 'The three public diabetes datasets. Positive rate = share of diabetic patients.')}
<p>The <b>Pima Indians Diabetes</b> data {cite('pima')} contains eight laboratory and body measurements of women of Pima
heritage aged 21 or older. The <b>Early Stage Diabetes Risk</b> data {cite('early')} contains questionnaire answers of
patients of Sylhet Diabetes Hospital: age, gender and fourteen yes/no symptoms. The <b>CDC Diabetes Health
Indicators</b> data {cite('cdc')} contains 21 answers from the 2015 Behavioral Risk Factor Surveillance System, a US
telephone health survey. The full attribute lists are in Appendix C.</p>
<p><b>Why these three.</b> They differ in exactly the ways that matter for this study: the number of attributes (8, 16,
21), the size (768 to 253,680 records), the type of attributes (continuous measurements, yes/no symptoms, mixed survey
answers) and the share of diabetic patients (14% to 62%). The largest one also tests whether SHAP is affordable on big
data.</p>
<h2>5.2 Data preparation</h2>
{T(prep, 'Data preparation steps (implemented in src/data.py).')}
<h2>5.3 Fuzzy sets and membership functions</h2>
<p>In ordinary logic a glucose value is either "high" or not. In fuzzy logic it belongs to "high" to a degree between 0
and 1, called its membership. ANFIS describes each rule's IF part with one membership function per attribute. We use
two shapes. With u = (x − c) / σ the distance of the value x from the centre c, measured in widths σ:</p>
{eq(1, "Gaussian: &nbsp; μ(x) = exp(−u² / 2)")}
{eq(2, "Generalised bell: &nbsp; μ(x) = 1 / (1 + (u²)<sup>b</sup>) = 1 / (1 + u⁴) &nbsp; with b = 2")}
<p>Both equal 1 at the centre and fall towards 0 away from it. The centre c says which value the rule is about, and the
width σ says how strict the rule is: a narrow width means the membership drops quickly, so the rule only accepts values
close to its centre. The bell curve has a flatter top and longer tails than the Gaussian, so rules built from it fire
more softly and share the vote more (this matters in Section 8.6). Example from the trained Pima model: Rule 1 has its
Glucose centre at {e['c'][0, j]:.1f} mg/dL ({cz:+.2f} in standardised units) with width {sg:.2f}. Our worked-example
patient has Glucose {e['orig'][j]:.0f} ({e['x'][j]:+.2f} standardised), so u = ({e['x'][j]:+.2f} − ({cz:+.2f})) / {sg:.2f}
= {u:+.2f} and μ = exp(−{u:.2f}²/2) = {e['mu'][0, j]:.3f}: the patient's glucose fits Rule 1 fairly well.</p>
<p><b>Why Gaussian as the default.</b> It is smooth everywhere (needed for gradient training), has only two parameters
per attribute, and is the most common choice in ANFIS work. The bell function is included to test whether the
conclusions depend on the shape.</p>""")

    sym = pd.DataFrame([
        ("x = (x<sub>1</sub>, …, x<sub>d</sub>)", "One patient's standardised attribute values", "d numbers", "From the data"),
        ("d", "Number of attributes", "8, 16 or 21", "From the data"),
        ("R", "Number of rules", "2 to 16", "Chosen by us (Section 5.6)"),
        ("c<sub>rj</sub>", "Centre of rule r's membership function for attribute j", "R × d", "k-means, then trained"),
        ("σ<sub>rj</sub>", "Width of that membership function (stored as log σ so it stays positive)", "R × d", "Cluster spread, then trained"),
        ("μ<sub>rj</sub>", "Membership of attribute j in rule r", "R × d per patient", "Computed, equation (1) or (2)"),
        ("w<sub>r</sub>", "Firing strength of rule r", "R per patient", "Computed, equation (3)"),
        ("w̄<sub>r</sub>", "Normalised firing strength (vote share)", "R per patient, sums to 1", "Computed, equation (4)"),
        ("a<sub>r</sub> = (a<sub>r1</sub>, …, a<sub>rd</sub>)", "Coefficients of rule r's THEN formula", "R × d", "Least squares, then trained"),
        ("b<sub>r</sub>", "Constant of rule r's THEN formula", "R", "Least squares, then trained"),
        ("f<sub>r</sub>", "Output (score) of rule r", "R per patient", "Computed, equation (5)"),
        ("z", "Model output in log-odds", "1 per patient", "Computed, equation (6)"),
        ("p", "Predicted probability of diabetes", "1 per patient", "Computed, equation (6)"),
    ], columns=["Symbol", "Meaning", "Size", "Where it comes from"])
    add(f"""
<h2>5.4 The ANFIS model, layer by layer</h2>
<p>ANFIS is a network of five layers. Each layer performs one step of fuzzy reasoning. For a patient x with d
attributes and a model with R rules:</p>
<p><b>Layer 1: membership.</b> Each rule r has its own membership function for every attribute j, with centre
c<sub>rj</sub> and width σ<sub>rj</sub>, giving μ<sub>rj</sub>(x<sub>j</sub>) from equation (1) or (2). <i>Role:</i>
decides how well each attribute value fits each rule.</p>
<p><b>Layer 2: firing strength (fuzzy AND).</b> The memberships of one rule are combined into one number:</p>
{eq(3, "w<sub>r</sub>(x) = ( μ<sub>r1</sub> × μ<sub>r2</sub> × … × μ<sub>rd</sub> )<sup>1/d</sup> = exp( (1/d) Σ<sub>j</sub> log μ<sub>rj</sub> )")}
<p><i>Role:</i> a rule fires strongly only if the patient fits it on all attributes together, which is what AND means.
Classic ANFIS uses the plain product; the d-th root (geometric mean) is the HTSK fix explained in Section 5.5.</p>
<p><b>Layer 3: normalisation.</b></p>
{eq(4, "w̄<sub>r</sub>(x) = w<sub>r</sub>(x) / Σ<sub>k=1..R</sub> w<sub>k</sub>(x)")}
<p><i>Role:</i> turns the firing strengths into vote shares that add up to 1, so the output is a weighted average of the
rules. Because w<sub>r</sub> = exp(log w<sub>r</sub>), equation (4) is exactly a softmax of the values log w<sub>r</sub>;
this is used in Section 5.7.</p>
<p><b>Layer 4: rule outputs (THEN parts).</b></p>
{eq(5, "first order: &nbsp; f<sub>r</sub>(x) = a<sub>r1</sub>x<sub>1</sub> + … + a<sub>rd</sub>x<sub>d</sub> + b<sub>r</sub> &nbsp;&nbsp;&nbsp; zero order: &nbsp; f<sub>r</sub> = b<sub>r</sub>")}
<p><i>Role:</i> each rule gives its own score. In a first-order TSK model the score is a linear formula of the
attributes, so each rule is a small linear model that is valid in its own region; in a zero-order model it is a single
number (a fixed risk level for patients who match the rule).</p>
<p><b>Layer 5: output.</b></p>
{eq(6, "z(x) = Σ<sub>r</sub> w̄<sub>r</sub>(x) · f<sub>r</sub>(x), &nbsp;&nbsp;&nbsp; p = P(diabetic | x) = 1 / (1 + e<sup>−z</sup>)")}
<p><i>Role:</i> the final score is the vote-weighted average of the rule scores, and the sigmoid turns it into a
probability. <b>Why the output is in log-odds.</b> The sigmoid is the standard link for a yes/no outcome (as in logistic
regression), and in log-odds the effects of attributes add up, which is what an attribution (one number per attribute)
needs.</p>
{T(sym, 'Every symbol of the model, its size and where its value comes from.')}""")

    hs = pd.DataFrame({"Rule": [f"Rule {k + 1}" for k in range(4)],
                       "Product w<sub>r</sub>": [f"{v:.2e}" for v in h["prod_one"]],
                       "Vote share with product": [f"{v:.7f}" for v in h["prod_share_one"]],
                       "Geometric mean w<sub>r</sub>": [f"{v:.3f}" for v in h["geo_one"]],
                       "Vote share with geometric mean": [f"{v:.3f}" for v in h["geo_share_one"]]})
    add(f"""
<h2>5.5 Why the geometric mean (HTSK) is needed</h2>
<p>With the plain product, a rule's firing strength is the product of d numbers that are all below 1. With 21
attributes this becomes extremely small, and the rule that happens to be least small takes almost the whole vote after
normalisation. The model then behaves like a hard switch between rules, gradients for the other rules vanish, and
training stalls. This is the "curse of dimensionality" for TSK models described by Cui, Wu and Xu {cite('cui')}. Taking
the d-th root, equation (3), keeps the firing strengths on a usable scale. It does not change which rule fits best
(the order of the rules stays the same), only how sharply the vote is concentrated. Measured on our trained CDC model
(21 attributes, 4 rules) for the first test patient:</p>
{T(hs, 'Firing strengths of one CDC patient with the plain product and with the geometric mean (HTSK).')}
<p>With the product, one rule receives {h['prod_share_one'].max() * 100:.5f}% of the vote; with the geometric mean the vote
is shared ({h['geo_share_one'].max():.2f} for the strongest rule). Over {h['n']:,} CDC test patients the strongest rule's
average share is {h['prod_max_share']:.2f} with the product and {h['geo_max_share']:.2f} with the geometric mean. In the
code, the geometric mean is computed as the average of the log-memberships (<code>.mean(axis=2)</code> in
<code>src/anfis.py</code>), which also avoids numerical underflow.</p>""")

    hp = pd.DataFrame([
        ("Number of rules R", "2, 3, 4, 6, 8, 12, 16 (4 in studies B and D)", "Covers small to large rule bases; 4 is the setting with good accuracy and readability"),
        ("Membership function", "Gaussian (bell with b = 2 in study D)", "Smooth and standard; bell tests the effect of the shape"),
        ("k-means restarts / sample", "5 restarts on up to 20,000 training patients", "Stable cluster centres; the sample keeps CDC fast"),
        ("Initial width", "Cluster standard deviation, clipped to [0.5, 3]", "Starts each rule as wide as its cluster, avoiding extreme widths"),
        ("Width limits during training", "σ kept within [0.15, 8]", "Prevents a rule collapsing to a point or becoming flat"),
        ("Least-squares targets / ridge", "t = +2 (diabetic), −2 (not), ridge 1.0", "+2 log-odds ≈ 88% probability: sensible starting scores; ridge keeps the solution stable"),
        ("Loss", "Binary cross-entropy + (λ/2)‖a‖², λ = 10<sup>−4</sup>", "Correct loss for probabilities; small penalty against large coefficients"),
        ("Optimiser", "Adam, η = 0.01, β<sub>1</sub> = 0.9, β<sub>2</sub> = 0.999, ε = 10<sup>−8</sup>", "Standard defaults; adapts the step size to each parameter"),
        ("Batch size / epochs", "32 / 300 (CDC: 1,024 / 40)", "Small batches for small data; large batches keep CDC (152,208 training records) fast"),
        ("Early stopping patience", "30 epochs (CDC: 6)", "Stops when validation loss no longer improves, avoiding overfitting"),
        ("Decision threshold", "Maximises F1 on validation data", "0.5 is a poor cut-off when classes are unequal (CDC: 14% diabetic)"),
        ("Data split / seeds", "60 / 20 / 20 stratified, seeds 0 to 4", "Separate data for training, tuning and testing; five repeats for stability"),
        ("Explained patients", f"{info['n_explain']} test patients per model (all 154 on Pima, 104 on Early Stage)", "Black-box SHAP is costly; 200 patients give stable averages"),
        ("SHAP background", f"{info['n_background']} random training patients", "Defines the 'average patient'; 100 is the usual compromise of accuracy and cost"),
        ("SHAP algorithm", "Exact for d ≤ 10; permutation with 20(2d + 1) evaluations above", "Exact where affordable (2<sup>8</sup> = 256 coalitions); sampling for 16 and 21 attributes"),
        ("LIME", "1,000 perturbed patients, kernel width 0.75√d, ridge 1.0", "The original LIME defaults for tabular data"),
        ("Deletion test", "k = 1, 2, 3 removed attributes; 20 background patients", "Top-1 to top-3 covers what a reader looks at; 20 replacements average out noise"),
        ("LightGBM", "Up to 2,000 trees, learning rate 0.03, 7 leaves (31 on CDC), early stopping 50", "A well-tuned strong baseline, small trees for small data"),
        ("Logistic regression", "Default L2 penalty, up to 2,000 iterations", "A linear reference model"),
    ], columns=["Setting", "Value", "Reason"])
    add(f"""
<h2>5.6 How the rules are created and trained</h2>
<p>Training finds the values of c, σ, a and b. It has four stages: place the rules, set their widths, solve for good
starting THEN parts, and then fine-tune everything by gradient descent.</p>
<h3>5.6.1 Placing the rules with k-means</h3>
<p>The number of rules R is chosen by us. The rules are then placed by grouping the training patients into R clusters
of similar patients with k-means, which minimises the distance of every patient to the centre of its cluster:</p>
{eq(7, "J = Σ<sub>i</sub> ‖ x<sub>i</sub> − c<sub>k(i)</sub> ‖², &nbsp;&nbsp; k(i) = the cluster closest to patient i")}
<p>Each cluster centre becomes one rule's centres c<sub>r</sub>, and the standard deviation of the cluster in each
attribute becomes the initial width σ<sub>rj</sub> (clipped to [0.5, 3]). <b>Why k-means and not a grid.</b> Classic
ANFIS splits every attribute into a few fuzzy sets and makes one rule for every combination. With two sets per attribute
this gives 2<sup>d</sup> rules: 256 for Pima, 65,536 for Early Stage and 2,097,152 for CDC, which is impossible to train
or read. With one rule per cluster the number of rules is independent of d, so we can study 2 to 16 rules on every
dataset, and each rule starts in a region where real patients are.</p>
<h3>5.6.2 Starting THEN parts by least squares</h3>
<p>Once the IF parts are fixed, the firing strengths w̄<sub>r</sub> are fixed numbers for each patient, and the output
z = Σ<sub>r</sub> w̄<sub>r</sub>(a<sub>r</sub>·x + b<sub>r</sub>) is <i>linear</i> in the unknown a and b. All of them can
therefore be found in one step. Write them as one vector θ, and for each patient a row Φ<sub>i</sub> containing
w̄<sub>r</sub>x<sub>j</sub> for every rule and attribute and w̄<sub>r</sub> for every rule, so that z<sub>i</sub> =
Φ<sub>i</sub>θ. We want z close to a target t<sub>i</sub> = +2 for diabetic and −2 for non-diabetic patients (log-odds
of about 88% and 12%). Minimising the squared error plus a small ridge penalty and setting the derivative to zero gives
the closed form:</p>
{eq(8, "θ* = argmin<sub>θ</sub> ‖Φθ − t‖² + ‖θ‖² &nbsp;&nbsp; ⇒ &nbsp;&nbsp; 2Φᵀ(Φθ − t) + 2θ = 0 &nbsp;&nbsp; ⇒ &nbsp;&nbsp; θ* = (ΦᵀΦ + I)<sup>−1</sup> Φᵀ t")}
<p><b>Why.</b> This is the least-squares half of Jang's hybrid learning {cite('jang')}. It puts the THEN parts close to a
good solution before gradient descent starts, so training is faster and more stable. The ridge term (+I) keeps the
matrix invertible when attributes are correlated.</p>
<h3>5.6.3 The loss function</h3>
<p>The model predicts a probability p<sub>i</sub> for each patient with label y<sub>i</sub> ∈ {{0, 1}}. The probability
the model gives to the true labels is Π p<sub>i</sub><sup>y<sub>i</sub></sup>(1 − p<sub>i</sub>)<sup>1−y<sub>i</sub></sup>.
Maximising it (maximum likelihood) is the same as minimising its negative logarithm divided by n, which is the binary
cross-entropy. A small penalty on the coefficients is added:</p>
{eq(9, "L = −(1/n) Σ<sub>i</sub> [ y<sub>i</sub> log p<sub>i</sub> + (1 − y<sub>i</sub>) log(1 − p<sub>i</sub>) ] + (λ/2) ‖a‖², &nbsp;&nbsp; λ = 10<sup>−4</sup>")}
<p><b>Why.</b> Cross-entropy is the correct loss for probabilities: it punishes confident wrong answers heavily, and
with the sigmoid its gradient is simply (p − y). The penalty discourages very large coefficients (overfitting).</p>
<h3>5.6.4 Gradients</h3>
<p>Training needs the derivative of L with respect to every parameter. We derived them by the chain rule through the
five layers (u<sub>rj</sub> = (x<sub>j</sub> − c<sub>rj</sub>)/σ<sub>rj</sub>):</p>
{eq(10, "∂L/∂z = (p − y)/n, &nbsp; ∂z/∂f<sub>r</sub> = w̄<sub>r</sub>, &nbsp; ∂z/∂a<sub>rj</sub> = w̄<sub>r</sub>x<sub>j</sub>, &nbsp; ∂z/∂log w<sub>r</sub> = w̄<sub>r</sub>(f<sub>r</sub> − z), &nbsp; ∂log w<sub>r</sub>/∂u²<sub>rj</sub> = −1/(2d) (Gaussian) or −2u²/(d(1 + u⁴)) (bell)")}
<p>and from u² = (x − c)²/σ²: ∂u²/∂c = −2(x − c)/σ² and ∂u²/∂log σ = −2u². The term w̄<sub>r</sub>(f<sub>r</sub> − z) is
derived in Section 5.7, because the same quantity gives the IF-part explanation. <b>Why derive them by hand.</b>
Implementing ANFIS ourselves lets us read every internal quantity, which the explanation needs. The gradients were
checked against numerical differences (Section 7.4).</p>
<h3>5.6.5 Parameter update (Adam) and early stopping</h3>
<p>All parameters are updated on small batches of patients with Adam {cite('adam')}, with g the gradient and t the step
number:</p>
{eq(11, "m ← β<sub>1</sub>m + (1 − β<sub>1</sub>)g, &nbsp; v ← β<sub>2</sub>v + (1 − β<sub>2</sub>)g², &nbsp; θ ← θ − η · m̂ / (√v̂ + ε), &nbsp; m̂ = m/(1 − β<sub>1</sub><sup>t</sup>), v̂ = v/(1 − β<sub>2</sub><sup>t</sup>)")}
<p>m is a running average of the gradient (momentum) and v of its square, so each parameter gets its own step size.
<b>Why Adam.</b> Centres, widths and coefficients have very different gradient sizes; plain gradient descent would need
a separate learning rate for each. After every epoch the loss on the validation data is computed; the parameters with
the lowest validation loss are kept, and training stops when it has not improved for a set number of epochs:</p>
{eq(12, "θ̂ = argmin over epochs e of L<sub>val</sub>(θ<sup>(e)</sup>)")}
<h3>5.6.6 Decision threshold and accuracy measures</h3>
<p>A probability is turned into a yes/no prediction with a threshold τ chosen on validation data to maximise F1, then
applied unchanged to the test data. With TP, FP, TN, FN the counts of true/false positives/negatives, precision P =
TP/(TP + FP) and recall (true positive rate) TPR = TP/(TP + FN):</p>
{eq(13, "τ* = argmax<sub>τ</sub> F1<sub>val</sub>(τ), &nbsp; F1 = 2PR/(P + R), &nbsp; accuracy = (TP + TN)/n, &nbsp; balanced accuracy = (TPR + TNR)/2")}
{eq(14, "ROC-AUC = P( p(diabetic patient) > p(non-diabetic patient) ) = area under the curve of TPR against FPR over all thresholds")}
<p><b>Why ROC-AUC is the main measure.</b> It needs no threshold and is not fooled by unequal classes: on CDC a model that
always says "not diabetic" has 86% accuracy but ROC-AUC 0.5. Balanced accuracy and F1 are reported as well.</p>
{T(hp, 'All settings used in the project, with their values and the reason for each.')}""")

    names = feats
    x_tab = pd.DataFrame([[f"{v:g}" for v in np.round(e["orig"], 2)], [f"{v:+.2f}" for v in e["x"]]], columns=names,
                         index=["Value", "Standardised x<sub>j</sub>"])
    mu_tab = pd.DataFrame(np.round(e["mu"], 3), columns=names, index=[f"Rule {k + 1}" for k in range(4)])
    mu_tab["w<sub>r</sub>"] = np.round(e["w"], 3)
    mu_tab["w̄<sub>r</sub>"] = np.round(e["wbar"], 3)
    mu_tab["f<sub>r</sub>"] = np.round(e["f"], 3)
    mu_tab["w̄<sub>r</sub>f<sub>r</sub>"] = np.round(e["wbar"] * e["f"], 3)
    att = pd.DataFrame({"Attribute": names, "THEN-part": np.round(e["then"], 3), "IF-part": np.round(e["if"], 3),
                        "IF + THEN": np.round(e["full"], 3), "SHAP": np.round(e["shap"], 3), "LIME": np.round(e["lime"], 3)})
    att.loc[len(att)] = ["<b>Sum</b>", f"<b>{e['then'].sum():.3f}</b>", f"<b>{e['if'].sum():.3f}</b>",
                         f"<b>{e['full'].sum():.3f}</b>", f"<b>{e['shap'].sum():.3f}</b>", f"<b>{e['lime'].sum():.3f}</b>"]
    met = pd.DataFrame([{"Explanation (vs SHAP)": k, "Spearman": f"{v['spearman_local']:.3f}", "Kendall": f"{v['kendall_local']:.3f}",
                         "Same top attribute": "yes" if v["top1_match"] else "no", "Top-3 overlap": f"{v['top3_overlap']:.2f}",
                         "Sign agreement": f"{v['sign_agreement']:.2f}"} for k, v in e["metrics"].items()])
    gl = 1
    contrib_then = e["wbar"] * e["a"][:, gl] * e["x"][gl]
    bias_part = float((e["wbar"] * e["b"]).sum())
    add(f"""
<h2>5.7 Turning the rules into an explanation</h2>
<p>This section is the core of the project. SHAP and LIME give an explanation as one number per attribute (an
attribution φ<sub>j</sub>). To compare the rules with them, the rules must give their explanation in the same form. We
therefore derive, from the model's own parameters, how much each attribute contributed to the output z of one patient.
All attributions are measured relative to a reference patient x̄, the average training patient; after standardisation
x̄ = 0, so a contribution says how much the attribute moved this patient away from the average patient.</p>
<h3>5.7.1 The two ways an attribute affects ANFIS</h3>
<p>Look again at equation (6): z(x) = Σ<sub>r</sub> w̄<sub>r</sub>(x) · f<sub>r</sub>(x). An attribute x<sub>j</sub>
appears in <b>both</b> factors:</p>
<ul>
<li><b>Inside the THEN formulas</b> f<sub>r</sub>(x) = a<sub>r</sub>·x + b<sub>r</sub>: a higher glucose raises the score
of every rule with a positive glucose coefficient. This is what a person reads from the fired rules.</li>
<li><b>Inside the vote shares</b> w̄<sub>r</sub>(x): a different glucose value changes how well the patient fits each
rule's IF part, so the vote moves from one rule to another (rule switching). This is not visible from the THEN
formulas.</li>
</ul>
<p>By the product rule, a small change in the inputs changes the output through these two routes:</p>
{eq(15, "dz = Σ<sub>r</sub> w̄<sub>r</sub> · df<sub>r</sub> &nbsp;+&nbsp; Σ<sub>r</sub> f<sub>r</sub> · dw̄<sub>r</sub> &nbsp;&nbsp;&nbsp; (THEN route + IF route)")}
<p>The two attributions below measure these two routes.</p>
<h3>5.7.2 The THEN-part attribution</h3>
<p>Put equation (5) into equation (6) and reorder the sum by attribute:</p>
{eq("16a", "z = Σ<sub>r</sub> w̄<sub>r</sub> ( Σ<sub>j</sub> a<sub>rj</sub>x<sub>j</sub> + b<sub>r</sub> ) = Σ<sub>j</sub> ( Σ<sub>r</sub> w̄<sub>r</sub> a<sub>rj</sub> ) x<sub>j</sub> + Σ<sub>r</sub> w̄<sub>r</sub> b<sub>r</sub>")}
<p>The output is a sum with one term per attribute plus a constant. The term of attribute j is its THEN-part
attribution (with x̄ = 0, x<sub>j</sub> − x̄<sub>j</sub> = x<sub>j</sub>):</p>
{eq(16, "φ<sup>THEN</sup><sub>j</sub>(x) = Σ<sub>r</sub> w̄<sub>r</sub>(x) · a<sub>rj</sub> · (x<sub>j</sub> − x̄<sub>j</sub>)")}
{eq(17, "Σ<sub>j</sub> φ<sup>THEN</sup><sub>j</sub> + Σ<sub>r</sub> w̄<sub>r</sub> b<sub>r</sub> = z(x) &nbsp;&nbsp; (exact)")}
<p><b>Meaning.</b> Vote share of each rule × that rule's coefficient for the attribute × how far the attribute is from
average, added over the rules. This is precisely what a person does when reading the fired rules: "these rules fired,
and in their formulas glucose has a large positive weight, and this patient's glucose is above average". Equation (17)
shows that it is exact: the THEN-part attributions plus the vote-weighted constants add up to the model output. It
holds the vote shares fixed at their values for this patient, so it measures the THEN route of equation (15) only.</p>
<h3>5.7.3 The IF-part attribution</h3>
<p>The IF route needs the derivative of the output with respect to the firing strengths. Write s<sub>r</sub> = log
w<sub>r</sub>. Equation (4) is a softmax of s, and the derivative of a softmax is ∂w̄<sub>k</sub>/∂s<sub>r</sub> =
w̄<sub>k</sub>(δ<sub>kr</sub> − w̄<sub>r</sub>), where δ<sub>kr</sub> is 1 if k = r and 0 otherwise. Therefore</p>
{eq(18, "∂z/∂s<sub>r</sub> = Σ<sub>k</sub> f<sub>k</sub> · w̄<sub>k</sub>(δ<sub>kr</sub> − w̄<sub>r</sub>) = w̄<sub>r</sub>f<sub>r</sub> − w̄<sub>r</sub> Σ<sub>k</sub> w̄<sub>k</sub>f<sub>k</sub> = w̄<sub>r</sub> ( f<sub>r</sub> − z )")}
<p>This says: increasing rule r's firing raises the output if the rule's score is above the current output
(f<sub>r</sub> &gt; z) and lowers it otherwise, in proportion to the rule's vote share. From equation (3), s<sub>r</sub> =
Σ<sub>j</sub> ℓ<sub>rj</sub> with ℓ<sub>rj</sub> = (1/d) log μ<sub>rj</sub>, so each attribute adds its own piece to
s<sub>r</sub>. Attribute j changes its piece from ℓ<sub>rj</sub>(x̄<sub>j</sub>) at the average patient to
ℓ<sub>rj</sub>(x<sub>j</sub>) for this patient. Multiplying the change by the sensitivity (18) and adding over the rules
gives the first-order (Taylor) estimate of how much attribute j changed the output through rule switching:</p>
{eq(19, "φ<sup>IF</sup><sub>j</sub>(x) = Σ<sub>r</sub> w̄<sub>r</sub>(x) · ( f<sub>r</sub>(x) − z(x) ) · ( ℓ<sub>rj</sub>(x<sub>j</sub>) − ℓ<sub>rj</sub>(x̄<sub>j</sub>) )")}
<p><b>Meaning.</b> An attribute gets IF-part credit when it moves the patient towards rules whose score differs from the
current output. It is an approximation (first order), because the softmax is not linear. The combined rule explanation
is</p>
{eq(20, "φ<sup>IF+THEN</sup><sub>j</sub> = φ<sup>THEN</sup><sub>j</sub> + φ<sup>IF</sup><sub>j</sub>")}
<h3>5.7.4 Why the IF part and the THEN part are kept separate</h3>
<ol>
<li><b>They answer different questions.</b> The THEN part answers "what do the rules that fired say about each
attribute?"; the IF part answers "how did each attribute decide which rules fired?". A reader of ANFIS rules naturally
does the first; the second is hidden.</li>
<li><b>The THEN part is the claim being tested.</b> The usual interpretability claim of ANFIS is that the fired rules and
their formulas explain the decision. Equation (16) is exactly that reading, so it must be measured on its own.</li>
<li><b>They are different in nature.</b> The THEN part is exact (equation 17); the IF part is a first-order
approximation. Mixing them would hide which part causes an error.</li>
<li><b>Separating them diagnoses the cause.</b> If the THEN part agrees well and the IF part poorly, the rule formulas
carry the explanation; if the IF part becomes more important, rule switching is driving the model. Chapter 8 uses exactly
this to explain why more rules make the explanation worse.</li>
<li><b>Zero-order models only have an IF part.</b> Their THEN part is a constant, so φ<sup>THEN</sup> = 0 and the rules
can explain only through which rules fire. Keeping the parts separate lets first- and zero-order models be compared on
equal terms.</li>
</ol>
<h3>5.7.5 Why they are also combined</h3>
<p>The real model uses both routes at once (equation 15), so the most complete explanation that can be read from the
rules is their sum, equation (20). Comparing IF + THEN with THEN alone tests whether rule switching matters enough to
change the explanation: if adding the IF part brings the rules closer to SHAP, switching is important for that dataset.
In our results this happens significantly only on CDC (Section 8.2).</p>
<h3>5.7.6 Worked example with a real patient</h3>
<p>The numbers below are computed by the project code for test patient 0 of the Pima data (seed 0, Gaussian
memberships, first order, 4 rules); the same patient is shown in the dashboard in Section 7.5. The patient is diabetic.</p>
{T(x_tab, 'The patient: original values and standardised values (0 = average training patient).', index=True)}
{T(mu_tab, 'Layers 1 to 5 for this patient: membership of each attribute in each rule, firing strength w (geometric mean of the row), vote share w̄, rule output f and its weighted contribution.', index=True)}
<p><b>Reading the table.</b> Rule 1 fits the patient best (w = {e['w'][0]:.3f}) because seven of its eight memberships are
above 0.74; Rule 3 fits worst because the patient's Pregnancies ({e['mu'][2, 0]:.3f}) and BMI ({e['mu'][2, 5]:.3f}) are
far from its centres. After normalisation the vote shares are {', '.join(f'{v:.3f}' for v in e['wbar'])}. The rule
outputs are {', '.join(f'{v:+.3f}' for v in e['f'])}: only Rule 2 votes towards diabetic. The output is z =
{' + '.join(f'({v:+.3f})' for v in e['wbar'] * e['f'])} = {e['z']:+.3f}, so p = 1/(1 + e<sup>−z</sup>) =
{e['p'] * 100:.1f}%. (LightGBM gives {e['p_lgbm'] * 100:.1f}% for the same patient.)</p>
<p><b>THEN-part for Glucose</b> (x = {e['x'][gl]:+.2f}), from equation (16): {' + '.join(f'{wv:.3f}×{av:.2f}×({e["x"][gl]:+.2f})' for wv, av in zip(e['wbar'], e['a'][:, gl]))}
= {' + '.join(f'({v:+.3f})' for v in contrib_then)} = {contrib_then.sum():+.3f}. Glucose is below average and every rule
has a positive glucose coefficient, so it pushes the score down.</p>
{T(att, 'All explanations of this patient (log-odds). Positive values push towards diabetic.')}
<p><b>Checks.</b> The THEN-part sums to {e['then'].sum():+.3f}, and adding the vote-weighted constants
Σw̄<sub>r</sub>b<sub>r</sub> = {bias_part:+.3f} gives {e['then'].sum() + bias_part:+.3f} = z, as equation (17) says. The
SHAP values sum to {e['shap'].sum():+.3f}, which is z minus the average output over the background patients
({e['z']:+.3f} − ({e['base_z']:+.3f})), as SHAP's efficiency property (equation 22) requires. All four rule-based and
post-hoc explanations name Glucose, BMI and DiabetesPedigree as the three most important attributes, with Glucose
pushing down and BMI and DiabetesPedigree pushing up.</p>
{T(met, 'Agreement of each explanation with SHAP for this patient (definitions in Section 5.11).')}
<p>For this patient the rules agree very well with SHAP. The IF-part alone ranks Pregnancies highest, because the
patient's 7 pregnancies make Rule 3 (centred on about 2 pregnancies, narrow width) fire weakly; its signs disagree with
SHAP, showing why the IF part alone is not a good explanation for a first-order model.</p>
<h3>5.7.7 A patient where the rules and SHAP disagree</h3>
<p>Agreement is not always this high. For Pima test patient 25 the THEN-part has Spearman {dis_then:.2f} with SHAP
(IF + THEN {dis_full:.2f}). The vote is spread over all four rules ({', '.join(f'{v:.2f}' for v in dis_w)}).</p>
{T(dis_t, 'Pima test patient 25, sorted by SHAP importance. Rank 1 = most important.')}
<p>Both name DiabetesPedigree first, but the THEN-part ranks Glucose second and Insulin third, while SHAP ranks them
fifth and eighth and puts Age second. With the vote spread over four rules, moving Age or Pregnancies shifts the patient
between rules whose scores differ, an effect that SHAP measures and the THEN-part reading does not. This is the
mechanism behind the partial agreement found in Chapter 8.</p>""")

    add(f"""
<h2>5.8 SHAP: the first independent explanation</h2>
<p><b>Idea.</b> SHAP treats the attributes as players in a team game whose payout is the model output. The fair share
of each player, defined by Shapley {cite('shapley')}, is its extra contribution averaged over every possible order in
which the players could join the team. For attribute j, with F the set of all d attributes and S a coalition not
containing j:</p>
{eq(21, "φ<sup>SHAP</sup><sub>j</sub>(x) = Σ<sub>S ⊆ F∖{j}</sub> [ |S|! (d − |S| − 1)! / d! ] · ( v(S ∪ {j}) − v(S) )")}
<p>The bracket is the fraction of all d! joining orders in which exactly the attributes of S come before j; v(S ∪ {{j}})
− v(S) is what j adds when it joins S. The value of a coalition is the model output when the attributes in S keep the
patient's values and the others are replaced by values of real background patients:</p>
{eq("21a", "v(S) = average over background patients x' of z( x<sub>S</sub>, x'<sub>not S</sub> )")}
{eq(22, "Σ<sub>j</sub> φ<sup>SHAP</sup><sub>j</sub>(x) = z(x) − average background output &nbsp;&nbsp; (efficiency)")}
<p><b>Why SHAP is the reference.</b> It is the only attribution with the four fairness properties of Section 2.2, it uses
nothing but the model's input-output behaviour (so it sees both routes of equation 15), and it is the most widely used
explanation method. <b>Settings.</b> {info['n_background']} random training patients form the background. For d ≤ 10
(Pima) all 2<sup>d</sup> coalitions are evaluated, giving exact values; for 16 and 21 attributes the values are
estimated by sampling random joining orders with 20(2d + 1) model evaluations per patient (permutation SHAP). <b>Cost.</b>
Exact SHAP needs 2<sup>d</sup> coalitions times the background size: 2<sup>21</sup> = 2,097,152 coalitions for CDC, which
is why it is applied to a sample of patients. For LightGBM, TreeSHAP {cite('treeshap')} computes the same values exactly
in polynomial time by following the paths of the trees.</p>
<h2>5.9 LIME: the second independent explanation</h2>
<p>LIME {cite('lime')} explains one patient by imitating the model near that patient with a straight-line model. We
implemented it ourselves from its definition:</p>
<ol><li>Create 1,000 artificial patients x<sup>(m)</sup> by taking each attribute from a random background patient.</li>
<li>Ask ANFIS for the output z(x<sup>(m)</sup>) of each one.</li>
<li>Give each artificial patient a weight that is larger the closer it is to the real patient (kernel π).</li>
<li>Fit a weighted ridge regression of z on the attributes; its slopes β<sub>j</sub> say how z changes with each
attribute near the patient.</li>
<li>The contribution of attribute j is its slope times how far the patient is from the background mean m̄.</li></ol>
{eq(23, "β = argmin<sub>β</sub> Σ<sub>m</sub> π<sub>m</sub> ( z(x<sup>(m)</sup>) − β<sub>0</sub> − β·(x<sup>(m)</sup> − m̄) )² + ‖β‖², &nbsp; π<sub>m</sub> = exp(−‖x − x<sup>(m)</sup>‖² / κ²), &nbsp; κ = 0.75√d, &nbsp; φ<sup>LIME</sup><sub>j</sub> = β<sub>j</sub>(x<sub>j</sub> − m̄<sub>j</sub>)")}
<p><b>Why LIME as well.</b> It works differently from SHAP (local regression instead of game theory). If the rules
disagree with SHAP and LIME in the same way, the conclusion does not depend on one particular reference. The kernel
width 0.75√d is LIME's default for tabular data; the ridge term keeps the fit stable.</p>
<h2>5.10 Deletion test</h2>
<p>Agreement with SHAP shows only that two explanations are similar, not which one is right. The deletion test needs no
reference. For each explanation φ and each patient: take the k attributes it ranks highest, replace their values with
values of 20 background patients, run ANFIS again, and measure how much the output changed:</p>
{eq(24, "Δ<sub>k</sub>(φ) = (1/N) Σ<sub>i</sub> | z(x<sub>i</sub>) − average over x' of z( x<sub>i</sub> with its top-k attributes under φ taken from x' ) |")}
<p>A faithful explanation names the attributes the model really depends on, so removing them changes the output a lot.
A random ranking is the lower control: an explanation must beat it to be useful. We use k = 1, 2, 3. Removing by
background values is SHAP's own convention; we therefore also report removal by the training mean (the reference x̄ of
the rule attributions), so neither method is favoured by the choice of convention.</p>""")

    metrics = pd.DataFrame([
        ("Spearman ρ", "ρ = 1 − 6Σd<sub>j</sub>² / (d(d² − 1)) on the ranks of |φ|", "−1 to 1", "Do the two explanations order the attributes the same way?", "Main measure; uses the whole ranking"),
        ("Kendall τ", "(concordant − discordant pairs) / all pairs", "−1 to 1", "Same, counted pair by pair", "More robust with few attributes"),
        ("Weighted Kendall τ", "Kendall τ with more weight on top-ranked attributes {}".format(cite('vigna')), "−1 to 1", "Do they agree on the most important attributes?", "A reader looks at the top first"),
        ("Top-1 match", "1 if both name the same most important attribute", "0 or 1", "Same main reason?", "The simplest question a doctor asks"),
        ("Top-3 overlap", "|T<sub>3</sub>(A) ∩ T<sub>3</sub>(B)| / 3", "0 to 1", "Same three main reasons?", "Ignores order, focuses on the set"),
        ("Sign agreement", "Share of SHAP's top-3 attributes where both have the same sign", "0 to 1", "Same direction (towards or away from diabetic)?", "Rank measures use |φ| and ignore direction"),
        ("Global Spearman", "Spearman between mean |φ| over all patients", "−1 to 1", "Same overall importance for the dataset?", "Separates patient-level from dataset-level agreement"),
    ], columns=["Metric", "Formula", "Range", "Question", "Why it is used"])
    add(f"""
<h2>5.11 Agreement metrics</h2>
<p>For two explanations A and B of the same patient we compute the measures below, then average over patients
(T<sub>k</sub>(·) is the set of the k attributes with largest |φ|, d<sub>j</sub> the difference in rank of attribute j):</p>
{eq(25, "ρ = 1 − 6 Σ<sub>j</sub> d<sub>j</sub>² / (d(d² − 1)), &nbsp;&nbsp; top-k overlap = |T<sub>k</sub>(A) ∩ T<sub>k</sub>(B)| / k")}
{eq(26, "sign agreement = (1/k) Σ<sub>j ∈ T<sub>k</sub>(B)</sub> 1[ sign(A<sub>j</sub>) = sign(B<sub>j</sub>) ]")}
{T(metrics, 'The seven agreement metrics.')}
<p>Rank measures use absolute values because importance and direction are separate questions; direction is measured by
the sign agreement. Using several measures avoids conclusions that depend on one definition.</p>""")

    studies = pd.DataFrame([
        ("A. Rule count", f"R ∈ {{{', '.join(map(str, r.rules))}}}", "All attributes, Gaussian, first order", "Does a bigger rule base explain better or worse? (RQ3)"),
        ("B. Attribute count", f"Top-k attributes by mutual information: Pima {r.subs['pima']}, Early Stage {r.subs['early']}, CDC {r.subs['cdc']}", "R = 4", "Does the number of inputs change faithfulness? (RQ4)"),
        ("C. SHAP cost", "Number of attributes; black-box SHAP, TreeSHAP, LIME", "R = 4", "How expensive is each explainer? (RQ4)"),
        ("D. Model variants", "Gaussian / bell × first / zero order", "R = 4, all attributes", "Does the rule form matter? (RQ5)"),
    ], columns=["Study", "What varies", "What is fixed", "Question"])
    add(f"""
<h2>5.12 Experimental design</h2>
{T(studies, 'The four studies. Every setting is run for 5 seeds.')}
<p><b>Why one factor at a time.</b> Each study changes one thing and keeps everything else fixed, so that any change in
agreement can be attributed to that factor. Studies B and D use 4 rules because Study A shows that 4 rules give the same
accuracy as larger rule bases with readable rules, and repeating every variant at all seven rule counts would multiply the
run time without answering a new question.</p>
<p><b>Mutual information</b> (study B) measures how much knowing an attribute reduces uncertainty about the label. It is
estimated on training data with the k-nearest-neighbour method {cite('kraskov')}:</p>
{eq(27, "I(X<sub>j</sub>; Y) = Σ p(x, y) log [ p(x, y) / (p(x) p(y)) ]")}
<p>It is used because it also detects non-linear relations, unlike correlation. The four most informative attributes
(seed 0) are: Pima {mi_top['pima']}; Early Stage {mi_top['early']}; CDC {mi_top['cdc']}.</p>
<h2>5.13 Statistical analysis</h2>
<p><b>Wilcoxon signed-rank test</b> {cite('wilcoxon')}. To compare two settings A and B, each patient is compared with
itself: the difference in agreement under A and under B is computed for every (seed, patient) pair; the differences are
ranked by size, and W<sup>+</sup> and W<sup>−</sup> are the sums of ranks of the positive and negative differences. If A
and B were equally good, W<sup>+</sup> and W<sup>−</sup> would be similar. <b>Why this test.</b> It is paired (removes
differences between patients) and does not assume normally distributed values, which correlation values are not.</p>
<p><b>Friedman test</b> {cite('friedman')}. The rank-based test for more than two related settings, used to test whether
agreement differs across all seven rule counts at once.</p>
<p><b>Effect size.</b> A p-value says whether a difference is real, not how large it is. The matched-pairs
rank-biserial correlation r gives the size, from −1 (B always better) to +1 (A always better):</p>
{eq(28, "r = (W<sup>+</sup> − W<sup>−</sup>) / (W<sup>+</sup> + W<sup>−</sup>)")}
<p><b>Holm correction</b> {cite('holm')}. We run {len(r.sig)} tests. At the 5% level, the chance that at least one of
{len(r.sig)} true "no difference" cases would look significant by luck is 1 − 0.95<sup>{len(r.sig)}</sup> ≈
{(1 - 0.95 ** len(r.sig)) * 100:.0f}%. Holm's method removes this risk: with the p-values sorted p<sub>(1)</sub> ≤ … ≤
p<sub>(m)</sub>,</p>
{eq(29, "p̃<sub>(i)</sub> = max<sub>j ≤ i</sub> min(1, (m − j + 1) · p<sub>(j)</sub>)")}
<p>and a difference is called significant only if p̃ &lt; 0.05.</p>""")

    allf = pd.DataFrame([
        ("1, 2", "Gaussian / bell membership", "Layer 1: how well a value fits a rule", "anfis.py: log_membership"),
        ("3", "Firing strength (geometric mean)", "Layer 2: fuzzy AND, HTSK", "anfis.py: forward"),
        ("4", "Normalisation (softmax)", "Layer 3: vote shares", "anfis.py: forward"),
        ("5", "Rule output", "Layer 4: THEN parts", "anfis.py: forward"),
        ("6", "Output and sigmoid", "Layer 5: prediction", "anfis.py: forward, predict_proba"),
        ("7", "k-means objective", "Places the rules", "anfis.py: _init_params"),
        ("8", "Ridge least squares", "Starting THEN parts", "anfis.py: _init_params"),
        ("9", "Cross-entropy + L2", "Training objective", "anfis.py: _bce, loss_and_grads"),
        ("10", "Gradients", "Training direction", "anfis.py: loss_and_grads"),
        ("11, 12", "Adam, early stopping", "Parameter update, stopping", "anfis.py: fit"),
        ("13, 14", "Threshold, F1, accuracy, ROC-AUC", "Accuracy measures", "run_experiments.py: score"),
        ("15 to 20", "Two routes, THEN-part, IF-part, IF + THEN", "Explanation from the rules", "attribution.py: rule_attribution"),
        ("21, 22", "SHAP", "First reference", "attribution.py: shap_blackbox, shap_tree"),
        ("23", "LIME", "Second reference", "attribution.py: lime_local"),
        ("24", "Deletion score", "Reference-free check", "attribution.py: deletion_test"),
        ("25, 26", "Agreement metrics", "Comparing explanations", "consistency.py: compare"),
        ("27", "Mutual information", "Attribute ranking (study B)", "run_experiments.py: run_one"),
        ("28, 29", "Rank-biserial r, Holm", "Statistics", "analyze.py"),
    ], columns=["Equation", "Name", "Purpose", "Where in the code"])
    add(f"""
<h2>5.14 Summary of all formulas</h2>
{T(allf, 'Every formula used in the project, its purpose and where it is implemented.')}""")

    nodes = pd.DataFrame([
        ("1. Membership", "R × d", "μ<sub>rj</sub> = exp(−u²/2)", "c<sub>rj</sub>, σ<sub>rj</sub>", "Yes"),
        ("2. Firing (AND)", "R", "w<sub>r</sub> = (Π<sub>j</sub> μ<sub>rj</sub>)<sup>1/d</sup>", "none", "No"),
        ("3. Normalisation", "R", "w̄<sub>r</sub> = w<sub>r</sub> / Σw<sub>k</sub>", "none", "No"),
        ("4. Consequent (THEN)", "R", "w̄<sub>r</sub> f<sub>r</sub>, f<sub>r</sub> = a<sub>r</sub>·x + b<sub>r</sub>", "a<sub>r</sub>, b<sub>r</sub>", "Yes"),
        ("5. Output", "1", "z = Σ w̄<sub>r</sub>f<sub>r</sub>, p = σ(z)", "none", "No"),
    ], columns=["Layer", "Nodes", "Node function", "Parameters", "Adaptive"])
    pc = pd.DataFrame([{"Dataset": nm, "d": d, "Premise 2Rd": 2 * 4 * d, "Consequent R(d + 1)": 4 * (d + 1),
                        "Total (R = 4)": 2 * 4 * d + 4 * (d + 1), "Total (R = 16)": 2 * 16 * d + 16 * (d + 1),
                        "Grid rules (2 sets)": f"{2 ** d:,}"} for nm, d in [("Pima", 8), ("Early Stage", 16), ("CDC", 21)]])
    layers = pd.DataFrame([
        ("1 Data", "Three CSV files", "Loading, typing", "Attribute table and label"),
        ("2 Preparation", "Attribute table", "Cleaning, stratified split, train-only imputation and scaling", "Train / validation / test arrays"),
        ("3 Models", "Training and validation arrays", "ANFIS (variants, rule counts), LightGBM, logistic regression", "Trained models, test scores"),
        ("4 Explanations", "Models, explained patients, background", "Rule attribution, SHAP, LIME, TreeSHAP", "Attribution matrices"),
        ("5 Evaluation", "Attribution matrices", "Agreement metrics, deletion test, statistics, diagnostics", "Result tables"),
        ("6 Outputs", "Result tables, saved models", "Figures, dashboard, report, article", "Deliverables"),
    ], columns=["Layer", "Input", "Processing", "Output"])
    add(f"""
<h1>6. Model / Architecture / Design</h1>
<h2>6.1 System architecture</h2>
<p>Figure {n['f'] + 1} shows the block diagram of the whole system. Data flows from left to right through six layers.
The experiment controller (<code>run_experiments.py</code>) drives the data, model, explanation and evaluation layers for
every dataset, seed and study, and writes all numbers to result tables, from which the figures, the dashboard and this
report are produced.</p>
{F(FIG / 'arch_system.png', 'Block diagram of the audit system, with the source file that implements each layer.')}
{T(layers, 'What each layer of the system does.')}
<h2>6.2 ANFIS network architecture</h2>
<p>Figure {n['f'] + 1} shows the ANFIS network as implemented. Unlike grid ANFIS, each rule has its own membership
function for every attribute (a scatter partition from k-means), so layer 1 has R × d adaptive nodes. In the
first-order model the inputs also go directly to layer 4, because the rule formulas use them.</p>
{F(FIG / 'arch_anfis.png', 'The five-layer ANFIS network (three rules and three inputs drawn; R rules and d inputs in general). Adaptive layers 1 and 4 hold the trained parameters.')}
{T(nodes, 'The layers of the network: number of nodes, node function, parameters, and whether they are learned.')}
<h2>6.3 Number of parameters</h2>
{T(pc, 'Number of trained parameters of our ANFIS compared with the number of rules a grid partition would need.')}
<p>Each rule has 2d premise parameters (a centre and a width per attribute) and d + 1 consequent parameters (d
coefficients and one constant). Even the largest model has fewer than 1,500 parameters, while a two-set grid partition
of CDC would need over two million rules.</p>""")

    modules = pd.DataFrame([
        ("<code>src/data.py</code>", "load_raw, make_split, Split.select, describe", "experiments, tests, report"),
        ("<code>src/anfis.py</code>", "ANFIS: fit, forward, loss_and_grads, _init_params, describe_rules", "experiments, dashboard"),
        ("<code>src/attribution.py</code>", "rule_attribution, shap_blackbox, shap_tree, lime_local, deletion_test", "experiments, dashboard"),
        ("<code>src/consistency.py</code>", "compare, per_patient", "experiments, dashboard"),
        ("<code>run_experiments.py</code>", "train_anfis, train_lgbm, score, audit, run_one", "writes results/tables and results/models"),
        ("<code>analyze.py</code>", "Wilcoxon, Friedman, Holm, rank-biserial, diagnostics", "writes significance and diagnostic tables"),
        ("<code>make_figures.py</code>", "figures 1 to 9, summary tables", "writes results/figures"),
        ("<code>app.py</code>", "Streamlit dashboard with three tabs", "the user"),
    ], columns=["Module", "Main functions", "Used by"])
    flow = pd.DataFrame([
        ("1", "Load and split the dataset for this seed; draw the explained patients and 100 background patients", "data.py: make_split"),
        ("2", "Train LightGBM and logistic regression; score them on the test data", "run_experiments.py: train_lgbm, score"),
        ("3", "Study A: for each R from 2 to 16, train ANFIS, score it, and audit it", "train_anfis, audit"),
        ("4", "Audit = rule attribution + SHAP + TreeSHAP + LIME + agreement metrics + deletion test + per-patient values", "run_experiments.py: audit"),
        ("5", "Study D: train the bell and zero-order variants at R = 4 and audit them", "run_one"),
        ("6", "Study B: rank attributes by mutual information; for each k train ANFIS and LightGBM on the top k and audit", "run_one"),
        ("7", "Seed 0 only: save models and data for the dashboard; time TreeSHAP on the full CDC test set", "joblib.dump"),
        ("8", "After all 15 (dataset, seed) jobs: write the result tables; analyze.py adds statistics; make_figures.py draws the figures", "main"),
    ], columns=["Step", "What happens", "Code"])
    add(f"""
<h2>6.4 Module design</h2>
{T(modules, 'Module design.')}
<h2>6.5 Flow of one experiment run</h2>
{T(flow, 'What run_experiments.py does for one dataset and one seed. The 15 jobs run in parallel.')}
<h2>6.6 Dashboard design</h2>
<p>The dashboard (<code>app.py</code>) has three tabs. <i>Explain a patient</i> takes four inputs (dataset, ANFIS
variant, number of rules, test patient) and shows the patient's values, both models' predictions, the five explanations
side by side, their agreement with SHAP for that patient, a live deletion check and the fired rules written as
sentences. <i>Experiment results</i> shows all figures and the statistical tables. <i>Datasets and models</i> shows the
dataset summary, accuracy over all seeds and run information. The explanations shown are computed live with the same
functions as the experiments, from the saved seed-0 models.</p>""")

    add(f"""
<h1>7. Implementation</h1>
<h2>7.1 Environment</h2>
<p>Python {info['python']} in a project virtual environment, with NumPy {info['numpy']}, scikit-learn
{info['scikit_learn']} {cite('sklearn')}, LightGBM {info['lightgbm']}, SHAP {info['shap']}, SciPy, pandas, Matplotlib and
Streamlit; exact versions are in <code>requirements.txt</code>. The ANFIS model and its gradients, the rule attribution,
LIME, the deletion test and the agreement metrics are our own code; SHAP and TreeSHAP use the shap library and k-means
uses scikit-learn. The full study (all datasets, seeds and studies) ran in {info['total_runtime_seconds'] / 60:.0f}
minutes on a 16-thread laptop processor, with the 15 (dataset, seed) jobs in parallel.</p>
<h2>7.2 Code organisation</h2>
<pre class="out">ANFIS-SHAP-Interpretability-Audit/
  data/raw/            the three datasets, unmodified
  src/data.py          loading, cleaning, splitting, scaling
  src/anfis.py         the ANFIS model and its training
  src/attribution.py   rule attribution, SHAP, TreeSHAP, LIME, deletion test
  src/consistency.py   agreement metrics
  run_experiments.py   runs studies A to D for all datasets and seeds
  analyze.py           significance tests and diagnostics
  make_figures.py      figures and summary tables
  app.py               interactive dashboard
  tests/test_anfis.py  correctness checks
  results/             tables/, figures/, models/
  docs/                report, article, slides and the scripts that build them</pre>
<h2>7.3 Key code</h2>
<p><b>Forward pass</b>, equations (1) to (6). Line by line: membership (as log μ), the geometric mean as an average of
logs (layer 2), subtracting the maximum to avoid overflow, exponent and normalisation (layer 3), rule formulas (layer 4)
and the weighted sum (layer 5):</p>
{snap('src/anfis.py', '    def log_membership', '    def _dlogmu_du2')}
{snap('src/anfis.py', '    def forward', '    def decision_function')}
<p><b>Rule placement and starting THEN parts</b>, equations (7) and (8): k-means on up to 20,000 patients, widths from
the clusters, then the ridge least-squares solution with targets ±2:</p>
{snap('src/anfis.py', '    def _init_params', '    def fit')}
<p><b>Gradients</b>, equations (9) and (10): <code>g</code> is ∂L/∂z, <code>g_logw</code> is ∂L/∂log w<sub>r</sub> from
equation (18), and the last two lines apply the chain rule to the centres and log-widths:</p>
{snap('src/anfis.py', '    def loss_and_grads', '    def _init_params')}
<p><b>Rule attribution</b>, equations (16) and (19): <code>consequent</code> is vote share × coefficient × value;
<code>antecedent</code> is w̄(f − z) times the change in the log-membership divided by d:</p>
{snap('src/attribution.py', 'def rule_attribution', 'def shap_blackbox')}
<p><b>LIME</b>, equation (23), written by us:</p>
{snap('src/attribution.py', 'def lime_local', 'def deletion_test')}
<h2>7.4 Verification</h2>
<p>Four kinds of checks are run by <code>tests/test_anfis.py</code>: (i) the hand-derived gradients are compared with
numerical (central finite-difference) gradients for every parameter group and all four model variants; (ii) the
THEN-part attribution plus the vote-weighted constants must reproduce the model output exactly (equation 17); (iii) SHAP
values must satisfy efficiency (equation 22); (iv) the data splits must have no missing values and correct scaling.
Output:</p>
<pre class="out">{tests}</pre>
<p>The largest gradient difference is about 10<sup>−10</sup>, so the training equations are implemented correctly.</p>""")

    add(f"""
<h2>7.5 The dashboard in use</h2>
<p>The dashboard is started with <code>streamlit run app.py</code> and opens at <code>localhost:8501</code>. The
screenshots below show the worked-example patient of Section 5.7.6 (Pima, 4 rules, test patient 0).</p>
{F(crops['r_controls'], 'Left column. Inputs: dataset, ANFIS variant, number of rules (2 to 16), test patient. Outputs: the true label, the probability from ANFIS (51.2%) and from LightGBM, and the patient\'s attribute values in original units.', 'small')}
{F(crops['r_charts'], 'The five explanations of this patient side by side, in log-odds. Each bar is one attribute\'s contribution; bars to the right push towards diabetic, bars to the left away from it; attributes are sorted by size. From left: rules THEN-part, rules IF + THEN, SHAP on ANFIS, LIME on ANFIS, TreeSHAP on LightGBM (a different model, shown for comparison).')}
{F(crops['r_agree'], 'Agreement of each explanation with SHAP on ANFIS for this patient (the metrics of Section 5.11). These are the per-patient values; Chapter 8 reports their averages.')}
{F(crops['r_delete'], 'Live deletion check (Section 5.10): each explanation\'s top 3 attributes are replaced with background values and ANFIS is run again. A larger drop means the explanation picked attributes the model really uses.')}
{F(crops['r_rules'], 'The fired rules written as sentences, strongest first, with each rule\'s vote share (firing strength) and output. The IF part lists the rule\'s four narrowest memberships with their centres in original units (LOW / MEDIUM / HIGH = centre below, near or above the average); the THEN part lists the constant and the four largest coefficients.')}
{F(crops['r_variant_rules'], 'The same view for the zero-order variant on Early Stage: each rule\'s THEN part is a constant, so one rule with vote share 0.999 decides the output. Yes/no attributes are shown as YES / NO.')}""")

    add(results_chapter(r, T, F))
    add(comparison_chapter(r, T))
    add(conclusion_chapter(r))
    add(future_chapter(r))

    files = pd.DataFrame([
        ("results/tables/performance.csv", "Accuracy, balanced accuracy, F1, ROC-AUC and threshold of every model, dataset and seed"),
        ("results/tables/consistency.csv", "All agreement metrics for every explanation, setting and seed"),
        ("results/tables/per_patient.csv.gz", "Per-patient agreement and firing concentration (input to the statistics)"),
        ("results/tables/deletion.csv", "Deletion test results for k = 1, 2, 3 and both removal conventions"),
        ("results/tables/shap_timing.csv", "Time per patient of black-box SHAP, TreeSHAP and LIME"),
        ("results/tables/significance.csv, friedman_rules.csv", "Wilcoxon and Friedman tests with Holm-corrected p-values and effect sizes"),
        ("results/tables/early_stage_diagnostics.csv", "Diagnostic quantities of Section 8.7"),
        ("results/tables/summary_*.csv", "Means and standard deviations over the five seeds"),
        ("results/figures/*.png", "Figures 1 to 9 and the two block diagrams"),
        ("results/models/*_seed0.joblib", "Saved seed-0 models and data used by the dashboard"),
    ], columns=["File", "Contents"])
    attrs = pd.DataFrame([{"Dataset": nm, "Attributes": ", ".join(raw[k].columns)} for k, nm in
                          [("pima", "Pima (8)"), ("early", "Early Stage (16)"), ("cdc", "CDC (21)")]])
    add(f"""
<h1>References</h1>
{references_html()}
<h1>Appendix</h1>
<h2>A. Reproducing the results</h2>
<table class="t"><thead><tr><th>Step</th><th>Command (from the project folder)</th></tr></thead>
<tr><td>Correctness checks</td><td><code>.venv\\Scripts\\python.exe tests\\test_anfis.py</code></td></tr>
<tr><td>All experiments (about {info['total_runtime_seconds'] / 60:.0f} min)</td><td><code>.venv\\Scripts\\python.exe run_experiments.py</code></td></tr>
<tr><td>Statistics</td><td><code>.venv\\Scripts\\python.exe analyze.py</code></td></tr>
<tr><td>Figures</td><td><code>.venv\\Scripts\\python.exe make_figures.py</code> and <code>docs\\make_diagrams.py</code></td></tr>
<tr><td>Dashboard</td><td><code>.venv\\Scripts\\streamlit.exe run app.py</code></td></tr>
<tr><td>This report</td><td><code>.venv\\Scripts\\python.exe docs\\build_report.py</code></td></tr></table>
<p>Every number in this report is read from the result files when the report is built, so the report always matches
the latest run.</p>
<h2>B. Files produced by the pipeline</h2>
{T(files, 'Output files.')}
<h2>C. Attributes of the datasets</h2>
{T(attrs, 'All attributes, in the order used by the models.')}""")
    return "\n".join(S)


def main():
    page = f"<!doctype html><html><head><meta charset='utf-8'><title>Project Report</title><style>{CSS}</style></head><body>{build()}</body></html>"
    html_path, pdf_path = DOCS / "Project_Report.html", DOCS / "Project_Report.pdf"
    html_path.write_text(page, encoding="utf-8")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--allow-file-access-from-files",
                    f"--print-to-pdf={pdf_path}", html_path.as_uri()], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("wrote", pdf_path, f"({pdf_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
