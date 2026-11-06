import subprocess

from report_common import CHROME, snap, team, ATTR_EQUATIONS, EQUATIONS, FIG, ROOT, Results, cite, references_html
from report_sections import article_discussion, article_results

DOCS = ROOT / "docs"

CSS = """
@page { size: A4; margin: 18mm 15mm 18mm 15mm; }
body { font-family: 'Times New Roman', Times, serif; font-size: 10pt; line-height: 1.32; color: #111; margin: 0; }
.head { text-align: center; margin-bottom: 5mm; }
.head h1 { font-size: 19pt; line-height: 1.2; margin: 0 0 3mm; font-weight: normal; }
.head .au { font-size: 11pt; margin: 0 0 1mm; }
.head .af { font-size: 9.5pt; font-style: italic; margin: 0; }
.abstract { font-size: 9.2pt; font-weight: bold; text-align: justify; margin: 0 0 2mm; }
.abstract i { font-weight: bold; }
.kw { font-size: 9.2pt; text-align: justify; margin: 0 0 4mm; }
.cols { columns: 2; column-gap: 6mm; text-align: justify; }
h2 { font-size: 10pt; text-transform: uppercase; text-align: center; margin: 4mm 0 1.5mm; font-weight: normal;
     letter-spacing: 0.5px; break-after: avoid; }
h3 { font-size: 10pt; font-style: italic; font-weight: normal; margin: 2.5mm 0 1mm; break-after: avoid; }
p { margin: 0 0 1.6mm; text-indent: 4mm; }
p.ni { text-indent: 0; }
ul { margin: 0 0 1.6mm; padding-left: 5mm; } li { margin-bottom: 0.6mm; }
table.t { border-collapse: collapse; width: 100%; margin: 1mm 0 0.5mm; font-size: 7.6pt; break-inside: avoid; }
table.t th { border-top: 1px solid #111; border-bottom: 0.6px solid #111; padding: 0.6mm 1mm; text-align: left; font-weight: bold; }
table.t td { padding: 0.5mm 1mm; }
table.t tr:last-child td { border-bottom: 1px solid #111; }
.cap { font-size: 8pt; text-align: center; margin: 0.5mm 0 3mm; text-indent: 0; }
.tcap { font-size: 8pt; text-align: center; margin: 2mm 0 0.5mm; text-indent: 0; font-variant: small-caps; }
figure { margin: 1.5mm 0 2.5mm; break-inside: avoid; text-align: center; }
figure img { width: 100%; }
figcaption { font-size: 8pt; text-align: justify; }
.wide { column-span: all; }
.eq { margin: 1.2mm 0 1.2mm 2mm; font-size: 9.2pt; position: relative; padding-right: 7mm; text-indent: 0; }
.eqn { position: absolute; right: 0; }
.refs { font-size: 8pt; padding-left: 5mm; margin: 0; } .refs li { margin-bottom: 0.8mm; text-align: left; }
"""


def table(df, caption):
    return f"<p class='tcap'>{caption}</p>" + df.to_html(index=False, border=0, classes="t", escape=True)


def build():
    r = Results()
    info = r.info
    lo, hi = r.rules[0], r.rules[-1]
    n = {"t": 0, "f": 0}
    roman = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]

    def T(df, caption, wide=False):
        n["t"] += 1
        html = table(df, f"Table {roman[n['t'] - 1]}. {caption}")
        return f"<div class='wide'>{html}</div>" if wide else html

    def F(name, caption, wide=False):
        n["f"] += 1
        html = f"<figure><img src='{(FIG / name).as_uri()}'><figcaption>Fig. {n['f']}. {caption}</figcaption></figure>"
        return f"<div class='wide'>{html}</div>" if wide else html

    sign_min = min(r.c(d, 'rule_consequent', 'sign_agreement', rules=k) for d in ["pima", "early", "cdc"] for k in r.rules)
    abstract = (
        "Adaptive neuro-fuzzy inference systems (ANFIS) are often chosen for clinical prediction because they explain "
        "each decision with IF-THEN rules, yet the faithfulness of that explanation is rarely tested. We audit it on "
        f"three public diabetes datasets with 8, 16 and 21 attributes (768 to 253,680 records). A first-order "
        "Takagi-Sugeno-Kang ANFIS with clustering-initialised rules and dimension-normalised firing is trained with "
        f"{lo} to {hi} rules, and a per-patient attribution is derived from its rule base, separately for the "
        "consequent and the antecedent. It is compared with SHAP and LIME attributions of the same model, and all "
        "explanations are scored by a reference-free deletion test, over five random splits, attribute subsets and "
        "four membership/TSK-order variants, with LightGBM and TreeSHAP as a reference model. ANFIS matches LightGBM "
        f"in ROC-AUC on all datasets. The consequent-based rule explanation agrees with SHAP on the direction of each "
        f"effect in at least {sign_min * 100:.0f}% of cases, but its per-patient rank agreement is only partial "
        f"(Spearman {r.c('early', 'rule_consequent'):.2f} to {r.c('cdc', 'rule_consequent'):.2f} with four rules) "
        "and decreases as rules are added while accuracy does not improve. ANFIS rule explanations should therefore be "
        "verified rather than assumed, and compact rule bases are preferable.")

    authors = f"<p class='au'>{', '.join(nm for nm, _ in team())}</p>" if team() else ""
    S = [f"""
<div class="head">
<h1>How Faithful Are ANFIS's Fuzzy Rules? A SHAP-Based<br>Consistency Audit on Diabetes Diagnosis</h1>
{authors}
<p class="af">School of Computer Science and Engineering, VIT-AP University, India</p>
</div>
<div class="cols">
<p class="abstract ni"><i>Abstract</i>—{abstract}</p>
<p class="kw ni"><b><i>Index Terms</i>—ANFIS, neuro-fuzzy systems, explainable AI, SHAP, LIME, faithfulness, diabetes.</b></p>

<h2>I. Introduction</h2>
<p>Interpretable models are preferred in clinical decision support because a clinician must understand why a patient
is flagged before acting. The adaptive neuro-fuzzy inference system (ANFIS) {cite('jang')} is a common choice: it
learns a Takagi-Sugeno-Kang (TSK) fuzzy rule base {cite('takagi')} from data and presents its reasoning as linguistic
IF-THEN rules. The rules are readable, but readability is not faithfulness. A trained ANFIS blends many partially fired
rules whose weights themselves depend on the input, so the explanation a reader takes from the fired rules may differ
from what actually moved the output.</p>
<p>We test this directly. For every explained patient we derive an attribution from the rule base and compare it with
post-hoc attributions of the same model (SHAP {cite('shap')}, LIME {cite('lime')}) and with a deletion test that needs
no reference explanation {cite('samek')}. Our contributions are: (i) a decomposition of the rule explanation into an
exact consequent part and a first-order antecedent part; (ii) a systematic audit over three diabetes datasets, seven
rule counts, attribute subsets and four model variants, repeated over five splits with corrected significance tests;
(iii) an open pipeline and interactive demonstrator.</p>

<h2>II. Related Work</h2>
<p>ANFIS {cite('jang')} trains TSK systems {cite('takagi')} by combining least squares for consequents with gradient
descent for memberships. In high dimension, products of many memberships collapse and normalised firing becomes a hard
maximum; Cui et al. {cite('cui')} address this by normalising firing by the input dimension (HTSK). Fuzzy-system
interpretability has mainly been measured structurally, by rule count, rule length and set distinguishability
{cite('gacto')}, which captures readability rather than faithfulness. SHAP {cite('shap')} assigns Shapley values
{cite('shapley')} to attributes; TreeSHAP {cite('treeshap')} computes them exactly and efficiently for tree ensembles
such as LightGBM {cite('lgbm')}. LIME {cite('lime')} fits a local weighted linear surrogate. Explainers often disagree
{cite('krishna')}, and removal-based tests {cite('samek', 'hooker')} are the usual way to evaluate explanations without
ground truth. To our knowledge, no prior study measures the per-patient agreement between an ANFIS rule base's own
explanation and independent attributions of the same model.</p>

<h2>III. Method</h2>
<h3>A. Model</h3>
<p class="ni">For x ∈ ℝ<sup>d</sup> and R rules:</p>
{EQUATIONS}
<p>Centres are initialised by k-means (one rule per cluster, so R is free; a grid would need 2<sup>d</sup> rules),
widths by within-cluster spread, consequents by ridge least squares; all parameters are then trained with Adam
{cite('adam')} on (5) with early stopping. Gradients for all variants were derived analytically and verified against
finite differences (maximum deviation below 10<sup>−9</sup>).</p>
<h3>B. Rule-based attribution</h3>
{ATTR_EQUATIONS.split('<div class="eq"><span class="eqn">(8)')[0]}
<p>Here x̄ is the training mean. The consequent part (6) is what a reader takes from the fired rules and satisfies
Σ<sub>j</sub>φ<sup>THEN</sup><sub>j</sub> + Σ<sub>r</sub>w̄<sub>r</sub>b<sub>r</sub> = z exactly; the antecedent part (7)
is the first-order effect through the firing strengths, since ∂z/∂log w<sub>r</sub> = w̄<sub>r</sub>(f<sub>r</sub> − z).
For zero-order models (6) vanishes and the rules explain only through (7).</p>
<h3>C. References and deletion test</h3>
{'<div class="eq"><span class="eqn">(8)' + ATTR_EQUATIONS.split('<div class="eq"><span class="eqn">(8)')[1]}
<p>SHAP (8) uses {info['n_background']} training patients as background, exact enumeration for d ≤ 10 and permutation
sampling otherwise. LIME (9) draws 1,000 perturbations per patient from background marginals with kernel width
κ = 0.75√d. The deletion score (10) replaces the top-k attributes with values of 20 background patients; a random
ranking is the control. All attributions are in log-odds.</p>
<h3>D. Agreement and statistics</h3>
<p>Per patient we compute Spearman ρ, Kendall τ and weighted Kendall τ {cite('vigna')} between absolute attributions,
top-1 match, top-3 overlap and sign agreement over the reference's top three attributes; global agreement is the
Spearman correlation of mean absolute attributions. Paired differences are tested with the Wilcoxon signed-rank test
{cite('wilcoxon')} on (seed, patient) units, differences across rule counts with the Friedman test {cite('friedman')},
with Holm correction {cite('holm')} and rank-biserial effect sizes.</p>

<h2>IV. Experimental Setup</h2>
<p>We use Pima Indians Diabetes {cite('pima')} (768 × 8, laboratory values), Early Stage Diabetes Risk {cite('early')}
(520 × 16, mostly yes/no symptoms) and CDC Diabetes Health Indicators {cite('cdc')} (253,680 × 21, survey answers).
Impossible zeros in Pima are imputed with the training median; splits are stratified 60/20/20 and standardised with
training statistics. Study A varies R ∈ {{{', '.join(map(str, r.rules))}}}; study B keeps the top-k attributes by mutual
information {cite('kraskov')}; study C records SHAP cost; study D compares Gaussian and bell (b = 2) memberships in first
and zero order at R = 4. Each setting is repeated for five seeds, with {info['n_explain']} explained test patients per
model. LightGBM and logistic regression {cite('sklearn')} are trained on the same splits.</p>
"""]
    S.append(article_results(r, T, F))
    S.append(article_discussion(r))
    S.append(f"""
<h2>References</h2>
{references_html()}
</div>""")
    return "\n".join(S)


def main():
    page = f"<!doctype html><html><head><meta charset='utf-8'><title>Research Article</title><style>{CSS}</style></head><body>{build()}</body></html>"
    html_path, pdf_path = DOCS / "Research_Article.html", DOCS / "Research_Article.pdf"
    html_path.write_text(page, encoding="utf-8")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf_path}", html_path.as_uri()], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("wrote", pdf_path, f"({pdf_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
