import html
import io
import json
import keyword
import tokenize
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TAB, FIG = ROOT / "results" / "tables", ROOT / "results" / "figures"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
esc = html.escape


def team():
    f = ROOT / "docs" / "team.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else []
DS = ["pima", "early", "cdc"]
NAME = {"pima": "Pima", "early": "Early Stage", "cdc": "CDC"}
FULL = {"pima": "Pima Indians Diabetes", "early": "Early Stage Diabetes Risk",
        "cdc": "CDC Diabetes Health Indicators"}
EXPL = {"rule_consequent": "Rules: THEN-part", "rule_antecedent": "Rules: IF-part",
        "rule_full": "Rules: IF + THEN", "lime_anfis": "LIME on ANFIS", "shap_lgbm": "SHAP on LightGBM"}


class Results:
    def __init__(self):
        self.info = json.loads((TAB / "run_info.json").read_text())
        self.perf = pd.read_csv(TAB / "performance.csv")
        self.cons_all = pd.read_csv(TAB / "consistency.csv")
        self.cons = self.cons_all[self.cons_all.reference == "shap_anfis"]
        self.time = pd.read_csv(TAB / "shap_timing.csv")
        self.dele = pd.read_csv(TAB / "deletion.csv")
        self.sig = pd.read_csv(TAB / "significance.csv")
        self.fried = pd.read_csv(TAB / "friedman_rules.csv")
        self.diag = pd.read_csv(TAB / "early_stage_diagnostics.csv").set_index("dataset")
        self.datasets = pd.read_csv(TAB / "datasets.csv")
        self.rules = self.info["rules"]
        self.subs = self.info["subset_sizes"]
        self.seeds = len(self.info["seeds"])

    def c(self, ds, ex, metric="spearman_local", rules=4, attrs=None, ref="shap_anfis", study=None, mf=None, order=None):
        d = self.cons_all[(self.cons_all.dataset == ds) & (self.cons_all.explanation == ex)
                          & (self.cons_all.reference == ref)]
        if study == "variants":
            d = d[(d.study == "variants") & (d.mf == mf) & (d.order == order)]
        elif attrs is not None:
            d = d[(d.study == "attributes") & (d.n_attributes == attrs)]
        else:
            d = d[(d.study == "rules") & (d.n_rules == rules)]
        return float(d[metric].mean())

    def auc(self, ds, model, rules=None):
        d = self.perf[(self.perf.dataset == ds) & (self.perf.model == model)]
        if rules is not None:
            d = d[d.n_rules == rules]
        return d.roc_auc

    def vauc(self, ds, mf, order):
        return self.perf[(self.perf.dataset == ds) & (self.perf.model == f"ANFIS ({mf}, order {order})")].roc_auc

    def ms(self, ds, attrs, col):
        d = self.time[(self.time.study == "attributes") & (self.time.dataset == ds) & (self.time.n_attributes == attrs)]
        return float(d[col].mean() * 1000)

    def deletion(self, ds, ex, k=3, removal="background", rules=4):
        d = self.dele[(self.dele.dataset == ds) & (self.dele.study == "rules") & (self.dele.n_rules == rules)
                      & (self.dele.explanation == ex) & (self.dele.k == k) & (self.dele.removal == removal)]
        return float(d.mean_abs_change.mean())

    def test(self, ds, contains):
        row = self.sig[(self.sig.dataset == ds) & self.sig.test.str.contains(contains, regex=False)]
        return row.iloc[0]

    def table_datasets(self):
        t = self.datasets.copy()
        t.columns = ["Dataset", "Records", "Attributes", "Binary attributes", "Missing cells", "Positive rate"]
        return t

    def table_accuracy(self):
        pm = lambda s: f"{s.mean():.3f} ± {s.std():.3f}"
        lo, hi = self.rules[0], self.rules[-1]
        rows = []
        for ds in DS:
            for label, model, k in [("Logistic regression", "LogisticRegression", None), ("LightGBM", "LightGBM", None),
                                    (f"ANFIS ({lo} rules)", "ANFIS", lo), ("ANFIS (4 rules)", "ANFIS", 4),
                                    (f"ANFIS ({hi} rules)", "ANFIS", hi)]:
                d = self.perf[(self.perf.dataset == ds) & (self.perf.model == model)]
                if k is not None:
                    d = d[d.n_rules == k]
                rows.append({"Dataset": NAME[ds], "Model": label, "ROC-AUC": pm(d.roc_auc), "F1": pm(d.f1),
                             "Balanced acc.": pm(d.balanced_accuracy)})
        return pd.DataFrame(rows)

    def table_agreement(self, keys=("rule_consequent", "rule_antecedent", "rule_full", "lime_anfis", "shap_lgbm")):
        rows = []
        for ds in DS:
            for key in keys:
                rows.append({"Dataset": NAME[ds], "Explanation vs SHAP on ANFIS": EXPL[key],
                             "Spearman": f"{self.c(ds, key):.2f}", "Kendall": f"{self.c(ds, key, 'kendall_local'):.2f}",
                             "Weighted τ": f"{self.c(ds, key, 'weighted_kendall'):.2f}",
                             "Top-1": f"{self.c(ds, key, 'top1_match'):.2f}", "Top-3": f"{self.c(ds, key, 'top3_overlap'):.2f}",
                             "Sign": f"{self.c(ds, key, 'sign_agreement'):.2f}",
                             "Global ρ": f"{self.c(ds, key, 'spearman_global'):.2f}"})
        return pd.DataFrame(rows)

    def table_rules(self, key="rule_consequent"):
        return pd.DataFrame([{"Dataset": NAME[ds], **{f"R={k}": f"{self.c(ds, key, rules=k):.2f}" for k in self.rules}}
                             for ds in DS])

    def table_attributes(self):
        rows = []
        for ds in DS:
            for k in self.subs[ds]:
                rows.append({"Dataset": NAME[ds], "k": k,
                             "THEN-part": f"{self.c(ds, 'rule_consequent', attrs=k):.2f}",
                             "IF + THEN": f"{self.c(ds, 'rule_full', attrs=k):.2f}",
                             "LIME": f"{self.c(ds, 'lime_anfis', attrs=k):.2f}",
                             "SHAP on LightGBM": f"{self.c(ds, 'shap_lgbm', attrs=k):.2f}"})
        return pd.DataFrame(rows)

    def table_variants(self):
        rows = []
        for ds in DS:
            for mf, order in [("gauss", 1), ("bell", 1), ("gauss", 0), ("bell", 0)]:
                key = "rule_consequent" if order == 1 else "rule_full"
                a = self.vauc(ds, mf, order)
                rows.append({"Dataset": NAME[ds], "Membership": "Gaussian" if mf == "gauss" else "Bell",
                             "TSK order": order, "ROC-AUC": f"{a.mean():.3f} ± {a.std():.3f}",
                             "Rule reading": "THEN-part" if order == 1 else "IF-part (= IF + THEN)",
                             "Spearman vs SHAP": f"{self.c(ds, key, study='variants', mf=mf, order=order):.2f}",
                             "Top-3 vs SHAP": f"{self.c(ds, key, 'top3_overlap', study='variants', mf=mf, order=order):.2f}"})
        return pd.DataFrame(rows)

    def table_lime(self):
        rows = []
        for ds in DS:
            rows.append({"Dataset": NAME[ds],
                         "THEN-part vs SHAP": f"{self.c(ds, 'rule_consequent'):.2f}",
                         "THEN-part vs LIME": f"{self.c(ds, 'rule_consequent', ref='lime_anfis'):.2f}",
                         "IF + THEN vs LIME": f"{self.c(ds, 'rule_full', ref='lime_anfis'):.2f}",
                         "LIME vs SHAP": f"{self.c(ds, 'lime_anfis'):.2f}"})
        return pd.DataFrame(rows)

    def table_deletion(self, removal="background"):
        rows = []
        for ds in DS:
            row = {"Dataset": NAME[ds]}
            for key, label in [("shap_anfis", "SHAP"), ("lime_anfis", "LIME"), ("rule_consequent", "THEN-part"),
                               ("rule_full", "IF + THEN"), ("rule_antecedent", "IF-part"), ("random", "Random")]:
                row[label] = f"{self.deletion(ds, key, removal=removal):.2f}"
            rows.append(row)
        return pd.DataFrame(rows)

    def table_significance(self):
        t = self.sig.copy()
        t["Dataset"] = t.dataset.map(NAME)
        fmt_p = lambda p: "< 0.001" if p < 0.001 else f"{p:.3f}"
        return pd.DataFrame({"Dataset": t.Dataset, "Comparison (A vs B)": t.test, "Pairs": t.n_pairs,
                             "Mean A": t.mean_a.map("{:.3f}".format), "Mean B": t.mean_b.map("{:.3f}".format),
                             "Effect r": t.rank_biserial.map("{:+.2f}".format),
                             "p (Holm)": t.p_holm.map(fmt_p),
                             "Significant": np.where(t["significant_at_0.05"], "yes", "no")})

    def table_diagnostics(self):
        d = self.diag
        rows = [("Binary (yes/no) attributes", "share_yes_no_attributes", "{:.0%}"),
                ("Mean |correlation| between attributes", "mean_abs_correlation_between_attributes", "{:.3f}"),
                ("Share of |SHAP| held by the top 3 attributes", "mean_shap_top3_share", "{:.2f}"),
                ("Attributes with |SHAP| < 10% of the largest", "share_attributes_below_10pct_of_top", "{:.0%}"),
                ("Mean largest firing strength", "mean_max_firing", "{:.2f}"),
                ("Agreement, patients with concentrated SHAP", "agreement_when_concentrated", "{:.2f}"),
                ("Agreement, patients with spread-out SHAP", "agreement_when_spread", "{:.2f}"),
                ("Spearman(top-3 share, agreement) over patients", "spearman_vs_top3_share", "{:+.2f}"),
                ("Spearman(largest firing, agreement) over patients", "spearman_vs_max_firing", "{:+.2f}")]
        return pd.DataFrame([{"Quantity (4 rules, THEN-part)": label, **{NAME[ds]: f.format(d.loc[ds, col]) for ds in DS}}
                             for label, col, f in rows])


EQUATIONS = """
<div class="eq"><span class="eqn">(1)</span>μ<sub>rj</sub>(x<sub>j</sub>) = exp(−(x<sub>j</sub> − c<sub>rj</sub>)² / 2σ<sub>rj</sub>²)
&nbsp; (Gaussian) &nbsp; or &nbsp; 1 / (1 + ((x<sub>j</sub> − c<sub>rj</sub>)/σ<sub>rj</sub>)<sup>2b</sup>) &nbsp; (bell, b = 2)</div>
<div class="eq"><span class="eqn">(2)</span>w<sub>r</sub>(x) = (∏<sub>j=1..d</sub> μ<sub>rj</sub>(x<sub>j</sub>))<sup>1/d</sup>,
&nbsp;&nbsp; w̄<sub>r</sub> = w<sub>r</sub> / Σ<sub>k</sub> w<sub>k</sub></div>
<div class="eq"><span class="eqn">(3)</span>f<sub>r</sub>(x) = a<sub>r</sub>·x + b<sub>r</sub> &nbsp; (first order) &nbsp; or &nbsp; f<sub>r</sub> = b<sub>r</sub> &nbsp; (zero order)</div>
<div class="eq"><span class="eqn">(4)</span>z(x) = Σ<sub>r</sub> w̄<sub>r</sub>(x) f<sub>r</sub>(x), &nbsp;&nbsp; P(y = 1 | x) = 1 / (1 + e<sup>−z</sup>)</div>
<div class="eq"><span class="eqn">(5)</span>L = −(1/n) Σ<sub>i</sub> [ y<sub>i</sub> log p<sub>i</sub> + (1 − y<sub>i</sub>) log(1 − p<sub>i</sub>) ] + (λ/2) ‖a‖²</div>
"""

ATTR_EQUATIONS = """
<div class="eq"><span class="eqn">(6)</span>φ<sup>THEN</sup><sub>j</sub>(x) = Σ<sub>r</sub> w̄<sub>r</sub>(x) · a<sub>rj</sub> · (x<sub>j</sub> − x̄<sub>j</sub>)</div>
<div class="eq"><span class="eqn">(7)</span>φ<sup>IF</sup><sub>j</sub>(x) = Σ<sub>r</sub> w̄<sub>r</sub>(x) · (f<sub>r</sub>(x) − z(x)) · (ℓ<sub>rj</sub>(x<sub>j</sub>) − ℓ<sub>rj</sub>(x̄<sub>j</sub>)),
&nbsp;&nbsp; ℓ<sub>rj</sub> = (1/d) log μ<sub>rj</sub></div>
<div class="eq"><span class="eqn">(8)</span>φ<sup>SHAP</sup><sub>j</sub>(x) = Σ<sub>S ⊆ F∖{j}</sub> [ |S|! (d − |S| − 1)! / d! ] · ( v(S ∪ {j}) − v(S) ),
&nbsp;&nbsp; v(S) = E<sub>x' ∼ background</sub>[ z(x<sub>S</sub>, x'<sub>S̄</sub>) ]</div>
<div class="eq"><span class="eqn">(9)</span>β = argmin<sub>β</sub> Σ<sub>m</sub> π(x, x<sup>(m)</sup>) (z(x<sup>(m)</sup>) − β<sub>0</sub> − β·(x<sup>(m)</sup> − m̄))² + ‖β‖²,
&nbsp; π = exp(−‖x − x<sup>(m)</sup>‖² / κ²), &nbsp; φ<sup>LIME</sup><sub>j</sub> = β<sub>j</sub>(x<sub>j</sub> − m̄<sub>j</sub>)</div>
<div class="eq"><span class="eqn">(10)</span>Δ<sub>k</sub>(φ) = (1/N) Σ<sub>i</sub> | z(x<sub>i</sub>) − E<sub>x' ∼ background</sub>[ z(x<sub>i</sub> with its top-k attributes under φ taken from x') ] |</div>
"""

REFERENCES = [
    "J.-S. R. Jang, \"ANFIS: Adaptive-network-based fuzzy inference system,\" <i>IEEE Transactions on Systems, Man, and Cybernetics</i>, vol. 23, no. 3, pp. 665–685, 1993.",
    "T. Takagi and M. Sugeno, \"Fuzzy identification of systems and its applications to modeling and control,\" <i>IEEE Transactions on Systems, Man, and Cybernetics</i>, vol. SMC-15, no. 1, pp. 116–132, 1985.",
    "Y. Cui, D. Wu, and Y. Xu, \"Curse of dimensionality for TSK fuzzy neural networks: Explanation and solutions,\" in <i>Proc. International Joint Conference on Neural Networks (IJCNN)</i>, 2021.",
    "M. J. Gacto, R. Alcalá, and F. Herrera, \"Interpretability of linguistic fuzzy rule-based systems: An overview of interpretability measures,\" <i>Information Sciences</i>, vol. 181, no. 20, pp. 4340–4360, 2011.",
    "L. S. Shapley, \"A value for n-person games,\" in <i>Contributions to the Theory of Games II</i>, Princeton University Press, 1953, pp. 307–317.",
    "S. M. Lundberg and S.-I. Lee, \"A unified approach to interpreting model predictions,\" in <i>Advances in Neural Information Processing Systems 30 (NeurIPS)</i>, 2017.",
    "S. M. Lundberg, G. Erion, H. Chen, A. DeGrave, J. M. Prutkin, B. Nair, R. Katz, J. Himmelfarb, N. Bansal, and S.-I. Lee, \"From local explanations to global understanding with explainable AI for trees,\" <i>Nature Machine Intelligence</i>, vol. 2, pp. 56–67, 2020.",
    "M. T. Ribeiro, S. Singh, and C. Guestrin, \"'Why should I trust you?': Explaining the predictions of any classifier,\" in <i>Proc. 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining</i>, 2016, pp. 1135–1144.",
    "G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y. Liu, \"LightGBM: A highly efficient gradient boosting decision tree,\" in <i>Advances in Neural Information Processing Systems 30 (NeurIPS)</i>, 2017.",
    "W. Samek, A. Binder, G. Montavon, S. Lapuschkin, and K.-R. Müller, \"Evaluating the visualization of what a deep neural network has learned,\" <i>IEEE Transactions on Neural Networks and Learning Systems</i>, vol. 28, no. 11, pp. 2660–2673, 2017.",
    "S. Hooker, D. Erhan, P.-J. Kindermans, and B. Kim, \"A benchmark for interpretability methods in deep neural networks,\" in <i>Advances in Neural Information Processing Systems 32 (NeurIPS)</i>, 2019.",
    "S. Krishna, T. Han, A. Gu, J. Pombra, S. Jabbari, S. Wu, and H. Lakkaraju, \"The disagreement problem in explainable machine learning: A practitioner's perspective,\" arXiv:2202.01602, 2022.",
    "J. W. Smith, J. E. Everhart, W. C. Dickson, W. C. Knowler, and R. S. Johannes, \"Using the ADAP learning algorithm to forecast the onset of diabetes mellitus,\" in <i>Proc. Annual Symposium on Computer Application in Medical Care</i>, 1988, pp. 261–265.",
    "M. M. F. Islam, R. Ferdousi, S. Rahman, and H. Y. Bushra, \"Likelihood prediction of diabetes at early stage using data mining techniques,\" in <i>Computer Vision and Machine Intelligence in Medical Image Analysis</i>, Springer, 2020, pp. 113–125.",
    "Centers for Disease Control and Prevention, \"CDC Diabetes Health Indicators\" (BRFSS 2015), UCI Machine Learning Repository, dataset 891, https://archive.ics.uci.edu/dataset/891.",
    "D. P. Kingma and J. Ba, \"Adam: A method for stochastic optimization,\" in <i>Proc. International Conference on Learning Representations (ICLR)</i>, 2015.",
    "A. Kraskov, H. Stögbauer, and P. Grassberger, \"Estimating mutual information,\" <i>Physical Review E</i>, vol. 69, 066138, 2004.",
    "F. Wilcoxon, \"Individual comparisons by ranking methods,\" <i>Biometrics Bulletin</i>, vol. 1, no. 6, pp. 80–83, 1945.",
    "M. Friedman, \"The use of ranks to avoid the assumption of normality implicit in the analysis of variance,\" <i>Journal of the American Statistical Association</i>, vol. 32, no. 200, pp. 675–701, 1937.",
    "S. Holm, \"A simple sequentially rejective multiple test procedure,\" <i>Scandinavian Journal of Statistics</i>, vol. 6, no. 2, pp. 65–70, 1979.",
    "S. Vigna, \"A weighted correlation index for rankings with ties,\" in <i>Proc. 24th International Conference on World Wide Web (WWW)</i>, 2015, pp. 1166–1176.",
    "F. Pedregosa et al., \"Scikit-learn: Machine learning in Python,\" <i>Journal of Machine Learning Research</i>, vol. 12, pp. 2825–2830, 2011.",
]
CITE = {k: i + 1 for i, k in enumerate(["jang", "takagi", "cui", "gacto", "shapley", "shap", "treeshap", "lime",
                                        "lgbm", "samek", "hooker", "krishna", "pima", "early", "cdc", "adam",
                                        "kraskov", "wilcoxon", "friedman", "holm", "vigna", "sklearn"])}


def cite(*keys):
    return "[" + ", ".join(str(CITE[k]) for k in keys) + "]"


def references_html():
    return "<ol class='refs'>" + "".join(f"<li>{r}</li>" for r in REFERENCES) + "</ol>"


def _table_literature(self):
    rows = [
        ("Jang, 1993 [1]", "ANFIS: TSK rules trained by hybrid learning", "Accuracy; readable rules", "No check that the rules explain the model faithfully"),
        ("Takagi and Sugeno, 1985 [2]", "TSK fuzzy model with functional consequents", "Modelling with fuzzy rules", "Basis of our model"),
        ("Cui, Wu and Xu, 2021 [3]", "HTSK: dimension-normalised firing", "Numerical behaviour in high dimension", "Adopted in our ANFIS"),
        ("Gacto et al., 2011 [4]", "Interpretability measures for fuzzy rule bases", "Structural readability (rule count, length)", "Measures readability, not faithfulness"),
        ("ANFIS variants surveyed in Review 1", "Rule simplification, structural relaxation, compact rule bases", "Smaller or more accurate rule bases", "Do not measure explanation faithfulness"),
        ("Lundberg and Lee, 2017 [6]", "SHAP (Shapley-value attribution)", "Model-agnostic explanation", "Our independent reference"),
        ("Lundberg et al., 2020 [7]", "TreeSHAP", "Exact, fast SHAP for trees", "Used for LightGBM on the large dataset"),
        ("Ribeiro et al., 2016 [8]", "LIME local surrogate", "Model-agnostic explanation", "Our second reference"),
        ("Samek et al., 2017 [10]; Hooker et al., 2019 [11]", "Removal-based evaluation", "Evaluating explanations without ground truth", "Basis of our deletion test"),
        ("Krishna et al., 2022 [12]", "Disagreement between explainers", "Explainer disagreement in practice", "Motivates using two references and a deletion test"),
    ]
    return pd.DataFrame(rows, columns=["Work", "Contribution", "Focus", "Relation to this project"])


Results.table_literature = _table_literature


def highlight(line):
    out, pos = [], 0
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(line).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return esc(line)
    for tok in toks:
        if tok.type in (tokenize.NEWLINE, tokenize.NL, tokenize.ENDMARKER, tokenize.INDENT, tokenize.DEDENT):
            continue
        (_, c0), (_, c1) = tok.start, tok.end
        if tok.start[0] != 1:
            continue
        out.append(esc(line[pos:c0]))
        text = esc(line[c0:c1])
        if tok.type == tokenize.COMMENT:
            text = f'<span class="cm">{text}</span>'
        elif tok.type == tokenize.STRING:
            text = f'<span class="st">{text}</span>'
        elif tok.type == tokenize.NUMBER:
            text = f'<span class="nu">{text}</span>'
        elif tok.type == tokenize.NAME and keyword.iskeyword(tok.string):
            text = f'<span class="kw">{text}</span>'
        out.append(text)
        pos = c1
    out.append(esc(line[pos:]))
    return "".join(out)


def snap(path, start, end=None, max_lines=60):
    lines = (ROOT / path).read_text(encoding="utf-8").splitlines()
    i = next(k for k, l in enumerate(lines) if start in l)
    j = len(lines)
    if end is not None:
        j = next(k for k in range(i + 1, len(lines)) if end in lines[k])
    j = min(j, i + max_lines)
    while j > i and not lines[j - 1].strip():
        j -= 1
    in_doc, rows = False, []
    for k in range(i, j):
        line = lines[k]
        quotes = line.count('"""')
        if in_doc or quotes:
            body = f'<span class="st">{esc(line)}</span>'
            if quotes == 1:
                in_doc = not in_doc
        else:
            body = highlight(line)
        rows.append(f'<span class="ln">{k + 1:>3}</span>{body}')
    return (f'<div class="snap"><div class="bar"><i></i><i></i><i></i><b>{esc(path)}</b>'
            f'<em>lines {i + 1} to {j}</em></div><pre>' + "\n".join(rows) + "</pre></div>")
