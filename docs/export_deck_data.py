import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TAB = ROOT / "results" / "tables"
perf = pd.read_csv(TAB / "performance.csv")
cons_all = pd.read_csv(TAB / "consistency.csv")
if "reference" not in cons_all:
    cons_all["reference"] = "shap_anfis"
cons = cons_all[cons_all.reference == "shap_anfis"]
timing = pd.read_csv(TAB / "shap_timing.csv")
imp = pd.read_csv(TAB / "global_importance_seed0_rules4.csv")
METRICS = ["spearman_local", "kendall_local", "top1_match", "top3_overlap", "sign_agreement", "spearman_global"]
if "weighted_kendall" in cons_all:
    METRICS.append("weighted_kendall")

out = {"info": json.loads((TAB / "run_info.json").read_text()),
       "datasets": pd.read_csv(TAB / "datasets.csv").to_dict("records"),
       "perf": {}, "rules": {}, "attributes": {}, "timing": {}, "importance": {},
       "lime_ref": {}, "variants": {}, "deletion": {}}
for ds, d in perf.groupby("dataset"):
    out["perf"][ds] = {}
    for (model, k), g in d.groupby(["model", d.n_rules.fillna(0).astype(int)]):
        name = f"ANFIS-{k}" if model == "ANFIS" else model
        out["perf"][ds][name] = {m: [float(g[m].mean()), float(g[m].std())] for m in ["roc_auc", "f1", "accuracy"]}
for study, key in [("rules", "n_rules"), ("attributes", "n_attributes")]:
    c = cons[cons.study == study]
    for (ds, ex, k), g in c.groupby(["dataset", "explanation", key]):
        out[study].setdefault(ds, {}).setdefault(ex, {})[int(k)] = {m: float(g[m].mean()) for m in METRICS}
lr = cons_all[(cons_all.reference == "lime_anfis") & (cons_all.study == "rules")]
for (ds, ex, k), g in lr.groupby(["dataset", "explanation", "n_rules"]):
    out["lime_ref"].setdefault(ds, {}).setdefault(ex, {})[int(k)] = {m: float(g[m].mean()) for m in METRICS}
v = cons[cons.study == "variants"]
for (ds, mf, order), g in v.groupby(["dataset", "mf", "order"]):
    key = f"{mf}-{int(order)}"
    a = perf[(perf.dataset == ds) & (perf.model == f"ANFIS ({mf}, order {int(order)})")].roc_auc
    out["variants"].setdefault(ds, {})[key] = {
        "auc": [float(a.mean()), float(a.std())],
        **{ex: {m: float(gg[m].mean()) for m in METRICS} for ex, gg in g.groupby("explanation")}}
t = timing[timing.study == "attributes"]
for (ds, k), g in t.groupby(["dataset", "n_attributes"]):
    out["timing"].setdefault(ds, {})[int(k)] = {
        "blackbox_ms": float(g.blackbox_shap_sec_per_patient.mean() * 1000),
        "tree_ms": float(g.tree_shap_sec_per_patient.mean() * 1000),
        "lime_ms": float(g.lime_sec_per_patient.mean() * 1000) if "lime_sec_per_patient" in g else None}
for ds, g in imp.groupby("dataset"):
    out["importance"][ds] = g.sort_values("shap_anfis", ascending=False).head(6).drop(columns="dataset").to_dict("records")
if (TAB / "deletion.csv").exists():
    de = pd.read_csv(TAB / "deletion.csv")
    de = de[(de.study == "rules") & (de.n_rules == 4)]
    for (ds, rem, ex, k), g in de.groupby(["dataset", "removal", "explanation", "k"]):
        out["deletion"].setdefault(ds, {}).setdefault(rem, {}).setdefault(ex, {})[int(k)] = float(g.mean_abs_change.mean())
for name, key in [("significance.csv", "significance"), ("friedman_rules.csv", "friedman")]:
    if (TAB / name).exists():
        out[key] = pd.read_csv(TAB / name).to_dict("records")
if (TAB / "early_stage_diagnostics.csv").exists():
    out["diagnostics"] = pd.read_csv(TAB / "early_stage_diagnostics.csv").set_index("dataset").to_dict("index")
if (TAB / "per_patient.csv.gz").exists():
    pp = pd.read_csv(TAB / "per_patient.csv.gz")
    fr = pp[pp.study == "rules"].groupby(["dataset", "n_rules"]).max_firing.mean()
    out["firing_rules"] = {ds: {int(k): float(v) for k, v in g.droplevel(0).items()} for ds, g in fr.groupby(level=0)}
    fv = pp[pp.study == "variants"].groupby(["dataset", "mf", "order"]).max_firing.mean()
    out["firing_variants"] = {}
    for (ds, mf, order), v in fv.items():
        out["firing_variants"].setdefault(ds, {})[f"{mf}-{int(order)}"] = float(v)
tests = subprocess.run([sys.executable, "-W", "ignore", str(ROOT / "tests" / "test_anfis.py")],
                       capture_output=True, text=True, cwd=ROOT).stdout.strip().splitlines()
out["tests"] = tests
(ROOT / "docs" / "deck_data.json").write_text(json.dumps(out, indent=1, default=str))
print("wrote deck_data.json")
