import argparse
import json
import platform
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
import sklearn
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_recall_curve, roc_auc_score)

from src.anfis import ANFIS
from src.attribution import deletion_test, lime_local, rule_attribution, shap_blackbox, shap_tree
from src.consistency import compare, per_patient
from src.data import DATASETS, describe, make_split

OUT = Path(__file__).resolve().parent / "results"
RULES = [2, 3, 4, 6, 8, 12, 16]
SUBSET_SIZES = {"pima": [3, 4, 6, 8], "early": [4, 8, 12, 16], "cdc": [4, 8, 12, 16, 21]}
SUBSET_RULES = 4
N_EXPLAIN = 200
N_BACKGROUND = 100
VARIANTS = [("gauss", 1), ("bell", 1), ("gauss", 0), ("bell", 0)]


def train_anfis(sp, n_rules, seed, mf="gauss", order=1):
    big = len(sp.y_train) > 20000
    model = ANFIS(n_rules=n_rules, seed=seed, lr=0.01, mf=mf, order=order,
                  epochs=40 if big else 300, batch_size=1024 if big else 32,
                  patience=6 if big else 30)
    start = time.perf_counter()
    model.fit(sp.X_train, sp.y_train, sp.X_val, sp.y_val)
    return model, time.perf_counter() - start


def train_lgbm(sp, seed):
    big = len(sp.y_train) > 20000
    model = lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=0.03, num_leaves=31 if big else 7,
        min_child_samples=50 if big else 10, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=1.0, random_state=seed, n_jobs=1, verbose=-1)
    start = time.perf_counter()
    model.fit(sp.X_train, sp.y_train, eval_X=sp.X_val, eval_y=sp.y_val,
              callbacks=[lgb.early_stopping(50, verbose=False)])
    return model, time.perf_counter() - start


def score(y_val, p_val, y_test, p_test):
    prec, rec, thr = precision_recall_curve(y_val, p_val)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    t = float(thr[np.argmax(f1)])
    pred = (p_test >= t).astype(int)
    return {"accuracy": accuracy_score(y_test, pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, pred),
            "f1": f1_score(y_test, pred), "roc_auc": roc_auc_score(y_test, p_test),
            "threshold": t}


def audit(sp, anfis, lgbm, explain_idx, background, seed):
    Xe = sp.X_test[explain_idx]
    rule = rule_attribution(anfis, Xe)
    if anfis.order == 0:
        rule = {"antecedent": rule["antecedent"], "full": rule["full"]}
    shap_a, t_black, method = shap_blackbox(anfis.decision_function, Xe, background, seed)
    shap_l, t_tree = shap_tree(lgbm, Xe, background)
    lime_a, t_lime = lime_local(anfis.decision_function, Xe, background, seed=seed)
    explanations = {**{f"rule_{k}": v for k, v in rule.items()}, "lime_anfis": lime_a, "shap_lgbm": shap_l}
    rows = [{"explanation": name, "reference": "shap_anfis", **compare(vals, shap_a)}
            for name, vals in explanations.items()]
    rows += [{"explanation": f"rule_{k}", "reference": "lime_anfis", **compare(v, lime_a)} for k, v in rule.items()]

    absv = np.abs(shap_a)
    top3_share = np.sort(absv, axis=1)[:, ::-1][:, :3].sum(axis=1) / np.clip(absv.sum(axis=1), 1e-12, None)
    max_fire = anfis.forward(Xe)[1].max(axis=1)
    pp = []
    for name, vals in explanations.items():
        rho, over = per_patient(vals, shap_a)
        pp += [{"explanation": name, "patient": int(i), "spearman": float(r_), "top3": float(o_),
                "shap_top3_share": float(t_), "max_firing": float(m_)}
               for i, r_, o_, t_, m_ in zip(explain_idx, rho, over, top3_share, max_fire)]
    dele = deletion_test(anfis.decision_function, Xe,
                         {**{f"rule_{k}": v for k, v in rule.items()}, "shap_anfis": shap_a, "lime_anfis": lime_a},
                         background, seed=seed)
    timing = {"shap_method_anfis": method,
              "blackbox_shap_sec_per_patient": t_black / len(Xe),
              "tree_shap_sec_per_patient": t_tree / len(Xe),
              "lime_sec_per_patient": t_lime / len(Xe)}
    return rows, timing, {"shap_anfis": shap_a, **explanations}, pp, dele


def run_one(name, seed):
    sp = make_split(name, seed)
    rng = np.random.default_rng(seed)
    explain_idx = rng.choice(len(sp.y_test), size=min(N_EXPLAIN, len(sp.y_test)), replace=False)
    background = sp.X_train[rng.choice(len(sp.y_train), size=N_BACKGROUND, replace=False)]
    d = len(sp.features)
    tag = {"dataset": name, "seed": seed, "n_attributes": d}
    perf, cons, timing, pp_all, del_all = [], [], [], [], []

    lgbm, t_fit = train_lgbm(sp, seed)
    perf.append({**tag, "model": "LightGBM", "n_rules": np.nan, "train_seconds": t_fit,
                 **score(sp.y_val, lgbm.predict_proba(sp.X_val)[:, 1],
                         sp.y_test, lgbm.predict_proba(sp.X_test)[:, 1])})
    start = time.perf_counter()
    logreg = LogisticRegression(max_iter=2000).fit(sp.X_train, sp.y_train)
    perf.append({**tag, "model": "LogisticRegression", "n_rules": np.nan,
                 "train_seconds": time.perf_counter() - start,
                 **score(sp.y_val, logreg.predict_proba(sp.X_val)[:, 1],
                         sp.y_test, logreg.predict_proba(sp.X_test)[:, 1])})

    saved = {}
    for R in RULES:
        anfis, t_fit = train_anfis(sp, R, seed)
        perf.append({**tag, "model": "ANFIS", "n_rules": R, "train_seconds": t_fit,
                     "epochs_run": anfis.epochs_run,
                     **score(sp.y_val, anfis.predict_proba(sp.X_val),
                             sp.y_test, anfis.predict_proba(sp.X_test))})
        rows, tm, attrs, pp, dele = audit(sp, anfis, lgbm, explain_idx, background, seed)
        max_fire = float(anfis.forward(sp.X_test[explain_idx])[1].max(axis=1).mean())
        st = {**tag, "study": "rules", "n_rules": R}
        cons += [{**st, "mean_max_firing": max_fire, **r} for r in rows]
        timing.append({**st, **tm})
        pp_all += [{**st, **r} for r in pp]
        del_all += [{**st, **r} for r in dele]
        if R == SUBSET_RULES:
            anfis_4 = anfis
        if seed == 0:
            saved[R] = {"params": anfis.get_params(), "attributions": attrs}

    variants = {}
    for mf, order in VARIANTS:
        if (mf, order) == ("gauss", 1):
            anfis, t_fit = anfis_4, np.nan
        else:
            anfis, t_fit = train_anfis(sp, SUBSET_RULES, seed, mf=mf, order=order)
        st = {**tag, "study": "variants", "n_rules": SUBSET_RULES, "mf": mf, "order": order}
        perf.append({**tag, "model": f"ANFIS ({mf}, order {order})", "n_rules": SUBSET_RULES, "mf": mf,
                     "order": order, "train_seconds": t_fit,
                     **score(sp.y_val, anfis.predict_proba(sp.X_val), sp.y_test, anfis.predict_proba(sp.X_test))})
        rows, tm, _, pp, dele = audit(sp, anfis, lgbm, explain_idx, background, seed)
        cons += [{**st, **r} for r in rows]
        pp_all += [{**st, **r} for r in pp]
        del_all += [{**st, **r} for r in dele]
        if seed == 0:
            variants[f"{mf}-{order}"] = anfis.get_params()

    sub = rng.choice(len(sp.y_train), size=min(20000, len(sp.y_train)), replace=False)
    mi = mutual_info_classif(sp.X_train[sub], sp.y_train[sub], random_state=seed)
    ranking = np.argsort(-mi)
    for k in SUBSET_SIZES[name]:
        sp_k = sp.select(sorted(ranking[:k]))
        bg_k = background[:, sorted(ranking[:k])]
        anfis, _ = train_anfis(sp_k, SUBSET_RULES, seed)
        lgbm_k, _ = train_lgbm(sp_k, seed)
        auc_a = roc_auc_score(sp_k.y_test, anfis.predict_proba(sp_k.X_test))
        auc_l = roc_auc_score(sp_k.y_test, lgbm_k.predict_proba(sp_k.X_test)[:, 1])
        rows, tm, _, pp, dele = audit(sp_k, anfis, lgbm_k, explain_idx, bg_k, seed)
        st = {**tag, "study": "attributes", "n_attributes": k, "n_rules": SUBSET_RULES}
        cons += [{**st, "anfis_auc": auc_a, "lgbm_auc": auc_l, **r} for r in rows]
        pp_all += [{**st, **r} for r in pp]
        del_all += [{**st, **r} for r in dele]
        timing.append({**tag, "study": "attributes", "n_attributes": k,
                       "n_rules": SUBSET_RULES, **tm})

    extra = {}
    if seed == 0:
        if name == "cdc":
            start = time.perf_counter()
            shap.TreeExplainer(lgbm).shap_values(sp.X_test, check_additivity=False)
            extra = {"cdc_full_test_rows": int(len(sp.X_test)),
                     "cdc_full_test_treeshap_seconds": time.perf_counter() - start}
        keep = slice(0, 3000)
        joblib.dump({"name": name, "features": sp.features, "mean": sp.mean, "std": sp.std,
                     "X_test": sp.X_test[keep], "y_test": sp.y_test[keep],
                     "background": background, "explain_idx": explain_idx,
                     "mi_ranking": ranking, "anfis": saved, "variants": variants, "lgbm": lgbm,
                     "split": sp if name != "cdc" else None},
                    OUT / "models" / f"{name}_seed0.joblib", compress=3)
    return perf, cons, timing, extra, pp_all, del_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    seeds = [0] if args.quick else [0, 1, 2, 3, 4]
    for sub in ["tables", "models", "figures"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    pd.DataFrame([describe(n) for n in DATASETS]).to_csv(OUT / "tables" / "datasets.csv", index=False)
    start = time.perf_counter()
    tasks = [(n, s) for s in seeds for n in DATASETS]
    out = joblib.Parallel(n_jobs=args.jobs, verbose=5)(joblib.delayed(run_one)(n, s) for n, s in tasks)

    perf = pd.DataFrame([r for o in out for r in o[0]])
    cons = pd.DataFrame([r for o in out for r in o[1]])
    timing = pd.DataFrame([r for o in out for r in o[2]])
    perf.to_csv(OUT / "tables" / "performance.csv", index=False)
    cons.to_csv(OUT / "tables" / "consistency.csv", index=False)
    timing.to_csv(OUT / "tables" / "shap_timing.csv", index=False)
    pd.DataFrame([r for o in out for r in o[4]]).to_csv(OUT / "tables" / "per_patient.csv.gz", index=False)
    pd.DataFrame([r for o in out for r in o[5]]).to_csv(OUT / "tables" / "deletion.csv", index=False)
    extra = {k: v for o in out for k, v in o[3].items()}
    meta = {"seeds": seeds, "rules": RULES, "subset_sizes": SUBSET_SIZES, "variants": VARIANTS,
            "n_explain": N_EXPLAIN, "n_background": N_BACKGROUND,
            "total_runtime_seconds": time.perf_counter() - start,
            "python": platform.python_version(), "numpy": np.__version__,
            "scikit_learn": sklearn.__version__, "lightgbm": lgb.__version__,
            "shap": shap.__version__, "machine": platform.processor(), **extra}
    (OUT / "tables" / "run_info.json").write_text(json.dumps(meta, indent=2))
    print(f"done in {meta['total_runtime_seconds']:.0f} s, tables written to {OUT / 'tables'}")


if __name__ == "__main__":
    main()
