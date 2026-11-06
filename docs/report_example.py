import sys

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr

from report_common import ROOT

sys.path.insert(0, str(ROOT))
from src.anfis import ANFIS
from src.attribution import lime_local, rule_attribution, shap_blackbox, shap_tree
from src.consistency import compare

SCREENS = ROOT / "docs" / "screens"


def _model(bundle, rules=4):
    m = ANFIS(n_rules=rules)
    m.set_params(bundle["anfis"][rules]["params"])
    return m


def worked_example(patient=0):
    b = joblib.load(ROOT / "results" / "models" / "pima_seed0.joblib")
    m = _model(b)
    F = b["features"]
    x = b["X_test"][patient:patient + 1]
    orig = x[0] * b["std"] + b["mean"]
    lm = m.log_membership(x)[0]
    mu = np.exp(lm)
    w = np.exp(lm.mean(axis=1))
    z, wbar, f = m.forward(x)
    rule = rule_attribution(m, x)
    shap_a, _, _ = shap_blackbox(m.decision_function, x, b["background"])
    lime_a, _ = lime_local(m.decision_function, x, b["background"])
    tree_a, _ = shap_tree(b["lgbm"], x, b["background"])
    base_z = float(m.decision_function(b["background"]).mean())
    e = {"F": F, "x": x[0], "orig": orig, "mu": mu, "w": w, "wbar": wbar[0], "f": f[0], "z": float(z[0]),
         "p": float(1 / (1 + np.exp(-z[0]))), "a": m.a, "b": m.b, "c": m.c * b["std"] + b["mean"], "cz": m.c,
         "sigma": m.sigma, "then": rule["consequent"][0], "if": rule["antecedent"][0], "full": rule["full"][0],
         "shap": shap_a[0], "lime": lime_a[0], "tree": tree_a[0], "base_z": base_z, "label": int(b["y_test"][patient]),
         "p_lgbm": float(b["lgbm"].predict_proba(x)[0, 1]), "mean": b["mean"], "std": b["std"]}
    e["metrics"] = {k: compare(v[None, :], shap_a) for k, v in
                    [("THEN-part", e["then"]), ("IF-part", e["if"]), ("IF + THEN", e["full"]), ("LIME", e["lime"])]}
    return e


def disagreement_example(patient=25):
    b = joblib.load(ROOT / "results" / "models" / "pima_seed0.joblib")
    m = _model(b)
    x = b["X_test"][patient:patient + 1]
    rule = rule_attribution(m, x)
    shap_a, _, _ = shap_blackbox(m.decision_function, x, b["background"])
    _, wbar, _ = m.forward(x)
    rank = lambda v: pd.Series(-np.abs(v)).rank(method="first").astype(int).to_numpy()
    t = pd.DataFrame({"Attribute": b["features"],
                      "THEN-part φ": np.round(rule["consequent"][0], 3), "Rank (THEN)": rank(rule["consequent"][0]),
                      "IF-part φ": np.round(rule["antecedent"][0], 3),
                      "SHAP φ": np.round(shap_a[0], 3), "Rank (SHAP)": rank(shap_a[0])}).sort_values("Rank (SHAP)")
    rho_then = spearmanr(np.abs(rule["consequent"][0]), np.abs(shap_a[0])).statistic
    rho_full = spearmanr(np.abs(rule["full"][0]), np.abs(shap_a[0])).statistic
    return t, float(rho_then), float(rho_full), np.round(wbar[0], 3)


def htsk_demo(n=2000):
    b = joblib.load(ROOT / "results" / "models" / "cdc_seed0.joblib")
    m = _model(b)
    lm = m.log_membership(b["X_test"][:n])
    prod = np.exp(lm.sum(axis=2))
    geo = np.exp(lm.mean(axis=2))
    share = lambda w: (w / w.sum(axis=1, keepdims=True))
    one = lm[0]
    return {"prod_one": np.exp(one.sum(axis=1)), "prod_share_one": share(np.exp(one.sum(axis=1))[None])[0],
            "geo_one": np.exp(one.mean(axis=1)), "geo_share_one": share(np.exp(one.mean(axis=1))[None])[0],
            "prod_max_share": float(share(prod).max(axis=1).mean()), "geo_max_share": float(share(geo).max(axis=1).mean()),
            "n": n}


CROPS = {
    "r_controls": ("patient_pima.png", (60, 270, 412, 1190)),
    "r_charts": ("patient_pima.png", (415, 285, 1425, 550)),
    "r_agree": ("patient_pima.png", (415, 555, 1425, 820)),
    "r_delete": ("patient_pima.png", (415, 835, 1425, 1160)),
    "r_rules": ("patient_pima.png", (415, 1180, 1425, 1680)),
    "r_header": ("patient_pima.png", (60, 100, 1440, 270)),
    "r_variant_rules": ("patient_variant.png", (415, 1195, 1440, 1730)),
    "r_data": ("data.png", (60, 250, 1440, 1400)),
}


def make_crops():
    out = {}
    for name, (src, box) in CROPS.items():
        im = Image.open(SCREENS / src)
        box = (box[0], box[1], min(box[2], im.width), min(box[3], im.height))
        path = SCREENS / f"{name}.png"
        im.crop(box).save(path)
        out[name] = path
    return out
