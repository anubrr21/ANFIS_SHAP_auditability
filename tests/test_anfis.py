import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.anfis import ANFIS
from src.attribution import rule_attribution, shap_blackbox
from src.data import DATASETS, make_split


def test_gradients():
    for mf in ("gauss", "bell"):
        for order in (1, 0):
            check_gradients(mf, order)


def check_gradients(mf, order):
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(40, 5)), rng.integers(0, 2, 40).astype(float)
    model = ANFIS(n_rules=3, mf=mf, order=order)
    model.c, model.log_s = rng.normal(size=(3, 5)), rng.normal(scale=0.3, size=(3, 5))
    model.a, model.b = rng.normal(size=(3, 5)), rng.normal(size=3)
    if order == 0:
        model.a = np.zeros((3, 5))
    _, grads = model.loss_and_grads(X, y)
    worst = 0.0
    for name in (["a"] if order else []) + ["b", "c", "log_s"]:
        p = getattr(model, name)
        num = np.zeros_like(p)
        for i in np.ndindex(p.shape):
            old = p[i]
            p[i] = old + 1e-6
            up = model.loss_and_grads(X, y)[0]
            p[i] = old - 1e-6
            down = model.loss_and_grads(X, y)[0]
            p[i] = old
            num[i] = (up - down) / 2e-6
        err = np.max(np.abs(num - grads[name]))
        worst = max(worst, err)
        assert err < 1e-6, f"gradient mismatch for {name}: {err}"
    print(f"gradient check passed for {mf} memberships, order {order} (largest difference {worst:.2e})")


def test_consequent_identity():
    sp = make_split("pima", 0)
    model = ANFIS(n_rules=4, epochs=30).fit(sp.X_train, sp.y_train, sp.X_val, sp.y_val)
    z, wbar, _ = model.forward(sp.X_test)
    cons = rule_attribution(model, sp.X_test)["consequent"]
    assert np.allclose(cons.sum(axis=1) + wbar @ model.b, z)
    print("consequent attribution reconstructs the model output exactly")


def test_shap_additivity():
    sp = make_split("pima", 0)
    model = ANFIS(n_rules=4, epochs=30).fit(sp.X_train, sp.y_train, sp.X_val, sp.y_val)
    bg = sp.X_train[:100]
    values, _, method = shap_blackbox(model.decision_function, sp.X_test[:20], bg)
    gap = values.sum(axis=1) - (model.decision_function(sp.X_test[:20])
                                - model.decision_function(bg).mean())
    assert method == "exact" and np.max(np.abs(gap)) < 1e-8
    print(f"SHAP additivity holds (largest gap {np.max(np.abs(gap)):.2e})")


def test_no_leakage_and_shapes():
    for name in DATASETS:
        sp = make_split(name, 0)
        assert not np.isnan(sp.X_train).any() and not np.isnan(sp.X_test).any()
        assert np.allclose(sp.X_train.mean(axis=0), 0, atol=1e-8)
        n = len(sp.y_train) + len(sp.y_val) + len(sp.y_test)
        print(f"{name}: {n} records, {len(sp.features)} attributes, "
              f"train/val/test = {len(sp.y_train)}/{len(sp.y_val)}/{len(sp.y_test)}")


if __name__ == "__main__":
    test_gradients()
    test_consequent_identity()
    test_shap_additivity()
    test_no_leakage_and_shapes()
    print("all checks passed")
