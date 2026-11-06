import time

import numpy as np
import shap


def rule_attribution(model, X, baseline=None):
    X = np.asarray(X, float)
    d = X.shape[1]
    base = np.zeros(d) if baseline is None else np.asarray(baseline, float)
    z, wbar, f = model.forward(X)
    consequent = np.einsum("nr,rj->nj", wbar, model.a) * (X - base)
    log_mu = model.log_membership(X) / d
    log_mu_base = model.log_membership(base[None, :]) / d
    antecedent = np.einsum("nr,nrj->nj", wbar * (f - z[:, None]), log_mu - log_mu_base)
    return {"consequent": consequent, "antecedent": antecedent, "full": consequent + antecedent}


def shap_blackbox(predict_fn, X, background, seed=0):
    d = X.shape[1]
    masker = shap.maskers.Independent(background, max_samples=len(background))
    start = time.perf_counter()
    if d <= 10:
        method = "exact"
        values = shap.Explainer(predict_fn, masker, algorithm="exact")(X, silent=True).values
    else:
        method = "permutation"
        explainer = shap.Explainer(predict_fn, masker, algorithm="permutation", seed=seed)
        values = explainer(X, max_evals=20 * (2 * d + 1), silent=True).values
    return np.asarray(values), time.perf_counter() - start, method


def shap_tree(lgbm_model, X, background):
    start = time.perf_counter()
    explainer = shap.TreeExplainer(lgbm_model, data=background,
                                   feature_perturbation="interventional", model_output="raw")
    values = explainer.shap_values(X, check_additivity=False)
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]
    return values, time.perf_counter() - start


def lime_local(predict_fn, X, background, n_samples=1000, seed=0):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    width = 0.75 * np.sqrt(d)
    mean = background.mean(axis=0)
    out = np.zeros((n, d))
    start = time.perf_counter()
    for i in range(n):
        rows = rng.integers(0, len(background), size=(n_samples, d))
        Z = background[rows, np.arange(d)]
        Z[0] = X[i]
        y = predict_fn(Z)
        dist = np.sqrt(((Z - X[i]) ** 2).sum(axis=1))
        w = np.exp(-dist ** 2 / width ** 2)
        Zc = np.column_stack([np.ones(n_samples), Z - mean])
        A = Zc.T @ (Zc * w[:, None]) + 1.0 * np.diag([0.0] + [1.0] * d)
        coef = np.linalg.solve(A, Zc.T @ (w * y))[1:]
        out[i] = coef * (X[i] - mean)
    return out, time.perf_counter() - start


def deletion_test(predict_fn, X, attributions, background, ks=(1, 2, 3), seed=0):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    attributions = dict(attributions)
    attributions["random"] = rng.random((n, d))
    z0 = predict_fn(X)
    bg = background[rng.choice(len(background), size=min(20, len(background)), replace=False)]
    rows = []
    for name, A in attributions.items():
        order = np.argsort(-np.abs(A), axis=1, kind="stable")
        for k in ks:
            if k > d:
                continue
            mask = np.zeros((n, d), bool)
            mask[np.arange(n)[:, None], order[:, :k]] = True
            Xm = np.where(mask, 0.0, X)
            rows.append({"explanation": name, "removal": "mean", "k": k,
                         "mean_abs_change": float(np.mean(np.abs(z0 - predict_fn(Xm))))})
            Xb = np.where(mask[None], bg[:, None, :], X[None])
            zb = predict_fn(Xb.reshape(-1, d)).reshape(len(bg), n).mean(axis=0)
            rows.append({"explanation": name, "removal": "background", "k": k,
                         "mean_abs_change": float(np.mean(np.abs(z0 - zb)))})
    return rows
