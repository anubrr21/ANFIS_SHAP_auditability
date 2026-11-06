import warnings

import numpy as np
from scipy.stats import kendalltau, spearmanr, weightedtau


def _topk(v, k):
    return set(np.argsort(-np.abs(v), kind="stable")[:k])


def per_patient(A, B, k=3):
    k = min(k, A.shape[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho = [spearmanr(np.abs(a), np.abs(b)).statistic for a, b in zip(A, B)]
    over = [len(_topk(a, k) & _topk(b, k)) / k for a, b in zip(A, B)]
    return np.asarray(rho, float), np.asarray(over, float)


def compare(A, B, k=3):
    k = min(k, A.shape[1])
    rho, tau, top1, topk, sign, wtau = [], [], [], [], [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for a, b in zip(A, B):
            rho.append(spearmanr(np.abs(a), np.abs(b)).statistic)
            tau.append(kendalltau(np.abs(a), np.abs(b)).statistic)
            wtau.append(weightedtau(np.abs(a), np.abs(b)).statistic)
            ta, tb = _topk(a, k), _topk(b, k)
            top1.append(float(_topk(a, 1) == _topk(b, 1)))
            topk.append(len(ta & tb) / k)
            idx = list(tb)
            sign.append(float(np.mean(np.sign(a[idx]) == np.sign(b[idx]))))
        glob = spearmanr(np.abs(A).mean(axis=0), np.abs(B).mean(axis=0)).statistic
    return {
        "spearman_local": float(np.nanmean(rho)),
        "kendall_local": float(np.nanmean(tau)),
        "top1_match": float(np.mean(top1)),
        "top3_overlap": float(np.mean(topk)),
        "sign_agreement": float(np.mean(sign)),
        "weighted_kendall": float(np.nanmean(wtau)),
        "spearman_global": float(glob),
    }
