import numpy as np
import torch
from sklearn.metrics import (average_precision_score, f1_score, precision_score, recall_score, roc_auc_score)


def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def rmse(pred, target):
    pred = _to_np(pred).astype(np.float64)
    target = _to_np(target).astype(np.float64)
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def mae(pred, target):
    pred = _to_np(pred).astype(np.float64)
    target = _to_np(target).astype(np.float64)
    return float(np.mean(np.abs(pred - target)))


def _flatten_samples(x):
    x = _to_np(x).astype(np.float64)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    if x.ndim == 2:
        return x
    return x.reshape(x.shape[0], -1)


def _pairwise_sq_dists(a, b):
    aa = np.sum(a * a, axis=1, keepdims=True)
    bb = np.sum(b * b, axis=1, keepdims=True).T
    return np.maximum(aa + bb - 2.0 * a @ b.T, 0.0)


def rbf_mmd(real, generated, bandwidths=None):
    X = _flatten_samples(real)
    Y = _flatten_samples(generated)

    Kxx = _pairwise_sq_dists(X, X)
    Kyy = _pairwise_sq_dists(Y, Y)
    Kxy = _pairwise_sq_dists(X, Y)

    if bandwidths is None:
        med = np.median(Kxx[Kxx > 0]) if np.any(Kxx > 0) else 1.0
        bandwidths = [0.25 * med, 0.5 * med, med, 2.0 * med, 4.0 * med]

    m, n = X.shape[0], Y.shape[0]
    mmd = 0.0
    for s in bandwidths:
        g = 1.0 / (2.0 * s + 1e-12)
        kxx = np.exp(-g * Kxx)
        kyy = np.exp(-g * Kyy)
        kxy = np.exp(-g * Kxy)
        np.fill_diagonal(kxx, 0.0)
        np.fill_diagonal(kyy, 0.0)
        mmd += kxx.sum() / (m * (m - 1) + 1e-12) + kyy.sum() / (n * (n - 1) + 1e-12) - 2.0 * kxy.mean()
    return float(mmd / len(bandwidths))


def crps_ensemble(samples, target):
    samples = _to_np(samples).astype(np.float64)
    target = _to_np(target).astype(np.float64)
    S = samples.shape[0]

    abs_err = np.abs(samples - target[None, ...]).mean(axis=0)
    pair_acc = np.zeros_like(abs_err)
    for i in range(S):
        pair_acc += np.abs(samples[i][None, ...] - samples).sum(axis=0)
    pair_term = 0.5 * pair_acc / (S * S)
    return float(np.mean(abs_err - pair_term))


def prediction_interval(samples, lower=0.05, upper=0.95):
    samples = _to_np(samples).astype(np.float64)
    return np.quantile(samples, lower, axis=0), np.quantile(samples, upper, axis=0)


def picp(samples, target, lower=0.05, upper=0.95):
    target = _to_np(target).astype(np.float64)
    lo, hi = prediction_interval(samples, lower, upper)
    return float(np.mean((target >= lo) & (target <= hi)))


def pinaw(samples, target, lower=0.05, upper=0.95):
    target = _to_np(target).astype(np.float64)
    lo, hi = prediction_interval(samples, lower, upper)
    M = target.shape[-1]
    width = (hi - lo).reshape(-1, M).mean(axis=0)
    t = target.reshape(-1, M)
    rng = t.max(axis=0) - t.min(axis=0)
    rng = np.where(rng > 1e-8, rng, 1.0)
    return float(np.mean(width / rng))


def calibration_curve(samples, target, levels=None):
    if levels is None:
        levels = np.linspace(0.1, 0.9, 9)
    nominal, empirical = [], []
    for p in levels:
        lower = (1.0 - p) / 2.0
        nominal.append(p)
        empirical.append(picp(samples, target, lower, 1.0 - lower))
    return np.array(nominal), np.array(empirical)


GRAPH_METRIC_KEYS = ("auroc", "auprc", "f1", "f1_macro", "precision", "recall", "accuracy", "shd")


def graph_metrics(scores, binary, gt_matrix, include_diagonal=False):
    s = _to_np(scores).astype(np.float64)
    b = _to_np(binary).astype(np.int64)
    g = _to_np(gt_matrix).astype(np.int64)
    mask = g >= 0
    if not include_diagonal:
        mask &= ~np.eye(g.shape[0], dtype=bool)
    s, b, g = s[mask], b[mask], g[mask]
    degenerate = g.min() == g.max()
    return {
        "auroc": float("nan") if degenerate else float(roc_auc_score(g, s)),
        "auprc": float("nan") if degenerate else float(average_precision_score(g, s)),
        "f1": float(f1_score(g, b, zero_division=0)),
        "f1_macro": float(f1_score(g, b, average="macro", zero_division=0)),
        "precision": float(precision_score(g, b, zero_division=0)),
        "recall": float(recall_score(g, b, zero_division=0)),
        "accuracy": float(np.mean(g == b)),
        "shd": int(np.sum(g != b)),
    }


def causal_contrast(scores, gt_matrix):
    s = _to_np(scores).astype(np.float64)
    g = _to_np(gt_matrix).astype(np.int64)
    off = ~np.eye(s.shape[0], dtype=bool)
    causal = s[(g == 1) & off]
    non_causal = s[(g == 0) & off]
    c_mean = float(causal.mean()) if causal.size else float("nan")
    n_mean = float(non_causal.mean()) if non_causal.size else float("nan")
    return {
        "causal_mean": c_mean,
        "non_causal_mean": n_mean,
        "contrast": c_mean / n_mean if n_mean > 0 else float("inf"),
    }


def discovery_metrics(scores, binary, gt_matrix):
    if gt_matrix is None:
        return {"variable_usage_pct": 100.0 * float(np.mean(_to_np(binary)))}
    out = graph_metrics(scores, binary, gt_matrix, include_diagonal=False)
    out.update({f"{k}_incl_diag": v
                for k, v in graph_metrics(scores, binary, gt_matrix, include_diagonal=True).items()})
    out.update({f"cc_{k}": v for k, v in causal_contrast(scores, gt_matrix).items()})
    out["variable_usage_pct"] = 100.0 * float(np.mean(_to_np(binary)[_to_np(gt_matrix) >= 0]))
    return out
