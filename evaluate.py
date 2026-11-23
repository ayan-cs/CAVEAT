import numpy as np
import torch

from metrics import calibration_curve, crps_ensemble, mae, picp, pinaw, rbf_mmd, rmse


@torch.no_grad()
def sample_forecast(clstm, errorc, X_left, future, n_samples=100, error_scale=0.01):
    X_left = X_left.to(next(clstm.parameters()).device)
    mu, logvar = clstm.encode(X_left)
    std = torch.exp(0.5 * logvar)
    err = error_scale * errorc.forecast(clstm.residual(X_left), future)
    out = []
    for _ in range(n_samples):
        yhat = clstm.rollout(mu + std * torch.randn_like(std), future) + err
        out.append(yhat.cpu())
    return torch.stack(out, dim=0).numpy()


def evaluate_probabilistic(clstm, errorc, dl, n_samples=100, error_scale=0.01, inverse=None, levels=None, mmd_cap=1000, return_forecasts=False):
    clstm.eval()
    errorc.eval()
    means, targets, ens_all, draws = [], [], [], []
    for X_l, X_r in dl:
        ens = sample_forecast(clstm, errorc, X_l, X_r.shape[1], n_samples=n_samples, error_scale=error_scale)
        means.append(ens.mean(axis=0))
        targets.append(X_r.numpy())
        ens_all.append(ens)
        draws.append(ens[0])

    mean = np.concatenate(means, axis=0)
    target = np.concatenate(targets, axis=0)
    ens = np.concatenate(ens_all, axis=1)
    draw = np.concatenate(draws, axis=0)

    idx = np.arange(target.shape[0])
    if len(idx) > mmd_cap:
        idx = np.random.default_rng(0).choice(len(idx), mmd_cap, replace=False)

    nominal, empirical = calibration_curve(ens[:, idx], target[idx], levels)
    out = {
        'rmse': rmse(mean, target),
        'mae': mae(mean, target),
        'mmd': rbf_mmd(target[idx], draw[idx]),
        'crps': crps_ensemble(ens[:, idx], target[idx]),
        'picp90': picp(ens[:, idx], target[idx], 0.05, 0.95),
        'pinaw90': pinaw(ens[:, idx], target[idx], 0.05, 0.95),
        'rmse_per_var': np.sqrt(np.mean((mean - target) ** 2, axis=(0, 1))).tolist(),
        'mae_per_var': np.mean(np.abs(mean - target), axis=(0, 1)).tolist(),
        'rmse_per_step': np.sqrt(np.mean((mean - target) ** 2, axis=(0, 2))).tolist(),
        'calibration': {'nominal': nominal.tolist(), 'empirical': empirical.tolist()},
        'n_windows': int(target.shape[0]),
        'n_samples': n_samples,
    }
    if inverse is not None:
        out['rmse_orig'] = rmse(inverse(mean), inverse(target))
        out['mae_orig'] = mae(inverse(mean), inverse(target))
    if return_forecasts:
        out['forecasts'] = {
            'y_true': target.astype(np.float32),
            'y_pred': mean.astype(np.float32),
            'y_q05': np.quantile(ens, 0.05, axis=0).astype(np.float32),
            'y_q95': np.quantile(ens, 0.95, axis=0).astype(np.float32),
        }
    return out
