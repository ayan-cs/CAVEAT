from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import load_json, registry_name


def dataset_dir(dataset, seed, artifacts_dir="artifacts"):
    return Path(artifacts_dir) / f"seed{seed}" / f"artifacts_{dataset}"


def load_registry(dataset, scale_mode, seed, stage="caveat", artifacts_dir="artifacts"):
    return load_json(dataset_dir(dataset, seed, artifacts_dir) / registry_name(stage, scale_mode))


def load_run(run_dir):
    run_dir = Path(run_dir)
    return {"run_dir": run_dir, "meta": load_json(run_dir / "meta.json"), "grid": pd.read_csv(run_dir / "grid_results.csv")}


def load_best(run_dir):
    d = Path(run_dir)
    if (d / "best").is_dir():
        d = d / "best"
    preds = None
    if (d / "test_predictions.npz").exists():
        with np.load(d / "test_predictions.npz") as z:
            preds = {k: z[k] for k in z.files}
    return {"dir": d, "config": load_json(d / "config.json"), "metrics": load_json(d / "metrics.json"),
            "history": load_json(d / "history.json"), "graph": load_json(d / "causal_graph.json"),
            "predictions": preds}


def load_trial(run_dir, trial):
    for p in Path(run_dir, "trials").glob("trial_*__history.json"):
        if int(p.name.split("__")[0][len("trial_"):]) == int(trial):
            return load_json(p)
    raise FileNotFoundError(f"no history for trial {trial} in {Path(run_dir, 'trials')}")


def summary_table(datasets, scale_modes, seeds, stage="caveat",
                  metrics=("auroc", "auprc", "f1", "f1_macro", "shd", "variable_usage_pct",
                           "test_rmse", "test_mae", "test_crps", "test_mmd", "test_picp90", "test_pinaw90",
                           "ood_rmse", "ood_mae", "ood_crps", "ood_mmd", "ood_picp90", "ood_pinaw90"),
                  artifacts_dir="artifacts"):
    rows = []
    for d in datasets:
        for m in scale_modes:
            for s in seeds:
                p = dataset_dir(d, s, artifacts_dir) / registry_name(stage, m)
                if p.exists():
                    sel = load_json(p)["selected"]
                    rows.append({"dataset": d, "scale_mode": m, "seed": s, **{k: sel.get(k) for k in metrics if k in sel}})
    per_seed = pd.DataFrame(rows)
    if per_seed.empty:
        return per_seed, per_seed
    cols = [c for c in metrics if c in per_seed]
    return per_seed, per_seed.groupby(["dataset", "scale_mode"])[cols].agg(["mean", "std"])


def plot_graph(gc, gt=None, title="GC", annotate=True, ax=None):
    gc = np.asarray(gc, dtype=np.float64)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(gc, cmap="Blues")
    if annotate:
        binary = np.all(np.isin(gc, (0, 1)))
        for i in range(gc.shape[0]):
            for j in range(gc.shape[1]):
                ax.text(j, i, f"{int(gc[i, j])}" if binary else f"{gc[i, j]:.2f}", ha="center", va="center")
    if gt is not None:
        gt = np.asarray(gt)
        for i, j in zip(*np.where(((gc > 0).astype(int) != gt) & (gt >= 0))):
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=1.5))
        for i, j in zip(*np.where(gt < 0)):
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="grey", hatch="//", linewidth=0))
    ax.set_xlabel("Cause")
    ax.set_ylabel("Effect")
    ax.set_title(title)
    return ax


def plot_gc_trajectories(history1, gt, target, optimal_epoch=None, start=0, ax=None, feature_names=None):
    gc_raw = np.asarray(history1["gc_raw"])[start:]
    x = np.asarray(history1["epoch"])[start:]
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    if gt is None:
        for j in range(gc_raw.shape[2]):
            ax.plot(x, gc_raw[:, target, j], alpha=0.8, label=feature_names[j] if feature_names else f"x{j}")
    else:
        gt = np.asarray(gt)
        for k, j in enumerate(np.where(gt[target] == 1)[0]):
            ax.plot(x, gc_raw[:, target, j], color="tab:blue", alpha=0.8, label="cause" if k == 0 else None)
        for k, j in enumerate(np.where(gt[target] == 0)[0]):
            ax.plot(x, gc_raw[:, target, j], color="tab:orange", alpha=0.5, label="non-cause" if k == 0 else None)
        for k, j in enumerate(np.where(gt[target] == -1)[0]):
            ax.plot(x, gc_raw[:, target, j], color="grey", alpha=0.3, label="unknown" if k == 0 else None)
    ax.plot(x, np.asarray(history1["threshold"], dtype=float)[start:], color="black", label="threshold")
    if optimal_epoch is not None:
        ax.axvline(optimal_epoch, color="green", linestyle="--", label="optimal epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Raw GC value")
    ax.set_title(f"Target {target}")
    ax.legend()
    return ax


def plot_losses(history, keys=("train_loss", "val_loss"), optimal_epoch=None, logy=True, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    for k in keys:
        ax.plot(history["epoch"], history[k], label=k)
    if optimal_epoch is not None:
        ax.axvline(optimal_epoch, color="green", linestyle="--", label="optimal epoch")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.legend()
    return ax


def plot_discovery_curves(history1, keys=("auroc", "f1", "shd", "variable_usage_pct"), optimal_epoch=None):
    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3.5))
    for ax, k in zip(np.atleast_1d(axes), keys):
        ax.plot(history1["epoch"], history1[k])
        if optimal_epoch is not None:
            ax.axvline(optimal_epoch, color="green", linestyle="--")
        ax.set_title(k)
        ax.set_xlabel("Epoch")
    fig.tight_layout()
    return fig


def plot_calibration(forecast_metrics, ax=None, label=None):
    cal = forecast_metrics["calibration"]
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":")
    ax.plot(cal["nominal"], cal["empirical"], marker="o", label=label)
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    return ax


def plot_forecast(predictions, window, variable, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3))
    h = np.arange(predictions["y_true"].shape[1])
    ax.plot(h, predictions["y_true"][window, :, variable], color="black", label="target")
    ax.plot(h, predictions["y_pred"][window, :, variable], color="tab:blue", label="mean")
    ax.fill_between(h, predictions["y_q05"][window, :, variable], predictions["y_q95"][window, :, variable],
                    color="tab:blue", alpha=0.2, label="5-95%")
    ax.set_xlabel("Horizon step")
    ax.set_title(f"window start {int(predictions['window_start'][window])}, x{variable}")
    ax.legend()
    return ax
