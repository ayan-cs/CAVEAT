import argparse
import csv
import gc
import os
import pickle
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

import config
from data import paths as data_paths
from data.datasets import FrozenDataset
from evaluate import evaluate_probabilistic
from metrics import discovery_metrics
from model import CausalLSTM, ErrorCompensation
from services import Criterion, EarlyStopper, ProximalGradientDescent
from utils import (Timer, build_run_name, columns, env_info, fmt_seconds, get_logger, load_json,
                   registry_name, resolve_device, run_prefix, save_json, set_deterministic, timestamp,
                   to_loader, unfinished_run, unique_dir, update_registry)

STAGE = "caveat"
CHECKPOINT = "best_so_far.pt"
SCALE_MODES = ("std", "norm", "raw")
LOSS_KEYS = ("loss", "mse", "kl")
FORECAST_KEYS = ("rmse", "mae", "crps", "mmd", "picp90", "pinaw90", "rmse_orig", "mae_orig")
REGISTRY_DISCOVERY_KEYS = ("auroc", "auprc", "f1", "f1_macro", "shd", "auroc_incl_diag", "f1_macro_incl_diag", "shd_incl_diag", "variable_usage_pct")


class _Running:
    def __init__(self):
        self.sums, self.n = {}, 0

    def add(self, loss, parts, bs):
        for k, v in {"loss": loss, **parts}.items():
            v = v.item() if torch.is_tensor(v) else float(v)
            self.sums[k] = self.sums.get(k, 0.0) + v * bs
        self.n += bs

    def mean(self):
        return {k: v / self.n for k, v in self.sums.items()}


def _prefixed(prefix, d):
    return {f"{prefix}_{k}": v for k, v in d.items()}


def _gt_summary(d):
    return f" | auroc {d['auroc']:.3f} f1 {d['f1']:.3f} shd {d['shd']}" if "auroc" in d else ""


def _cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


@torch.no_grad()
def loss_pass(model, dl, criterion, device):
    model.eval()
    agg = _Running()
    for X_l, X_r in dl:
        X_l, X_r = X_l.to(device), X_r.to(device)
        pred, mu, logvar = model(X_l, X_r)
        loss, parts = criterion(pred, X_r, mu, logvar, model)
        agg.add(loss, parts, X_l.shape[0])
    return agg.mean()


def phase1_epoch(model, train_dl, val_dl, prox, criterion, device):
    model.train()
    tr = _Running()
    for X_l, X_r in train_dl:
        prox.zero_grad()
        X_l, X_r = X_l.to(device), X_r.to(device)
        pred, mu, logvar = model(X_l, X_r)
        loss, parts = criterion(pred, X_r, mu, logvar, model)
        loss.backward()
        tr.add(loss, parts, X_l.shape[0])
        prox.step(model)
    return tr.mean(), loss_pass(model, val_dl, criterion, device)


def phase2_epoch(clstm, errorc, train_dl, val_dl, optimizer, criterion_e, device):
    clstm.eval()
    errorc.train()
    tr = _Running()
    for X_l, X_r in train_dl:
        optimizer.zero_grad()
        X_l, X_r = X_l.to(device), X_r.to(device)
        with torch.no_grad():
            pred, _, _ = clstm(X_l, X_r)
        error = X_r - pred
        pred_e, mu_e, logvar_e = errorc(error)
        loss_e, parts = criterion_e(pred_e, error, mu_e, logvar_e, errorc)
        loss_e.backward()
        optimizer.step()
        tr.add(loss_e, parts, X_l.shape[0])
    return tr.mean(), errorc_loss_pass(clstm, errorc, val_dl, criterion_e, device)


@torch.no_grad()
def errorc_loss_pass(clstm, errorc, dl, criterion_e, device):
    clstm.eval()
    errorc.eval()
    agg = _Running()
    for X_l, X_r in dl:
        X_l, X_r = X_l.to(device), X_r.to(device)
        pred, _, _ = clstm(X_l, X_r)
        error = X_r - pred
        pred_e, mu_e, logvar_e = errorc(error)
        loss_e, parts = criterion_e(pred_e, error, mu_e, logvar_e, errorc)
        agg.add(loss_e, parts, X_l.shape[0])
    return agg.mean()


def train_phase2(clstm, errorc, train_dl, val_dl, optimizer, criterion_e, epochs, phase_cfg, device,
                 logger, tag="phase2"):
    stopper = EarlyStopper(patience=phase_cfg["patience"])
    scheduler = plateau_scheduler(optimizer, phase_cfg)
    rows, best_val, best_epoch, best_state = [], np.inf, 0, None
    with Timer() as t:
        for epoch in range(epochs):
            with Timer() as et:
                tr, va = phase2_epoch(clstm, errorc, train_dl, val_dl, optimizer, criterion_e, device)
            lr = optimizer.param_groups[0]["lr"]
            scheduler.step(va["loss"])
            rows.append({"epoch": epoch + 1, "lr": lr, **_prefixed("train", tr), **_prefixed("val", va),
                         "epoch_time_s": et.elapsed})
            if epoch % config.TRAIN["log_every"] == 0:
                logger.info(f"    [{tag}] epoch {epoch + 1:5d} | train {tr['loss']:.6f} | val {va['loss']:.6f} "
                            f"| lr {lr:.2e}")
            if va["loss"] < best_val:
                best_val, best_epoch, best_state = va["loss"], epoch + 1, _cpu_state(errorc)
            if stopper.early_stop(va["loss"]):
                logger.info(f"    [{tag}] early stop at epoch {epoch + 1} (best epoch {best_epoch})")
                break
    errorc.load_state_dict(best_state)
    return rows, best_epoch, t.elapsed


def evaluate_forecasts(clstm, errorc, splits, batch_size, inverse):
    ev = config.EVAL
    results, forecasts = {}, None
    for split in (s for s in ev["splits"] if s in splits):
        m = evaluate_probabilistic(clstm, errorc, to_loader(splits[split], batch_size),
                                   n_samples=ev["n_samples"], error_scale=ev["error_scale"],
                                   inverse=inverse, return_forecasts=(split == "test"))
        forecasts = m.pop("forecasts", forecasts)
        results[split] = m
    return results, forecasts


def plateau_scheduler(optimizer, phase_cfg):
    return ReduceLROnPlateau(optimizer, mode="min", factor=phase_cfg["lr_factor"],
                             patience=phase_cfg["lr_patience"], min_lr=phase_cfg["min_lr"])


def gmm_params(gmm):
    if gmm is None:
        return None
    means = gmm.means_.flatten()
    return {"weights": gmm.weights_.tolist(), "means": means.tolist(),
            "covariances": gmm.covariances_.flatten().tolist(),
            "noise_component": int(np.argmin(means)), "signal_component": int(np.argmax(means)),
            "converged": bool(gmm.converged_), "n_iter": int(gmm.n_iter_)}


def run_trial(hp, ds, splits, seed, device, logger):
    set_deterministic(seed)
    tc = config.TRAIN
    gt = ds.adjacency
    M = ds.n_vars

    train_dl = to_loader(splits['train'], hp['batch_size'], shuffle=True)
    val_dl = to_loader(splits['val'], hp['batch_size'], shuffle=False)

    clstm = CausalLSTM(M, hp['hidden_size'], np.ones([M, M])).to(device)
    prox = ProximalGradientDescent(clstm.parameters(), lr=hp['lr'], lam=hp['lam'])
    criterion = Criterion(hp['beta_kl'], hp['lam_ridge'], model='clstm')
    stopper = EarlyStopper(patience=tc["phase1"]["patience"])
    scheduler = plateau_scheduler(prox, tc["phase1"])

    rows1, best_val, best_epoch, best_state = [], np.inf, 0, None
    with Timer() as t1:
        for epoch in range(tc["epochs"]):
            with Timer() as et:
                tr, va = phase1_epoch(clstm, train_dl, val_dl, prox, criterion, device)
            lr = prox.param_groups[0]["lr"]
            scheduler.step(va["loss"])
            raw, binary, thr = clstm.get_causal_matrix()
            raw, binary = raw.numpy(), binary.numpy()
            dm = discovery_metrics(raw, binary, gt)
            rows1.append({"epoch": epoch + 1, "lr": lr, **_prefixed("train", tr), **_prefixed("val", va),
                          "group_norm": float(raw.sum()), "threshold": thr, **dm,
                          "gc_raw": raw, "gc_est": binary, "epoch_time_s": et.elapsed})
            if epoch % tc["log_every"] == 0:
                logger.info(
                    f"    [phase1] epoch {epoch + 1:5d} | train {tr['loss']:.6f} | val {va['loss']:.6f} "
                    f"(mse {va['mse']:.6f}) | usage {dm['variable_usage_pct']:.2f}% (thr={thr:.4g})"
                    f"{_gt_summary(dm)} | lr {lr:.2e} | {et.elapsed:.2f}s/epoch")
            if va["loss"] < best_val:
                best_val, best_epoch, best_state = va["loss"], epoch + 1, _cpu_state(clstm)
            if stopper.early_stop(va["loss"]):
                logger.info(f"    [phase1] early stop at epoch {epoch + 1} (best epoch {best_epoch})")
                break

    clstm.load_state_dict(best_state)
    for p in clstm.parameters():
        p.requires_grad = False
    best = rows1[best_epoch - 1]
    raw, binary, thr, gmm = clstm.get_causal_matrix(return_gmm=True)
    assert np.allclose(raw.numpy(), best["gc_raw"]), "reloaded weights do not reproduce the best epoch's graph"
    test_loss = loss_pass(clstm, to_loader(splits['test'], hp['batch_size']), criterion, device)

    errorc = ErrorCompensation(M, hp['hidden_size']).to(device)
    adam = torch.optim.Adam(errorc.parameters(), lr=hp['lr'])
    criterion_e = Criterion(hp['beta_e'], hp['lam_ridge'], model='errorc')
    rows2, best_epoch2, t2 = train_phase2(clstm, errorc, train_dl, val_dl, adam, criterion_e,
                                          tc["epochs"], tc["phase2"], device, logger)
    best2 = rows2[best_epoch2 - 1]

    with Timer() as te:
        forecast, forecasts = evaluate_forecasts(clstm, errorc, splits, hp['batch_size'], ds.inverse_transform)
    logger.info("    " + " | ".join(f"{s} rmse {m['rmse']:.6f} crps {m['crps']:.6f}" for s, m in forecast.items())
                + f" | {fmt_seconds(te.elapsed)}")

    metrics = {
        "selection": {"metric": tc["select_metric"], "score": best[tc["select_metric"]],
                      "best_epoch": best_epoch},
        "epochs": {"phase1_run": len(rows1), "phase1_best": best_epoch,
                   "phase2_run": len(rows2), "phase2_best": best_epoch2},
        "loss": {"train": {k: best[f"train_{k}"] for k in LOSS_KEYS},
                 "val": {k: best[f"val_{k}"] for k in LOSS_KEYS},
                 "test": {k: test_loss[k] for k in LOSS_KEYS}},
        "phase2_loss": {"train": {k: best2[f"train_{k}"] for k in LOSS_KEYS},
                        "val": {k: best2[f"val_{k}"] for k in LOSS_KEYS}},
        "discovery": {"threshold": thr, **discovery_metrics(raw.numpy(), binary.numpy(), gt)},
        "forecast": forecast,
        "timing": {"phase1_s": t1.elapsed, "phase2_s": t2, "eval_s": te.elapsed,
                   "avg_epoch_phase1_s": t1.elapsed / len(rows1), "avg_epoch_phase2_s": t2 / len(rows2)},
    }
    graph = {"GC_raw": raw.numpy(), "GC_est": binary.numpy(), "GC_true": gt,
             "orientation": "A[i, j] = 1 means j causes i (rows are effects)", "feature_names": ds.feature_names,
             "threshold": thr, "gmm": gmm_params(gmm)}
    out = {"metrics": metrics, "rows1": rows1, "rows2": rows2, "graph": graph, "gmm": gmm,
           "clstm_state": _cpu_state(clstm), "errorc_state": _cpu_state(errorc), "forecasts": forecasts}
    del clstm, errorc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def trial_row(trial, hp, m):
    row = {"trial": trial, "status": "OK", **{f"hp_{k}": v for k, v in hp.items()},
           f"select_{m['selection']['metric']}": m["selection"]["score"], **m["epochs"],
           "train_time_s": round(m["timing"]["phase1_s"] + m["timing"]["phase2_s"], 2)}
    for split, d in m["loss"].items():
        row.update({f"{split}_{k}": v for k, v in d.items()})
    row.update({f"phase2_{split}_loss": d["loss"] for split, d in m["phase2_loss"].items()})
    row.update(m.get("discovery", {}))
    for split, d in m["forecast"].items():
        row.update({f"{split}_{k}": d.get(k) for k in FORECAST_KEYS})
    return row


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    tmp = Path(path).with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def save_checkpoint(trials_dir, best):
    path = Path(trials_dir) / CHECKPOINT
    tmp = path.with_suffix(".pt.tmp")
    torch.save(best, tmp)
    os.replace(tmp, path)


def resume_grid(run_dir, trials, select_metric, logger):
    with open(run_dir / "grid_results.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        i = int(r["trial"])
        if i > len(trials) or any(r.get(f"hp_{k}") != str(v) for k, v in trials[i - 1].items()):
            raise RuntimeError(f"{run_dir} was run with a different hyperparameter grid (trial {i}); "
                               f"restore that grid in config.py or set RESUME = False")
    failed = [int(r["trial"]) for r in rows if r["status"] == "FAILED"]
    ok = [r for r in rows if r["status"] == "OK"]
    col, best = f"select_{select_metric}", None
    if ok:
        if col not in ok[0]:
            raise RuntimeError(f"{run_dir} was selected by a different metric (no column {col!r}); "
                               f"restore select_metric in config.py or set RESUME = False")
        top = min(ok, key=lambda r: float(r[col]))
        trial = int(top["trial"])
        ckpt = run_dir / "trials" / CHECKPOINT
        if ckpt.exists():
            saved = torch.load(ckpt, map_location="cpu", weights_only=False)
            if saved["trial"] == trial:
                best = saved
        if best is None:
            best = {"score": float(top[col]), "trial": trial, "hp": trials[trial - 1], "out": None}
    logger.info(f"[resume] {len(rows)}/{len(trials)} trials already done ({len(failed)} failed)"
                + (f" | best so far: trial {best['trial']} ({col}={best['score']:.6f})" if best else ""))
    return rows, failed, best


def save_best(best_dir, out, hp, trial, ds, run_info):
    best_dir = Path(best_dir)
    best_dir.mkdir(parents=True, exist_ok=True)
    m = out["metrics"]
    torch.save({"clstm_state_dict": out["clstm_state"], "errorc_state_dict": out["errorc_state"],
                "model_hp": {"num_features": ds.n_vars, "hidden_size": hp["hidden_size"]},
                "causal_graph": np.ones([ds.n_vars, ds.n_vars]), "hp": hp, "trial": trial, "metrics": m},
               best_dir / "model.pt")
    if out["gmm"] is not None:
        with open(best_dir / "gmm.pkl", "wb") as f:
            pickle.dump(out["gmm"], f)
    save_json(out["graph"], best_dir / "causal_graph.json")
    save_json({"hp": hp, "trial": trial, **run_info,
               "select_metric": m["selection"]["metric"], "select_score": m["selection"]["score"],
               "train_config": config.TRAIN, "eval_config": config.EVAL}, best_dir / "config.json")
    save_json({"phase1": columns(out["rows1"]), "phase2": columns(out["rows2"])}, best_dir / "history.json")
    save_json(m, best_dir / "metrics.json")
    if out["forecasts"] is not None:
        np.savez_compressed(best_dir / "test_predictions.npz", window_start=ds.window_starts("test"),
                            norm_mean=ds.norm_mean, norm_std=ds.norm_std, **out["forecasts"])


def run_grid(args):
    dataset, scale_mode, seed = args.dataset, args.scale_mode, args.seed
    trials = config.expand_grid()
    ds = FrozenDataset(dataset, scale_mode, args.data_dir)
    splits = ds.as_splits()
    sizes = {s: len(splits[s][0]) for s in ds.splits}
    run_info = {"dataset": dataset, "scale_mode": scale_mode, "seed": seed,
                "npz": data_paths.dataset_npz(dataset, scale_mode, args.data_dir),
                "has_ground_truth": ds.has_ground_truth, "feature_names": ds.feature_names,
                "data": load_json(data_paths.dataset_manifest(dataset, scale_mode, args.data_dir))}

    ds_dir = Path(args.artifacts_dir) / f"artifacts_{dataset}"
    prefix = run_prefix(STAGE, scale_mode, seed)
    run_dir = unfinished_run(ds_dir, [f"{prefix}__*"]) if args.resume else None
    resumed = run_dir is not None
    run_dir = run_dir or unique_dir(ds_dir / build_run_name(prefix))
    trials_dir = run_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)
    width = max(2, len(str(len(trials))))

    logger = get_logger(f"caveat.{dataset}.{scale_mode}.seed{seed}", run_dir / "run.log")
    device = resolve_device(args.device)
    logger.info("=" * 88)
    logger.info(f"CAVEAT grid search | dataset={dataset} | scale_mode={scale_mode} "
                f"| seed={seed} "
                f"| device={device} | started {timestamp()}")
    logger.info(f"run_dir={run_dir}" + (" (resumed)" if resumed else ""))
    logger.info(f"[data] {run_info['npz']} | n_dim={ds.n_vars} | lookback={ds.lookback} | horizon={ds.horizon} "
                f"| sizes={sizes}")
    logger.info(f"{len(trials)} trials | select_metric={config.TRAIN['select_metric']} "
                f"| config={args.config or 'config.py'}")

    csv_path = run_dir / "grid_results.csv"
    rows, failed, best = (resume_grid(run_dir, trials, config.TRAIN["select_metric"], logger) if resumed
                          else ([], [], None))
    done = {int(r["trial"]) for r in rows}
    with Timer() as grid_timer:
        for i, hp in enumerate(trials, start=1):
            if i in done:
                continue
            logger.info("-" * 88)
            logger.info(f"  trial {i:0{width}d}/{len(trials)} | started {timestamp()} | hp={hp}")
            try:
                out = run_trial(hp, ds, splits, seed, device, logger)
            except Exception as e:
                logger.error(f"[FAILED] trial {i:0{width}d} | hp={hp}\n{traceback.format_exc()}")
                rows.append({"trial": i, "status": "FAILED", **{f"hp_{k}": v for k, v in hp.items()},
                             "error": f"{type(e).__name__}: {e}"})
                write_csv(csv_path, rows)
                failed.append(i)
                continue

            m = out["metrics"]
            score = m["selection"]["score"]
            if best is None or score < best["score"] - 1e-12:
                best = {"score": score, "trial": i, "hp": hp, "out": out}
                save_checkpoint(trials_dir, best)
            save_json({"trial": i, "hp": hp,
                       "history": {"phase1": columns(out["rows1"], drop=("gc_raw", "gc_est")),
                                   "phase2": columns(out["rows2"])},
                       "metrics": m}, trials_dir / f"trial_{i:0{width}d}__history.json")
            rows.append(trial_row(i, hp, m))
            write_csv(csv_path, rows)
            d, f = m["discovery"], m["forecast"].get("test", {})
            logger.info(f"  trial {i:0{width}d} done | {m['selection']['metric']}={m['selection']['score']:.6f}"
                        f"{_gt_summary(d)} | usage={d['variable_usage_pct']:.2f}% "
                        f"| test_rmse={f.get('rmse', float('nan')):.6f} "
                        f"| {fmt_seconds(m['timing']['phase1_s'] + m['timing']['phase2_s'])}")
            del out

        if best is not None and best["out"] is None:
            logger.info("-" * 88)
            logger.info(f"[resume] re-training trial {best['trial']} to recover its weights "
                        f"(not saved before the interruption) | hp={best['hp']}")
            best["out"] = run_trial(best["hp"], ds, splits, seed, device, logger)
            best["score"] = best["out"]["metrics"]["selection"]["score"]

    if best is None:
        logger.error(f"[ABORT] every trial failed for {dataset}/{scale_mode} seed {seed}")
        raise RuntimeError(f"all {len(trials)} trials failed; see {run_dir / 'run.log'}")

    best_dir = run_dir / "best"
    save_best(best_dir, best["out"], best["hp"], best["trial"], ds, run_info)
    bm = best["out"]["metrics"]

    save_json({**run_info, "stage": STAGE, "device": str(device), "run_dir": run_dir, "args": vars(args),
               "n_dim": ds.n_vars, "lookback": ds.lookback, "horizon": ds.horizon, "sizes": sizes,
               "n_trials": len(trials), "n_ok": len(trials) - len(failed), "failed_trials": failed,
               "grid_time_s": round(grid_timer.elapsed, 2), "grid_time_h": round(grid_timer.elapsed / 3600, 4),
               "best_trial": best["trial"], "best_hp": best["hp"],
               "select_metric": config.TRAIN["select_metric"], "best_score": best["score"],
               "param_grid": config.PARAM_GRID, "train_config": config.TRAIN, "eval_config": config.EVAL,
               "resumed": resumed, "env": env_info()}, run_dir / "meta.json")
    (trials_dir / CHECKPOINT).unlink(missing_ok=True)

    logger.info("=" * 88)
    logger.info(f"BEST | trial {best['trial']} | hp={best['hp']} | "
                f"{config.TRAIN['select_metric']}={best['score']:.6f}")
    logger.info(f"grid finished in {fmt_seconds(grid_timer.elapsed)} | {len(failed)}/{len(trials)} failed "
                f"| artifacts at {run_dir}")

    entry = {
        "dataset": dataset, "scale_mode": scale_mode, "seed": seed,
        "run_dir": run_dir.as_posix(), "model_path": (best_dir / "model.pt").as_posix(),
        "best_trial": best["trial"], "hp": best["hp"],
        "select_metric": config.TRAIN["select_metric"], "select_score": best["score"],
        **{k: bm["discovery"].get(k) for k in REGISTRY_DISCOVERY_KEYS},
        **{f"test_{k}": bm["forecast"]["test"].get(k) for k in FORECAST_KEYS if "test" in bm["forecast"]},
        **{f"ood_{k}": bm["forecast"]["ood"].get(k) for k in FORECAST_KEYS if "ood" in bm["forecast"]},
        "n_dim": ds.n_vars, "lookback": ds.lookback, "horizon": ds.horizon, **{f"n_{s}": n for s, n in sizes.items()},
        "n_trials": len(trials), "n_failed": len(failed), "grid_time_s": round(grid_timer.elapsed, 2),
        "npz": run_info["npz"],
    }
    update_registry(ds_dir / registry_name(STAGE, scale_mode), run_dir.name, entry,
                    promote=args.promote, logger=logger)
    return entry


def validate_job(dataset, scale_mode, data_dir=data_paths.DATA_ROOT):
    if scale_mode not in SCALE_MODES:
        raise ValueError(f"unknown scale_mode {scale_mode!r}; expected one of {list(SCALE_MODES)}")
    npz = data_paths.dataset_npz(dataset, scale_mode, data_dir)
    if not npz.exists():
        raise FileNotFoundError(f"{npz} not found; expected the frozen dataset at "
                                f"<data-dir>/{dataset}/dataset_{scale_mode}.npz")


def parse_args(argv=None, description="Grid search for one (dataset, scale_mode, seed) job.", feedback=False):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--dataset", required=True,
                   help="folder under --data-dir holding dataset_<scale_mode>.npz and dataset_manifest_<scale_mode>.json")
    p.add_argument("--scale-mode", required=True, choices=SCALE_MODES)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--data-dir", default=str(data_paths.DATA_ROOT))
    p.add_argument("--artifacts-dir", required=True,
                   help="this seed's artifact root, e.g. artifacts/seed42")
    p.add_argument("--device", default="auto", help='"auto" | "cuda" | "cuda:1" | "cpu"')
    p.add_argument("--promote", action=argparse.BooleanOptionalAction, default=True,
                   help="make this run the selected run in its registry")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False,
                   help="continue the newest unfinished run of this job, keeping its completed trials")
    p.add_argument("--config", default=None,
                   help="JSON snapshot of config.py (written by run.py); default: config.py as is")
    if feedback:
        p.add_argument("--source-run", default=None,
                       help="finished Stage-I run folder whose graph is fed back; default: the registry's selected run")
    args = p.parse_args(argv)
    if args.config:
        config.apply_snapshot(load_json(args.config))
        args.config = "run.py snapshot"
    validate_job(args.dataset, args.scale_mode, args.data_dir)
    return args


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    run_grid(parse_args())
