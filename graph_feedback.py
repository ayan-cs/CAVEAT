import os
import traceback
from itertools import product
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW

import config
from data.datasets import FrozenDataset
from metrics import discovery_metrics
from model import CausalLSTM, ErrorCompensation
from services import Criterion, EarlyStopper
from data import paths as data_paths
from train import (CHECKPOINT, FORECAST_KEYS, LOSS_KEYS, REGISTRY_DISCOVERY_KEYS, STAGE, _cpu_state, _gt_summary,
                   _prefixed, _Running, evaluate_forecasts, loss_pass, parse_args, plateau_scheduler, resume_grid,
                   save_checkpoint, train_phase2, trial_row, write_csv)
from utils import (Timer, columns, env_info, fmt_seconds, get_logger, load_json, registry_name,
                   resolve_device, save_json, set_deterministic, timestamp, to_loader, unfinished_run, unique_dir,
                   update_registry)

FEEDBACK_STAGE = "feedback"


def hard_mask(graph):
    g = np.array(graph, dtype=np.int64)
    for d in range(g.shape[0]):
        if g[d, :].sum() == 0:
            g[d, d] = 1
    return g


def usage_pct(graph, adjacency):
    known = np.ones_like(graph, dtype=bool) if adjacency is None else adjacency >= 0
    return float(100 * graph[known].mean())


def feedback_phase1_epoch(model, train_dl, val_dl, optimizer, criterion, grad_clip, device):
    model.train()
    tr = _Running()
    for X_l, X_r in train_dl:
        optimizer.zero_grad()
        X_l, X_r = X_l.to(device), X_r.to(device)
        pred, mu, logvar = model(X_l, X_r)
        loss, parts = criterion(pred, X_r, mu, logvar, model)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        tr.add(loss, parts, X_l.shape[0])
    return tr.mean(), loss_pass(model, val_dl, criterion, device)


def run_graph_feedback(ds, splits, discovered_graph, hp, seed, device, logger):
    set_deterministic(seed)
    fb = config.FEEDBACK
    M = ds.n_vars
    graph = hard_mask(discovered_graph)
    assert graph.shape == (M, M), f"graph {graph.shape} != ({M},{M})"

    train_dl = to_loader(splits['train'], hp['batch_size'], shuffle=True)
    val_dl = to_loader(splits['val'], hp['batch_size'], shuffle=False)

    clstm = CausalLSTM(M, hp['hidden_size'], graph.T).to(device)
    criterion = Criterion(hp['beta_kl'], hp['lam_ridge'], model='clstm')
    optimizer = AdamW(clstm.parameters(), lr=hp['lr'])
    scheduler = plateau_scheduler(optimizer, fb['phase1'])
    stopper = EarlyStopper(patience=fb['phase1']['patience'])

    rows1, best_val, best_epoch, best_state = [], np.inf, 0, None
    with Timer() as t1:
        for epoch in range(fb['epochs']):
            with Timer() as et:
                tr, va = feedback_phase1_epoch(clstm, train_dl, val_dl, optimizer, criterion, fb['grad_clip'], device)
            lr = optimizer.param_groups[0]['lr']
            scheduler.step(va['loss'])
            rows1.append({"epoch": epoch + 1, "lr": lr, **_prefixed("train", tr), **_prefixed("val", va),
                          "epoch_time_s": et.elapsed})
            if epoch % config.TRAIN['log_every'] == 0:
                logger.info(f"    [fb phase1] epoch {epoch + 1:5d} | train {tr['loss']:.6f} "
                            f"| val {va['loss']:.6f} | lr {lr:.2e}")
            if va['loss'] < best_val:
                best_val, best_epoch, best_state = va['loss'], epoch + 1, _cpu_state(clstm)
            if stopper.early_stop(va['loss']):
                logger.info(f"    [fb phase1] early stop at epoch {epoch + 1} (best epoch {best_epoch})")
                break

    clstm.load_state_dict(best_state)
    for p in clstm.parameters():
        p.requires_grad = False
    best = rows1[best_epoch - 1]
    test_loss = loss_pass(clstm, to_loader(splits['test'], hp['batch_size']), criterion, device)

    errorc = ErrorCompensation(M, hp['hidden_size']).to(device)
    optimizer_e = AdamW(errorc.parameters(), lr=hp['lr'])
    criterion_e = Criterion(hp['beta_e'], hp['lam_ridge'], model='errorc')
    rows2, best_epoch2, t2 = train_phase2(clstm, errorc, train_dl, val_dl, optimizer_e, criterion_e, fb['epochs'], fb['phase2'], device, logger, tag="fb phase2")
    best2 = rows2[best_epoch2 - 1]

    with Timer() as te:
        forecast, forecasts = evaluate_forecasts(clstm, errorc, splits, hp['batch_size'], ds.inverse_transform)

    metrics = {
        "selection": {"metric": fb["select_metric"], "score": best[fb["select_metric"]], "best_epoch": best_epoch},
        "epochs": {"phase1_run": len(rows1), "phase1_best": best_epoch, "phase2_run": len(rows2), "phase2_best": best_epoch2},
        "loss": {"train": {k: best[f"train_{k}"] for k in LOSS_KEYS}, "val": {k: best[f"val_{k}"] for k in LOSS_KEYS}, "test": {k: test_loss[k] for k in LOSS_KEYS}},
        "phase2_loss": {"train": {k: best2[f"train_{k}"] for k in LOSS_KEYS}, "val": {k: best2[f"val_{k}"] for k in LOSS_KEYS}},
        "variable_usage_pct": usage_pct(graph, ds.adjacency),
        "forecast": forecast,
        "timing": {"phase1_s": t1.elapsed, "phase2_s": t2, "eval_s": te.elapsed},
    }
    return {"metrics": metrics, "rows1": rows1, "rows2": rows2, "graph": graph, "arch_graph": graph.T, "clstm_state": _cpu_state(clstm), "errorc_state": _cpu_state(errorc), "forecasts": forecasts}


def source_run_dir(args, ds_dir):
    if args.source_run:
        src = Path(args.source_run)
    else:
        reg = ds_dir / registry_name(STAGE, args.scale_mode)
        if not reg.exists():
            raise FileNotFoundError(f"{reg} not found; run the Stage-I grid search first (run.py STAGE = \"train\")")
        src = Path(load_json(reg)["selected"]["run_dir"])
    missing = [p for p in ("meta.json", "best/causal_graph.json", "best/config.json") if not (src / p).exists()]
    if missing:
        raise FileNotFoundError(f"{src} is not a finished grid-search run (missing {missing})")
    return src


def feedback_grid(base_hp):
    ov = {k: (v if isinstance(v, (list, tuple)) else [v]) for k, v in config.FEEDBACK["overrides"].items()}
    return [{**base_hp, **dict(zip(ov, values))} for values in product(*ov.values())]


def save_best(best_dir, out, hp, trial, ds, source_graph, stage2, info):
    best_dir = Path(best_dir)
    best_dir.mkdir(parents=True, exist_ok=True)
    m = out["metrics"]
    torch.save({"clstm_state_dict": out["clstm_state"], "errorc_state_dict": out["errorc_state"],
                "model_hp": {"num_features": ds.n_vars, "hidden_size": hp["hidden_size"]},
                "causal_graph": out["arch_graph"], "graph_used": out["graph"], "hp": hp, "trial": trial,
                "metrics": m}, best_dir / "model.pt")
    save_json({"graph_used": out["graph"], "stage2_graph": stage2["graph"],
               "threshold": stage2["threshold"],
               "source_GC_raw": source_graph["GC_raw"], "source_GC_est": source_graph["GC_est"],
               "GC_true": source_graph["GC_true"], "orientation": source_graph["orientation"],
               "feature_names": source_graph.get("feature_names")}, best_dir / "causal_graph.json")
    save_json({"hp": hp, "trial": trial, **info,
               "select_metric": m["selection"]["metric"], "select_score": m["selection"]["score"],
               "feedback_config": config.FEEDBACK, "eval_config": config.EVAL}, best_dir / "config.json")
    save_json({"phase1": columns(out["rows1"]), "phase2": columns(out["rows2"])}, best_dir / "history.json")
    save_json(m, best_dir / "metrics.json")
    if out["forecasts"] is not None:
        np.savez_compressed(best_dir / "test_predictions.npz", window_start=ds.window_starts("test"), norm_mean=ds.norm_mean, norm_std=ds.norm_std, **out["forecasts"])


def feedback_for(args):
    dataset, scale_mode, seed = args.dataset, args.scale_mode, args.seed
    device = resolve_device(args.device)
    ds_dir = Path(args.artifacts_dir) / f"artifacts_{dataset}"
    src = source_run_dir(args, ds_dir)
    source_graph = load_json(src / "best" / "causal_graph.json")
    source_cfg = load_json(src / "best" / "config.json")
    ds = FrozenDataset(dataset, scale_mode, args.data_dir)
    splits = ds.as_splits()

    graph, thr = np.array(source_graph["GC_est"], dtype=np.int64), source_graph.get("threshold")
    stage2 = {"graph": graph, "threshold": thr}
    discovery = {"threshold": thr, **discovery_metrics(np.array(source_graph["GC_raw"]), graph, ds.adjacency)}
    trials = feedback_grid(source_cfg["hp"])

    info = {"dataset": dataset, "scale_mode": scale_mode, "seed": seed,
            "source_run": src.as_posix(), "source_trial": source_cfg["trial"],
            "npz": data_paths.dataset_npz(dataset, scale_mode, args.data_dir),
            "has_ground_truth": ds.has_ground_truth, "feature_names": ds.feature_names}

    run_dir = unfinished_run(src, ["feedback", "feedback__r*"]) if args.resume else None
    resumed = run_dir is not None
    run_dir = run_dir or unique_dir(src / "feedback")
    trials_dir = run_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)
    width = max(2, len(str(len(trials))))
    logger = get_logger(f"caveat.feedback.{dataset}.{scale_mode}.seed{seed}", run_dir / "run.log")
    logger.info("=" * 88)
    logger.info(f"graph feedback | dataset={dataset} | scale_mode={scale_mode} | seed={seed} "
                f"| source={src.name} (trial {source_cfg['trial']}) | device={device} | started {timestamp()}")
    logger.info(f"run_dir={run_dir}" + (" (resumed)" if resumed else ""))
    logger.info(f"[stage2] threshold={thr if thr is None else f'{thr:.4g}'} (adaptive) "
                f"| usage {discovery['variable_usage_pct']:.2f}%{_gt_summary(discovery)}")
    logger.info(f"{len(trials)} trials | select_metric={config.FEEDBACK['select_metric']} "
                f"| overrides={config.FEEDBACK['overrides']} | config={args.config or 'config.py'}")

    csv_path = run_dir / "grid_results.csv"
    rows, failed, best = (resume_grid(run_dir, trials, config.FEEDBACK["select_metric"], logger) if resumed
                          else ([], [], None))
    done = {int(r["trial"]) for r in rows}
    with Timer() as grid_timer:
        for i, hp in enumerate(trials, start=1):
            if i in done:
                continue
            logger.info("-" * 88)
            logger.info(f"  trial {i:0{width}d}/{len(trials)} | started {timestamp()} | hp={hp}")
            try:
                out = run_graph_feedback(ds, splits, graph, hp, seed, device, logger)
            except Exception as e:
                logger.error(f"[FAILED] trial {i:0{width}d} | hp={hp}\n{traceback.format_exc()}")
                rows.append({"trial": i, "status": "FAILED", **{f"hp_{k}": v for k, v in hp.items()}, "error": f"{type(e).__name__}: {e}"})
                write_csv(csv_path, rows)
                failed.append(i)
                continue

            m = out["metrics"]
            m["discovery"] = discovery
            score = m["selection"]["score"]
            if best is None or score < best["score"] - 1e-12:
                best = {"score": score, "trial": i, "hp": hp, "out": out}
                save_checkpoint(trials_dir, best)
            save_json({"trial": i, "hp": hp,
                       "history": {"phase1": columns(out["rows1"]), "phase2": columns(out["rows2"])},
                       "metrics": m}, trials_dir / f"trial_{i:0{width}d}__history.json")
            rows.append({**trial_row(i, hp, m), "variable_usage_pct": m["variable_usage_pct"]})
            write_csv(csv_path, rows)
            t = m["forecast"].get("test", {})
            logger.info(f"  trial {i:0{width}d} done | {m['selection']['metric']}={m['selection']['score']:.6f}"
                        f" | usage={m['variable_usage_pct']:.2f}% | test_rmse={t.get('rmse', float('nan')):.6f} "
                        f"| test_crps={t.get('crps', float('nan')):.6f} "
                        f"| {fmt_seconds(m['timing']['phase1_s'] + m['timing']['phase2_s'] + m['timing']['eval_s'])}")
            del out

        if best is not None and best["out"] is None:
            logger.info("-" * 88)
            logger.info(f"[resume] re-training trial {best['trial']} to recover its weights "
                        f"(not saved before the interruption) | hp={best['hp']}")
            best["out"] = run_graph_feedback(ds, splits, graph, best["hp"], seed, device, logger)
            best["out"]["metrics"]["discovery"] = discovery
            best["score"] = best["out"]["metrics"]["selection"]["score"]

    if best is None:
        logger.error(f"[ABORT] every feedback trial failed for {dataset}/{scale_mode} seed {seed}")
        raise RuntimeError(f"all {len(trials)} feedback trials failed; see {run_dir / 'run.log'}")

    best_dir = run_dir / "best"
    save_best(best_dir, best["out"], best["hp"], best["trial"], ds, source_graph, stage2, info)
    bm = best["out"]["metrics"]
    save_json({**info, "stage": FEEDBACK_STAGE, "device": str(device), "run_dir": run_dir, "args": vars(args),
               "n_dim": ds.n_vars, "lookback": ds.lookback, "horizon": ds.horizon,
               "n_trials": len(trials), "n_ok": len(trials) - len(failed), "failed_trials": failed,
               "grid_time_s": round(grid_timer.elapsed, 2), "grid_time_h": round(grid_timer.elapsed / 3600, 4),
               "best_trial": best["trial"], "best_hp": best["hp"],
               "select_metric": config.FEEDBACK["select_metric"], "best_score": best["score"],
               "feedback_config": config.FEEDBACK, "eval_config": config.EVAL,
               "resumed": resumed, "env": env_info()}, run_dir / "meta.json")
    (trials_dir / CHECKPOINT).unlink(missing_ok=True)

    t = bm["forecast"].get("test", {})
    logger.info("=" * 88)
    logger.info(f"BEST | trial {best['trial']} | hp={best['hp']} | "
                f"{config.FEEDBACK['select_metric']}={best['score']:.6f} | test rmse {t.get('rmse', float('nan')):.6f} "
                f"| crps {t.get('crps', float('nan')):.6f}")
    logger.info(f"feedback finished in {fmt_seconds(grid_timer.elapsed)} | {len(failed)}/{len(trials)} failed "
                f"| artifacts at {run_dir}")
    entry = {"dataset": dataset, "scale_mode": scale_mode, "seed": seed,
             "run_dir": run_dir.as_posix(), "model_path": (best_dir / "model.pt").as_posix(),
             "source_run": src.as_posix(), "best_trial": best["trial"], "hp": best["hp"],
             "select_metric": config.FEEDBACK["select_metric"], "select_score": best["score"],
             "variable_usage_pct": bm["variable_usage_pct"],
             "threshold": thr,
             **{f"stage2_{k}": discovery.get(k) for k in REGISTRY_DISCOVERY_KEYS},
             **{f"test_{k}": t.get(k) for k in FORECAST_KEYS},
             **{f"ood_{k}": bm["forecast"]["ood"].get(k) for k in FORECAST_KEYS if "ood" in bm["forecast"]},
             "n_trials": len(trials), "n_failed": len(failed), "grid_time_s": round(grid_timer.elapsed, 2)}
    update_registry(ds_dir / registry_name(FEEDBACK_STAGE, scale_mode),
                    f"{src.name}/{run_dir.name}", entry, promote=args.promote, logger=logger)
    return entry


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    feedback_for(parse_args(description="Graph-feedback (Stage II + III) for one (dataset, scale_mode, seed) job.", feedback=True))
