import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import config
from utils import registry_name, run_prefix, unfinished_run

# ======================================================================== #
# Edit here                                                                #
# ======================================================================== #
STAGE = "both"             # "train" | "feedback" | "both" (grid search, then feedback on it)

SEEDS = [42]               # run in sequence; or one window per seed: 42, 71, 97, 101, 209
DATASETS = [               # folder names under DATA_DIR, each holding dataset_<scale_mode>.npz
    "henon",
]
SCALE_MODES = ["std"]

DATA_DIR = "data"
ARTIFACTS_DIR = "artifacts"   # per-seed roots artifacts/seed<S>/ are added automatically

DEVICE = "cuda"            # "auto" | "cuda" | "cpu"
GPU = None                 # e.g. 0 to pin this window to one card; None = default
PROMOTE = True             # each new run becomes the selected run in its registry
SKIP_IF_DONE = True        # skip a stage this job has already finished (checked per stage)
RESUME = True              # continue an interrupted run from its completed trials instead of starting over
# ======================================================================== #

MODULES = {"train": "train", "feedback": "graph_feedback"}
STAGES = {"train": ("train",), "feedback": ("feedback",), "both": ("train", "feedback")}
TRAIN_PREFIX = "caveat"
FEEDBACK_PATTERNS = ("feedback", "feedback__r*")


def getTime():
    return datetime.now().strftime("%d-%m-%Y_%H-%M-%S")


def seed_root(seed):
    return Path(ARTIFACTS_DIR) / f"seed{seed}"


def write_snapshot():
    folder = Path(ARTIFACTS_DIR) / "config_snapshots"
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"config_snapshot__{STAGE}__{getTime()}"
    for k in range(100):
        path = folder / (f"{stem}__r{k}.json" if k else f"{stem}.json")
        try:
            with open(path, "x") as f:
                json.dump(config.snapshot(), f, indent=2)
            return path
        except FileExistsError:
            continue
    raise RuntimeError(f"cannot allocate a unique snapshot name for {stem}")


def remove_snapshot(path):
    path.unlink(missing_ok=True)
    try:
        path.parent.rmdir()
    except OSError:
        pass


def dataset_root(dataset, seed):
    return seed_root(seed) / f"artifacts_{dataset}"


def train_prefix(scale_mode, seed):
    return run_prefix(TRAIN_PREFIX, scale_mode, seed)


def newest_dir(parent, patterns, since):
    dirs = [d for pat in patterns for d in Path(parent).glob(pat) if d.is_dir() and d.stat().st_mtime >= since]
    return max(dirs, key=lambda d: d.stat().st_mtime, default=None)


def train_done(dataset, scale_mode, seed):
    return any(dataset_root(dataset, seed).glob(f"{train_prefix(scale_mode, seed)}__*/meta.json"))


def selected_train_run(dataset, scale_mode, seed):
    reg = dataset_root(dataset, seed) / registry_name(TRAIN_PREFIX, scale_mode)
    return Path(json.loads(reg.read_text())["selected"]["run_dir"]) if reg.exists() else None


def train_finished(run_dir):
    return run_dir is not None and all((Path(run_dir) / p).exists() for p in ("meta.json", "best/causal_graph.json"))


def feedback_done(source_run):
    return any(any(Path(source_run).glob(f"{pat}/meta.json")) for pat in FEEDBACK_PATTERNS)


def main():
    if STAGE not in STAGES:
        print(f"unknown STAGE {STAGE!r}; expected one of {list(STAGES)}", flush=True)
        return 1
    jobs = [(d, m, s) for s in SEEDS for d in DATASETS for m in SCALE_MODES]
    if not jobs:
        print("no jobs queued; check SEEDS / DATASETS / SCALE_MODES", flush=True)
        return 1

    snapshot_path = write_snapshot()
    try:
        return run_queue(jobs, snapshot_path)
    finally:
        remove_snapshot(snapshot_path)


def run_job(stage, tag, dataset, scale_mode, seed, snapshot_path, env, source_run=None):
    cmd = [
        sys.executable, "-m", MODULES[stage],
        "--dataset", dataset,
        "--scale-mode", scale_mode,
        "--seed", str(seed),
        "--data-dir", DATA_DIR,
        "--artifacts-dir", str(seed_root(seed)),
        "--device", DEVICE,
        "--promote" if PROMOTE else "--no-promote",
        "--resume" if RESUME else "--no-resume",
        "--config", str(snapshot_path),
    ]
    if source_run is not None:
        cmd += ["--source-run", Path(source_run).as_posix()]

    parent = dataset_root(dataset, seed) if stage == "train" else Path(source_run)
    patterns = [f"{train_prefix(scale_mode, seed)}__*"] if stage == "train" else FEEDBACK_PATTERNS
    resume_dir = unfinished_run(parent, patterns) if RESUME else None

    print(f"\nRunning: {tag} | {stage}   [{getTime()}]", flush=True)
    if resume_dir is not None:
        print(f"[RESUME] {tag} {stage}: continuing {resume_dir}", flush=True)
    t1, started = time.perf_counter(), time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, env=env)
    try:
        code = proc.wait()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Ctrl+C received. Terminating current run ...", flush=True)
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        raise

    minutes = (time.perf_counter() - t1) / 60
    run_dir = resume_dir or newest_dir(parent, patterns, started)
    log = run_dir / "run.log" if run_dir else "the error above (no run folder was created)"
    if code == 0:
        print(f"[OK] {tag} {stage} finished in {minutes:.1f} min | log -> {log}", flush=True)
    else:
        print(f"[ERROR] {tag} {stage} failed with code {code} in {minutes:.1f} min; see {log}", flush=True)
    return code, minutes, run_dir


def run_queue(jobs, snapshot_path):
    stages = STAGES[STAGE]
    print(f"stage {STAGE} ({' -> '.join(stages)}) | {len(jobs)} jobs "
          f"| artifacts -> {ARTIFACTS_DIR}/seed<S> | config snapshot (removed when the queue ends) -> {snapshot_path}")
    print(f"queue: {jobs}", flush=True)

    env = os.environ.copy()
    if GPU is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(GPU)

    t0 = time.perf_counter()
    results, failed, skipped, missing = [], [], [], []
    try:
        for dataset, scale_mode, seed in jobs:
            tag = f"{dataset}/{scale_mode} seed{seed}"
            source = None
            if "train" in stages:
                if SKIP_IF_DONE and train_done(dataset, scale_mode, seed):
                    print(f"[SKIP] {tag} train: already has a grid-search run", flush=True)
                    skipped.append(f"{tag} train")
                else:
                    code, minutes, source = run_job("train", tag, dataset, scale_mode, seed, snapshot_path, env)
                    results.append((f"{tag} train", code, minutes))
                    if code != 0:
                        failed.append(f"{tag} train")
                        if "feedback" in stages:
                            print(f"[NOT RUN] {tag} feedback: its grid search did not finish", flush=True)
                        continue

            if "feedback" in stages:
                source = source or selected_train_run(dataset, scale_mode, seed)
                if not train_finished(source):
                    print(f"[MISSING] {tag} feedback: no finished grid-search run; "
                          f"run STAGE = \"train\" first", flush=True)
                    missing.append(tag)
                    continue
                if SKIP_IF_DONE and feedback_done(source):
                    print(f"[SKIP] {tag} feedback: {source.name} already has a finished feedback", flush=True)
                    skipped.append(f"{tag} feedback")
                    continue
                code, minutes, _ = run_job("feedback", tag, dataset, scale_mode, seed, snapshot_path, env,
                                           source_run=source)
                results.append((f"{tag} feedback", code, minutes))
                if code != 0:
                    failed.append(f"{tag} feedback")
    except KeyboardInterrupt:
        print("[STOPPED] Exiting run.py.", flush=True)
        return 130

    print(f"\nExecution summary ({STAGE}):", flush=True)
    for tag, code, minutes in sorted(results, key=lambda r: -r[2]):
        print(f"  {tag:40s} {minutes:7.1f} min  {'OK' if code == 0 else 'FAILED'}")
    print(f"Total wall time: {(time.perf_counter() - t0) / 60:.1f} min for {len(results)} runs "
          f"({len(skipped)} skipped, {len(missing)} missing prerequisites)")
    print(f"Failed runs: {failed if failed else None}", flush=True)
    print(f"Missing prerequisites: {missing if missing else None}", flush=True)
    return 1 if (failed or missing) else 0


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    raise SystemExit(main())
