import json
import logging
import os
import platform
import random
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def set_deterministic(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device="auto"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def to_loader(split, batch_size, shuffle=False):
    Xl, Xr = split
    ds = TensorDataset(torch.FloatTensor(np.asarray(Xl, dtype=np.float32)), torch.FloatTensor(np.asarray(Xr, dtype=np.float32)))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def timestamp():
    return datetime.now().strftime("%d-%m-%Y_%H-%M-%S")


def run_prefix(stage, scale_mode, seed):
    return f"{stage}__{scale_mode}__seed{seed}"


def registry_name(stage, scale_mode):
    return f"{stage}_registry__{scale_mode}.json"


def unfinished_run(parent, patterns):
    dirs = [d for pat in patterns for d in Path(parent).glob(pat)
            if (d / "grid_results.csv").exists() and not (d / "meta.json").exists()]
    return max(dirs, key=lambda d: (d / "grid_results.csv").stat().st_mtime, default=None)


def build_run_name(*parts):
    return "__".join([*map(str, parts), timestamp()])


def unique_dir(path):
    path = Path(path)
    if not path.exists():
        return path
    for k in range(1, 100):
        cand = path.with_name(f"{path.name}__r{k}")
        if not cand.exists():
            return cand
    raise RuntimeError(f"cannot allocate a unique run directory near {path}")


def get_logger(name, log_file, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter(fmt="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logging.captureWarnings(True)
    warn_logger = logging.getLogger("py.warnings")
    warn_logger.handlers.clear()
    warn_logger.propagate = False
    eh = logging.StreamHandler(sys.stderr)
    eh.setFormatter(fmt)
    warn_logger.addHandler(fh)
    warn_logger.addHandler(eh)

    def excepthook(exc_type, exc, tb):
        if not issubclass(exc_type, KeyboardInterrupt):
            fh.handle(logger.makeRecord(name, logging.CRITICAL, __file__, 0, "uncaught exception",
                                        None, (exc_type, exc, tb)))
        sys.__excepthook__(exc_type, exc, tb)
    sys.excepthook = excepthook
    return logger


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.t0
        return False


def fmt_seconds(s):
    h, r = divmod(int(s), 3600)
    m, sec = divmod(r, 60)
    return f"{h:d}h{m:02d}m{sec:02d}s" if h else f"{m:d}m{sec:02d}s"


class _NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, torch.Tensor):
            return o.detach().cpu().tolist()
        if isinstance(o, Path):
            return o.as_posix()
        return super().default(o)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, cls=_NpEncoder)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def columns(rows, drop=()):
    return {k: [r[k] for r in rows] for k in (rows[0] if rows else {}) if k not in drop}


def update_registry(registry_path, run_name, entry, promote=True, logger=None):
    registry_path = Path(registry_path)
    reg = load_json(registry_path) if registry_path.exists() else {"selected": None, "runs": {}}
    entry = {**entry, "run_name": run_name, "recorded_at": timestamp()}
    reg["runs"][run_name] = entry
    if reg["selected"] is None:
        reg["selected"] = entry
        action = "promoted (first run)"
    elif promote:
        previous = reg["selected"]["run_name"]
        reg["selected"] = entry
        action = f"promoted (replaces {previous})"
    else:
        action = f"recorded only; selected run remains {reg['selected']['run_name']}"
    save_json(reg, registry_path)
    if logger is not None:
        logger.info(f"[registry] {run_name}: {action}")
    return reg


def env_info():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                         stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        commit = "n/a"
    info = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "git_commit": commit,
        "command": " ".join(sys.argv),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_total_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
    return info
