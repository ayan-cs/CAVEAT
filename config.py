from itertools import product

# PARAM_GRID = {
#     "lr":          [0.01, 0.001],
#     "batch_size":  [2048],
#     "hidden_size": [256, 512],
#     "lam":         [0.1, 0.01, 0.005, 0.001],
#     "lam_ridge":   [0.0, 0.1, 0.01, 0.001],
#     "beta_kl":     [0.1, 0.05, 0.01, 0.005, 0.001],
#     "beta_e":      [1.0],
# }
PARAM_GRID = {
    "lr":          [0.01],
    "batch_size":  [2048],
    "hidden_size": [256],
    "lam":         [0.01, 0.005],
    "lam_ridge":   [0.1, 0.01],
    "beta_kl":     [0.01, 0.005],
    "beta_e":      [1.0],
}

TRAIN = dict(
    epochs=10000,
    phase1=dict(patience=200, lr_patience=30, lr_factor=0.1, min_lr=1e-6),
    phase2=dict(patience=100, lr_patience=30, lr_factor=0.1, min_lr=1e-6),
    log_every=20,
    select_metric="val_mse",   # "val_mse" | "val_loss"
)

EVAL = dict(
    splits=("val", "test", "ood"),
    n_samples=100,
    error_scale=0.01,
)

FEEDBACK = dict(
    epochs=1000,
    phase1=dict(patience=100, lr_patience=20, lr_factor=0.1, min_lr=0.0),
    phase2=dict(patience=100, lr_patience=20, lr_factor=0.1, min_lr=0.0),
    grad_clip=5.0,
    overrides={},           # e.g. {"batch_size": 4096}; a list value, e.g. {"lr": [1e-3, 1e-2]}, is searched over
    select_metric="val_mse",
)

SNAPSHOT_KEYS = ("PARAM_GRID", "TRAIN", "EVAL", "FEEDBACK")


def expand_grid() -> list[dict]:
    return [dict(zip(PARAM_GRID, values)) for values in product(*PARAM_GRID.values())]


def snapshot() -> dict:
    return {k: globals()[k] for k in SNAPSHOT_KEYS}


def apply_snapshot(snap: dict) -> None:
    globals().update({k: snap[k] for k in SNAPSHOT_KEYS if k in snap})
