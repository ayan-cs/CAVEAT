from __future__ import annotations

from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent


def dataset_dir(dataset: str, root: str | Path = DATA_ROOT) -> Path:
    return Path(root) / dataset


def dataset_npz(dataset: str, scale_mode: str, root: str | Path = DATA_ROOT) -> Path:
    return dataset_dir(dataset, root) / f"dataset_{scale_mode}.npz"


def dataset_manifest(dataset: str, scale_mode: str, root: str | Path = DATA_ROOT) -> Path:
    return dataset_dir(dataset, root) / f"dataset_manifest_{scale_mode}.json"
