from __future__ import annotations

import numpy as np

from data import paths


def _normalise(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


def _stack_windows(X: np.ndarray, starts: np.ndarray, length: int) -> np.ndarray:
    if len(starts) == 0:
        return np.zeros((0, length, X.shape[1]), dtype=X.dtype)
    return np.stack([X[s:s + length] for s in starts], axis=0)


class FrozenDataset:

    def __init__(self, dataset: str, scale_mode: str = "std", data_dir=paths.DATA_ROOT):
        npz = np.load(paths.dataset_npz(dataset, scale_mode, data_dir), allow_pickle=False)

        self._name = str(npz["name"])
        self._n_vars = int(npz["n_vars"])
        self._has_gt = bool(npz["has_ground_truth"]) if "has_ground_truth" in npz.files else True
        self._adjacency = npz["adjacency"].astype(np.int64) if self._has_gt else None
        self._feature_names = ([str(f) for f in npz["feature_names"]] if "feature_names" in npz.files
                               else [f"x{i}" for i in range(self._n_vars)])
        self._lookback = int(npz["lookback"])
        self._horizon = int(npz["horizon"])

        self._norm_mean = npz["norm_mean"].astype(np.float32)
        self._norm_std = npz["norm_std"].astype(np.float32)
        self._scale_mode = str(npz["scale_mode"])

        self._x_main = npz["x_main"].astype(np.float32)

        self._win = {s: npz[f"win_{s}"].astype(np.int64)
                     for s in ("train", "val", "test", "ood") if f"win_{s}" in npz.files}

    @property
    def name(self) -> str:
        return self._name

    @property
    def n_vars(self) -> int:
        return self._n_vars

    @property
    def has_ground_truth(self) -> bool:
        return self._has_gt

    @property
    def adjacency(self) -> np.ndarray | None:
        return self._adjacency

    @property
    def eval_mask(self) -> np.ndarray | None:
        return None if self._adjacency is None else self._adjacency >= 0

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names

    @property
    def splits(self) -> tuple[str, ...]:
        return tuple(self._win)

    @property
    def lookback(self) -> int:
        return self._lookback

    @property
    def horizon(self) -> int:
        return self._horizon

    @property
    def scale_mode(self) -> str:
        return self._scale_mode

    def parents(self, i: int) -> list[int]:
        self._require_gt()
        return [j for j in range(self._n_vars) if self._adjacency[i, j] == 1 and j != i]

    def nonparents(self, i: int) -> list[int]:
        self._require_gt()
        return [j for j in range(self._n_vars) if self._adjacency[i, j] == 0 and j != i]

    def _require_gt(self) -> None:
        if not self._has_gt:
            raise ValueError(f"{self._name} has no ground-truth graph")

    @property
    def norm_mean(self) -> np.ndarray:
        return self._norm_mean

    @property
    def norm_std(self) -> np.ndarray:
        return self._norm_std

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X * self._norm_std + self._norm_mean

    def window_starts(self, split: str) -> np.ndarray:
        return self._win[split]

    def windows(self, split: str) -> np.ndarray:
        starts = self._win[split]
        X = _normalise(self._x_main, self._norm_mean, self._norm_std)
        return _stack_windows(X, starts, self._lookback)

    def targets(self, split: str) -> np.ndarray:
        starts = self._win[split] + self._lookback
        X = _normalise(self._x_main, self._norm_mean, self._norm_std)
        return _stack_windows(X, starts, self._horizon)

    def as_splits(self) -> dict:
        splits = {s: (self.windows(s), self.targets(s)) for s in self.splits}
        splits["gt_matrix"] = self._adjacency
        splits["feature_names"] = self._feature_names
        return splits
