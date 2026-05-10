# normalize.py
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class NormalizationStats:
    mean: np.ndarray  # shape (n_channels,)
    std: np.ndarray  # shape (n_channels,)

    def __post_init__(self):
        if (self.std <= 0).any():
            raise ValueError(
                "Encountered non-positive std; channel is constant or empty."
            )


def fit_normalizer(x_train: np.ndarray, *, eps: float = 1e-8) -> NormalizationStats:
    flat = x_train.transpose(1, 0, 2).reshape(x_train.shape[1], -1)
    mean = flat.mean(axis=1)
    std = flat.std(axis=1) + eps
    return NormalizationStats(mean=mean.astype(np.float32), std=std.astype(np.float32))


def apply_normalizer(x: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    return ((x - stats.mean[None, :, None]) / stats.std[None, :, None]).astype(
        x.dtype, copy=False
    )


def save_normalizer(stats: NormalizationStats, path: Path) -> None:
    np.savez(
        path,
        mean=stats.mean,
        std=stats.std,
        schema_version=np.array(1),
    )


def load_normalizer(path: Path) -> NormalizationStats:
    data = np.load(path)
    if int(data["schema_version"]) != 1:
        raise ValueError(
            f"Unsupported normalizer schema: {int(data['schema_version'])}"
        )
    return NormalizationStats(mean=data["mean"], std=data["std"])
