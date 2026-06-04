"""BCI IV 2a preprocessing pipeline — bandpass filter then normalisation."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .filters import BandpassParams, bandpass_filter
from .normalizer import (
    NormalizationStats,
    apply_normalizer,
    fit_normalizer,
)


@dataclass(frozen=True)
class PipelineConfig:
    """All preprocessing hyperparameters. Everything that can be configured
    without seeing data goes here."""

    bandpass: BandpassParams


@dataclass(frozen=True)
class FittedPipeline:
    """A pipeline with all fittable state populated. Immutable: the only thing
    you can do with this is transform new data."""

    config: PipelineConfig
    normalizer: NormalizationStats

    def transform(self, epochs: np.ndarray) -> np.ndarray:
        """Apply the full preprocessing pipeline to new data.

        Args:
            epochs: shape (n_trials, n_channels, n_samples), float32.

        Returns:
            Preprocessed epochs, same shape and dtype.
        """
        x = bandpass_filter(epochs, self.config.bandpass)
        x = apply_normalizer(x, self.normalizer)
        return x


def fit_pipeline(train_epochs: np.ndarray, config: PipelineConfig) -> FittedPipeline:
    """Fit the pipeline on training epochs.

    The order matters: filter first, then fit normalizer on filtered data.
    The normalizer must see the same distribution at fit time that it will see
    at transform time, which means filtering happens before stats computation.
    """
    x = bandpass_filter(train_epochs, config.bandpass)
    normalizer = fit_normalizer(x)
    return FittedPipeline(config=config, normalizer=normalizer)


def save_pipeline(pipeline: FittedPipeline, path: Path) -> None:
    """Save a fitted pipeline to a directory.

    Layout:
        path/
            config.json       — preprocessing hyperparameters
            normalizer.npz    — fitted statistics
    """
    path.mkdir(parents=True, exist_ok=True)
    config_dict = {
        "schema_version": 1,
        "bandpass": {
            "low_hz": pipeline.config.bandpass.low_hz,
            "high_hz": pipeline.config.bandpass.high_hz,
            "fs_hz": pipeline.config.bandpass.fs_hz,
            "order": pipeline.config.bandpass.order,
        },
    }
    (path / "config.json").write_text(json.dumps(config_dict, indent=2))
    np.savez(
        path / "normalizer.npz",
        mean=pipeline.normalizer.mean,
        std=pipeline.normalizer.std,
        schema_version=np.array(1),
    )


def load_pipeline(path: Path) -> FittedPipeline:
    config_dict = json.loads((path / "config.json").read_text())
    if config_dict["schema_version"] != 1:
        raise ValueError(
            f"Unsupported pipeline schema: {config_dict['schema_version']}"
        )
    config = PipelineConfig(
        bandpass=BandpassParams(**config_dict["bandpass"]),
    )
    norm_data = np.load(path / "normalizer.npz")
    if int(norm_data["schema_version"]) != 1:
        raise ValueError(
            f"Unsupported normalizer schema: {int(norm_data['schema_version'])}"
        )
    normalizer = NormalizationStats(
        mean=norm_data["mean"],
        std=norm_data["std"],
    )
    return FittedPipeline(config=config, normalizer=normalizer)
