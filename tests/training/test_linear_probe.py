"""Tests for the linear-probe evaluation."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest
import torch

from neurostream.models.mae import EEGMaskedAutoencoder
from neurostream.training.linear_probe import (
    ProbeConfig,
    run_probe,
    run_pretrained_vs_random,
)


def _make_synthetic_loader(
    n_subjects: int = 3,
    n_trials_per_session: int = 60,
    seed: int = 0,
) -> "tuple[callable, dict]":
    """Build a synthetic Phase 1 loader and the cached data behind it.

    Each subject's labels are linearly separable in *some* projection of
    the raw signal — guarantees the linear probe can hit > chance and the
    test can assert meaningful accuracy gaps.
    """
    rng = np.random.RandomState(seed)
    storage: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}

    for subject_id in range(1, n_subjects + 1):
        # Per-subject class offsets — SAME across T and E sessions so the
        # linear probe trained on T can transfer to E.
        class_offsets = rng.randn(4, 22, 1).astype(np.float32) * 2.0
        for session in ("T", "E"):
            labels = rng.randint(0, 4, size=n_trials_per_session).astype(np.int64)
            base = rng.randn(n_trials_per_session, 22, 1000).astype(np.float32) * 0.5
            epochs = base + class_offsets[labels]
            storage[(subject_id, session)] = (epochs, labels)

    def loader(
        subject_id: int, session: Literal["T", "E"]
    ) -> tuple[np.ndarray, np.ndarray]:
        return storage[(subject_id, session)]

    return loader, storage


@pytest.fixture
def small_encoder() -> EEGMaskedAutoencoder:
    return EEGMaskedAutoencoder(
        n_channels=22, n_samples=1000, patch_samples=25,
        encoder_dim=64, encoder_depth=2, encoder_heads=4,
        decoder_dim=32, decoder_depth=1, decoder_heads=2,
    )


@pytest.fixture
def probe_cfg() -> ProbeConfig:
    return ProbeConfig(
        pool="mean",
        batch_size=32,
        standardize=True,
        logreg_max_iter=2000,
        subjects=(1, 2, 3),
    )


# ---------------------------------------------------------------------
def test_run_probe_returns_well_formed_report(
    small_encoder: EEGMaskedAutoencoder, probe_cfg: ProbeConfig
) -> None:
    loader, _ = _make_synthetic_loader()
    report = run_probe(small_encoder, loader, probe_cfg, label="test")

    assert report.pretrained_or_random == "test"
    assert len(report.subjects) == 3
    assert report.feature_dim == 64  # encoder_dim
    assert 0.0 <= report.mean_accuracy <= 1.0


def test_run_probe_freezes_encoder(
    small_encoder: EEGMaskedAutoencoder, probe_cfg: ProbeConfig
) -> None:
    """After ``run_probe``, no encoder params should have grad."""
    loader, _ = _make_synthetic_loader()
    _ = run_probe(small_encoder, loader, probe_cfg)
    for p in small_encoder.parameters():
        assert not p.requires_grad


def test_run_probe_uses_train_session_T_eval_session_E(
    small_encoder: EEGMaskedAutoencoder, probe_cfg: ProbeConfig
) -> None:
    """Verify the probe calls the loader with the correct sessions."""
    calls: list[tuple[int, str]] = []

    def spy_loader(subject_id: int, session: Literal["T", "E"]):
        calls.append((subject_id, session))
        return (
            np.random.randn(20, 22, 1000).astype(np.float32),
            np.random.randint(0, 4, 20).astype(np.int64),
        )

    cfg = ProbeConfig(subjects=(1,), batch_size=8, logreg_max_iter=200)
    _ = run_probe(small_encoder, spy_loader, cfg)
    assert (1, "T") in calls
    assert (1, "E") in calls


def test_run_probe_handles_label_shifts(
    small_encoder: EEGMaskedAutoencoder, probe_cfg: ProbeConfig
) -> None:
    """Probe should work even if test set has a different label distribution."""

    def skewed_loader(subject_id: int, session: Literal["T", "E"]):
        if session == "T":
            labels = np.arange(60) % 4  # balanced
        else:
            labels = np.ones(60, dtype=np.int64) * 2  # all one class
        epochs = np.random.randn(60, 22, 1000).astype(np.float32)
        return epochs, labels.astype(np.int64)

    report = run_probe(small_encoder, skewed_loader, probe_cfg)
    assert all(s.accuracy >= 0.0 for s in report.subjects)


def test_run_pretrained_vs_random_returns_two_reports(
    tmp_path, small_encoder: EEGMaskedAutoencoder, probe_cfg: ProbeConfig
) -> None:
    """The wrapper should produce both pretrained and random-init reports."""
    # Save a checkpoint we can reload.
    ckpt_path = tmp_path / "test.pt"
    torch.save(
        {
            "step": 0,
            "model": small_encoder.state_dict(),
            "config": {
                "model": {
                    "n_channels": 22, "n_samples": 1000, "patch_samples": 25,
                    "encoder_dim": 64, "encoder_depth": 2, "encoder_heads": 4,
                    "decoder_dim": 32, "decoder_depth": 1, "decoder_heads": 2,
                }
            },
        },
        ckpt_path,
    )
    loader, _ = _make_synthetic_loader()
    pretrained, random_ = run_pretrained_vs_random(
        str(ckpt_path), loader, probe_cfg
    )
    assert pretrained.pretrained_or_random == "pretrained"
    assert random_.pretrained_or_random == "random"
    assert pretrained.checkpoint_path == str(ckpt_path)
    assert random_.checkpoint_path is None


def test_random_init_separable_data_beats_chance(probe_cfg: ProbeConfig) -> None:
    """With class-separable synthetic data, even random-init should beat 25% chance."""
    # Use a larger encoder for this — gives the probe more features to work with.
    encoder = EEGMaskedAutoencoder(
        n_channels=22, n_samples=1000, patch_samples=25,
        encoder_dim=128, encoder_depth=2, encoder_heads=4,
        decoder_dim=32, decoder_depth=1, decoder_heads=2,
    )
    loader, _ = _make_synthetic_loader(n_trials_per_session=120, seed=42)
    report = run_probe(encoder, loader, probe_cfg)
    # 25% is chance for 4-class; with separable data and any reasonable encoder,
    # we expect well above chance even without training.
    assert report.mean_accuracy > 0.30, (
        f"random-init probe with separable data should beat chance, "
        f"got {report.mean_accuracy:.3f}"
    )
