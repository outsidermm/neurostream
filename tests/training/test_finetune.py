"""Tests for the MAE fine-tuning loop."""

from __future__ import annotations

from typing import cast

import numpy as np
import torch
import torch.nn as nn

from neurostream.models.mae import EEGMaskedAutoencoder
from neurostream.models.mae_classifier import MAEClassifier
from neurostream.training.finetune import (
    EarlyStopping,
    FinetuneConfig,
    freeze_unused_decoder,
    stratified_train_val_split,
    train_one_subject,
)


def _small_clf() -> MAEClassifier:
    enc = EEGMaskedAutoencoder(
        n_channels=22,
        n_samples=1000,
        patch_samples=25,
        encoder_dim=64,
        encoder_depth=2,
        encoder_heads=4,
        decoder_dim=32,
        decoder_depth=1,
        decoder_heads=2,
    )
    return MAEClassifier(enc, n_classes=4)


def _synthetic(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 4, size=n)
    # Give each class a faint mean offset so the task is weakly learnable.
    x = rng.standard_normal((n, 22, 1000)).astype(np.float32)
    x += (y[:, None, None] - 1.5) * 0.05
    return x, y.astype(np.int64)


# ── EarlyStopping ────────────────────────────────────────────────────────────


def test_early_stopping_triggers_after_patience() -> None:
    es = EarlyStopping(patience=2)
    assert es.update(1.0) is True  # first value always improves
    assert not es.should_stop
    assert es.update(0.9) is True  # improved
    assert es.update(0.95) is False  # bad 1
    assert not es.should_stop
    assert es.update(0.96) is False  # bad 2
    assert not es.should_stop
    assert es.update(0.97) is False  # bad 3 > patience
    assert es.should_stop
    assert es.best == 0.9


# ── Helpers ──────────────────────────────────────────────────────────────────


def test_freeze_unused_decoder() -> None:
    clf = _small_clf()
    freeze_unused_decoder(clf)
    for name, p in clf.named_parameters():
        if "decoder" in name or name.endswith("mask_token"):
            assert not p.requires_grad, name
    # Encoder + head stay trainable.
    assert clf.encoder.patch_embed.proj.weight.requires_grad
    assert clf.head[-1].weight.requires_grad


def test_stratified_split_keeps_all_classes() -> None:
    x, y = _synthetic(80, seed=1)
    xtr, ytr, xval, yval = stratified_train_val_split(x, y, val_fraction=0.25, seed=0)
    assert len(xtr) + len(xval) == 80
    assert set(np.unique(ytr)) == set(np.unique(y))
    assert set(np.unique(yval)) == set(np.unique(y))
    # No index overlap by reconstructing counts.
    assert abs(len(xval) / 80 - 0.25) < 0.1


# ── End-to-end smoke ─────────────────────────────────────────────────────────


def test_train_one_subject_smoke() -> None:
    torch.manual_seed(0)
    clf = _small_clf()
    freeze_unused_decoder(clf)

    train_x, train_y = _synthetic(48, seed=2)
    val_x, val_y = _synthetic(24, seed=3)
    test_x, test_y = _synthetic(24, seed=4)

    cfg = FinetuneConfig(
        base_lr=1e-3,
        epochs=3,
        warmup_epochs=1,
        patience=5,
        batch_size=8,
        mixup_alpha=0.2,
    )

    head_weight = cast(nn.Linear, clf.head[-1]).weight
    before = head_weight.detach().clone()
    result = train_one_subject(
        clf,
        train_x,
        train_y,
        val_x,
        val_y,
        test_x,
        test_y,
        cfg,
        device=torch.device("cpu"),
        subject_id=3,
    )

    # Training actually moved the head weights.
    assert not torch.equal(before, head_weight.detach())
    assert 0.0 <= result.test_accuracy <= 1.0
    assert 1 <= result.epochs_run <= cfg.epochs
    assert np.isfinite(result.best_val_loss)


def test_train_one_subject_early_stops() -> None:
    """With patience 0 the loop must stop before exhausting all epochs."""
    torch.manual_seed(0)
    clf = _small_clf()
    freeze_unused_decoder(clf)
    train_x, train_y = _synthetic(32, seed=5)
    val_x, val_y = _synthetic(16, seed=6)
    cfg = FinetuneConfig(
        epochs=40, warmup_epochs=1, patience=0, batch_size=8, mixup_alpha=0.0
    )
    result = train_one_subject(
        clf,
        train_x,
        train_y,
        val_x,
        val_y,
        val_x,
        val_y,
        cfg,
        device=torch.device("cpu"),
        subject_id=1,
    )
    assert result.epochs_run < cfg.epochs
