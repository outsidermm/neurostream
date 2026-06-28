"""Tests for the MAE fine-tuning classifier head."""

from __future__ import annotations

import pytest
import torch

from neurostream.models.mae import EEGMaskedAutoencoder
from neurostream.models.mae_classifier import MAEClassifier


@pytest.fixture
def small_encoder() -> EEGMaskedAutoencoder:
    """Small encoder for fast tests (same shape conventions as the real one)."""
    return EEGMaskedAutoencoder(
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


def test_forward_shape(small_encoder: EEGMaskedAutoencoder) -> None:
    clf = MAEClassifier(small_encoder, n_classes=4)
    logits = clf(torch.randn(8, 22, 1000))
    assert logits.shape == (8, 4)


def test_rejects_wrong_sample_length(small_encoder: EEGMaskedAutoencoder) -> None:
    """The encoder's patch-embed hard-requires n_samples=1000."""
    clf = MAEClassifier(small_encoder, n_classes=4)
    with pytest.raises(ValueError):
        clf(torch.randn(2, 22, 256))


def test_gradients_reach_encoder_when_unfrozen(
    small_encoder: EEGMaskedAutoencoder,
) -> None:
    clf = MAEClassifier(small_encoder, n_classes=4)
    for p in clf.parameters():
        p.grad = None

    logits = clf(torch.randn(4, 22, 1000))
    logits.sum().backward()

    # A representative encoder weight must receive a gradient.
    enc_grad = clf.encoder.patch_embed.proj.weight.grad
    assert enc_grad is not None and torch.any(enc_grad != 0)
    # Head must receive a gradient too.
    assert clf.head[-1].weight.grad is not None


def test_exposes_encoder_and_head(small_encoder: EEGMaskedAutoencoder) -> None:
    """LLRD grouping relies on `.encoder` / `.head` being addressable."""
    clf = MAEClassifier(small_encoder, n_classes=4)
    assert clf.encoder is small_encoder
    head_names = [n for n, _ in clf.named_parameters() if n.startswith("head.")]
    assert head_names, "head params must be namespaced under 'head.'"
