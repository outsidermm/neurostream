"""Tests for feature extraction utilities."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from neurostream.models.mae import EEGMaskedAutoencoder
from neurostream.training.feature_extract import (
    extract_features,
    load_encoder_from_checkpoint,
    make_random_init_encoder,
)


@pytest.fixture
def small_encoder() -> EEGMaskedAutoencoder:
    """Small encoder for fast tests."""
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


def test_extract_features_shape_mean_pool(small_encoder: EEGMaskedAutoencoder) -> None:
    x = torch.randn(10, 22, 1000)
    feats = extract_features(small_encoder, x, pool="mean", batch_size=4)
    assert feats.shape == (10, 64)  # encoder_dim
    assert feats.dtype.name == "float32"


def test_extract_features_shape_cls_pool(small_encoder: EEGMaskedAutoencoder) -> None:
    x = torch.randn(10, 22, 1000)
    feats = extract_features(small_encoder, x, pool="cls", batch_size=4)
    assert feats.shape == (10, 64)


def test_extract_features_shape_concat_pool(small_encoder: EEGMaskedAutoencoder) -> None:
    x = torch.randn(10, 22, 1000)
    feats = extract_features(small_encoder, x, pool="concat", batch_size=4)
    assert feats.shape == (10, 128)  # 2x encoder_dim


def test_extract_features_no_grad_required(small_encoder: EEGMaskedAutoencoder) -> None:
    """Encoder parameters must not accumulate gradients during extraction."""
    for p in small_encoder.parameters():
        p.grad = None
    x = torch.randn(4, 22, 1000)
    _ = extract_features(small_encoder, x, pool="mean")
    for p in small_encoder.parameters():
        assert p.grad is None, "feature extraction should not produce gradients"


def test_extract_features_deterministic(small_encoder: EEGMaskedAutoencoder) -> None:
    """Same input + same encoder = same features (encoder is in eval mode)."""
    x = torch.randn(4, 22, 1000)
    feats_a = extract_features(small_encoder, x, pool="mean")
    feats_b = extract_features(small_encoder, x, pool="mean")
    assert (feats_a == feats_b).all()


def test_extract_features_rejects_bad_shape(small_encoder: EEGMaskedAutoencoder) -> None:
    with pytest.raises(ValueError):
        extract_features(small_encoder, torch.randn(22, 1000), pool="mean")  # 2D


def test_extract_features_rejects_bad_pool(small_encoder: EEGMaskedAutoencoder) -> None:
    with pytest.raises(ValueError):
        extract_features(small_encoder, torch.randn(4, 22, 1000), pool="bogus")  # type: ignore[arg-type]


def test_load_encoder_from_checkpoint_roundtrip(
    small_encoder: EEGMaskedAutoencoder, tmp_path: Path
) -> None:
    """Save a model + config, load it back, verify same features."""
    ckpt_path = tmp_path / "test.pt"
    state = {
        "step": 1000,
        "model": small_encoder.state_dict(),
        "config": {
            "model": {
                "n_channels": 22,
                "n_samples": 1000,
                "patch_samples": 25,
                "encoder_dim": 64,
                "encoder_depth": 2,
                "encoder_heads": 4,
                "decoder_dim": 32,
                "decoder_depth": 1,
                "decoder_heads": 2,
            }
        },
    }
    torch.save(state, ckpt_path)

    loaded = load_encoder_from_checkpoint(ckpt_path)
    x = torch.randn(4, 22, 1000)
    feats_a = extract_features(small_encoder, x)
    feats_b = extract_features(loaded, x)
    assert (feats_a == feats_b).all()


def test_load_encoder_strips_ddp_prefix(
    small_encoder: EEGMaskedAutoencoder, tmp_path: Path
) -> None:
    """A state dict saved with DDP 'module.' prefix should still load."""
    ckpt_path = tmp_path / "ddp.pt"
    msd = {f"module.{k}": v for k, v in small_encoder.state_dict().items()}
    torch.save(
        {
            "step": 1,
            "model": msd,
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
    loaded = load_encoder_from_checkpoint(ckpt_path)
    assert sum(p.numel() for p in loaded.parameters()) > 0


def test_load_encoder_missing_config_raises(tmp_path: Path) -> None:
    ckpt_path = tmp_path / "bare.pt"
    torch.save({"step": 0, "model": {}}, ckpt_path)
    with pytest.raises(ValueError, match="no 'config' key"):
        load_encoder_from_checkpoint(ckpt_path)


def test_make_random_init_encoder_has_same_architecture(
    small_encoder: EEGMaskedAutoencoder, tmp_path: Path
) -> None:
    ckpt_path = tmp_path / "ref.pt"
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
    random_encoder = make_random_init_encoder(ckpt_path, seed=42)

    # Same parameter count and shapes...
    pretrained_params = list(small_encoder.parameters())
    random_params = list(random_encoder.parameters())
    assert len(pretrained_params) == len(random_params)
    for a, b in zip(pretrained_params, random_params):
        assert a.shape == b.shape

    # ...but different weight values (very unlikely to collide).
    x = torch.randn(4, 22, 1000)
    feats_pretrained = extract_features(small_encoder, x)
    feats_random = extract_features(random_encoder, x)
    assert not (feats_pretrained == feats_random).all()
