"""Shape and parameter-count tests for the EEG MAE."""

from __future__ import annotations

import pytest
import torch

from neurostream.models.mae import EEGMaskedAutoencoder


@pytest.fixture
def model() -> EEGMaskedAutoencoder:
    return EEGMaskedAutoencoder()


def test_forward_returns_expected_shapes(model: EEGMaskedAutoencoder) -> None:
    x = torch.randn(4, 22, 1000)
    loss, pred, mask = model(x)
    assert loss.shape == ()
    assert loss.dtype == torch.float32
    assert pred.shape == (4, 40, 22 * 25)
    assert mask.shape == (4, 40)


def test_parameter_count_in_target_range(model: EEGMaskedAutoencoder) -> None:
    """Default config should land at ~5.4M parameters."""
    n_params = sum(p.numel() for p in model.parameters())
    assert 4_000_000 < n_params < 6_500_000, f"unexpected parameter count: {n_params:,}"


def test_unpatchify_inverts_patchify(model: EEGMaskedAutoencoder) -> None:
    x = torch.randn(2, 22, 1000)
    assert torch.allclose(model.unpatchify(model.patchify(x)), x, atol=1e-6)


def test_input_validation_rejects_wrong_shape(
    model: EEGMaskedAutoencoder,
) -> None:
    with pytest.raises(ValueError):
        model(torch.randn(4, 21, 1000))  # wrong n_channels
    with pytest.raises(ValueError):
        model(torch.randn(4, 22, 999))  # wrong n_samples
    with pytest.raises(ValueError):
        model(torch.randn(22, 1000))  # wrong rank


def test_constructor_validates_patch_divisibility() -> None:
    with pytest.raises(ValueError):
        EEGMaskedAutoencoder(n_samples=1001, patch_samples=25)


def test_constructor_validates_mask_ratio() -> None:
    with pytest.raises(ValueError):
        EEGMaskedAutoencoder(mask_ratio=1.0)
    with pytest.raises(ValueError):
        EEGMaskedAutoencoder(mask_ratio=-0.1)


def test_constructor_validates_head_divisibility() -> None:
    with pytest.raises(ValueError):
        EEGMaskedAutoencoder(encoder_dim=256, encoder_heads=7)


def test_encode_returns_all_tokens(model: EEGMaskedAutoencoder) -> None:
    """`encode` should return CLS + n_patches tokens without any masking."""
    x = torch.randn(3, 22, 1000)
    z = model.encode(x)
    assert z.shape == (3, 41, 256)  # 1 CLS + 40 patch tokens


def test_state_dict_roundtrip(model: EEGMaskedAutoencoder) -> None:
    """A fresh instance can load weights from another and produce same output."""
    x = torch.randn(2, 22, 1000)
    torch.manual_seed(0)
    loss_a, _, _ = model(x)

    twin = EEGMaskedAutoencoder()
    twin.load_state_dict(model.state_dict())
    torch.manual_seed(0)
    loss_b, _, _ = twin(x)

    assert torch.allclose(loss_a, loss_b, atol=1e-6)
