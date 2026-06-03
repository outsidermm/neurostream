"""Masking-behaviour tests for the EEG MAE."""

from __future__ import annotations

import torch

from neurostream.models.mae import EEGMaskedAutoencoder


def test_mask_fraction_is_exact() -> None:
    """Each sample should mask exactly ``n_patches - int(n * (1 - r))`` tokens."""
    model = EEGMaskedAutoencoder(mask_ratio=0.5)
    x = torch.randn(8, 22, 1000)
    _, _, mask = model(x)
    n_patches = mask.shape[1]
    n_keep = int(n_patches * 0.5)
    expected_masked = n_patches - n_keep
    assert torch.all(mask.sum(dim=1) == expected_masked)


def test_masks_differ_across_batch_samples() -> None:
    """Independent random masks: no two consecutive samples should match."""
    torch.manual_seed(0)
    model = EEGMaskedAutoencoder(mask_ratio=0.5)
    x = torch.randn(16, 22, 1000)
    _, _, mask = model(x)
    for i in range(15):
        assert not torch.all(mask[i] == mask[i + 1]), (
            f"samples {i} and {i + 1} produced identical masks"
        )


def test_mask_values_are_binary() -> None:
    model = EEGMaskedAutoencoder(mask_ratio=0.5)
    _, _, mask = model(torch.randn(2, 22, 1000))
    assert torch.all((mask == 0.0) | (mask == 1.0))


def test_random_masking_unit() -> None:
    """Unit test for the static masking utility itself."""
    torch.manual_seed(42)
    x = torch.randn(4, 10, 8)
    visible, mask, ids_restore = EEGMaskedAutoencoder.random_masking(x, 0.4)

    n_keep = int(10 * (1.0 - 0.4))
    assert visible.shape == (4, n_keep, 8)
    assert mask.shape == (4, 10)
    assert torch.all(mask.sum(dim=1) == 10 - n_keep)

    # ids_restore must be a valid permutation of {0..9} per row.
    for row in ids_restore:
        assert set(row.tolist()) == set(range(10))


def test_per_call_mask_ratio_override() -> None:
    """Passing mask_ratio to forward() should override the default."""
    model = EEGMaskedAutoencoder(mask_ratio=0.5)
    _, _, mask_a = model(torch.randn(4, 22, 1000), mask_ratio=0.25)
    n_patches = mask_a.shape[1]
    expected = n_patches - int(n_patches * 0.75)
    assert torch.all(mask_a.sum(dim=1) == expected)


def test_zero_mask_ratio_keeps_all_tokens() -> None:
    """At mask_ratio=0, no patches are masked and the mask is all zeros."""
    model = EEGMaskedAutoencoder(mask_ratio=0.0)
    _, _, mask = model(torch.randn(2, 22, 1000))
    assert mask.sum().item() == 0.0
