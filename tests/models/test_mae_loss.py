"""Loss-computation correctness tests for the EEG MAE."""

from __future__ import annotations

import torch

from neurostream.models.mae import EEGMaskedAutoencoder


def test_zero_mask_ratio_gives_zero_loss_safely() -> None:
    """No masked patches => no loss to compute, and no div-by-zero."""
    model = EEGMaskedAutoencoder(mask_ratio=0.0)
    loss, _, mask = model(torch.randn(2, 22, 1000))
    assert torch.isfinite(loss)
    assert mask.sum().item() == 0.0
    assert loss.item() == 0.0


def test_loss_is_computed_only_on_masked_patches() -> None:
    """Recomputing the masked MSE manually should match the model's loss exactly.

    This is the single most important correctness test for the MAE:
    a bug in the loss-masking masks (e.g., wrong sign, wrong axis,
    averaging over all patches) is silent and devastating for downstream
    transfer. We re-derive the loss from ``pred`` and ``mask`` and assert
    equality.
    """
    torch.manual_seed(0)
    model = EEGMaskedAutoencoder(mask_ratio=0.5)
    x = torch.randn(4, 22, 1000)
    loss_model, pred, mask = model(x)

    target = model.patchify(x)
    if model.norm_pix_loss:
        m = target.mean(dim=-1, keepdim=True)
        v = target.var(dim=-1, keepdim=True, unbiased=False)
        target = (target - m) / torch.sqrt(v + 1e-6)
    per_patch = (pred - target).pow(2).mean(dim=-1)
    loss_manual = (per_patch * mask).sum() / mask.sum().clamp_min(1.0)

    assert torch.allclose(loss_model, loss_manual, atol=1e-5)


def test_visible_patch_predictions_do_not_affect_loss() -> None:
    """Perturbing predictions at *visible* positions should leave loss unchanged."""
    torch.manual_seed(0)
    model = EEGMaskedAutoencoder(mask_ratio=0.5)
    x = torch.randn(4, 22, 1000)
    _, pred, mask = model(x)

    # Compute the baseline loss from the same pred/mask via forward_loss.
    loss_a = model.forward_loss(x, pred, mask)

    # Perturb only visible positions (where mask == 0) and recompute.
    perturbed = pred.clone()
    visible = (mask == 0.0).unsqueeze(-1)  # (B, N, 1)
    perturbed = torch.where(
        visible, perturbed + 100.0 * torch.randn_like(perturbed), perturbed
    )
    loss_b = model.forward_loss(x, perturbed, mask)

    assert torch.allclose(loss_a, loss_b, atol=1e-5)


def test_norm_pix_loss_changes_loss_value() -> None:
    """norm_pix_loss should produce a different loss from the un-normalised version."""
    torch.manual_seed(0)
    x = torch.randn(2, 22, 1000)

    m1 = EEGMaskedAutoencoder(mask_ratio=0.5, norm_pix_loss=True)
    m2 = EEGMaskedAutoencoder(mask_ratio=0.5, norm_pix_loss=False)
    m2.load_state_dict(m1.state_dict())  # identical weights apart from the flag

    # Reset RNG so masking is identical across the two forwards.
    torch.manual_seed(123)
    l1, _, _ = m1(x)
    torch.manual_seed(123)
    l2, _, _ = m2(x)

    assert not torch.allclose(l1, l2)
