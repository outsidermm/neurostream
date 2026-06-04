"""Gradient-flow tests for the EEG MAE.

These tests catch the single most common class of MAE bugs: parameters
that look reachable but receive no gradient because the masking logic
drops their contribution from the loss-relevant path.
"""

from __future__ import annotations

import torch

from neurostream.models.mae import EEGMaskedAutoencoder


def test_gradients_flow_to_all_parameters() -> None:
    """Every learnable parameter must receive a non-zero gradient."""
    torch.manual_seed(0)
    model = EEGMaskedAutoencoder(mask_ratio=0.5)
    x = torch.randn(2, 22, 1000)
    loss, _, _ = model(x)
    loss.backward()

    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"params without grad: {missing}"

    zero = [
        n
        for n, p in model.named_parameters()
        if p.grad is not None and p.grad.abs().sum().item() == 0.0
    ]
    assert not zero, f"params with exactly-zero grad: {zero}"


def test_mask_token_receives_gradient() -> None:
    """The shared mask token must be reached by gradient when mask_ratio > 0."""
    torch.manual_seed(0)
    model = EEGMaskedAutoencoder(mask_ratio=0.5)
    loss, _, _ = model(torch.randn(2, 22, 1000))
    loss.backward()
    assert model.mask_token.grad is not None
    assert model.mask_token.grad.abs().sum().item() > 0.0


def test_loss_decreases_on_single_batch() -> None:
    """Sanity check: optimizer can drive loss down on a memorisation task."""
    torch.manual_seed(0)
    model = EEGMaskedAutoencoder(mask_ratio=0.5)
    x = torch.randn(4, 22, 1000)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses: list[float] = []
    for _ in range(20):
        loss, _, _ = model(x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.8, (
        f"loss did not meaningfully decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    )


def test_no_nan_grads_with_amp_dtype() -> None:
    """bf16 inputs should not produce NaN gradients (mixed-precision smoke test)."""
    if not torch.cuda.is_available():
        return  # bf16 autocast smoke check only meaningful on CUDA
    torch.manual_seed(0)
    model = EEGMaskedAutoencoder(mask_ratio=0.5).cuda()
    x = torch.randn(2, 22, 1000, device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss, _, _ = model(x)
    loss.backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"NaN/Inf grad in {name}"
