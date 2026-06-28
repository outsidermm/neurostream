"""Tests for selective weight-decay parameter splitting."""

from __future__ import annotations

from typing import cast

import pytest
import torch
import torch.nn as nn

from neurostream.models.mae import EEGMaskedAutoencoder
from neurostream.models.mae_classifier import MAEClassifier
from neurostream.training.optim import (
    param_groups_llrd,
    split_weight_decay_param_groups,
)


class _Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(8, 16)  # weight decays, bias doesn't
        self.norm = nn.LayerNorm(16)  # neither weight nor bias decays
        self.conv = nn.Conv1d(16, 32, 3)  # weight decays, bias doesn't
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 16))  # no decay
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 16))  # no decay


def test_groups_decay_2d_weights() -> None:
    groups = split_weight_decay_param_groups(_Toy(), weight_decay=0.05)
    decay_group, no_decay_group = groups
    assert decay_group["weight_decay"] == 0.05
    assert no_decay_group["weight_decay"] == 0.0

    decay_params = decay_group["params"]
    # Linear weight + Conv1d weight = 2 decayed params.
    assert len(decay_params) == 2
    assert all(p.ndim >= 2 for p in decay_params)


def test_groups_exempt_biases_and_norms_and_special_tokens() -> None:
    model = _Toy()
    groups = split_weight_decay_param_groups(model, weight_decay=0.05)
    no_decay_params = groups[1]["params"]

    # Linear bias, LayerNorm gain, LayerNorm bias, Conv1d bias, CLS, mask = 6
    assert len(no_decay_params) == 6


def test_frozen_parameters_excluded() -> None:
    model = _Toy()
    # Freeze linear.weight — it should appear in neither group.
    model.linear.weight.requires_grad = False
    groups = split_weight_decay_param_groups(model, weight_decay=0.05)
    flat_ids = {id(p) for g in groups for p in g["params"]}
    assert id(model.linear.weight) not in flat_ids


def test_passes_into_adamw_without_error() -> None:
    """Smoke test: groups should be directly usable by AdamW."""
    model = _Toy()
    groups = split_weight_decay_param_groups(model, weight_decay=0.05)
    opt = torch.optim.AdamW(groups, lr=1e-3, betas=(0.9, 0.95))
    # Run one synthetic step to ensure update is well-defined.
    x = torch.randn(2, 8)
    y = model.linear(x).sum()
    y.backward()
    opt.step()


# ── Layer-wise LR decay (fine-tuning) ────────────────────────────────────────


def _llrd_clf(encoder_depth: int = 4) -> MAEClassifier:
    enc = EEGMaskedAutoencoder(
        n_channels=22,
        n_samples=1000,
        patch_samples=25,
        encoder_dim=64,
        encoder_depth=encoder_depth,
        encoder_heads=4,
        decoder_dim=32,
        decoder_depth=1,
        decoder_heads=2,
    )
    return MAEClassifier(enc, n_classes=4)


def _scale_by_param_id(groups: list[dict]) -> dict[int, float]:
    return {id(p): g["lr_scale"] for g in groups for p in g["params"]}


def test_llrd_scales_head_block_and_stem() -> None:
    """Head gets full LR; each lower encoder block is decayed one extra power."""
    depth, decay = 4, 0.7
    clf = _llrd_clf(encoder_depth=depth)
    # Decoder is unused in the classifier forward; exclude it from optimisation.
    for name, p in clf.encoder.named_parameters():
        if "decoder" in name or name == "mask_token":
            p.requires_grad = False

    groups = param_groups_llrd(clf, base_lr=1e-3, decay=decay, weight_decay=0.05)
    scale = _scale_by_param_id(groups)

    # Head (top): scale 1.0
    assert scale[id(clf.head[-1].weight)] == pytest.approx(1.0)
    # Last encoder block (layer depth): decay**1
    assert scale[
        id(cast(nn.LayerNorm, clf.encoder.encoder_blocks[depth - 1].norm1).weight)
    ] == pytest.approx(decay**1)
    # First encoder block (layer 1): decay**(depth-0)
    assert scale[
        id(cast(nn.LayerNorm, clf.encoder.encoder_blocks[0].norm1).weight)
    ] == pytest.approx(decay**depth)
    # Patch-embed stem (layer 0): decay**(depth+1)
    assert scale[id(clf.encoder.patch_embed.proj.weight)] == pytest.approx(
        decay ** (depth + 1)
    )
    # cls_token shares the stem layer.
    assert scale[id(clf.encoder.cls_token)] == pytest.approx(decay ** (depth + 1))


def test_llrd_preserves_no_decay_rule() -> None:
    """Biases / norms / special tokens still get weight_decay=0 within their layer."""
    clf = _llrd_clf()
    for name, p in clf.encoder.named_parameters():
        if "decoder" in name or name == "mask_token":
            p.requires_grad = False

    groups = param_groups_llrd(clf, base_lr=1e-3, decay=0.7, weight_decay=0.05)
    wd_by_id = {id(p): g["weight_decay"] for g in groups for p in g["params"]}

    # Linear weight in the head decays; its bias and the LayerNorm gain do not.
    assert wd_by_id[id(clf.head[-1].weight)] == 0.05
    assert wd_by_id[id(clf.head[-1].bias)] == 0.0
    assert wd_by_id[id(clf.head[0].weight)] == 0.0  # LayerNorm gain


def test_llrd_excludes_frozen_params() -> None:
    clf = _llrd_clf()
    for name, p in clf.encoder.named_parameters():
        if "decoder" in name or name == "mask_token":
            p.requires_grad = False
    groups = param_groups_llrd(clf, base_lr=1e-3, decay=0.7, weight_decay=0.05)
    flat_ids = {id(p) for g in groups for p in g["params"]}
    # No decoder param should appear.
    assert (
        id(cast(nn.LayerNorm, clf.encoder.decoder_blocks[0].norm1).weight)
        not in flat_ids
    )


def test_llrd_groups_usable_by_adamw_with_apply_lr() -> None:
    from neurostream.training.scheduler import apply_lr

    clf = _llrd_clf()
    for name, p in clf.encoder.named_parameters():
        if "decoder" in name or name == "mask_token":
            p.requires_grad = False
    groups = param_groups_llrd(clf, base_lr=1e-3, decay=0.7, weight_decay=0.05)
    opt = torch.optim.AdamW(groups, lr=1e-3, betas=(0.9, 0.95))
    apply_lr(opt, 1e-3)
    # Per-group LR is base_lr * lr_scale.
    for pg in opt.param_groups:
        assert pg["lr"] == pytest.approx(1e-3 * pg["lr_scale"])
    clf(torch.randn(2, 22, 1000)).sum().backward()
    opt.step()
