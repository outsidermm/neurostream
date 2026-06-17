"""Tests for selective weight-decay parameter splitting."""

from __future__ import annotations

import torch
import torch.nn as nn

from neurostream.training.optim import split_weight_decay_param_groups


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
