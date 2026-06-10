"""Optimizer setup utilities.

Centralises the "which parameters get weight decay" decision. Following
He et al. 2022 (and BERT/ViT before): biases, LayerNorm gains/shifts,
positional embeddings, and the CLS/mask special tokens are excluded from
weight decay. Everything else (Linear weights, conv weights) gets decayed.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch.nn as nn


def split_weight_decay_param_groups(
    model: nn.Module,
    weight_decay: float,
    no_decay_names: Iterable[str] = ("cls_token", "mask_token"),
) -> list[dict[str, object]]:
    """Build optimizer ``param_groups`` with selective weight decay.

    A parameter is placed in the ``no_decay`` group if any of the following
    is true:
      * Its ``.ndim`` is 1 (biases, LayerNorm gains, 1D embeddings).
      * Its qualified name contains any of ``no_decay_names``.

    Args:
        model: The model being optimized.
        weight_decay: Weight decay value applied to the decayed group.
        no_decay_names: Substrings that exempt a parameter from decay.

    Returns:
        A two-group list ready to pass as the first argument to
        :class:`torch.optim.AdamW`. Frozen parameters are excluded.
    """
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    no_decay_substrings = tuple(no_decay_names)

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or any(s in name for s in no_decay_substrings):
            no_decay.append(p)
        else:
            decay.append(p)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


__all__ = ["split_weight_decay_param_groups"]
