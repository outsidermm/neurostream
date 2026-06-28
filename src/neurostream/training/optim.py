"""Optimizer setup utilities.

Centralises the "which parameters get weight decay" decision. Following
He et al. 2022 (and BERT/ViT before): biases, LayerNorm gains/shifts,
positional embeddings, and the CLS/mask special tokens are excluded from
weight decay. Everything else (Linear weights, conv weights) gets decayed.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch.nn as nn

from neurostream.models.mae_classifier import MAEClassifier


def split_weight_decay_param_groups(
    model: nn.Module,
    weight_decay: float,
    no_decay_names: Iterable[str] = ("cls_token", "mask_token"),
) -> list[dict[str, Any]]:
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


def _llrd_layer_id(name: str, num_blocks: int) -> int:
    """Map a classifier parameter name to its layer-wise-decay depth index.

    Layers, deepest-from-the-head first:
      * 0                — patch-embed stem + CLS token (gets the smallest LR).
      * ``1 .. num_blocks`` — encoder block ``i`` is layer ``i + 1``.
      * ``num_blocks + 1``  — encoder norm + classification head (full LR).
    """
    if "patch_embed" in name or "cls_token" in name:
        return 0
    if "encoder_blocks." in name:
        # name like "encoder.encoder_blocks.<i>.<...>"
        idx = int(name.split("encoder_blocks.")[1].split(".")[0])
        return idx + 1
    # encoder_norm, head, and anything else ride at the top.
    return num_blocks + 1


def param_groups_llrd(
    model: MAEClassifier,
    base_lr: float,
    decay: float = 0.70,
    weight_decay: float = 0.05,
    no_decay_names: Iterable[str] = ("cls_token", "mask_token"),
) -> list[dict[str, Any]]:
    """Build AdamW ``param_groups`` with layer-wise learning-rate decay (LLRD).

    Lower encoder layers — closer to the input, further from the loss — get
    exponentially smaller learning rates than the head (He et al. 2022 §A.2).
    Each group carries an ``lr_scale`` (``decay ** (top - layer_id)``) rather
    than an absolute LR, so the shared warmup→cosine schedule can drive every
    group through :func:`neurostream.training.scheduler.apply_lr`.

    The selective-weight-decay rule (biases / norms / special tokens excluded)
    is preserved *within* each layer, so a layer can yield up to two groups.
    Frozen parameters (``requires_grad=False`` — e.g. the unused MAE decoder)
    are skipped.

    Args:
        model: The :class:`MAEClassifier` being fine-tuned.
        base_lr: Peak LR for the head (``lr_scale == 1.0``).
        decay: Per-layer multiplicative decay in ``(0, 1]``.
        weight_decay: Weight decay for the decayed group.
        no_decay_names: Substrings exempting a parameter from weight decay.

    Returns:
        A list of param-group dicts, each with ``params``, ``weight_decay`` and
        ``lr_scale`` keys, ready for ``torch.optim.AdamW``.
    """
    if not 0.0 < decay <= 1.0:
        raise ValueError(f"decay must be in (0, 1], got {decay}")

    num_blocks = len(model.encoder.encoder_blocks)
    top = num_blocks + 1
    no_decay_substrings = tuple(no_decay_names)

    # (layer_id, is_no_decay) -> list[param]
    buckets: dict[tuple[int, bool], list[nn.Parameter]] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        layer_id = _llrd_layer_id(name, num_blocks)
        is_no_decay = p.ndim <= 1 or any(s in name for s in no_decay_substrings)
        buckets.setdefault((layer_id, is_no_decay), []).append(p)

    groups: list[dict[str, Any]] = []
    for (layer_id, is_no_decay), params in sorted(buckets.items()):
        groups.append(
            {
                "params": params,
                "weight_decay": 0.0 if is_no_decay else weight_decay,
                "lr_scale": decay ** (top - layer_id),
                "lr": base_lr * decay ** (top - layer_id),
            }
        )
    return groups


__all__ = ["split_weight_decay_param_groups", "param_groups_llrd"]
