"""Learning rate schedules for MAE pretraining and downstream fine-tuning.

The pretraining schedule is linear warmup followed by cosine decay to
zero — the He et al. 2022 recipe, identical to the original Transformer
and ViT schedules. The fine-tuning schedule is exposed too because it
shares the same warmup-then-cosine structure.
"""

from __future__ import annotations

import math


def warmup_cosine_lr(
    step: int,
    base_lr: float,
    warmup_steps: int,
    total_steps: int,
    min_lr: float = 0.0,
) -> float:
    """Compute the learning rate at a given step.

    Phases:
      * ``step < warmup_steps``: linear ramp from 0 to ``base_lr``.
      * ``warmup_steps <= step < total_steps``: cosine decay from
        ``base_lr`` to ``min_lr``.
      * ``step >= total_steps``: clamp at ``min_lr``.

    Args:
        step: Current step (0-indexed).
        base_lr: Peak learning rate, reached at end of warmup.
        warmup_steps: Number of linear warmup steps.
        total_steps: Total number of training steps. Must be greater
            than ``warmup_steps``.
        min_lr: Floor of the cosine decay. Defaults to 0.

    Returns:
        Learning rate at this step.
    """
    if total_steps <= warmup_steps:
        raise ValueError(
            f"total_steps ({total_steps}) must exceed warmup_steps ({warmup_steps})"
        )
    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}")
    if base_lr < min_lr:
        raise ValueError(f"base_lr ({base_lr}) must be >= min_lr ({min_lr})")

    if step < warmup_steps:
        # Linear warmup. Use (step + 1) so step 0 isn't exactly 0
        # (avoids a "zero learning step" at the very first iteration).
        return base_lr * (step + 1) / max(1, warmup_steps)

    if step >= total_steps:
        return min_lr

    # Cosine decay over the remaining steps.
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def apply_lr(optimizer: object, lr: float) -> None:
    """Set ``lr`` on every parameter group of an optimizer.

    Use this with optimizers that have a single LR (the pretraining
    case). For layer-wise LR decay (fine-tuning), use the per-group
    LRs set up in :mod:`neurostream.training.optim`.
    """
    # Typed loosely (``object``) because torch optimizers don't share a
    # clean public base type for the param_groups attribute.
    for pg in optimizer.param_groups:  # type: ignore[attr-defined]
        # Scale by any group-local lr_scale (used for LLRD fine-tuning).
        scale = pg.get("lr_scale", 1.0)
        pg["lr"] = lr * scale


__all__ = ["warmup_cosine_lr", "apply_lr"]
