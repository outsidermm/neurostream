"""Mixup augmentation for fine-tuning (Zhang et al. 2018).

Convex-combines pairs of windows and their labels, regularising the
classifier toward linear behaviour between training examples. The Phase 2
fine-tune recipe uses ``alpha=0.2``. Randomness flows through an explicit
:class:`torch.Generator` so runs stay reproducible without touching global
RNG state.
"""

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


def mixup_batch(
    x: Tensor,
    y: Tensor,
    alpha: float = 0.2,
    *,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, Tensor, float]:
    """Mix a batch with a Beta(alpha, alpha)-sampled ratio.

    Args:
        x: Inputs ``(B, ...)``.
        y: Integer class labels ``(B,)``.
        alpha: Beta concentration. ``0`` disables mixing (``lam == 1``).
        generator: Seeded RNG for the permutation and the mix ratio.

    Returns:
        ``(mixed_x, y_a, y_b, lam)`` where the mixed input is
        ``lam * x + (1 - lam) * x[perm]`` and the loss should blend the two
        label sets with weights ``lam`` / ``1 - lam``.
    """
    if alpha <= 0.0:
        return x, y, y, 1.0

    # numpy's Beta is seeded off the torch generator for one reproducible draw.
    seed = int(torch.randint(0, 2**31 - 1, (1,), generator=generator).item())
    lam = float(np.random.default_rng(seed).beta(alpha, alpha))

    perm = torch.randperm(x.shape[0], generator=generator)
    mixed = lam * x + (1.0 - lam) * x[perm]
    return mixed, y, y[perm], lam


def mixup_loss(
    criterion: nn.Module,
    logits: Tensor,
    y_a: Tensor,
    y_b: Tensor,
    lam: float,
) -> Tensor:
    """Blend a criterion over the two label sets: ``lam*L(y_a) + (1-lam)*L(y_b)``."""
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)


__all__ = ["mixup_batch", "mixup_loss"]
