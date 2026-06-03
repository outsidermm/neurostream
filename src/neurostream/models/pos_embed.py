"""1D positional embeddings for transformer models.

Provides fixed sinusoidal positional encodings as used in Vaswani et al. 2017
and adopted by ViT (Dosovitskiy et al. 2020) and MAE (He et al. 2022).
"""

from __future__ import annotations

import torch
from torch import Tensor


def build_1d_sincos_pos_embed(n_positions: int, embed_dim: int) -> Tensor:
    """Build a 1D sinusoidal positional embedding table.

    Uses the alternating sin/cos formulation from Vaswani et al. 2017
    with a base period of 10,000. The first half of dimensions are sin
    components, the second half are cos components.

    Args:
        n_positions: Number of positions to embed (including the CLS slot
            if used).
        embed_dim: Dimension of each positional vector. Must be even.

    Returns:
        Tensor of shape ``(n_positions, embed_dim)``, dtype ``float32``,
        located on CPU. The caller is expected to register this as a
        non-persistent buffer so it follows ``.to(device)`` without
        bloating the checkpoint.

    Raises:
        ValueError: If ``embed_dim`` is not divisible by 2, or
            ``n_positions`` is non-positive.
    """
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even, got {embed_dim}")
    if n_positions <= 0:
        raise ValueError(f"n_positions must be positive, got {n_positions}")

    pos = torch.arange(n_positions, dtype=torch.float32).unsqueeze(1)
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega = 1.0 / (10_000.0 ** (omega / float(embed_dim // 2)))
    angles = pos * omega.unsqueeze(0)
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
