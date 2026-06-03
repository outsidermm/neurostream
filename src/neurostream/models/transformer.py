"""Pre-norm transformer block used in both encoder and decoder of the MAE."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class TransformerBlock(nn.Module):
    """A single pre-norm transformer block.

    Layout: ``x + Attn(LN(x))`` followed by ``x + MLP(LN(x))``. Pre-norm
    is empirically more stable than post-norm for transformers trained
    from scratch (Xiong et al. 2020) and is the standard configuration
    for ViT and MAE.

    Uses ``nn.MultiheadAttention`` which dispatches to fused
    scaled-dot-product-attention kernels (Flash Attention 2 /
    memory-efficient attention) on CUDA when available.

    Attributes:
        dim: Token embedding dimension.
        n_heads: Number of attention heads. Must divide ``dim``.
        mlp_ratio: MLP hidden-to-input dimension ratio.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim <= 0 or n_heads <= 0:
            raise ValueError(
                f"dim and n_heads must be positive; got dim={dim}, n_heads={n_heads}"
            )
        if dim % n_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by n_heads ({n_heads})")
        if mlp_ratio <= 0.0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.dim = dim
        self.n_heads = n_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=dropout,
            bias=qkv_bias,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)

        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply attention + MLP with residual connections.

        Args:
            x: Tensor of shape ``(batch, n_tokens, dim)``.

        Returns:
            Tensor of the same shape as ``x``.
        """
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x
