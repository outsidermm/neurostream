"""Fine-tuning classifier built on a pretrained MAE encoder.

Wraps an :class:`EEGMaskedAutoencoder`, runs its (no-masking) ``encode``
path, pools the patch tokens, and applies a small classification head.

Pooling modes:
  ``"mean"``  — average over patch tokens (matches the linear probe default).
  ``"cls"``   — use the CLS token at index 0 directly.
  ``"both"``  — concatenate CLS and mean-patch vectors (2 × encoder_dim head input).

The encoder and head are exposed as named submodules (``.encoder`` /
``.head``) so layer-wise LR decay can target each block; see
:func:`neurostream.training.optim.param_groups_llrd`.
"""

import torch
import torch.nn as nn
from torch import Tensor

from neurostream.models.mae import EEGMaskedAutoencoder

Pool = str  # "mean" | "cls" | "both"


class MAEClassifier(nn.Module):
    """Pretrained MAE encoder + pooling + linear classification head.

    Args:
        encoder: A (typically pretrained) MAE. The decoder is unused here.
        n_classes: Number of output classes (BCI IV 2a has 4).
        dropout: Dropout applied to the pooled features before the head.
        pool: Pooling strategy — ``"mean"``, ``"cls"``, or ``"both"``.
    """

    def __init__(
        self,
        encoder: EEGMaskedAutoencoder,
        n_classes: int = 4,
        dropout: float = 0.5,
        pool: Pool = "mean",
        head_hidden_dim: int = 0,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.pool = pool
        in_dim = encoder.encoder_dim * 2 if pool == "both" else encoder.encoder_dim
        if head_hidden_dim > 0:
            self.head = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Dropout(dropout),
                nn.Linear(in_dim, head_hidden_dim),
                nn.GELU(),
                nn.LayerNorm(head_hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(head_hidden_dim, n_classes),
            )
        else:
            self.head = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Dropout(dropout),
                nn.Linear(in_dim, n_classes),
            )

    def forward(self, x: Tensor) -> Tensor:
        """Classify a batch of EEG windows ``(B, n_channels, n_samples)``."""
        tokens = self.encoder.encode(x)  # (B, 1 + n_patches, encoder_dim)
        cls = tokens[:, 0, :]
        mean = tokens[:, 1:, :].mean(dim=1)
        if self.pool == "cls":
            pooled = cls
        elif self.pool == "both":
            pooled = torch.cat([cls, mean], dim=-1)
        else:
            pooled = mean
        return self.head(pooled)


__all__ = ["MAEClassifier"]
