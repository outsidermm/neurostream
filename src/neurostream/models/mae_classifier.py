"""Fine-tuning classifier built on a pretrained MAE encoder.

Wraps an :class:`EEGMaskedAutoencoder`, runs its (no-masking) ``encode``
path, mean-pools the patch tokens — matching the linear probe's default
``pool="mean"`` — and applies a small classification head.

The encoder and head are exposed as named submodules (``.encoder`` /
``.head``) so layer-wise LR decay can target each block; see
:func:`neurostream.training.optim.param_groups_llrd`.
"""

import torch.nn as nn
from torch import Tensor

from neurostream.models.mae import EEGMaskedAutoencoder


class MAEClassifier(nn.Module):
    """Pretrained MAE encoder + mean-pool + linear classification head.

    Args:
        encoder: A (typically pretrained) MAE. The decoder is unused here.
        n_classes: Number of output classes (BCI IV 2a has 4).
        dropout: Dropout applied to the pooled features before the head.
    """

    def __init__(
        self,
        encoder: EEGMaskedAutoencoder,
        n_classes: int = 4,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.LayerNorm(encoder.encoder_dim),
            nn.Dropout(dropout),
            nn.Linear(encoder.encoder_dim, n_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Classify a batch of EEG windows ``(B, n_channels, n_samples)``."""
        tokens = self.encoder.encode(x)  # (B, 1 + n_patches, encoder_dim)
        pooled = tokens[:, 1:, :].mean(dim=1)  # mean over patch tokens
        return self.head(pooled)


__all__ = ["MAEClassifier"]
