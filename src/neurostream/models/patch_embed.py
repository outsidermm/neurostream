"""Patch embedding for the EEG masked autoencoder.

Converts an EEG window of shape ``(channels, samples)`` into a sequence
of token embeddings by chunking along the time axis and projecting each
chunk to the model dimension.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class EEGPatchEmbed(nn.Module):
    """Patchify and embed EEG into transformer tokens.

    Implemented as a strided 1D convolution over the time axis with
    ``kernel_size == stride == patch_samples``. This is mathematically
    equivalent to "reshape into non-overlapping ``(channels,
    patch_samples)`` patches, flatten, then apply a linear projection",
    but compiles to a single fused operation on GPU.

    Attributes:
        n_channels: Number of EEG channels in the input.
        n_samples: Number of time samples in each input window.
        patch_samples: Number of time samples per patch.
        embed_dim: Output token dimension.
        n_patches: ``n_samples // patch_samples``.
    """

    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        patch_samples: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        if n_channels <= 0 or n_samples <= 0 or embed_dim <= 0:
            raise ValueError(
                "n_channels, n_samples, embed_dim must all be positive; "
                f"got n_channels={n_channels}, n_samples={n_samples}, "
                f"embed_dim={embed_dim}"
            )
        if patch_samples <= 0:
            raise ValueError(f"patch_samples must be positive, got {patch_samples}")
        if n_samples % patch_samples != 0:
            raise ValueError(
                f"n_samples ({n_samples}) must be divisible by "
                f"patch_samples ({patch_samples})"
            )

        self.n_channels = n_channels
        self.n_samples = n_samples
        self.patch_samples = patch_samples
        self.embed_dim = embed_dim
        self.n_patches = n_samples // patch_samples

        self.proj = nn.Conv1d(
            in_channels=n_channels,
            out_channels=embed_dim,
            kernel_size=patch_samples,
            stride=patch_samples,
            bias=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Embed a batch of EEG windows.

        Args:
            x: Tensor of shape ``(batch, n_channels, n_samples)``.

        Returns:
            Tensor of shape ``(batch, n_patches, embed_dim)``.

        Raises:
            ValueError: If the input rank or per-sample shape does not
                match the configured ``(n_channels, n_samples)``.
        """
        if x.ndim != 3:
            raise ValueError(f"expected 3D input (B, C, T), got {x.ndim}D")
        if x.shape[1] != self.n_channels or x.shape[2] != self.n_samples:
            raise ValueError(
                f"expected (B, {self.n_channels}, {self.n_samples}), "
                f"got {tuple(x.shape)}"
            )
        x = self.proj(x)  # (B, embed_dim, n_patches)
        return x.transpose(1, 2).contiguous()  # (B, n_patches, embed_dim)
