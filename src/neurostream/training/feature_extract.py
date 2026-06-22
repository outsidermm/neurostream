"""Feature extraction from a (frozen) MAE encoder.

Used by the linear-probe and fine-tuning code paths to convert raw EEG
windows into encoder-feature vectors. The encoder is run in eval mode
with no gradients; pooling over patch tokens is configurable.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from neurostream.models.mae import EEGMaskedAutoencoder

PoolMode = Literal["cls", "mean", "concat"]


@torch.no_grad()
def extract_features(
    encoder: EEGMaskedAutoencoder,
    x: torch.Tensor,
    pool: PoolMode = "mean",
    batch_size: int = 64,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """Run the frozen encoder over a batch of windows and pool to per-sample vectors.

    Args:
        encoder: Pretrained MAE. Will be moved to ``device`` and set to eval mode.
        x: Tensor of shape ``(N, n_channels, n_samples)``.
        pool: How to reduce the per-token outputs to a single vector per sample:
              ``"cls"``    — use the CLS token (position 0).
              ``"mean"``   — mean-pool over patch tokens (positions 1..n_patches).
              ``"concat"`` — concatenate CLS with the mean of patch tokens.
        batch_size: Mini-batch size for the encoder forward pass.
        device: Device to run the encoder on.

    Returns:
        ``(N, feature_dim)`` numpy array. ``feature_dim`` equals the encoder dim
        for ``"cls"``/``"mean"`` and twice the encoder dim for ``"concat"``.
    """
    if x.ndim != 3:
        raise ValueError(f"expected 3D input (N, C, T), got {x.ndim}D")
    if pool not in {"cls", "mean", "concat"}:
        raise ValueError(f"unknown pool mode: {pool}")

    device = torch.device(device)
    encoder = encoder.to(device).eval()

    loader = DataLoader(
        TensorDataset(x),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    chunks: list[np.ndarray] = []
    for (batch,) in loader:
        batch = batch.to(device, non_blocking=True)
        tokens = encoder.encode(batch)        # (B, 1 + n_patches, encoder_dim)
        feat = _pool_tokens(tokens, pool)     # (B, feature_dim)
        chunks.append(feat.cpu().numpy())

    return np.concatenate(chunks, axis=0)


def _pool_tokens(tokens: torch.Tensor, pool: PoolMode) -> torch.Tensor:
    """Reduce ``(B, 1+N, D)`` token outputs to ``(B, feature_dim)``."""
    cls_token = tokens[:, 0, :]            # (B, D)
    patch_tokens = tokens[:, 1:, :]        # (B, N, D)

    if pool == "cls":
        return cls_token
    if pool == "mean":
        return patch_tokens.mean(dim=1)
    if pool == "concat":
        return torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=-1)
    raise AssertionError(f"unreachable: pool={pool}")


def load_encoder_from_checkpoint(
    checkpoint_path: str,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> EEGMaskedAutoencoder:
    """Reconstruct an MAE from a Phase 2 pretraining checkpoint.

    The checkpoint must contain the resolved Hydra config under ``state["config"]``
    so we know which model configuration to instantiate.

    Args:
        checkpoint_path: Path to the ``.pt`` file written by ``CheckpointManager``.
        map_location: Where to map tensors during load.
        strict: If True, refuse mismatched state_dicts. Set False when loading
            a partial state (e.g., encoder-only into a model that still has
            decoder params).

    Returns:
        An ``EEGMaskedAutoencoder`` with the checkpoint's weights loaded.
    """
    state = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

    if "config" not in state:
        raise ValueError(
            f"checkpoint at {checkpoint_path} has no 'config' key — "
            "can't reconstruct model architecture"
        )
    model_cfg = state["config"]["model"]

    # Build kwargs by stripping Hydra's _target_ field.
    kwargs = {k: v for k, v in model_cfg.items() if not k.startswith("_")}
    model = EEGMaskedAutoencoder(**kwargs)

    # Strip DDP prefix if present.
    msd = {k.removeprefix("module."): v for k, v in state["model"].items()}
    model.load_state_dict(msd, strict=strict)
    return model


def make_random_init_encoder(
    reference_checkpoint_path: str,
    *,
    seed: int = 0,
) -> EEGMaskedAutoencoder:
    """Construct a randomly-initialised encoder with the same architecture as a checkpoint.

    Used as the control in linear-probe evaluation: pretrained linear-probe accuracy
    must exceed random-init linear-probe accuracy by ≥15pp for the pretraining to
    be considered useful.
    """
    state = torch.load(reference_checkpoint_path, map_location="cpu", weights_only=False)
    model_cfg = state["config"]["model"]
    kwargs = {k: v for k, v in model_cfg.items() if not k.startswith("_")}

    with torch.random.fork_rng():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None
        return EEGMaskedAutoencoder(**kwargs)


__all__ = [
    "PoolMode",
    "extract_features",
    "load_encoder_from_checkpoint",
    "make_random_init_encoder",
]
