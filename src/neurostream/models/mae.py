"""EEG masked autoencoder (MAE) for self-supervised pretraining.

Adapted from He et al. 2022 ("Masked Autoencoders Are Scalable Vision
Learners") for EEG signals. Key adaptations:

  * Patches are **temporal slices spanning all channels**
    ``(C, patch_samples)``, not 2D spatial squares. Channels are a small
    fixed positional axis on the scalp; time is where the signal evolves.
  * Lower default mask ratio (0.50 vs. 0.75 in vision MAE) reflecting
    EEG's lower per-patch dimensionality and stronger temporal redundancy.
  * 1D sinusoidal positional encoding over the time axis only.

The encoder operates **only on the visible subset** of patches — this is
the central compute saving of MAE. The decoder receives the encoded
visible tokens plus a shared learnable mask token at the hidden
positions, with positional embeddings disambiguating them. Reconstruction
loss is computed **only at masked positions**.
"""

import torch
import torch.nn as nn
from torch import Tensor

from neurostream.models.patch_embed import EEGPatchEmbed
from neurostream.models.pos_embed import build_1d_sincos_pos_embed
from neurostream.models.transformer import TransformerBlock


class EEGMaskedAutoencoder(nn.Module):
    """Masked autoencoder for EEG self-supervised pretraining.

    Default configuration (~5.4M parameters total, ~4.9M in the encoder)
    targets a single-GPU pretraining budget of 2–3 days for ~300k steps.

    Args:
        n_channels: Number of EEG channels in the input.
        n_samples: Number of time samples per input window.
            Must be divisible by ``patch_samples``.
        patch_samples: Number of time samples per patch.
        encoder_dim: Encoder token dimension.
        encoder_depth: Number of encoder transformer blocks.
        encoder_heads: Number of attention heads per encoder block.
            Must divide ``encoder_dim``.
        encoder_mlp_ratio: MLP hidden-to-input ratio in encoder blocks.
        decoder_dim: Decoder token dimension (typically half of
            ``encoder_dim`` — the decoder is discarded after pretraining,
            so its capacity is minimised).
        decoder_depth: Number of decoder transformer blocks.
        decoder_heads: Number of attention heads per decoder block.
            Must divide ``decoder_dim``.
        decoder_mlp_ratio: MLP hidden-to-input ratio in decoder blocks.
        mask_ratio: Default fraction of patches to mask during forward.
            Must be in ``[0, 1)``. Can be overridden per-call.
        norm_pix_loss: If True, normalise each target patch to zero mean
            and unit variance before computing MSE. Improves downstream
            transfer (He et al. 2022) by removing per-patch DC offsets so
            the model focuses on signal shape rather than absolute level.
        dropout: Dropout probability inside attention and MLP layers.
    """

    encoder_pos_embed: Tensor
    decoder_pos_embed: Tensor

    def __init__(
        self,
        n_channels: int = 22,
        n_samples: int = 1000,
        patch_samples: int = 25,
        encoder_dim: int = 256,
        encoder_depth: int = 6,
        encoder_heads: int = 8,
        encoder_mlp_ratio: float = 4.0,
        decoder_dim: int = 128,
        decoder_depth: int = 2,
        decoder_heads: int = 4,
        decoder_mlp_ratio: float = 4.0,
        mask_ratio: float = 0.50,
        norm_pix_loss: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= mask_ratio < 1.0:
            raise ValueError(f"mask_ratio must be in [0, 1), got {mask_ratio}")
        if encoder_depth <= 0 or decoder_depth <= 0:
            raise ValueError(
                "encoder_depth and decoder_depth must be positive; "
                f"got encoder_depth={encoder_depth}, "
                f"decoder_depth={decoder_depth}"
            )

        self.n_channels = n_channels
        self.n_samples = n_samples
        self.patch_samples = patch_samples
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss

        # ---- Encoder ----------------------------------------------------
        self.patch_embed = EEGPatchEmbed(
            n_channels, n_samples, patch_samples, encoder_dim
        )
        self.n_patches: int = self.patch_embed.n_patches
        self.patch_dim: int = n_channels * patch_samples  # raw values/patch

        # CLS token: trainable, prepended to the visible-token sequence.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_dim))

        # Positional embedding includes a slot for CLS at index 0.
        # Stored as a non-persistent buffer (moves with .to(device) but
        # not saved into the checkpoint — it's deterministic from config).
        self.register_buffer(
            "encoder_pos_embed",
            build_1d_sincos_pos_embed(self.n_patches + 1, encoder_dim).unsqueeze(0),
            persistent=False,
        )

        self.encoder_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=encoder_dim,
                    n_heads=encoder_heads,
                    mlp_ratio=encoder_mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(encoder_depth)
            ]
        )
        self.encoder_norm = nn.LayerNorm(encoder_dim)

        # ---- Decoder ----------------------------------------------------
        # Project encoder latents into the (typically narrower) decoder dim.
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim, bias=True)

        # ONE learnable vector, copied to every masked position. The
        # decoder distinguishes positions via the additive pos embed below.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))

        self.register_buffer(
            "decoder_pos_embed",
            build_1d_sincos_pos_embed(self.n_patches + 1, decoder_dim).unsqueeze(0),
            persistent=False,
        )

        self.decoder_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=decoder_dim,
                    n_heads=decoder_heads,
                    mlp_ratio=decoder_mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_norm = nn.LayerNorm(decoder_dim)

        # Final projection: decoder_dim -> raw patch values (C * patch_samples)
        self.decoder_pred = nn.Linear(decoder_dim, self.patch_dim, bias=True)

        self._init_weights()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        """ViT/MAE-standard initialisation."""
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        # Apply Linear/LayerNorm inits recursively.
        self.apply(self._init_module)
        # Conv patch embedding: init like a flattened linear layer.
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))

    @staticmethod
    def _init_module(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    # ------------------------------------------------------------------
    # Patchify / unpatchify (targets and visualisation only)
    # ------------------------------------------------------------------
    def patchify(self, x: Tensor) -> Tensor:
        """Reshape ``(B, C, T)`` to ``(B, n_patches, C * patch_samples)``.

        This is a **pure reshape** used to construct reconstruction
        targets. It is intentionally distinct from
        :class:`EEGPatchEmbed` — the latter is a learned projection used
        on the encoder side. Conflating the two is a common bug that
        causes the loss to compare against transformed targets rather
        than raw signal.

        Args:
            x: Tensor of shape ``(B, n_channels, n_samples)``.

        Returns:
            Tensor of shape ``(B, n_patches, n_channels * patch_samples)``.
        """
        if x.ndim != 3:
            raise ValueError(f"expected 3D input, got {x.ndim}D")
        b, c, t = x.shape
        p = self.patch_samples
        n = t // p
        x = x.reshape(b, c, n, p)
        x = x.permute(0, 2, 1, 3).contiguous()  # (B, N, C, P)
        return x.reshape(b, n, c * p)

    def unpatchify(self, x: Tensor) -> Tensor:
        """Inverse of :meth:`patchify`. Used for visualisation only."""
        b, n, _ = x.shape
        c, p = self.n_channels, self.patch_samples
        x = x.reshape(b, n, c, p)
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.reshape(b, c, n * p)

    # ------------------------------------------------------------------
    # Random masking
    # ------------------------------------------------------------------
    @staticmethod
    def random_masking(x: Tensor, mask_ratio: float) -> tuple[Tensor, Tensor, Tensor]:
        """Per-sample random masking via the noise-shuffle idiom.

        Each sample in the batch receives an independent random mask
        without Python-level loops. Implementation follows He et al. 2022.

        Args:
            x: Tensor of shape ``(B, N, D)``.
            mask_ratio: Fraction in ``[0, 1)`` of tokens to hide.

        Returns:
            A tuple of:
              * ``x_visible``: ``(B, n_keep, D)`` — only the kept tokens.
              * ``mask``: ``(B, N)`` — value ``1`` where the token is
                masked, ``0`` where visible. Same dtype as ``x``.
              * ``ids_restore``: ``(B, N)`` long tensor — permutation
                that undoes the shuffle, used by the decoder to reorder
                tokens back into their original temporal positions.
        """
        b, n, d = x.shape
        n_keep = int(n * (1.0 - mask_ratio))

        noise = torch.rand(b, n, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)  # ascending
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # inverse perm

        ids_keep = ids_shuffle[:, :n_keep]
        x_visible = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, d)
        )

        mask = torch.ones(b, n, device=x.device, dtype=x.dtype)
        mask[:, :n_keep] = 0.0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_visible, mask, ids_restore

    # ------------------------------------------------------------------
    # Encoder / decoder
    # ------------------------------------------------------------------
    def forward_encoder(
        self, x: Tensor, mask_ratio: float
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run the encoder on the visible subset of patches.

        Args:
            x: Input EEG, ``(B, n_channels, n_samples)``.
            mask_ratio: Fraction of tokens to hide.

        Returns:
            ``(latent, mask, ids_restore)`` where ``latent`` has shape
            ``(B, 1 + n_keep, encoder_dim)`` (CLS + visible tokens).
        """
        x = self.patch_embed(x)  # (B, N, D_enc)
        # Pos embed: skip CLS slot (added separately below).
        x = x + self.encoder_pos_embed[:, 1:, :]

        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # Prepend CLS token (with its own positional embedding).
        cls = self.cls_token + self.encoder_pos_embed[:, :1, :]
        cls = cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, 1+n_keep, D)

        for blk in self.encoder_blocks:
            x = blk(x)
        x = self.encoder_norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x: Tensor, ids_restore: Tensor) -> Tensor:
        """Reconstruct the full token sequence from encoded visible tokens.

        Args:
            x: Encoder output, ``(B, 1 + n_keep, encoder_dim)``.
            ids_restore: ``(B, N)`` — inverse permutation from masking.

        Returns:
            ``(B, N, patch_dim)`` — predicted raw patches in original
            temporal order (CLS dropped).
        """
        x = self.decoder_embed(x)  # (B, 1+n_keep, D_dec)

        b = x.shape[0]
        n_total = ids_restore.shape[1]
        n_masked = n_total - (x.shape[1] - 1)  # exclude CLS
        mask_tokens = self.mask_token.expand(b, n_masked, -1)

        # Place visible tokens first, mask tokens after, then unshuffle
        # back into the original positional order.
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # (B, N, D_dec)
        x_ = torch.gather(
            x_,
            dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, x_.shape[-1]),
        )
        x = torch.cat([x[:, :1, :], x_], dim=1)  # re-attach CLS

        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)  # (B, 1+N, patch_dim)
        return x[:, 1:, :]  # drop CLS

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def forward_loss(self, x: Tensor, pred: Tensor, mask: Tensor) -> Tensor:
        """MSE on masked patches only.

        If ``norm_pix_loss`` is set, the **target** (not the prediction)
        is per-patch normalised before MSE is computed. This removes
        per-patch DC offsets that vary across electrodes and recordings.

        Args:
            x: Raw input, ``(B, n_channels, n_samples)``.
            pred: Predicted patches, ``(B, N, patch_dim)``.
            mask: ``(B, N)`` — 1 where masked.

        Returns:
            Scalar tensor: mean per-patch MSE averaged over masked patches.
        """
        target = self.patchify(x)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True, unbiased=False)
            target = (target - mean) / torch.sqrt(var + 1e-6)

        per_patch = (pred - target).pow(2).mean(dim=-1)  # (B, N)
        return (per_patch * mask).sum() / mask.sum().clamp_min(1.0)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def forward(
        self, x: Tensor, mask_ratio: float | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """End-to-end forward pass.

        Args:
            x: Input EEG, ``(B, n_channels, n_samples)``.
            mask_ratio: Optional override of the model's default mask ratio.

        Returns:
            ``(loss, pred, mask)`` where:
              * ``loss``: scalar reconstruction MSE on masked patches.
              * ``pred``: ``(B, n_patches, patch_dim)`` patch predictions.
              * ``mask``: ``(B, n_patches)`` masking pattern (1 = masked).
        """
        r = self.mask_ratio if mask_ratio is None else mask_ratio
        latent, mask, ids_restore = self.forward_encoder(x, r)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(x, pred, mask)
        return loss, pred, mask

    # ------------------------------------------------------------------
    # Downstream feature extraction (no masking)
    # ------------------------------------------------------------------
    def encode(self, x: Tensor) -> Tensor:
        """Encode the full input (no masking) for downstream use.

        Used by the linear-probe protocol (Days 10–11) and as the
        backbone of the fine-tuning model (Days 12–14). The caller
        controls gradient flow via ``torch.no_grad()`` or by freezing
        encoder parameters as appropriate.

        Args:
            x: Input EEG, ``(B, n_channels, n_samples)``.

        Returns:
            ``(B, 1 + n_patches, encoder_dim)`` — per-token
            representations, with the CLS token at index 0 and patch
            tokens at indices ``1..n_patches``.
        """
        x = self.patch_embed(x)
        x = x + self.encoder_pos_embed[:, 1:, :]
        cls = (self.cls_token + self.encoder_pos_embed[:, :1, :]).expand(
            x.shape[0], -1, -1
        )
        x = torch.cat([cls, x], dim=1)
        for blk in self.encoder_blocks:
            x = blk(x)
        return self.encoder_norm(x)
