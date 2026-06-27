# ADR 0002: Masked autoencoder over contrastive methods for self-supervised pretraining

## Status
Accepted

## Context
Phase 2 needs a self-supervised pretraining method for EEG representation
learning. The encoder will be pretrained on ~290 subjects of unlabeled
motor-imagery EEG, then evaluated via linear probe and fine-tuning on BCI
Competition IV Dataset 2a. The method must produce transferable features from
a heterogeneous multi-source corpus.

## Decision
Masked Autoencoder (MAE), following He et al. 2022, adapted for EEG:
- Temporal patches (22 channels × 25 samples = 100ms), not spatial patches.
- 50% mask ratio (lower than vision MAE's 75%, since EEG is lower-dimensional
  per token).
- Asymmetric encoder-decoder: 6-block encoder (256-dim, 8 heads), 2-block
  decoder (128-dim, 4 heads). ~5.4M parameters total.
- Reconstruction loss on masked patches only, with per-patch normalization.

## Alternatives considered
- **SimCLR / MoCo**: require strong augmentations. EEG augmentations (channel
  dropout, time masking, jitter) are semantically weaker than image
  augmentations (crop, color jitter). Lower expected ceiling for EEG
  specifically.
- **BYOL / DINO**: no negative samples, more stable training. But require
  careful momentum-encoder tuning — higher engineering cost for a comparable
  ceiling on this data modality.
- **JEPA**: predicts in representation space rather than pixel space. Growing
  evidence it outperforms MAE for low-dimensional signals. Strongest
  alternative on quality grounds. Rejected for Phase 2 only because MAE has
  more reproducible reference implementations to compare against — not
  rejected as a method. Worth revisiting in a future ablation if time
  permits.

## Consequences
- MAE's asymmetric design means compute scales with visible (unmasked) tokens
  only — efficient for pretraining at batch 64 (our VRAM-constrained batch
  size).
- Reconstruction is in raw signal space, which means loss values are
  interpretable but also means the model can drive loss down by learning
  trivial local-autocorrelation features without learning semantically useful
  representations. Linear probe is the only reliable signal for downstream
  usefulness — reconstruction loss alone is necessary but not sufficient.
- The honest framing: "we picked MAE because it's the most reproducible
  starting point for this modality," not "MAE is the best method."
