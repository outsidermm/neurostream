# Architecture and Decisions

Condensed reference for model configs, the hyperparameters the design commits
to, and the reasoning behind the key choices. Not ceremonial ADR writeups —
just the design facts and rationale needed before changing any of this.

## Phase 1: EEGNet

- ~2,000 parameters. 3-layer CNN: temporal conv → depthwise spatial conv →
  separable conv → linear classifier. Implemented from the Lawhern et al.
  2018 paper, not copied from a reference repo.
- Trained within-subject: 9 independent runs (train on session T, eval on
  session E), reported as the mean ± std across subjects.
- Result: ~67–68% mean accuracy, matching the published 68.1% benchmark.
- Why EEGNet over ShallowConvNet / Deep4Net / EEG-Conformer: smallest,
  fastest to train, most-published track record on this exact dataset for
  benchmark verification. Higher-capacity alternatives were rejected
  specifically because the baseline's job is to be a fast, trivial-to-verify
  reference point, not to be competitive on its own. EEG-Conformer was also
  rejected because it's transformer-based, which would have muddied the later
  Phase 1-vs-Phase 2 comparison (better to contrast a non-transformer baseline
  against Phase 2's transformer encoder).

## Phase 2: MAE encoder

**Architecture** (`EEGMaskedAutoencoder`, `src/neurostream/models/mae.py`):

| Param | Value |
|---|---|
| n_channels | 22 |
| n_samples | 1000 |
| patch_samples | 25 |
| encoder_dim | 256 |
| encoder_depth | 6 |
| encoder_heads | 8 |
| decoder_dim | 128 |
| decoder_depth | 2 |
| decoder_heads | 4 |
| mask_ratio | 0.50 |
| norm_pix_loss | true |
| Total params | ~5.4M |

Patches are temporal (22 channels × 25 samples = a 100ms window of all
channels), not spatial — channels are kept together in each token since the
interesting structure in EEG is temporal, not spatial-only. Mask ratio 0.50 is
a starting point (lower than vision MAE's 75%, since EEG is lower-dimensional
per-token than images) and is the subject of the Days 15–19 ablation.

**`n_samples=1000` is a sample count, not a duration.** It corresponds to
whatever wall-clock duration 1000 samples spans at the corpus's actual sample
rate — see the sample-rate decision below.

## Pretraining corpus

- Five MOABB-wrapped open motor-imagery datasets: PhysionetMI, Cho2017,
  Lee2019_MI, Stieger2021, Schirrmeister2017. ~290 subjects combined.
- Harmonised to: 22 channels matching the BCI IV 2a montage (a deliberate
  selection to align with the downstream task's channel layout, not the
  intersection ceiling), band-pass 0.5–45 Hz, common average reference (CAR),
  sharded into ~2GB memmap `.npy` files with a sidecar JSON.

### Sample rate: 128 Hz (open decision)

The corpus was built and the encoder trained at **128 Hz**
(`configs/pretrain_corpus.yaml`, confirmed against the checkpoint's saved
config — trust the checkpoint, not loose YAML comments). At 128 Hz the
encoder's 1000-sample window is **7.81 s** of EEG. Some stale comments still
say 250 Hz / 4 s. Reconciling this — correct the comments to 128 Hz, or rebuild
and re-pretrain at 250 Hz — is an explicit open decision; see
`02_PHASE_2_PLAN.md` step 2.

## Pretraining hyperparameters

Deviate from the original recipe due to the 8GB VRAM ceiling:

| Param | Recipe | Used | Why |
|---|---|---|---|
| batch_size (per GPU) | 256 | **64** | 8GB VRAM ceiling |
| total_steps | ~300k | **1.2M** | ~4×, to compensate for smaller batch |
| base learning rate | 1.5e-4 | **3.75e-5** | linearly scaled down for batch 64 |
| warmup steps | 5% of total | 60k | |
| milestones | 50k…300k | 200k/400k/600k/800k/1.2M | |

The 4×-steps-for-4×-smaller-batch compensation is a heuristic (linear LR
scaling), not a guarantee of equivalent training quality — small-batch
transformer pretraining has a real quality ceiling more steps can't always
overcome. Whether this setup is near that ceiling is part of what the
fixed-preprocessing milestone sweep (`02_PHASE_2_PLAN.md` step 4) is meant to
establish.

## Why MAE over contrastive methods (SimCLR/MoCo/BYOL/DINO/JEPA)

- SimCLR/MoCo need strong augmentations; EEG augmentations (channel dropout,
  time masking, jitter) are weaker/less semantically meaningful than image
  augmentations — likely a lower ceiling for EEG.
- BYOL/DINO avoid negative samples and are more stable, but need careful
  momentum-encoder tuning — higher engineering cost for a comparable ceiling.
- JEPA is newer with growing evidence of outperforming MAE for low-dimensional
  signals, but was set aside for this phase only because MAE has more
  reproducible reference implementations to verify against — not on quality
  grounds. Worth reconsidering for a future ablation.

## Why an aggregated open corpus over TUH EEG

- TUH requires institutional registration — not reproducible by an arbitrary
  reader with just a GitHub account.
- TUH is dominantly clinical seizure data — weaker task-family match to motor
  imagery than the chosen open aggregate.
- The open aggregate forces real multi-source harmonisation engineering
  (different sampling rates, channel layouts, reference schemes across 5
  datasets) — a stronger, more representative signal of real data-engineering
  skill than ingesting one pre-cleaned source.
- Traded off: smaller total size (~80GB vs. TUH's ~1.7TB) and narrower clinical
  diversity. Accepted as a reasonable trade for reproducibility.

## Linear probe protocol

Frozen encoder + per-subject `sklearn.LogisticRegression` on mean-pooled
patch-token features (CLS-token and CLS+mean-concat pooling also implemented;
mean is the default, matching the MAE paper's linear-probe protocol). Features
standardized (`StandardScaler`, fit on the train session only, applied to both
train and test — same data-leakage discipline as Phase 1's normalization rule).
The random-init-encoder control uses an identical architecture with untrained
weights, built via `torch.random.fork_rng()` for reproducibility without
disturbing global RNG state. Acceptance bar: pretrained mean accuracy should
exceed random-init by ≥15 percentage points across the 9 subjects before
fine-tuning is considered justified.
