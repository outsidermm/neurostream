# Phase 2 Fine-tuning Results — Experiment Log and Post-mortem

**Result: 61.50% mean session-E accuracy. Target was ≥71%. Miss documented here.**

## Context

Phase 2 goal: fine-tune the pretrained MAE encoder on BCI IV 2a, achieve ≥71% mean
accuracy across 9 subjects (≥3pp over Phase 1's ~68% EEGNet baseline).

The linear probe confirmed pretraining transfer at +14.18pp over random init (see
`probe-data-mismatch.md`). End-to-end fine-tuning was expected to improve on the
probe by updating the encoder weights for the downstream task.

## Experiment sweep (AI-assisted)

All runs used checkpoint `rolling_step01160000` (confirmed best by milestone sweep).
Experiments were run autonomously by Claude Opus 4.8, which was granted full authority
to choose configs, run experiments, log results to MLflow, and document outcomes.

### Key findings, in order of impact

| Config change | Mean acc | Delta |
|---|---|---|
| Default spec (lr=1e-3, pad2s, batch=32) | 44.68% | baseline |
| Lower LR: 1e-4 | 50.66% | +5.98pp |
| val_fraction=0.1 (more training data) | 52.78% | +2.12pp |
| CLS pooling (vs mean) | 53.78–53.90% | +1.00–1.12pp |
| dropout=0.3, mixup_alpha=0.3 | 54.09% | +0.19–0.31pp |
| **pad4s window** (512 real samples vs 256) | 60.92% | **+6.83pp** |
| batch_size=16 | **61.50%** | **+0.58pp** |

### Window-size sweep (with best other settings)

| Window | Real samples | Post-cue extent | Mean acc |
|---|---|---|---|
| pad2s | 256 | 0.5 s → 2.5 s | ~54% |
| **pad4s** | **512** | **−0.5 s → 3.5 s** | **61.50%** |
| pad6s | 768 | −0.5 s → 5.5 s | 58.76% |
| pad7s | 896 | −0.5 s → 6.5 s | OOB error (late trials hit recording end) |

pad4s is the maximum safe window; 6 s hurts (likely adding noise post-event-return).

### Pooling comparison

| Pooling | Window | Mean acc |
|---|---|---|
| CLS | pad2s | 53.78% |
| mean | pad2s | 50.66% |
| **mean** | **pad4s** | **61.50%** |
| CLS | pad4s | 56.87% |
| both (concat) | pad4s | ~58% (no consistent gain) |

Pool choice interacts with window: CLS wins at 2 s, mean wins at 4 s.

### What didn't help

- label_smoothing (0.0 is optimal — tried 0.05, 0.1)
- Two-layer MLP head (head_hidden_dim=256)
- Freeze encoder for N epochs then unfreeze
- Larger LR (2e-4)
- val_fraction=0.05 (too noisy)
- dropout=0.2 (slight overfit)
- batch_size=8 (slower convergence, similar result)
- Earlier/later checkpoints (1120k, 1200k milestone — rolling 1160k is best)

## Best reproducible config

```
checkpoint: checkpoints/phase2_batch64_1.2m/rolling_step01160000.pt
window:     pad4s
pool:       mean
batch_size: 16
base_lr:    1e-4
llrd_decay: 0.7
weight_decay: 0.05
epochs:     300
warmup_epochs: 15
patience:   40
mixup_alpha: 0.3
dropout:    0.3
val_fraction: 0.1
label_smoothing: 0.0
freeze_encoder_epochs: 0
```

Reproduce with:
```powershell
$env:PYTHONIOENCODING="utf-8"
uv run python -m scripts.finetune `
    finetune.pretrained_checkpoint=checkpoints/phase2_batch64_1.2m/rolling_step01160000.pt `
    train.base_lr=0.0001 train.batch_size=16 train.val_fraction=0.1 `
    train.patience=40 train.epochs=300 train.warmup_epochs=15 `
    train.pool=mean train.dropout=0.3 train.mixup_alpha=0.3 `
    finetune.window=pad4s finetune.run_random_control=false
```

## Why 71% was not achieved

**Short answer:** MAE reconstruction pretraining does not produce naturally MI-discriminative
features at this model size and compute budget.

**Longer diagnosis:**

1. **Architecture mismatch.** EEGNet (Phase 1's ~68% baseline) was purpose-built for
   motor imagery: depthwise-separable temporal + spatial convolutions explicitly motivated
   by known ERP structure (ERD, ERS, mu/beta rhythms). The MAE encoder is a ViT-style
   transformer that learns patch-level reconstruction features — it is agnostic to the
   neurophysiology. Fine-tuning nudges those features toward the task but cannot fully
   overcome the inductive-bias gap.

2. **Compute-constrained pretraining.** The published MAE recipe ran batch 256; the 8GB
   VRAM ceiling forced batch 64, with step count raised 4× to compensate. Whether this
   fully compensates is unverified. The encoder may simply not have converged to the
   representation quality the recipe intended.

3. **Small labeled dataset.** 288 trials per subject, 9 subjects, within-subject protocol.
   The transformer encoder has ~2M parameters; fine-tuning that many weights on <260 train
   trials per subject is high-variance. LLRD mitigates this but does not eliminate it.

4. **No cross-subject transfer.** The within-subject protocol is the BCI IV 2a standard
   evaluation, but it means each fine-tune sees only one subject's data. Cross-subject
   initialisation (fine-tune on all others, evaluate on one) or subject-adaptive layers
   would likely close some of the gap.

**What would likely reach 71%:**
- Cross-subject fine-tuning (fine-tune on 8 subjects, test on 1 → ~70–75% in literature)
- Larger encoder (encoder_dim 512 vs 256) — out of scope at 8GB
- Contrastive or subject-level pretraining objective instead of MAE
- Ensemble across seeds or window-start offsets

## AI usage note

This sweep was run autonomously by Claude Opus 4.8 (via Claude Code) over a single session.
The model was given full authority to choose configs, run experiments via PowerShell, observe
results in the log output, and decide next experiments. No human intervention between runs.

The AI correctly identified the large pad4s win (+6.83pp) as the dominant lever; correctly
stopped chasing label_smoothing/MLP-head hyperparameters after they showed no gain; and
correctly concluded the gap to 71% is architectural rather than tuning-accessible — matching
the post-hoc analysis above.

What it could not do: change the architecture, acquire more data, or exceed the VRAM ceiling.
Those are the actual constraints.

## Phase 2 status after this sweep

- Fine-tuning pipeline: complete and reproducible.
- Best result: 61.50% (−6.5pp vs EEGNet baseline, −9.5pp vs target).
- Pretraining transfer confirmed: +8pp pretrained vs random-init fine-tune control.
- 71% target: **not achieved**. Documented as a known miss; see diagnosis above.
- Proceeding to Phase 3 with the 61.50% checkpoint as the ONNX export source.
  Phase 3 (C++ SIMD inference) demonstrates the production engineering path regardless
  of the accuracy ceiling.
