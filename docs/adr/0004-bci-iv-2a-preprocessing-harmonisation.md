# ADR 0004: Harmonise BCI IV 2a preprocessing to match the pretraining corpus

## Status
Accepted — established by ablation, not by upfront design.

## Context
After completing 1.2M-step MAE pretraining on the open motor-imagery corpus,
the linear probe evaluation showed a pretrained-vs-random-init gap of only
~5 percentage points — far below the 15pp threshold required by the Phase 2
spec. The gap was flat across the entire milestone sweep (400k→1.2M steps),
ruling out undertraining.

This ADR documents the investigation that identified the root cause and the
resulting preprocessing decision.

## The investigation

### Hypothesis 1 (wrong): window-length mismatch

The BCI IV 2a loader extracts 2-second epochs (tmin=0.5, tmax=2.5 relative
to cue). The MAE encoder expects 1000-sample inputs. The initial assumption
was that the corpus was harmonised at 250 Hz (per PHASE_2.md's stated spec),
making 1000 samples = 4 seconds. This would mean BCI IV 2a's 2-second epochs
were being zero-padded by ~50%.

A "fix" was attempted: extract a 4-second window (tmin=-0.5, tmax=3.5)
from BCI IV 2a instead.

**Result: the gap got WORSE (3.7pp, down from 5.1pp).** Both the pretrained
and random-init arms dropped. The longer window diluted actual motor-imagery
signal with pre-cue baseline and post-trial rest period.

### Discovery: the corpus is actually at 128 Hz, not 250 Hz

A parallel investigation (Claude Code CLI session) discovered that
`configs/pretrain_corpus.yaml` has `target_fs: 128`, contradicting
PHASE_2.md's stated 250 Hz. This was confirmed by reading the checkpoint's
saved config directly — the encoder's 1000-sample window actually represents
7.81 seconds at 128 Hz, not 4 seconds at 250 Hz.

This made the window-length hypothesis structurally wrong: at 128 Hz, BCI IV
2a's 2-second epoch provides only 256 real samples out of 1000 (74%
zero-padding), and the "4-second fix" only reduced this to 512/1000 (49%)
while introducing non-task signal.

### Hypothesis 2 (correct): preprocessing mismatch

The pretraining corpus harmonisation pipeline applies bandpass 0.5–45 Hz and
common average reference (CAR) to every recording. The BCI IV 2a probe
loader applied neither — only resampling and per-window z-score
normalization. Per-channel z-scoring is an affine transform that cannot undo
a missing CAR (a spatial operation across channels) or a missing bandpass
(a spectral operation). The encoder was receiving structurally different input
from what it was trained on.

### Ablation confirming the root cause

Four configurations, all on the 1.2M-step checkpoint:

| Config | Preprocessing | Window | Gap (pretrained − random) |
|---|---|---|---|
| a | raw (no filter/CAR) | 2s padded | +3.16pp |
| b | **+bandpass +CAR** | **2s padded** | **+13.46pp** |
| c | +bandpass +CAR | continuous 7.8s | +6.44pp |
| d | +bandpass +CAR | 4s padded | +13.31pp |

Key findings:
- **a→b: preprocessing is the dominant lever** (+3pp → +13.5pp, ~4x
  improvement), lifting the pretrained arm specifically.
- **b≈d: window length is not a lever** once preprocessing is fixed (2-second
  and 4-second padded windows perform identically).
- **c: continuous long windows actively hurt** — diluting task-relevant signal
  with non-imagery EEG is worse than zero-padding.

## Decision
All downstream evaluation and fine-tuning pipelines that feed data to the
pretrained MAE encoder must apply the same preprocessing as the pretraining
corpus harmonisation pipeline: bandpass 0.5–45 Hz followed by common average
reference (CAR), before any further normalization.

This is a constraint that follows from how the encoder was trained, not a
tuning choice. Omitting it collapses the learned representation advantage
toward random.

## Consequences
**Positive:**
- The pretrained-vs-random gap rises from ~5pp (meaningless) to ~13.5pp
  (approaching the 15pp spec threshold), validating that pretraining did
  learn useful EEG representations.
- The ablation provides a clean, documented explanation for a result that
  initially appeared to indicate pretraining failure.

**Negative:**
- The BCI IV 2a preprocessing pipeline now diverges from Phase 1's
  convention (Phase 1 used 8–30 Hz bandpass, which is task-specific and
  was applied in the model's preprocessing, not the broader 0.5–45 Hz
  corpus-level filter). Phase 2 evaluation uses 0.5–45 Hz + CAR to match
  the pretrained encoder's expected input distribution. This is documented
  but adds complexity.
- The fix has only been applied to the probe-specific loader
  (`bci_iv_harmonised.py`). The canonical loader (`bci_iv_loader.py`) used
  by the training script still lacks the fix. This MUST be resolved before
  fine-tuning.

## Lessons learned

1. **Preprocessing consistency is load-bearing in transfer learning.** A
   frozen encoder is a function that expects a specific input distribution.
   Feeding it inputs from a different distribution (different filtering,
   different spatial reference) collapses any learned advantage, regardless
   of how well the encoder was trained. This is not a subtle effect — it
   was a ~4x difference in the diagnostic metric.

2. **When a frozen-encoder probe underperforms, check preprocessing parity
   with the pretraining corpus before investigating architecture, training
   duration, or window-length issues.** The wrong initial hypothesis
   (window-length mismatch) was plausible but cost a full day of
   investigation before being falsified. The correct hypothesis
   (preprocessing mismatch) was cheaper to test and produced an
   unambiguous result.

3. **Config files can drift from what was actually executed.** The 250 Hz
   figure in PHASE_2.md and several config comments turned out to be stale —
   the real rate was 128 Hz, confirmed only by reading the checkpoint's
   saved config directly. Treat design docs and YAML comments as claims that
   need verification, not as ground truth.
