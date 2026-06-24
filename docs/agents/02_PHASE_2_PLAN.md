# Phase 2 Plan — MAE Pretraining → Fine-tune

The active phase. This file is the ordered plan to finish it. For where
execution actually stands right now (what's built, what's blocked), see
[`phase-2-status.md`](../phase-2-notes/phase-2-status.md).

## Goal

Show that self-supervised MAE pretraining on an aggregated open EEG corpus
transfers to BCI IV 2a motor imagery: fine-tuned encoder ≥ **71% mean accuracy**
(≥3pp over Phase 1's ~68% EEGNet baseline), within-subject, fully reproducible.

## Definition of done

- Fine-tuned model clears ≥71% mean accuracy across the 9 subjects.
- The full pipeline (pretrain → probe → fine-tune → eval) reproduces from a
  short, documented script sequence with no manual intervention.
- `v0.2.0` tagged, README + ADRs updated to match what was actually run.

## Foundations already in place

MAE pretrained (1.2M steps, batch 64 — see `03_ARCHITECTURE_AND_DECISIONS.md`
for why these differ from the recipe), linear probe built and run, and the
probe data-mismatch root-caused and fixed in the probe path. The fix lifted the
probe gap to ~+14pp, at the 15pp bar within measurement resolution — pretraining
demonstrably does substantial work. Details and error bars:
[`probe-data-mismatch.md`](../phase-2-notes/probe-data-mismatch.md).

## The one hard gate

**Port the band-pass + CAR fix into the canonical loader before any fine-tuning
runs.** This is the single dependency every source agrees on: fine-tuning
against the unfixed canonical loader silently re-inherits the out-of-distribution
bug the probe investigation just diagnosed, and a bad fine-tune is far harder to
notice than a flat probe curve. Step 1 below.

## Ordered plan

### 1. Port band-pass + CAR into the canonical loader — **BLOCKING**

The fix lives only in the probe-only `data/bci_iv_harmonised.py`. The canonical
`data/bci_iv_loader.py` (used by `training/train.py`, and what fine-tuning will
use) still lacks it. Port band-pass (0.5–45 Hz) + CAR over — or route the
canonical loader through the harmonised preprocessing — **with its own tests**
(this touches the supervised training path, so it is a separate, independently
tested change, not bundled into the probe finding). Verify parity: the two
loaders should produce statistically equivalent per-channel mean/std on a shared
sanity check. **Nothing in steps 4+ runs until this lands.**

### 2. Decide: 128 Hz vs. 250 Hz — **DECISION REQUIRED**

The corpus was built and the encoder trained at **128 Hz**, but the spec and
several stale comments say 250 Hz (see `03_ARCHITECTURE_AND_DECISIONS.md`).
Make an explicit, recorded decision:

- **(a)** Correct the stale doc/code comments to 128 Hz; leave corpus +
  checkpoint as-is. Cheap. 128 Hz comfortably covers the 0.5–45 Hz band per
  Nyquist, so representation quality is likely unaffected (untested).
- **(b)** Treat 128 Hz as the bug; rebuild the corpus + re-pretrain at 250 Hz
  to match the original spec. Expensive (full rebuild + ~1.2M-step re-pretrain).

Default toward (a) unless evidence says 128 Hz hurts. Record the decision as an
ADR — do not leave it ambiguous, since this drift already caused one debugging
detour.

### 3. (Optional) Formalise the probe gap CI

A 3-seed control sweep already puts the gap at +14.18pp (range +13.2…+15.9),
at threshold within resolution. A formal paired bootstrap over the 9 subjects
(resample subjects with replacement, recompute mean gap ~1000×, take 2.5/97.5
percentiles) would tighten this to a single honest CI — a ~10-line addition to
the existing probe/ablation scripts. **This is a nice-to-have, not a gate:** the
worked-through analysis already recommends proceeding to fine-tuning.

### 4. Re-run the milestone sweep with fixed preprocessing

The existing 400k→1.2M sweep showed a flat ~5pp gap, but used the broken
(no band-pass/CAR) probe pipeline. Re-run it with the fixed preprocessing to get
an honest "probe accuracy vs. pretraining duration" curve — and to learn whether
the encoder genuinely plateaued early or whether that plateau was an artifact of
the preprocessing bug suppressing signal equally at every checkpoint.

### 5. Days 12–14 — Fine-tuning

Pretrained encoder + new classification head, end-to-end fine-tune:
- Layer-wise LR decay (decay factor 0.65–0.75).
- Mixup (alpha=0.2).
- Per-subject independent fine-tuning (9 runs), reported as mean ± std.
- Target: ≥71% mean accuracy (≥3pp over Phase 1).

Uses the fixed canonical loader from step 1.

### 6. Days 15–19 — Ablation study

- Mask ratio (0.25 / 0.50 / 0.75) × pretraining duration (reusing existing
  milestone checkpoints — no extra pretraining for the duration axis).
- The patch-size axis is **out of scope** under the batch-64 compute budget
  (anticipated fallback). The evaluation harness here depends on the same
  probe/fine-tune pipeline, so it follows steps 1, 4, and 5.

### 7. Days 20–21 — Documentation and release

README updates, additional ADRs (including the step-2 decision), `v0.2.0`
release tag. Blocked on the above producing a real, verified result to document.

## Dependency summary

```
1 (loader fix) ──┬──> 4 (re-run sweep)
                 └──> 5 (fine-tune) ──> 6 (ablation) ──> 7 (docs + v0.2.0)
2 (Hz decision) ─────> 7 (record as ADR)
3 (bootstrap CI) ── optional, independent
```
