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
- Linear-probe and full-fine-tune numbers reported **separately** (per the
  original spec), each against the random-init control.
- The "held-out" claim is backed by a source-isolation test (step 4b) — no
  pretraining shard contains BCI IV 2a data.
- The full pipeline reproduces from a short, documented script sequence with no
  manual intervention — a colleague with Docker + a CUDA GPU can run
  `download-pretrained → finetune → evaluate` and land within ±2pp per subject,
  and separately re-run the corpus build + pretrain end-to-end. The scripts that
  exist (`pretrain.sh`, `ingest_open_corpus.py`, `probe_*`) need a fine-tune
  script and an `evaluate`/report script added to complete this surface.
- Corpus shards + milestone checkpoints tracked in DVC (not git); sidecar JSON
  committed to git. DVC is not yet initialised.
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

### 4b. Source-isolation test — defends the "held-out" claim

A one-time corpus audit: assert that no pretraining shard contains BCI IV 2a
data (the held-out evaluation set). The five corpus sources use different
subjects and equipment, so overlap is unlikely — but a one-line assertion turns
"unlikely" into "verified" and bulletproofs the held-out claim against scrutiny.
Cheap, independent of the loader fix; do it before reporting any transfer number.

### 5. Days 12–14 — Fine-tuning

Pretrained encoder (decoder dropped) + new mean-pool classification head,
end-to-end fine-tune, per-subject (9 independent runs, reported mean ± std).
Recipe from the original spec (not yet run — treat as the starting recipe, not a
validated config):

| Hyperparameter | Value |
|---|---|
| optimizer | AdamW, weight_decay 0.05 |
| base LR | 1e-3 |
| layer-wise LR decay | 0.7 (range 0.65–0.75) — lower encoder layers get smaller LRs, head gets base LR |
| batch size | 32 |
| epochs | 100, early stopping (patience 15 on val loss) |
| schedule | 10-epoch linear warmup, then cosine decay |
| mixup | alpha 0.2 |

Target: ≥71% mean accuracy (≥3pp over Phase 1), and above EEGNet on ≥6/9
subjects (subject variance is real — you won't dominate every subject).

Uses the fixed canonical loader from step 1. If fine-tune underperforms,
diagnose in order: did it converge? → is LLRD breaking (try uniform LR)? → are
the pretrained weights actually loaded (compare an encoder weight sum before/
after `load_state_dict`)? → is the mask ratio wrong (defer to step 6)?

### 6. Days 15–19 — Ablation study

- Mask ratio (0.25 / 0.50 / 0.75) × pretraining duration (reusing existing
  milestone checkpoints — no extra pretraining for the duration axis).
- The patch-size axis is **out of scope** under the batch-64 compute budget
  (anticipated fallback). The evaluation harness here depends on the same
  probe/fine-tune pipeline, so it follows steps 1, 4, and 5.

### 7. Days 20–21 — Documentation and release

- README updates (Phase 2 results table alongside Phase 1, link to the ablation
  report), additional ADRs including the step-2 decision.
- **Reproducibility scripts** to complete the DoD surface: a per-subject
  fine-tune script and an `evaluate`/HTML-report script (the ablation report
  lands in `results/`), plus a `download-pretrained` path so a reviewer can
  reproduce eval without re-pretraining.
- **DVC**: initialise DVC, configure a non-git remote, and track the harmonised
  corpus shards + milestone checkpoints (sidecar JSON stays in git). Raw MOABB
  sources and BCI IV 2a are re-downloadable and stay untracked.
- `v0.2.0` release tag; upload the winning ablation checkpoint as a release
  asset. Blocked on the above producing a real, verified result to document.

## Dependency summary

```
1 (loader fix) ──┬──> 4 (re-run sweep)
                 └──> 5 (fine-tune) ──> 6 (ablation) ──> 7 (docs + v0.2.0)
2 (Hz decision) ─────> 7 (record as ADR)
3 (bootstrap CI) ─── optional, independent
4b (source-isolation) ─── independent; precedes reporting any transfer number
```
