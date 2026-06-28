# Phase 2 — Current Status Snapshot

Point-in-time status of where execution actually stands. The forward-looking
plan lives in `docs/agents/02_PHASE_2_PLAN.md`; this file is the "where are we
right now" companion to it. When status changes, update this file — not the
plan docs, which describe intent, not state.

For the full root-cause analysis of the linear-probe data mismatch and its
ablation table, see [`probe-data-mismatch.md`](probe-data-mismatch.md); this
file only summarises it.

## Phase 1 — complete

Tagged `v0.1.0`. EEGNet baseline reproduced at ~67–68% mean accuracy
(within-subject), matching Lawhern et al. 2018 within tolerance. A repo
hygiene pass (secrets sweep, dataset terms compliance, no private paths in
committed docs) before making the repo public has **not** been done yet.

## Phase 2 — pretraining done, evaluation in progress

### Done

- **MAE pretrained: 1.2M steps at batch 64** (not the recipe's batch 256 —
  forced down by the 8GB VRAM ceiling; LR linearly scaled down, step count
  raised ~4× to compensate).
- Checkpoints at `checkpoints/phase2_batch64_1.2m/`. Milestone checkpoints at
  200k/400k/600k/800k/1.2M. **Not all confirmed present on disk:** 400k, 600k,
  800k, 1.2M have been loaded and probed; 200k has not been confirmed. Rolling
  checkpoints (last 5 only) are mostly pruned by now.
- **Linear probe built and run** (frozen encoder + per-subject
  `LogisticRegression` on mean-pooled patch tokens, with a random-init
  control).

### The linear-probe data mismatch (root cause found)

Initial probe showed a flat ~3–5pp pretrained-vs-random gap across all
milestones — "pretraining did nothing." Root cause was **preprocessing
parity**, not window length: the probe loader resampled + z-scored but never
applied the corpus's band-pass (0.5–45 Hz) + CAR, feeding the frozen encoder
out-of-distribution inputs. Fixing preprocessing lifted the gap to **+14.18pp
(3-seed sweep, range +13.2…+15.9)** — at the 15pp threshold within measurement
resolution. Window length was shown *not* to be a meaningful lever once
preprocessing was fixed.

Full diagnosis, ablation table, and error bars:
[`probe-data-mismatch.md`](probe-data-mismatch.md).

Conclusion from that note: the gap is statistically indistinguishable from the
15pp bar after fixing a genuine bug → **proceed to fine-tuning**; do not gate
it on chasing the last ~1pp.

### Blocking item before fine-tuning

The fix lives only in the probe-only loader `data/bci_iv_harmonised.py`. The
**canonical** loader `data/bci_iv_loader.py` (used by `training/train.py`, and
what Days 12–14 fine-tuning would use) still lacks band-pass + CAR. Fine-tuning
against it would silently re-inherit the same OOD bug — and a bad fine-tune
result is far harder to spot than a flat probe curve. **The loader fix is the
one hard gate before Days 12–14.** Tracked as step 1 in the Phase 2 plan.

### Unmade decision — 128 Hz actual vs. 250 Hz documented

The corpus was actually built and the encoder actually trained at **128 Hz**
(`configs/pretrain_corpus.yaml`, cross-checked against the checkpoint's saved
config). Several places — stale comments in `configs/model/mae_base.yaml`,
`window_dataset.py`'s docstring, and the project's prose — still say 250 Hz. At
128 Hz the encoder's 1000-sample window is 7.81 s, not the "4 s" those comments
imply. This drift caused the early window-length misdiagnosis. **No decision
has been made** between (a) correct the stale comments to 128 Hz, or (b) rebuild
the corpus + re-pretrain at 250 Hz. Tracked as a decision item in the plan.

### Not yet done

- Bootstrap CI on the preprocessing-fixed gap — *substantively already done*
  via the 3-seed control sweep in `probe-data-mismatch.md`; only a formal
  paired bootstrap over the 9 subjects remains, and it is optional, not a gate.
- Port the band-pass + CAR fix into `bci_iv_loader.py` — **superseded for the
  fine-tune path.** The canonical loader emits 256-sample (2 s) epochs, but the
  encoder hard-requires 1000 samples (`patch_embed.py` raises otherwise), so
  filtering it would not unblock fine-tuning. The fine-tune trainer instead
  reuses the tested harmonised adapter
  (`make_probe_adapter(harmonise=True, window="pad2s")`) — exact parity with the
  validated probe distribution. Porting the fix into the canonical loader
  remains worthwhile only for Phase-1-style EEGNet consistency, not as a gate.
- Re-run the milestone sweep (400k→1.2M) with fixed preprocessing — the
  existing flat-gap sweep used the broken probe pipeline.
- Days 12–14 fine-tuning — **framework landed, run pending.** Implemented and
  tested (TDD): `models/mae_classifier.py` (encoder + mean-pool + head),
  `training/optim.py::param_groups_llrd` (layer-wise LR decay via the
  scheduler's `lr_scale`), `training/mixup.py`, `training/finetune.py`
  (per-subject T-train/val → E-test, warmup→cosine, early stopping),
  `configs/finetune.yaml` + `configs/train/finetune.yaml`, `scripts/finetune.py`.
  Real-data wiring verified on A03 (adapter → `(288, 22, 1000)` → trainer →
  session-E number). The per-subject mean±std vs the ≥71% target still needs a
  run against the pretrained checkpoint on the GPU host (no checkpoint on the
  dev Mac).
- Days 15–19: ablation (mask ratio × pretraining duration; patch-size axis
  already dropped from scope under the batch-64 compute budget).
- Days 20–21: README updates, additional ADRs, `v0.2.0` release tag.

### Investigation artefacts now in the repo

- `src/neurostream/data/bci_iv_harmonised.py` — probe-only loader with the fix
  (TDD, 11 tests).
- `src/neurostream/data/window_extract.py` — continuous-window utility (the
  "config c" approach; kept though it wasn't the right lever for this problem).
- `scripts/probe_ablation.py` — reproduces the 4-config ablation table.
- `docs/phase-2-notes/probe-data-mismatch.md` — full writeup with error bars.
