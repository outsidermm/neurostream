# Data Pipeline

The data architecture: the loaders that exist, the corpus harmonisation
pipeline, and which loader is meant for which task. For the current state of
the canonical-loader fix and the preprocessing-mismatch root cause, see
[`phase-2-status.md`](../phase-2-notes/phase-2-status.md) and
[`probe-data-mismatch.md`](../phase-2-notes/probe-data-mismatch.md).

## BCI Competition IV 2a loaders — two exist, by design

| File | Role | Preprocessing |
|---|---|---|
| `bci_iv_loader.py` | Canonical — used by `training/train.py` (Phase 1 EEGNet, and Phase 2 fine-tuning) | epoch + resample + z-score |
| `bci_iv_harmonised.py` | Probe path — used by `scripts/linear_probe.py`, `scripts/probe_ablation.py` | adds band-pass + CAR |

The key design requirement: **any path that feeds the pretrained encoder must
harmonise its inputs the same way the pretraining corpus was** (band-pass +
CAR), or the encoder sees out-of-distribution data and the value of pretraining
collapses. The probe path already does this; bringing the canonical loader up to
the same harmonisation is the gating step before fine-tuning (`02_PHASE_2_PLAN.md`
step 1).

**`bci_iv_loader.py` (canonical):**
- `mne.Epochs(raw, events, tmin=0.5, tmax=2.5)` — 2-second window relative to cue.
- Drops EOG channels, keeps 22 EEG channels.
- Resamples to the target rate.
- Per-window, per-channel z-score normalization.

**`bci_iv_harmonised.py` (probe path):**
- Same epoch extraction (or continuous-window extraction via `window_extract.py`,
  config-selectable).
- Adds band-pass 0.5–45 Hz and CAR re-reference, to match the corpus
  harmonisation pipeline.
- Same z-score normalization, applied after the above.

**`window_extract.py` (utility):**
- Extracts continuous windows directly from raw recordings (bypassing
  `mne.Epochs`'s trial-boundary limitation) — slices `n_samples` real samples
  from the continuous signal at an absolute index.
- Hypothesized as the fix for zero-padding, but the ablation showed it is not
  the right lever for the probe problem (a longer real window dilutes the
  motor-imagery signal with non-task EEG). Kept as a utility for future use.

## Open corpus aggregation pipeline (`src/neurostream/data/open_corpus.py`)

Builds the pretraining corpus. Wraps 5 MOABB datasets (PhysionetMI, Cho2017,
Lee2019_MI, Stieger2021, Schirrmeister2017) into a single harmonised shard set.

`harmonise(raw, target_channels, target_fs)`, per recording:
1. Select/reorder to the 22 BCI IV 2a channels (a selection aligned with the
   downstream task's channel layout, not the full cross-dataset intersection).
2. Resample to `target_fs` via `scipy.signal.resample_poly` (polyphase
   filtering, chosen over FFT-based resampling for better biosignal fidelity).
3. Band-pass filter 0.5–45 Hz.
4. Re-reference to CAR.
5. Reject recordings with >10% NaN, >5% samples with peak-to-peak amplitude
   >500µV, or duration <60s.

Output: ~30–40 shards of memory-mapped `.npy` files (~2GB each), each with a
sidecar JSON listing recording boundaries and source-dataset provenance. The
provenance sidecar is what makes a cheap "drop source X" ablation possible
later, and is the authoritative record of what rate/format a shard was actually
built at — read it directly rather than trusting config files in isolation.

`target_fs` is **128 Hz** (see `03_ARCHITECTURE_AND_DECISIONS.md`; reconciling
the stale 250 Hz comments is an open decision in `02_PHASE_2_PLAN.md` step 2).

## Pretraining dataloader (`src/neurostream/data/window_dataset.py`)

`EEGWindowDataset` — an `IterableDataset` that samples random 1000-sample
windows from the shard memmaps, with:
- **Per-source weighting** (default: equal weight per source — oversampling
  small sources, undersampling large ones, analogous to per-domain weighting in
  LLM pretraining).
- **Per-worker RNG seeding** via `worker_init_fn` reading
  `torch.utils.data.get_worker_info()` (PyTorch's default `IterableDataset`
  worker-init would otherwise hand every DataLoader worker identical windows).

The dataloader is rate-agnostic — it grabs 1000 contiguous samples regardless of
what they represent in wall-clock time.

## Why pretraining samples continuous windows, not trial epochs

Pretraining never sees trial-bounded data — it samples arbitrary continuous
windows from multi-minute recordings, so there is no "real vs. padded"
distinction during pretraining; every sample is real signal. BCI IV 2a, once run
through `mne.Epochs(tmin, tmax)`, is fundamentally trial-bounded — you cannot
extract more real samples than the epoch window contains. This mismatch is
structural, which is why the right fix for probe transfer was matching
*preprocessing* (band-pass + CAR), orthogonal to the window-length question.

## Which loader to use for what

| Task | Loader |
|---|---|
| Phase 1 EEGNet training | `bci_iv_loader.py` (self-consistent, supervised end-to-end — no frozen pretrained encoder involved) |
| Linear probe / probe ablation | `bci_iv_harmonised.py` |
| Phase 2 fine-tuning | The canonical loader **once it has the band-pass + CAR fix** (`02_PHASE_2_PLAN.md` step 1) |
| Any Phase 2 work feeding the pretrained encoder | A harmonised loader — never a path that skips band-pass + CAR |
