# 05 — Resampling protocol: 250 Hz → 128 Hz, window [0.5, 2.5] s

*Even after `max_norm`, the temporal receptive field was wrong by 2×. Matching the paper's resample step fixed it.*

## What we observed

- After [log 04](04-paper-faithful-reproduction.md)'s fixes, the model still trained on 250 Hz data with `kernel_length = fs // 2 = 125` samples
- 125 samples at 250 Hz = **500 ms** wall-clock receptive field
- Lawhern's reference example uses `kernel_length = 32` at 128 Hz = **250 ms**
- Our convolutional layer was filtering at twice the paper's temporal scale — no error, just silently underfitting

## What caused it

Two missed steps from the paper:

| | Paper (Lawhern 2018) | Ours (pre-fix) |
|---|---|---|
| Sample rate | resampled to **128 Hz** | kept at 250 Hz |
| Trial window | **[0.5, 2.5]** s post-cue (2.0 s) | [0, 3.9] s post-cue (3.9 s) |
| Samples per trial | 256 | 975 |
| Kernel length | 32 (= 250 ms) | 125 (= 500 ms) |

The paper resamples *before* anything else, so every downstream constant is at 128 Hz.

## What we did

- File: [`src/neurostream/data/loader.py`](../../src/neurostream/data/loader.py)
- File: [`configs/train.yaml`](../../configs/train.yaml)
- Commit: `7245be9` — "feat: resample into 128 hz and shorter cue period"

Changes:

```python
# loader.py
TARGET_SFREQ = 128
TMIN, TMAX = 0.5, 2.5

raw = mne.io.read_raw_gdf(path, preload=True, verbose="ERROR")
raw.set_channel_types({name: "eog" for name in EOG_CHANNELS})
raw.resample(TARGET_SFREQ, verbose="ERROR")      # ← new
events, event_id_map = mne.events_from_annotations(raw, verbose="ERROR")
```

```yaml
# configs/train.yaml
model:
  fs: 128
  kernel_length: 32         # paper default at 128 Hz
preprocessing:
  bandpass:
    fs_hz: 128.0            # match resampled rate
```

- Bumped `CACHE_VERSION = "v3"` to invalidate the 250 Hz `.npz` caches
- Verified trial count preserved: **288 per session × 9 subjects** still passes the assertion
- Output shape: `(288, 22, 256)` — same 22 channels post-EOG fix, now 256 samples at 128 Hz

## What this signals

Implementation details compound. The model architecture was correct after [log 04](04-paper-faithful-reproduction.md); the *temporal frame of reference* was still wrong. The paper specifies "kernel length = fs/2 samples" — but `fs` is the paper's fs (128), not whatever your loader happens to return (250). Matching a paper requires matching its *entire pipeline*, not just the model. A 2× sample-rate mismatch is the kind of bug that doesn't fail any assertion and never shows up in unit tests — it just quietly trains on the wrong temporal scale.
