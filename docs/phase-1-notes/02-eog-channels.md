# 02 — EOG channels: 25 → 22

*GDF tags every channel as type "eeg"; without explicit relabelling the model trains on eye-movement artifacts.*

## What we observed

- Loader returned shape `(288, 25, 975)` instead of expected `(288, 22, 975)`
- `model.n_channels: 22` in `configs/train.yaml` silently disagreed with the data shape
- Model would have trained successfully — but on 22 EEG + 3 EOG, learning a known confound

## What caused it

- BCI IV 2a uses **22 EEG + 3 EOG** electrodes (EOG = electrooculogram, eye-movement reference channels)
- The GDF file format has no native EOG marker — every channel is written with type `"eeg"`
- MNE's `picks="eeg"` therefore picks all 25 channels indiscriminately

Channel layout discovered via `raw.ch_names`:

| Index range | Channel names | True type |
|---|---|---|
| 0–21 | `EEG-Fz`, `EEG-0`…`EEG-16`, `EEG-C3`, `EEG-Cz`, `EEG-C4`, `EEG-Pz` | EEG |
| 22–24 | `EOG-left`, `EOG-central`, `EOG-right` | EOG |

## What we did

- File: [`src/neurostream/data/loader.py`](../../src/neurostream/data/loader.py)
- Commit: `702a88a` — "feat: enhance data loader — EOG exclusion + session E labels"
- Added a `set_channel_types` call right after `read_raw_gdf`:

```python
EOG_CHANNELS = ("EOG-left", "EOG-central", "EOG-right")

raw = mne.io.read_raw_gdf(path, preload=True, verbose="ERROR")
raw.set_channel_types({name: "eog" for name in EOG_CHANNELS})
# picks="eeg" in mne.Epochs(...) now drops the relabelled EOG channels automatically
```

- Bumped `CACHE_VERSION = "v2"` to invalidate the 25-channel `.npz` files already on disk

## What this signals

EOG contamination is a textbook EEG confound — eye movements have predictable temporal and spectral structure that a CNN will happily exploit instead of true motor-imagery signal. The model wouldn't have *crashed*; it would have *worked* and reported inflated accuracy on subjects who blink consistently per class. Catching this requires either careful dataset-doc reading or noticing the channel-count mismatch. The habit of bumping `CACHE_VERSION` on any loader-semantics change is the kind of small discipline that prevents "why is my baseline different today" mysteries weeks later.
