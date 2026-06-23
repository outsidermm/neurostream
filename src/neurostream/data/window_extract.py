"""Per-trial window extraction from a continuous EEG recording.

Two strategies, both producing ``(n_trials, n_channels, n_samples)`` arrays
ready for the MAE encoder:

* ``centered_windows`` — slice ``n_samples`` *real* samples centred on each cue.
  Matches the pretraining distribution (arbitrary continuous chunk, no padding).
  Only clamps + zero-pads at the hard recording edges, and reports how many
  trials needed it so silent truncation can't hide.
* ``padded_windows`` — slice the short Lawhern-style window around each cue and
  centre-zero-pad it up to ``n_samples``. Reproduces the original (padded) probe
  behaviour for the ablation baseline.

Both z-score per window per channel, mirroring ``EEGWindowDataset._z_score``
used during pretraining.
"""

import numpy as np

from neurostream.data.bci_iv_loader import TARGET_SFREQ

# Cue-relative epoch windows (tmin, tmax) in seconds, supported by padded_windows.
_PROBE_WINDOWS = {2.0: (0.5, 2.5), 4.0: (-0.5, 3.5)}


def zscore_per_window(windows: np.ndarray) -> np.ndarray:
    """Per-window, per-channel z-score over the last axis (matches pretraining)."""
    mean = windows.mean(axis=-1, keepdims=True)
    std = windows.std(axis=-1, keepdims=True) + 1e-6
    return ((windows - mean) / std).astype(np.float32)


def centered_windows(
    data: np.ndarray,
    cue_samples: np.ndarray,
    n_samples: int = 1000,
) -> tuple[np.ndarray, int]:
    """Extract ``n_samples`` centred on each cue from a continuous recording.

    Args:
        data: ``(n_channels, n_times)`` continuous recording.
        cue_samples: integer sample index of each trial's cue.
        n_samples: window length (the encoder's fixed input length).

    Returns:
        ``(windows, n_padded)`` — z-scored ``(n_trials, n_channels, n_samples)``
        and the count of trials that hit a recording edge and needed any zero
        padding (0 means every window is fully real).
    """
    n_channels, n_times = data.shape
    half = n_samples // 2
    out = np.zeros((len(cue_samples), n_channels, n_samples), dtype=np.float32)
    n_padded = 0
    for i, cue in enumerate(cue_samples):
        start = int(cue) - half
        stop = start + n_samples
        s0, s1 = max(start, 0), min(stop, n_times)
        if s0 > start or s1 < stop:
            n_padded += 1
        out[i, :, (s0 - start) : (s1 - start)] = data[:, s0:s1]
    return zscore_per_window(out), n_padded


def padded_windows(
    data: np.ndarray,
    cue_samples: np.ndarray,
    window_seconds: float = 2.0,
    n_samples: int = 1000,
) -> np.ndarray:
    """Slice a short cue-relative window and centre-zero-pad it to ``n_samples``.

    Reproduces the original probe adapter: a 2 s ([0.5, 2.5] s) window at
    TARGET_SFREQ is 256 real samples, centre-padded with zeros to 1000, then
    z-scored over the full (mostly-zero) window.
    """
    try:
        tmin, tmax = _PROBE_WINDOWS[window_seconds]
    except KeyError:
        raise ValueError(
            f"window_seconds must be one of {sorted(_PROBE_WINDOWS)}, "
            f"got {window_seconds}"
        ) from None
    a = int(round(tmin * TARGET_SFREQ))
    b = int(round(tmax * TARGET_SFREQ))  # exclusive; (b - a) == window_seconds * fs
    win_len = b - a
    if win_len > n_samples:
        raise ValueError(
            f"{window_seconds}s window is {win_len} samples > n_samples={n_samples}"
        )
    n_channels, n_times = data.shape
    before = (n_samples - win_len) // 2
    out = np.zeros((len(cue_samples), n_channels, n_samples), dtype=np.float32)
    for i, cue in enumerate(cue_samples):
        lo, hi = int(cue) + a, int(cue) + b
        if lo < 0 or hi > n_times:
            raise ValueError(f"trial {i}: window [{lo}, {hi}) out of bounds {n_times}")
        out[i, :, before : before + win_len] = data[:, lo:hi]
    return zscore_per_window(out)
