"""Tests for per-trial window extraction (pure, no real data)."""

import numpy as np
import pytest

from neurostream.data.window_extract import (
    centered_windows,
    padded_windows,
    zscore_per_window,
)


def _ramp(n_channels=22, n_times=20000):
    # Distinct, non-constant per-channel signal so z-score is well-defined and
    # slices are identifiable.
    base = np.arange(n_times, dtype=np.float32)[None, :]
    offsets = np.arange(n_channels, dtype=np.float32)[:, None] * 1000.0
    return base + offsets


def test_zscore_is_unit_per_window_per_channel():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((5, 22, 1000)).astype(np.float32) * 3 + 7
    z = zscore_per_window(w)
    np.testing.assert_allclose(z.mean(axis=-1), 0.0, atol=1e-5)
    np.testing.assert_allclose(z.std(axis=-1), 1.0, atol=1e-3)


def test_centered_windows_shape_and_no_padding_in_interior():
    data = _ramp(n_times=20000)
    cues = np.array([5000, 10000, 15000], dtype=np.int64)
    windows, n_padded = centered_windows(data, cues, n_samples=1000)
    assert windows.shape == (3, 22, 1000)
    assert n_padded == 0  # all cues are >=500 from both edges


def test_centered_windows_are_real_and_centered():
    # Use a clean ramp; the centred window must be the 1000 samples [cue-500, cue+500).
    data = _ramp(n_times=20000)
    cue = 8000
    # Pre-zscore content check: reconstruct what slice was taken via argmax of std structure.
    windows, n_padded = centered_windows(data, np.array([cue]), n_samples=1000)
    assert n_padded == 0
    # Channel 0 of the raw slice is a strict ramp; after z-score it stays strictly
    # increasing, so the window is contiguous & real (no zero gaps in the middle).
    ch0 = windows[0, 0]
    assert np.all(np.diff(ch0) > 0)


def test_centered_windows_pad_only_at_edges():
    data = _ramp(n_times=20000)
    # Cue too close to the start -> needs left padding.
    cues = np.array([100, 10000], dtype=np.int64)
    windows, n_padded = centered_windows(data, cues, n_samples=1000)
    assert n_padded == 1
    # start = 100-500 = -400 -> first 400 samples are padding. After z-score the
    # padded region is a constant (-mean/std), so it reads as flat, while the
    # real tail keeps the ramp's strictly-increasing structure.
    left_pad = windows[0, 0, :400]
    assert np.allclose(left_pad, left_pad[0])
    assert np.all(np.diff(windows[0, 0, 400:]) > 0)


def test_padded_windows_2s_has_256_real_centered_samples():
    data = _ramp(n_times=20000)
    cues = np.array([5000, 10000], dtype=np.int64)
    out = padded_windows(data, cues, window_seconds=2.0, n_samples=1000)
    assert out.shape == (2, 22, 1000)
    # 2 s at 128 Hz = 256 real samples, centre-padded: 372 zeros | 256 real | 372 zeros.
    before = (1000 - 256) // 2
    # Real region is non-constant; padded flanks are constant (zero before z-score).
    real = out[0, 0, before : before + 256]
    left_pad = out[0, 0, :before]
    assert np.all(np.diff(real) > 0)
    assert np.allclose(left_pad, left_pad[0])  # flat (all-equal) padded region


def test_padded_windows_rejects_out_of_bounds():
    # cue near the end: window hi = 900 + 2.5*128 = 1220 > n_times=1000.
    data = _ramp(n_times=1000)
    with pytest.raises(ValueError, match="out of bounds"):
        padded_windows(data, np.array([900]), window_seconds=2.0)
