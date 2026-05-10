# test_filters.py
import pytest
import numpy as np
from neurostream.preprocessing.filters import (
    BandpassParams,
    bandpass_filter,
    design_bandpass_sos,
)


def test_passband_preserved():
    """A 15 Hz sine wave (in passband) should survive with amplitude ~1."""
    fs, duration = 250.0, 4.0
    t = np.arange(0, duration, 1 / fs)
    x = np.sin(2 * np.pi * 15 * t).astype(np.float32)
    y = bandpass_filter(x, BandpassParams(8, 30, fs))

    # Skip edge transients (first/last 0.5s)
    edge = int(0.5 * fs)
    assert np.abs(y[edge:-edge]).max() == pytest.approx(1.0, abs=0.05)


def test_stopband_attenuated():
    """5 Hz (below) and 60 Hz (above, line noise) should drop >20 dB."""
    fs, duration = 250.0, 4.0
    t = np.arange(0, duration, 1 / fs)
    for freq in [5.0, 60.0]:
        x = np.sin(2 * np.pi * freq * t).astype(np.float32)
        y = bandpass_filter(x, BandpassParams(8, 30, fs))

        # Skip edge transients (first/last 0.5s)
        edge = int(0.5 * fs)
        attenuation_db = 20 * np.log10(np.abs(y[edge:-edge]).max() + 1e-12)
        assert attenuation_db < -20, f"{freq} Hz only attenuated to {attenuation_db} dB"


def test_zero_phase():
    """Output of a symmetric input should be symmetric (zero phase delay)."""
    fs = 250.0
    x = np.zeros(1001, dtype=np.float32)
    x[500] = 1.0  # impulse at center
    y = bandpass_filter(x, BandpassParams(8, 30, fs))

    # Symmetric around the impulse location
    np.testing.assert_allclose(y[:500], y[501:][::-1], atol=1e-6)


def test_handles_3d_input():
    """Should filter along last axis for batched input."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((10, 22, 1000)).astype(np.float32)
    y = bandpass_filter(x, BandpassParams(8, 30, 250.0))
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_invalid_band_raises():
    with pytest.raises(ValueError):
        design_bandpass_sos(BandpassParams(low_hz=30, high_hz=8, fs_hz=250))
    with pytest.raises(ValueError):
        design_bandpass_sos(
            BandpassParams(low_hz=8, high_hz=130, fs_hz=250)
        )  # > Nyquist
