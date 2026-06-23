"""Tests for the BCI IV 2a subject loader, focused on the window_seconds knob."""

import numpy as np
import pytest

from neurostream.data import bci_iv_loader as L


# ---- Pure window -> (tmin, tmax) mapping --------------------------------
def test_window_2s_maps_to_lawhern_window():
    assert L._window_to_tmin_tmax(2.0) == (0.5, 2.5)


def test_window_4s_maps_to_wide_probe_window():
    assert L._window_to_tmin_tmax(4.0) == (-0.5, 3.5)


def test_unsupported_window_raises():
    with pytest.raises(ValueError, match="window_seconds"):
        L._window_to_tmin_tmax(3.0)


# ---- Cache keying must separate windows ---------------------------------
def test_default_window_keeps_legacy_cache_name():
    # The 2.0 s default must reuse the existing v3 filename so already-cached
    # subjects stay valid.
    p = L._cache_path(1, "T", 2.0)
    assert p.name == f"A01T_{L.CACHE_VERSION}.npz"


def test_nondefault_window_uses_distinct_cache_name():
    # A 4 s request must NOT collide with the 2 s cache, otherwise a 256-sample
    # window would be returned for a 512-sample request.
    two = L._cache_path(1, "T", 2.0)
    four = L._cache_path(1, "T", 4.0)
    assert two != four
    assert "4" in four.name


# ---- Real-data integration (needs the .gdf files; slow) -----------------
@pytest.mark.slow
@pytest.mark.parametrize("session", ["T", "E"])
@pytest.mark.parametrize(
    "window_seconds,expected_samples",
    [(2.0, 256), (4.0, 512)],  # both at TARGET_SFREQ = 128 Hz
)
def test_load_subject_shapes(session, window_seconds, expected_samples):
    epochs, labels = L.load_subject(
        1, session, window_seconds=window_seconds, use_cache=False
    )
    assert epochs.shape == (288, 22, expected_samples)
    assert labels.shape == (288,)
    assert epochs.dtype == np.float32
    assert labels.dtype == np.int64
    assert set(np.unique(labels)).issubset({0, 1, 2, 3})


@pytest.mark.slow
@pytest.mark.parametrize("subject_id", range(1, 10))
@pytest.mark.parametrize("session", ["T", "E"])
def test_wide_window_keeps_all_288_trials_every_subject(subject_id, session):
    # The 4 s window uses tmin=-0.5 (pre-cue territory no prior window touched).
    # The probe runs subjects 1-9 with a hard `assert len(epochs) == 288`, so
    # confirm no subject drops a boundary epoch at this window.
    epochs, labels = L.load_subject(
        subject_id, session, window_seconds=4.0, use_cache=False
    )
    assert epochs.shape == (288, 22, 512)
    assert labels.shape == (288,)
