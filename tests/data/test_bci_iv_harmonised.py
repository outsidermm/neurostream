"""Integration tests for the harmonised probe data path (needs real .gdf; slow)."""

import numpy as np
import pytest

from neurostream.data.bci_iv_harmonised import load_continuous, make_probe_adapter


@pytest.mark.slow
def test_load_continuous_shapes_and_labels():
    load_continuous.cache_clear()
    data, cue, labels = load_continuous(1, "T", harmonise=True)
    assert data.shape[0] == 22
    assert data.dtype == np.float32
    assert cue.shape == (288,) and labels.shape == (288,)
    assert set(np.unique(labels)).issubset({0, 1, 2, 3})
    # Cues sit inside the recording with room for a centred 1000-sample window.
    assert cue.min() >= 500
    assert data.shape[1] - cue.max() >= 500


@pytest.mark.slow
def test_harmonise_applies_common_average_reference():
    load_continuous.cache_clear()
    raw_data, _, _ = load_continuous(1, "T", harmonise=False)
    car_data, _, _ = load_continuous(1, "T", harmonise=True)
    # CAR forces the per-timestep mean across channels to ~0; the un-harmonised
    # path does not.
    assert np.abs(car_data.mean(axis=0)).max() < 1e-3
    assert np.abs(raw_data.mean(axis=0)).max() > 1e-3
    # Band-pass + CAR genuinely change the signal.
    assert not np.allclose(raw_data, car_data)


@pytest.mark.slow
@pytest.mark.parametrize("window", ["pad2s", "continuous"])
def test_adapter_output_shape(window):
    load_continuous.cache_clear()
    adapter = make_probe_adapter(harmonise=True, window=window)
    epochs, labels = adapter(1, "E")
    assert epochs.shape == (288, 22, 1000)
    assert epochs.dtype == np.float32
    assert labels.shape == (288,)


@pytest.mark.slow
def test_continuous_windows_need_no_padding_any_subject():
    # The whole point of the continuous path: zero padding across all trials.
    from neurostream.data.bci_iv_harmonised import load_continuous as lc
    from neurostream.data.window_extract import centered_windows

    lc.cache_clear()
    for subject_id in range(1, 10):
        for session in ("T", "E"):
            data, cue, _ = lc(subject_id, session, harmonise=True)
            _, n_padded = centered_windows(data, cue, n_samples=1000)
            assert n_padded == 0, f"A0{subject_id}{session}: {n_padded} padded"
