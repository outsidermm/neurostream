"""Tests for preprocessing.missing_channels — FCz interpolation for Lee2019_MI."""

import mne
import numpy as np
import pytest

from neurostream.data.channels import BCI_IV_2A_22_CHANNELS
from neurostream.preprocessing.missing_channels import ensure_channels


def _make_raw_without_fcz(fs: float = 128.0, n_seconds: float = 10.0) -> mne.io.RawArray:
    channels = [ch for ch in BCI_IV_2A_22_CHANNELS if ch != "FCz"]
    n_samples = int(fs * n_seconds)
    data = np.random.default_rng(0).standard_normal((len(channels), n_samples)) * 10e-6
    info = mne.create_info(ch_names=list(channels), sfreq=fs, ch_types="eeg")
    return mne.io.RawArray(data, info, verbose="ERROR")


def test_ensure_channels_is_noop_when_all_present(make_raw):
    """FCz already present → no interpolation, same channel list."""
    raw = make_raw(fs=128, n_seconds=10)
    result = ensure_channels(raw, BCI_IV_2A_22_CHANNELS, "Lee2019_MI")
    assert set(result.ch_names) >= set(BCI_IV_2A_22_CHANNELS)


def test_ensure_channels_is_noop_for_unknown_source():
    """Unknown source → no interpolation even if FCz is missing."""
    raw = _make_raw_without_fcz()
    result = ensure_channels(raw, BCI_IV_2A_22_CHANNELS, "PhysionetMI")
    assert "FCz" not in result.ch_names


def test_ensure_channels_adds_missing_fcz():
    raw = _make_raw_without_fcz()
    assert "FCz" not in raw.ch_names

    result = ensure_channels(raw, BCI_IV_2A_22_CHANNELS, "Lee2019_MI")

    assert "FCz" in result.ch_names


def test_ensure_channels_fcz_not_in_bads_after():
    raw = _make_raw_without_fcz()
    result = ensure_channels(raw, BCI_IV_2A_22_CHANNELS, "Lee2019_MI")
    assert "FCz" not in result.info["bads"]


def test_ensure_channels_does_not_modify_input():
    raw = _make_raw_without_fcz()
    names_before = list(raw.ch_names)
    ensure_channels(raw, BCI_IV_2A_22_CHANNELS, "Lee2019_MI")
    assert list(raw.ch_names) == names_before


def test_ensure_channels_fcz_is_mean_of_fc1_fc2():
    """FCz should be the (nan)mean of FC1 and FC2."""
    channels = [ch for ch in BCI_IV_2A_22_CHANNELS if ch != "FCz"]
    n_samples = 128
    data = np.zeros((len(channels), n_samples))
    fc1_idx = channels.index("FC1")
    fc2_idx = channels.index("FC2")
    data[fc1_idx, :] = 2.0
    data[fc2_idx, :] = 4.0
    info = mne.create_info(ch_names=list(channels), sfreq=128.0, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose="ERROR")

    result = ensure_channels(raw, BCI_IV_2A_22_CHANNELS, "Lee2019_MI")
    fcz_data = result.get_data(picks=["FCz"])[0]
    np.testing.assert_allclose(fcz_data, 3.0)  # mean of 2.0 and 4.0


def test_ensure_channels_fcz_has_finite_values():
    raw = _make_raw_without_fcz()
    result = ensure_channels(raw, BCI_IV_2A_22_CHANNELS, "Lee2019_MI")
    idx = result.ch_names.index("FCz")
    fcz_data = result.get_data()[idx]
    assert np.all(np.isfinite(fcz_data))


def test_ensure_channels_tolerates_nan_in_fc1_fc2():
    """NaN in FC1/FC2 should not crash — nanmean propagates NaN only if both are NaN."""
    channels = [ch for ch in BCI_IV_2A_22_CHANNELS if ch != "FCz"]
    n_samples = 128
    data = np.zeros((len(channels), n_samples))
    data[channels.index("FC1"), :] = np.nan
    info = mne.create_info(ch_names=list(channels), sfreq=128.0, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose="ERROR")

    result = ensure_channels(raw, BCI_IV_2A_22_CHANNELS, "Lee2019_MI")
    assert "FCz" in result.ch_names  # no crash
    assert "FCz" not in result.info["bads"]


def test_ensure_channels_tolerates_line_freq_in_source_raw():
    """add_channels fails if line_freq mismatches between source and placeholder.
    Lee2019_MI raws carry line_freq=60.0; the placeholder must inherit it."""
    channels = [ch for ch in BCI_IV_2A_22_CHANNELS if ch != "FCz"]
    n_samples = int(128.0 * 10.0)
    data = np.zeros((len(channels), n_samples))
    info = mne.create_info(ch_names=list(channels), sfreq=128.0, ch_types="eeg")
    info["line_freq"] = 60.0
    raw = mne.io.RawArray(data, info, verbose="ERROR")

    result = ensure_channels(raw, BCI_IV_2A_22_CHANNELS, "Lee2019_MI")
    assert "FCz" in result.ch_names


def test_ensure_channels_preserves_existing_channel_count(make_raw):
    raw = make_raw(fs=128, n_seconds=10)
    n_before = len(raw.ch_names)
    result = ensure_channels(raw, BCI_IV_2A_22_CHANNELS, "Lee2019_MI")
    # existing channels preserved (may have extras from make_raw)
    assert len(result.ch_names) >= n_before
