"""Tests for preprocessing.channel_selection and the channel montage."""

import mne
import numpy as np

from neurostream.data.channels import BCI_IV_2A_22_CHANNELS
from neurostream.preprocessing.channel_selection import select_channels


def test_channels_constant_has_22_unique_names():
    assert len(BCI_IV_2A_22_CHANNELS) == 22
    assert len(set(BCI_IV_2A_22_CHANNELS)) == 22


def test_channels_constant_uses_canonical_case():
    # Midline 'z' markers must be lowercase (Fz, Cz, ...), not 'Z'.
    for name in BCI_IV_2A_22_CHANNELS:
        assert "Z" not in name, f"{name}: midline 'z' should be lowercase"


def test_select_channels_picks_target_set(make_raw):
    raw = make_raw(fs=500, n_seconds=10)  # 22 target channels + extras
    data = select_channels(raw, BCI_IV_2A_22_CHANNELS)
    assert data is not None
    assert data.shape[0] == 22


def test_select_channels_returns_none_when_channel_missing(make_raw):
    raw = make_raw(fs=500, n_seconds=10)
    raw.drop_channels(["C3"])
    assert select_channels(raw, BCI_IV_2A_22_CHANNELS) is None


def test_select_channels_does_not_modify_input(make_raw):
    raw = make_raw(fs=500, n_seconds=10)
    names_before = list(raw.ch_names)
    select_channels(raw, BCI_IV_2A_22_CHANNELS)
    assert list(raw.ch_names) == names_before


def test_select_channels_reorders_to_target_order():
    """Channels inserted in reversed order must come back in target order.

    Each channel carries a constant equal to its index in the target montage;
    after selection, row i must hold the constant i.
    """
    target = BCI_IV_2A_22_CHANNELS
    order = list(reversed(target))
    data = np.zeros((22, 256))
    for row, name in enumerate(order):
        data[row, :] = target.index(name)
    info = mne.create_info(list(order), sfreq=250.0, ch_types="eeg")
    raw = mne.io.RawArray(data * 1e-6, info, verbose="ERROR")

    out = select_channels(raw, target)
    assert out is not None
    for i in range(22):
        np.testing.assert_allclose(out[i], i * 1e-6, atol=1e-12)
