"""Reconstruct structurally-absent EEG channels from spatial neighbours.

Lee2019_MI uses a 62-channel BrainAmp montage that omits FCz, which is present
in the BCI-IV-2A target montage. Approximating FCz as the nanmean of FC1 and
FC2 (its immediate frontocentral neighbours) recovers the channel without
discarding an otherwise good recording.

Only channels listed in _PER_SOURCE_MISSING are ever reconstructed. Channels
missing for other reasons (bad electrode, wrong dataset) are still rejected by
the amplitude/selection checks downstream.
"""

import logging

import mne
import numpy as np

log = logging.getLogger(__name__)

# Channels known to be structurally absent from a given source.
_PER_SOURCE_MISSING: dict[str, list[str]] = {
    "Lee2019_MI": ["FCz"],
}

# Nearest neighbours used to approximate each missing channel.
_NEAREST_NEIGHBOURS: dict[str, list[str]] = {
    "FCz": ["FC1", "FC2"],
}


def ensure_channels(
    raw: mne.io.BaseRaw,
    target_channels: tuple[str, ...],
    source: str = "",
) -> mne.io.BaseRaw:
    """Add structurally-missing target channels from their spatial neighbours.

    For each channel in _PER_SOURCE_MISSING[source] that is absent from *raw*,
    a new channel is created as the nanmean of the channels listed in
    _NEAREST_NEIGHBOURS. Returns *raw* unchanged if all target channels are
    present or the source has no entry.

    Args:
        raw: MNE Raw from any source. Never modified in-place.
        target_channels: used to filter which source-specific channels to add.
        source: dataset name — controls which channels are eligible.

    Returns:
        Raw with structurally-missing channels approximated (or *raw* unchanged).
    """
    expected_missing = _PER_SOURCE_MISSING.get(source, [])
    target_set = set(target_channels)
    available = set(raw.ch_names)
    to_add = [ch for ch in expected_missing if ch in target_set and ch not in available]
    if not to_add:
        return raw

    log.debug(f"[{source}] approximating missing channels from neighbours: {to_add}")
    raw = raw.copy()

    for ch in to_add:
        neighbours = _NEAREST_NEIGHBOURS.get(ch, [])
        present = [n for n in neighbours if n in available]
        if present:
            approx = np.nanmean(raw.get_data(picks=present), axis=0)
        else:
            approx = np.zeros(raw.n_times)

        new_info = mne.create_info([ch], sfreq=raw.info["sfreq"], ch_types="eeg")
        new_info["line_freq"] = raw.info.get("line_freq")
        placeholder = mne.io.RawArray(approx[np.newaxis, :], new_info, verbose="ERROR")
        raw.add_channels([placeholder])

    return raw
