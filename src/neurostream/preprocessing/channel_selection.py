"""Channel renaming and selection to a canonical montage.

Different MOABB datasets use slightly different channel-naming conventions.
This module canonicalises names, then picks and reorders to a fixed montage —
the encoder treats channels positionally, so order must be deterministic.
"""

import logging

import mne
import numpy as np

log = logging.getLogger(__name__)


# Per-source channel rename maps. Empty by default; fill if a source's
# naming convention needs fixing up (visible as MISSING_CHANNELS rejections).
PER_SOURCE_RENAMES: dict[str, dict[str, str]] = {
    "PhysionetMI": {},
    "Cho2017": {},
    "Lee2019_MI": {},
    "Schirrmeister2017": {},
}

# Common-case rename applied to every source — catches mis-cased midline names.
_CANONICAL_RENAMES: dict[str, str] = {
    "FZ": "Fz",
    "CZ": "Cz",
    "PZ": "Pz",
    "POZ": "POz",
    "FCZ": "FCz",
    "CPZ": "CPz",
}


def _canonical_name(name: str, source: str) -> str:
    """Map one raw channel name to its canonical form."""
    canonical = _CANONICAL_RENAMES.get(name, name)
    return PER_SOURCE_RENAMES.get(source, {}).get(canonical, canonical)


def select_channels(
    raw: mne.io.BaseRaw,
    target_channels: tuple[str, ...],
    source: str = "",
) -> np.ndarray | None:
    """Rename, validate, pick and reorder ``raw`` to ``target_channels``.

    Operates on a copy — the caller's ``raw`` is left untouched.

    Args:
        raw: an MNE Raw from any source.
        target_channels: the montage to select, in the desired order.
        source: dataset name, used to pick a per-source rename map.

    Returns:
        Picked data as ``(len(target_channels), n_samples)`` in MNE's native
        volts, or ``None`` if any target channel is absent.
    """
    raw = raw.copy()
    renames = {name: _canonical_name(name, source) for name in map(str, raw.ch_names)}
    renames = {old: new for old, new in renames.items() if old != new}
    if renames:
        raw.rename_channels(renames)

    available = {str(c) for c in raw.ch_names}
    missing = [ch for ch in target_channels if ch not in available]
    if missing:
        log.debug(f"[{source}] missing channels {missing}")
        return None

    raw.pick(list(target_channels))
    # MNE pick orders channels in the order provided (mne>=1.0).
    assert tuple(raw.ch_names) == tuple(target_channels), (
        f"channel order mismatch after pick: {raw.ch_names}"
    )
    return np.asarray(raw.get_data())
