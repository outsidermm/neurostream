import os
from pathlib import Path
from typing import Literal

import mne
import numpy as np

# src/neurostream/data/loader.py → parents[3] is the repo root
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_RAW = Path(os.getenv("NEUROSTREAM_DATA_RAW", _PROJECT_ROOT / "data" / "raw"))
DATA_CACHE = Path(
    os.getenv("NEUROSTREAM_DATA_CACHE", _PROJECT_ROOT / "data" / "processed")
)

_DATASET = "bci_iv_2a"

CACHE_VERSION = "v1"  # bump when loader logic changes

CUE_EVENT_IDS = {"769": 0, "770": 1, "771": 2, "772": 3}  # left, right, feet, tongue
SFREQ = 250
# Window relative to cue onset. BCI IV 2a recordings end exactly 5.908 s after
# the last cue across every training session, so a 6.0 s tmax drops the final
# trial as TOO_SHORT. 5.9 s sits inside that margin and keeps all 288 trials.
TMIN, TMAX = 0, 3.9


def _cache_path(subject_id: int, session: str) -> Path:
    return DATA_CACHE / _DATASET / f"A0{subject_id}{session}_{CACHE_VERSION}.npz"


def load_subject(
    subject_id: int,
    session: Literal["T", "E"],
    use_cache: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    cache_path = _cache_path(subject_id, session)

    if use_cache and cache_path.exists():
        with np.load(cache_path) as f:
            return f["epochs"].astype(np.float32), f["labels"].astype(np.int64)

    epochs, labels = _load_from_gdf(subject_id, session)

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(".tmp")
        # Pass a file object — np.savez auto-appends ".npz" when handed a
        # filename that doesn't end in .npz, so writing via the path directly
        # would produce {tmp}.npz and the rename below would miss.
        with tmp.open("wb") as f:
            np.savez(f, epochs=epochs, labels=labels)
        tmp.replace(cache_path)  # atomic on POSIX

    return epochs, labels


def _load_from_gdf(
    subject_id: int,
    session: Literal["T", "E"],
) -> tuple[np.ndarray, np.ndarray]:
    """Load one session for one subject. Returns (epochs, labels)."""
    if not 1 <= subject_id <= 9:
        raise ValueError(f"subject_id must be in 1..9, got {subject_id}")

    path = DATA_RAW / _DATASET / f"A0{subject_id}{session}.gdf"
    raw = mne.io.read_raw_gdf(path, preload=True, verbose="ERROR")

    events, event_id_map = mne.events_from_annotations(raw, verbose="ERROR")
    # Keep only cue events that are in our four classes
    cue_codes = {k: v for k, v in event_id_map.items() if k in CUE_EVENT_IDS}

    # tmax is shaved by one sample so the window is exactly (TMAX - TMIN) * SFREQ
    # samples — without this MNE includes the t=TMAX endpoint, giving one extra.
    epochs = mne.Epochs(
        raw,
        events,
        event_id=cue_codes,
        tmin=TMIN,
        tmax=TMAX - 1.0 / SFREQ,
        picks="eeg",
        baseline=None,
        preload=True,
        verbose="ERROR",
    )
    assert len(epochs) == 288, f"Expected 288 trials, got {len(epochs)}"

    # Remap MNE's internal event ids back to our 0..3 class labels
    inverse = {v: CUE_EVENT_IDS[k] for k, v in cue_codes.items()}
    labels = np.array([inverse[e] for e in epochs.events[:, 2]], dtype=np.int64)

    data = epochs.get_data().astype(np.float32)  # (288, 22, n_samples)
    return data, labels
