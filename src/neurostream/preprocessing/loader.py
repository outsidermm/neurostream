from pathlib import Path
from typing import Literal
import numpy as np
import mne

DATA_ROOT = Path("data/raw/bci_iv_2a")
CUE_EVENT_IDS = {"769": 0, "770": 1, "771": 2, "772": 3}  # left, right, feet, tongue
SFREQ = 250
TMIN, TMAX = 2.0, 6.0  # 4s window starting at cue; we slice the imagery portion later

def load_from_gdf(
    subject_id: int,
    session: Literal["T", "E"],
) -> tuple[np.ndarray, np.ndarray]:
    """Load one session for one subject. Returns (epochs, labels)."""
    if not 1 <= subject_id <= 9:
        raise ValueError(f"subject_id must be in 1..9, got {subject_id}")

    path = DATA_ROOT / f"A0{subject_id}{session}.gdf"
    raw = mne.io.read_raw_gdf(path, preload=True, verbose="ERROR")

    events, event_id_map = mne.events_from_annotations(raw, verbose="ERROR")
    # Keep only cue events that are in our four classes
    cue_codes = {k: v for k, v in event_id_map.items() if k in CUE_EVENT_IDS}

    epochs = mne.Epochs(
        raw, events, event_id=cue_codes,
        tmin=TMIN, tmax=TMAX, picks="eeg",
        baseline=None, preload=True, verbose="ERROR",
    )
    assert len(epochs) == 288, f"Expected 288 trials, got {len(epochs)}"

    # Remap MNE's internal event ids back to our 0..3 class labels
    inverse = {v: CUE_EVENT_IDS[k] for k, v in cue_codes.items()}
    labels = np.array([inverse[e] for e in epochs.events[:, 2]], dtype=np.int64)

    data = epochs.get_data().astype(np.float32)  # (288, 22, n_samples)
    return data, labels