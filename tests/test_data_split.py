# tests/data/test_data_splits.py
import json
from pathlib import Path

import numpy as np

from neurostream.preprocessing.data_split import (
    load_split,
    make_within_subject_random_split,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
SPLIT_CONFIG_PATH = Path(_PROJECT_ROOT / "src" / "neurostream" / "data")


def test_split_partitions_completely(split_path: Path):
    split = load_split(split_path)
    combined = np.concatenate([split.train, split.val])
    np.testing.assert_array_equal(np.sort(combined), np.arange(288))


def test_split_no_overlap(split_path: Path):
    split = load_split(split_path)
    assert len(set(split.train) & set(split.val)) == 0


def test_split_is_deterministic():
    """Same seed → same split, every time."""
    a = make_within_subject_random_split(n_trials=288, val_fraction=0.2, seed=42)
    b = make_within_subject_random_split(n_trials=288, val_fraction=0.2, seed=42)
    assert a == b


def test_committed_split_matches_seed(split_path: Path):
    """The committed JSON must match what the script produces with its declared seed.
    Catches: someone hand-edited the JSON, or seed changed without regenerating."""
    payload = json.loads(split_path.read_text())
    regenerated = make_within_subject_random_split(
        n_trials=payload["n_train_session_trials"],
        val_fraction=payload["val_fraction"],
        seed=payload["seed"],
    )
    assert regenerated["train"] == payload["split"]["train"]
    assert regenerated["val"] == payload["split"]["val"]
