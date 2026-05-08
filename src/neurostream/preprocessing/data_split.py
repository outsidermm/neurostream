# scripts/make_splits.py
"""Generate the train/val split for BCI IV 2a session T.

Run once to produce splits/bci_iv_2a_v1.json. Committed to the repo.
Re-running with the same args produces a byte-identical file.
"""

import argparse
import json
from pathlib import Path
from dataclasses import dataclass
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
SPLIT_CONFIG_PATH = Path(_PROJECT_ROOT / "src" / "neurostream" / "data")


@dataclass(frozen=True)
class TrainValSplit:
    train: np.ndarray  # int64 indices
    val: np.ndarray
    seed: int
    version: str


def load_split(path: Path) -> TrainValSplit:
    payload = json.loads(path.read_text())
    return TrainValSplit(
        train=np.array(payload["split"]["train"], dtype=np.int64),
        val=np.array(payload["split"]["val"], dtype=np.int64),
        seed=payload["seed"],
        version=payload["version"],
    )


def make_within_subject_random_split(
    n_trials: int,
    val_fraction: float,
    seed: int,
) -> dict[str, list[int]]:
    """
    Random split of trial indices into train and val.

    Note: 'within-subject' means the split applies to one subject's session T.
    The same indices are used for every subject — i.e., trial 5 is in val for
    every subject. This is the convention used in the EEGNet paper and most
    published BCI IV 2a benchmarks.
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(n_trials)
    rng.shuffle(indices)
    n_val = int(round(n_trials * val_fraction))
    val_idx = sorted(indices[:n_val].tolist())
    train_idx = sorted(indices[n_val:].tolist())
    return {"train": train_idx, "val": val_idx}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--n-trials", type=int, default=288)
    parser.add_argument(
        "--output", type=Path, default=SPLIT_CONFIG_PATH / "bci_iv_2a_v1.json"
    )
    args = parser.parse_args()

    split = make_within_subject_random_split(
        n_trials=args.n_trials,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    payload = {
        "version": "v1",
        "seed": args.seed,
        "dataset": "bci_iv_2a",
        "strategy": "within_subject_random",
        "val_fraction": args.val_fraction,
        "n_train_session_trials": args.n_trials,
        "split": split,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Wrote split to {args.output}")
    print(f"  train: {len(split['train'])} trials")
    print(f"  val:   {len(split['val'])} trials")


if __name__ == "__main__":
    main()
