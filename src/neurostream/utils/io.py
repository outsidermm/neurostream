"""Atomic filesystem writes — write to a temp sibling, then rename.

``Path.replace`` is atomic on POSIX, so a reader never observes a
half-written file even if the process dies mid-write.
"""

from pathlib import Path

import numpy as np


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    """Save ``array`` to ``path`` (.npy) atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.save(f, array)
    tmp.replace(path)


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)
