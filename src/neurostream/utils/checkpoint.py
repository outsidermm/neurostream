"""Per-subject ingest checkpoint — survives process crashes and interrupts.

A subject is recorded in the checkpoint only *after* its shard has been
flushed to disk, so the invariant is always:

    checkpointed  <=>  data is on disk in a named shard

Reloading the manager from the same path restores that set, letting
``ingest_corpus`` skip already-written subjects on restart.

Rejected recording entries are co-stored so the final manifest is complete
even when a run is resumed after an earlier crash.
"""

import json
from pathlib import Path

from neurostream.utils.io import atomic_write_text


class CheckpointManager:
    """Tracks completed (source, subject) pairs and accumulated rejects."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._done: set[tuple[str, int]] = set()
        self._rejected: list[dict] = []
        if path.exists():
            data = json.loads(path.read_text())
            for entry in data.get("completed", []):
                self._done.add((str(entry[0]), int(entry[1])))
            self._rejected = list(data.get("rejected", []))

    def is_done(self, source: str, subject: int) -> bool:
        return (source, subject) in self._done

    def mark_done(
        self, source: str, subject: int, rejected: list[dict] | None = None
    ) -> None:
        """Mark subject complete, append its rejects, and persist atomically."""
        self._done.add((source, subject))
        if rejected:
            self._rejected.extend(rejected)
        _persist(self._path, self._done, self._rejected)

    @property
    def completed_count(self) -> int:
        return len(self._done)

    @property
    def rejected(self) -> list[dict]:
        return list(self._rejected)


def _persist(
    path: Path, done: set[tuple[str, int]], rejected: list[dict]
) -> None:
    data = {
        "version": 1,
        "completed": sorted([s, n] for s, n in done),
        "rejected": rejected,
    }
    atomic_write_text(path, json.dumps(data, indent=2, default=str))
