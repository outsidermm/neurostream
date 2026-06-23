"""Streaming dataset over the harmonised sharded EEG corpus.

Designed for MAE pretraining: yields random windows from memory-mapped
shard files, with source-weighted sampling and correct per-worker seeding
under PyTorch's ``DataLoader`` (which is a classic footgun for
``IterableDataset``).

Expected sidecar JSON schema (emitted by the Days 1-4 harmonisation
pipeline)::

    {
      "version": 1,
      "window_samples": 1000,
      "sampling_rate_hz": 128,
      "n_channels": 22,
      "dtype": "float32",
      "shards": [
        {"path": "shard_0000.npy", "shape": [22, 1234567]},
        ...
      ],
      "segments_by_source": {
        "PhysionetMI": [
          {"shard": 0, "start": 0, "end": 500000},
          ...
        ],
        "Cho2017": [...],
        ...
      }
    }

Segments are contiguous regions of one source within one shard. Sampling
a window means: pick a source (weighted), pick a segment from that source
weighted by segment length, then a random offset within that segment.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info


@dataclass(frozen=True)
class _Segment:
    """One contiguous region of a single source inside one shard."""

    shard_idx: int
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


class EEGWindowDataset(IterableDataset):
    """Infinite stream of random EEG windows from a sharded corpus.

    Each ``__iter__`` call returns a generator that yields windows
    indefinitely. The training loop is responsible for counting steps
    and stopping when ``total_steps`` is reached.

    Args:
        index_path: Path to the sidecar JSON describing shards and
            per-source segments.
        window_samples: Length of each window in samples.
        source_weights: Optional per-source sampling weights. ``None``
            means uniform-per-source (equal weight regardless of source
            size), which oversamples small sources and undersamples
            large ones. Pass a dict to override.
        seed: Base RNG seed. Worker and rank are XOR-mixed in for
            distributed-safe per-worker uniqueness.
        normalize: If True, z-score each window per channel before
            returning. Independently per window — handles per-recording
            amplitude variation.
        min_window_segment_samples: Minimum segment length to be eligible
            for sampling. Defaults to ``window_samples`` (segments
            smaller than one window are dropped).
    """

    def __init__(
        self,
        index_path: Path | str,
        window_samples: int,
        source_weights: Mapping[str, float] | None = None,
        seed: int = 0,
        normalize: bool = True,
        min_window_segment_samples: int | None = None,
    ) -> None:
        super().__init__()
        index_path = Path(index_path)
        if not index_path.exists():
            raise FileNotFoundError(f"sidecar JSON not found: {index_path}")
        if window_samples <= 0:
            raise ValueError(f"window_samples must be positive, got {window_samples}")

        self.window_samples = window_samples
        self.seed = seed
        self.normalize = normalize
        self.min_segment = (
            min_window_segment_samples
            if min_window_segment_samples is not None
            else window_samples
        )

        self._index_path = index_path
        self._shard_dir = index_path.parent
        self._index = json.loads(index_path.read_text())
        self._validate_index(self._index)

        # Group segments by source, filtering out segments too short
        # to contain a window.
        self._segments_by_source: dict[str, list[_Segment]] = {}
        for source, raw in self._index["segments_by_source"].items():
            segs = [_Segment(s["shard"], s["start"], s["end"]) for s in raw]
            segs = [s for s in segs if s.length >= self.min_segment]
            if segs:
                self._segments_by_source[source] = segs

        if not self._segments_by_source:
            raise ValueError(
                "no usable segments after filtering by min_segment "
                f"({self.min_segment} samples)"
            )

        # Resolve source weights to a normalised probability vector.
        self._sources: list[str] = sorted(self._segments_by_source.keys())
        self._source_probs = self._resolve_source_weights(source_weights)

        # Per source: cumulative segment lengths for one-call segment sampling
        # via np.searchsorted.
        self._segment_cum_lengths: dict[str, np.ndarray] = {}
        for source, segs in self._segments_by_source.items():
            lengths = np.array(
                [s.length - self.window_samples + 1 for s in segs],
                dtype=np.int64,
            )
            self._segment_cum_lengths[source] = np.cumsum(lengths)

        # Worker-local state populated by worker_init_fn (or fallback below).
        self._worker_seed: int | None = None
        self._memmaps: dict[int, np.memmap] | None = None

    # ------------------------------------------------------------------
    # Validation / setup helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_index(index: dict[str, object]) -> None:
        required = {"shards", "segments_by_source", "window_samples"}
        missing = required - index.keys()
        if missing:
            raise ValueError(f"sidecar JSON missing required keys: {missing}")
        if not isinstance(index["shards"], list) or not index["shards"]:
            raise ValueError("'shards' must be a non-empty list")
        if (
            not isinstance(index["segments_by_source"], dict)
            or not index["segments_by_source"]
        ):
            raise ValueError("'segments_by_source' must be a non-empty dict")

    def _resolve_source_weights(
        self, weights: Mapping[str, float] | None
    ) -> np.ndarray:
        if weights is None:
            # Equal weight per source (default — analogue of LLM domain weighting).
            probs = np.ones(len(self._sources), dtype=np.float64)
        else:
            probs = np.array(
                [weights.get(s, 0.0) for s in self._sources], dtype=np.float64
            )
            if not (probs >= 0.0).all():
                raise ValueError(f"source weights must be non-negative: {weights}")
            if probs.sum() <= 0.0:
                raise ValueError(f"sum of source weights must be positive: {weights}")
        return probs / probs.sum()

    def _open_memmaps(self) -> dict[int, np.memmap]:
        """Lazily open all shards as memory-maps. Called once per worker.

        Uses ``np.load(mmap_mode="r")`` rather than a raw ``np.memmap``: the
        shards are ``.npy`` files (written by the harmonisation pipeline via
        ``np.save``), so they carry a header. A raw ``np.memmap`` reads from
        byte 0 and would reinterpret that header as data — shifting every
        sample by the header size and injecting garbage values (which then
        overflow when squared during z-scoring).
        """
        memmaps: dict[int, np.memmap] = {}
        for shard_idx, shard in enumerate(self._index["shards"]):
            path = self._shard_dir / shard["path"]
            memmaps[shard_idx] = np.load(path, mmap_mode="r")
        return memmaps

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def _sample_window(
        self, rng: np.random.Generator, memmaps: dict[int, np.memmap]
    ) -> np.ndarray:
        # 1. Pick a source according to weights.
        source = self._sources[rng.choice(len(self._sources), p=self._source_probs)]
        segments = self._segments_by_source[source]
        cum_lengths = self._segment_cum_lengths[source]
        total_valid = int(cum_lengths[-1])

        # 2. Pick a segment weighted by valid-window count.
        u = rng.integers(0, total_valid)
        seg_idx = int(np.searchsorted(cum_lengths, u, side="right"))
        segment = segments[seg_idx]

        # 3. Pick a window offset within that segment.
        max_offset = segment.end - self.window_samples
        offset = int(rng.integers(segment.start, max_offset + 1))

        # 4. Slice from the memmap. Copy out of the read-only mmap so the
        #    returned window is writable (torch.from_numpy requires it).
        shard = memmaps[segment.shard_idx]
        window = np.array(
            shard[:, offset : offset + self.window_samples], dtype=np.float32
        )
        return window

    def _z_score(self, window: np.ndarray) -> np.ndarray:
        mean = window.mean(axis=-1, keepdims=True)
        std = window.std(axis=-1, keepdims=True) + 1e-6
        return (window - mean) / std

    # ------------------------------------------------------------------
    # IterableDataset protocol
    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[Tensor]:
        info = get_worker_info()
        if info is None:
            # Main-process iteration (no DataLoader workers).
            worker_seed = self._worker_seed or self.seed
        else:
            # ``worker_init_fn`` should have set ``self._worker_seed``;
            # fall back to seed + worker_id if it didn't.
            worker_seed = (
                self._worker_seed
                if self._worker_seed is not None
                else self.seed + info.id
            )

        # Memmaps must be opened per-worker (file handles don't fork safely).
        memmaps = self._open_memmaps()
        rng = np.random.default_rng(worker_seed)

        while True:
            try:
                window = self._sample_window(rng, memmaps)
            except (IndexError, ValueError) as e:
                # Skip pathologically small segments / bad slices.
                # In practice these should already be filtered out.
                raise RuntimeError(f"window sampling failed: {e}") from e
            if self.normalize:
                window = self._z_score(window)
            yield torch.from_numpy(window)


def worker_init_fn(worker_id: int) -> None:
    """Seed each DataLoader worker uniquely, including across DDP ranks.

    Without this, all workers receive the same RNG state and produce
    identical windows — a notorious ``IterableDataset`` footgun.
    """
    info = get_worker_info()
    if info is None:
        return
    dataset = info.dataset
    if not isinstance(dataset, EEGWindowDataset):
        return

    # Mix in distributed rank if available so different ranks see different windows.
    try:
        import torch.distributed as dist

        rank = dist.get_rank() if dist.is_initialized() else 0
    except Exception:
        rank = 0

    # Combine base seed, rank, worker id into a unique per-worker seed.
    # The 10_000 multiplier gives plenty of headroom for typical
    # num_workers (8-16) and world sizes (1-64).
    dataset._worker_seed = (  # noqa: SLF001 — intentional cross-call hand-off
        dataset.seed + 1_000_000 * rank + worker_id
    )


def estimate_corpus_hours(index_path: Path | str, sample_rate_hz: int = 128) -> float:
    """Diagnostic helper: how many hours of EEG does the index describe?"""
    idx = json.loads(Path(index_path).read_text())
    total_samples = 0
    for source_segs in idx["segments_by_source"].values():
        for seg in source_segs:
            total_samples += seg["end"] - seg["start"]
    return total_samples / sample_rate_hz / 3600.0


__all__ = ["EEGWindowDataset", "worker_init_fn", "estimate_corpus_hours"]
