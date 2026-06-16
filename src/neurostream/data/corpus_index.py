"""Build the EEGWindowDataset sidecar index from a harmonised manifest.

The Days 1-4 harmonisation pipeline (``corpus_pipeline.py``) emits a
``manifest.json`` describing shards and the recordings packed into each.
``EEGWindowDataset`` instead wants a sidecar with ``shards`` (path + shape)
and ``segments_by_source``. This module bridges the two — one contiguous
recording becomes one segment.

The manifest's ``byte_offset`` is a *sample* offset along the time axis
(see ``corpus_pipeline.flush_shard``: ``offset += n_samples``), so a
segment is simply ``[byte_offset, byte_offset + n_samples)``.
"""

from collections.abc import Mapping
from typing import Any


def build_corpus_index(
    manifest: Mapping[str, Any],
    window_samples: int,
    sampling_rate_hz: int | None = None,
) -> dict[str, Any]:
    """Convert a harmonisation ``manifest.json`` into a dataset sidecar index.

    Args:
        manifest: Parsed ``manifest.json`` (must contain ``shards`` with
            per-shard ``recordings``).
        window_samples: Window length recorded in the index (informational;
            the dataset uses its own constructor value).
        sampling_rate_hz: Optional sampling rate to stamp into the index.

    Returns:
        A dict matching the schema validated by ``EEGWindowDataset``.
    """
    shards_meta = sorted(manifest["shards"], key=lambda s: s["shard_idx"])

    shards: list[dict[str, Any]] = []
    segments_by_source: dict[str, list[dict[str, int]]] = {}
    n_channels: int | None = None

    for position, shard in enumerate(shards_meta):
        shard_idx = int(shard["shard_idx"])
        if shard_idx != position:
            raise ValueError(
                f"shard_idx {shard_idx} is not contiguous at position {position}; "
                "the dataset keys memmaps by list position"
            )
        n_channels = int(shard["n_channels"])
        shards.append(
            {"path": shard["shard_name"], "shape": [n_channels, int(shard["total_samples"])]}
        )
        for rec in shard["recordings"]:
            start = int(rec["byte_offset"])
            end = start + int(rec["n_samples"])
            segments_by_source.setdefault(rec["source"], []).append(
                {"shard": shard_idx, "start": start, "end": end}
            )

    if not shards:
        raise ValueError("manifest has no shards")

    index: dict[str, Any] = {
        "version": 1,
        "window_samples": window_samples,
        "n_channels": n_channels,
        "dtype": "float32",
        "shards": shards,
        "segments_by_source": segments_by_source,
    }
    if sampling_rate_hz is not None:
        index["sampling_rate_hz"] = sampling_rate_hz
    return index
