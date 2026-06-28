"""Tests for building the dataset sidecar index from a manifest."""

import pytest

from neurostream.data.corpus_index import build_corpus_index


def _manifest() -> dict:
    return {
        "shards": [
            {
                "shard_name": "shard_000.npy",
                "shard_idx": 0,
                "n_channels": 22,
                "total_samples": 32000,
                "recordings": [
                    {"source": "PhysionetMI", "byte_offset": 0, "n_samples": 16000},
                    {"source": "PhysionetMI", "byte_offset": 16000, "n_samples": 16000},
                ],
            },
            {
                "shard_name": "shard_001.npy",
                "shard_idx": 1,
                "n_channels": 22,
                "total_samples": 20000,
                "recordings": [
                    {"source": "Cho2017", "byte_offset": 0, "n_samples": 20000},
                ],
            },
        ]
    }


def test_index_has_required_keys():
    index = build_corpus_index(_manifest(), window_samples=1000)
    for key in ("shards", "segments_by_source", "window_samples"):
        assert key in index


def test_shards_carry_path_and_shape():
    index = build_corpus_index(_manifest(), window_samples=1000)
    assert index["shards"][0] == {"path": "shard_000.npy", "shape": [22, 32000]}
    assert index["shards"][1] == {"path": "shard_001.npy", "shape": [22, 20000]}


def test_segments_are_grouped_by_source_with_sample_offsets():
    index = build_corpus_index(_manifest(), window_samples=1000)
    assert index["segments_by_source"]["PhysionetMI"] == [
        {"shard": 0, "start": 0, "end": 16000},
        {"shard": 0, "start": 16000, "end": 32000},
    ]
    assert index["segments_by_source"]["Cho2017"] == [
        {"shard": 1, "start": 0, "end": 20000},
    ]


def test_dtype_and_optional_sampling_rate():
    index = build_corpus_index(_manifest(), window_samples=1000, sampling_rate_hz=128)
    assert index["dtype"] == "float32"
    assert index["sampling_rate_hz"] == 128


def test_non_contiguous_shard_idx_raises():
    manifest = _manifest()
    manifest["shards"][1]["shard_idx"] = 5
    with pytest.raises(ValueError, match="contiguous"):
        build_corpus_index(manifest, window_samples=1000)
