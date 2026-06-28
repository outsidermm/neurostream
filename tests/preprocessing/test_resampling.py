"""Tests for preprocessing.resampling."""

import numpy as np
import pytest

from neurostream.preprocessing.resampling import resample_to_fs


@pytest.mark.parametrize("source_fs", [160.0, 250.0, 500.0, 512.0, 1000.0])
def test_resample_to_128hz_yields_expected_length(source_fs: float):
    n_seconds = 70
    data = np.random.default_rng(0).standard_normal((22, int(source_fs * n_seconds)))
    out = resample_to_fs(data, source_fs, 128)
    assert out.shape == (22, n_seconds * 128), (
        f"source_fs={source_fs}: expected 8960 samples, got {out.shape[1]}"
    )


def test_resample_is_noop_when_rates_match():
    data = np.random.default_rng(0).standard_normal((22, 1280))
    out = resample_to_fs(data, 128, 128)
    assert out is data  # returned unchanged, no copy


def test_resample_rejects_non_integer_source_fs():
    data = np.zeros((4, 1000))
    with pytest.raises(ValueError, match="Non-integer source fs"):
        resample_to_fs(data, 160.5, 128)
