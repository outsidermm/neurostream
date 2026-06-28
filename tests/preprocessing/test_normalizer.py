import numpy as np
import pytest

from neurostream.preprocessing.normalizer import (
    apply_normalizer,
    fit_normalizer,
    load_normalizer,
    save_normalizer,
)


def test_fit_produces_unit_normal_on_training():
    rng = np.random.default_rng(0)
    # Construct data with known per-channel mean and std
    x_train = rng.standard_normal((100, 22, 750)).astype(np.float32) * 5 + 2
    stats = fit_normalizer(x_train)
    x_norm = apply_normalizer(x_train, stats)
    # After normalization, training data has mean ~0, std ~1 per channel
    np.testing.assert_allclose(x_norm.mean(axis=(0, 2)), 0.0, atol=1e-5)
    np.testing.assert_allclose(x_norm.std(axis=(0, 2)), 1.0, atol=1e-3)


def test_apply_does_not_recompute():
    rng = np.random.default_rng(0)
    x_train = rng.standard_normal((100, 22, 750)).astype(np.float32)  # mean 0, std 1
    x_test = (
        rng.standard_normal((30, 22, 750)).astype(np.float32) * 3 + 5
    )  # mean 5, std 3
    stats = fit_normalizer(x_train)
    x_test_norm = apply_normalizer(x_test, stats)
    # Test set's actual mean (5) and std (3) differ from train's (0, 1).
    # After applying train stats to test, test should NOT be unit-normal.
    assert not np.allclose(x_test_norm.mean(axis=(0, 2)), 0.0, atol=0.5)
    assert not np.allclose(x_test_norm.std(axis=(0, 2)), 1.0, atol=0.5)


def test_constant_channel_raises():
    x = np.ones((10, 22, 750), dtype=np.float32)
    with pytest.raises(ValueError):
        fit_normalizer(x, eps=0)


def test_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    x_train = rng.standard_normal((50, 22, 750)).astype(np.float32) * 3 + 1
    stats = fit_normalizer(x_train)

    save_normalizer(stats, tmp_path / "norm.npz")
    loaded = load_normalizer(tmp_path / "norm.npz")

    np.testing.assert_array_equal(stats.mean, loaded.mean)
    np.testing.assert_array_equal(stats.std, loaded.std)

    # And applying the loaded stats produces identical output
    x = rng.standard_normal((10, 22, 750)).astype(np.float32)
    np.testing.assert_array_equal(
        apply_normalizer(x, stats),
        apply_normalizer(x, loaded),
    )
