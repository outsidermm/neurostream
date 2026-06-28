"""Tests for preprocessing.referencing."""

import numpy as np

from neurostream.preprocessing.referencing import common_average_reference


def test_car_zeroes_per_timestep_mean():
    data = np.random.default_rng(0).standard_normal((22, 5000))
    out = common_average_reference(data)
    np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-12)


def test_car_preserves_shape():
    data = np.random.default_rng(0).standard_normal((22, 500))
    assert common_average_reference(data).shape == data.shape


def test_car_removes_signal_common_to_all_channels():
    # A signal shared by every channel should be cancelled by CAR:
    # CAR(base + common) == CAR(base).
    rng = np.random.default_rng(0)
    base = rng.standard_normal((8, 1000))
    common = rng.standard_normal(1000)
    np.testing.assert_allclose(
        common_average_reference(base + common),
        common_average_reference(base),
        atol=1e-10,
    )
