# tests/test_bci_pipeline.py

import numpy as np

from neurostream.preprocessing.bci_pipeline import (
    PipelineConfig,
    fit_pipeline,
    load_pipeline,
    save_pipeline,
)
from neurostream.preprocessing.filters import BandpassParams


def test_fit_pipeline_is_deterministic():
    """Same input + same config → identical fitted pipeline."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((100, 22, 750)).astype(np.float32)
    config = PipelineConfig(bandpass=BandpassParams(8, 30, 250.0))

    p1 = fit_pipeline(x, config)
    p2 = fit_pipeline(x, config)

    np.testing.assert_array_equal(p1.normalizer.mean, p2.normalizer.mean)
    np.testing.assert_array_equal(p1.normalizer.std, p2.normalizer.std)


def test_transform_is_deterministic():
    """Same fitted pipeline + same input → bit-identical output."""
    rng = np.random.default_rng(0)
    x_train = rng.standard_normal((100, 22, 750)).astype(np.float32)
    x_test = rng.standard_normal((30, 22, 750)).astype(np.float32)

    config = PipelineConfig(bandpass=BandpassParams(8, 30, 250.0))
    pipeline = fit_pipeline(x_train, config)

    out1 = pipeline.transform(x_test)
    out2 = pipeline.transform(x_test)

    np.testing.assert_array_equal(out1, out2)


def test_pipeline_no_train_test_leakage():
    """Train comes out unit-normal; test (with different distribution) does not."""
    rng = np.random.default_rng(0)
    x_train = rng.standard_normal((100, 22, 750)).astype(np.float32)
    x_test = rng.standard_normal((30, 22, 750)).astype(np.float32) * 4

    config = PipelineConfig(bandpass=BandpassParams(8, 30, 250.0))
    pipeline = fit_pipeline(x_train, config)

    train_out = pipeline.transform(x_train)
    test_out = pipeline.transform(x_test)

    # Train: roughly unit-normal post-pipeline (filtering shifts std slightly,
    # so use loose tolerance).
    np.testing.assert_allclose(train_out.mean(axis=(0, 2)), 0.0, atol=0.05)
    np.testing.assert_allclose(train_out.std(axis=(0, 2)), 1.0, atol=0.1)

    # Test: std should be ~4 (NOT 1), because we used train's σ to normalize.
    # If apply_normalizer were buggy and recomputed stats on test, std would be ~1.
    test_std_per_channel = test_out.std(axis=(0, 2))
    assert (test_std_per_channel > 2.5).all()


def test_save_load_roundtrip(tmp_path):
    """A loaded pipeline produces identical output to the original."""
    rng = np.random.default_rng(0)
    x_train = rng.standard_normal((50, 22, 750)).astype(np.float32)
    x_test = rng.standard_normal((10, 22, 750)).astype(np.float32)
    config = PipelineConfig(bandpass=BandpassParams(8, 30, 250.0))
    pipeline = fit_pipeline(x_train, config)

    save_pipeline(pipeline, tmp_path / "pipe")
    loaded = load_pipeline(tmp_path / "pipe")

    np.testing.assert_array_equal(
        pipeline.transform(x_test),
        loaded.transform(x_test),
    )
