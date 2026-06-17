"""Tests for the warmup-cosine learning rate schedule."""

from __future__ import annotations
import pytest
from neurostream.training.scheduler import apply_lr, warmup_cosine_lr


def test_warmup_starts_above_zero() -> None:
    """Step 0 should be a non-zero fraction of base_lr (not exactly 0)."""
    lr = warmup_cosine_lr(0, base_lr=1.5e-4, warmup_steps=100, total_steps=1000)
    assert 0.0 < lr < 1.5e-4


def test_warmup_reaches_base_lr_at_end_of_warmup() -> None:
    """Last warmup step should equal base_lr."""
    lr = warmup_cosine_lr(99, base_lr=1.5e-4, warmup_steps=100, total_steps=1000)
    assert lr == pytest.approx(1.5e-4, rel=1e-6)


def test_cosine_decays_to_min_lr_at_total_steps() -> None:
    """At total_steps and beyond, lr should equal min_lr."""
    lr_end = warmup_cosine_lr(
        1000, base_lr=1.5e-4, warmup_steps=100, total_steps=1000, min_lr=1e-6
    )
    assert lr_end == pytest.approx(1e-6, rel=1e-6)

    lr_after = warmup_cosine_lr(
        5000, base_lr=1.5e-4, warmup_steps=100, total_steps=1000, min_lr=1e-6
    )
    assert lr_after == pytest.approx(1e-6, rel=1e-6)


def test_warmup_is_monotonically_non_decreasing() -> None:
    lrs = [
        warmup_cosine_lr(s, 1.5e-4, warmup_steps=100, total_steps=1000)
        for s in range(0, 100)
    ]
    for a, b in zip(lrs, lrs[1:]):
        assert b >= a


def test_cosine_is_monotonically_non_increasing() -> None:
    lrs = [
        warmup_cosine_lr(s, 1.5e-4, warmup_steps=100, total_steps=1000)
        for s in range(100, 1001, 50)
    ]
    for a, b in zip(lrs, lrs[1:]):
        assert b <= a + 1e-12


def test_cosine_midpoint_is_halfway_value() -> None:
    """The midpoint of cosine decay should be (base + min) / 2."""
    base = 1.0
    min_lr = 0.0
    warmup = 100
    total = 1100
    # Midpoint = (warmup + total) / 2 = 600. Cosine value there = 0.5.
    lr = warmup_cosine_lr(
        600, base_lr=base, warmup_steps=warmup, total_steps=total, min_lr=min_lr
    )
    assert lr == pytest.approx(0.5, abs=1e-6)


def test_apply_lr_sets_all_param_groups() -> None:
    class _MockOpt:
        def __init__(self) -> None:
            self.param_groups = [{"lr": 0.0}, {"lr": 0.0}]

    opt = _MockOpt()
    apply_lr(opt, 1.5e-4)
    assert all(pg["lr"] == 1.5e-4 for pg in opt.param_groups)


def test_apply_lr_respects_lr_scale() -> None:
    """Param groups with lr_scale should be multiplied — used for LLRD."""

    class _MockOpt:
        def __init__(self) -> None:
            self.param_groups = [
                {"lr": 0.0, "lr_scale": 1.0},
                {"lr": 0.0, "lr_scale": 0.5},
            ]

    opt = _MockOpt()
    apply_lr(opt, 1e-3)
    assert opt.param_groups[0]["lr"] == 1e-3
    assert opt.param_groups[1]["lr"] == pytest.approx(5e-4)


def test_invalid_total_steps_raises() -> None:
    with pytest.raises(ValueError):
        warmup_cosine_lr(0, base_lr=1e-3, warmup_steps=100, total_steps=50)


def test_negative_step_raises() -> None:
    with pytest.raises(ValueError):
        warmup_cosine_lr(-1, base_lr=1e-3, warmup_steps=100, total_steps=1000)


def test_min_lr_above_base_lr_raises() -> None:
    with pytest.raises(ValueError):
        warmup_cosine_lr(
            500, base_lr=1e-4, warmup_steps=100, total_steps=1000, min_lr=1e-3
        )
