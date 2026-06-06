"""Tests for the checkpoint manager."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from neurostream.training.checkpoint import CheckpointManager


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _make(tmp_path: Path, **kwargs: object) -> CheckpointManager:
    return CheckpointManager(ckpt_dir=tmp_path, **kwargs)


def test_rolling_save_writes_file(tmp_path: Path) -> None:
    ckpt = _make(tmp_path, rolling_interval=100, milestones=[])
    model = _Tiny()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = ckpt.maybe_save(step=100, model=model, optimizer=opt, config={})
    assert path is not None
    assert path.exists()
    assert "rolling" in path.name


def test_milestone_save_writes_file(tmp_path: Path) -> None:
    ckpt = _make(tmp_path, rolling_interval=10_000, milestones=[50])
    model = _Tiny()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = ckpt.maybe_save(step=50, model=model, optimizer=opt, config={})
    assert path is not None
    assert "milestone" in path.name


def test_no_save_at_non_trigger_step(tmp_path: Path) -> None:
    ckpt = _make(tmp_path, rolling_interval=100, milestones=[50])
    model = _Tiny()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    assert ckpt.maybe_save(step=37, model=model, optimizer=opt, config={}) is None


def test_rolling_keeps_only_latest_n(tmp_path: Path) -> None:
    """After 5 rolling saves with keep=3, only the latest 3 remain."""
    ckpt = _make(tmp_path, rolling_interval=100, rolling_keep=3, milestones=[])
    model = _Tiny()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in (100, 200, 300, 400, 500):
        ckpt.maybe_save(step=step, model=model, optimizer=opt, config={})

    rolling = sorted(tmp_path.glob("rolling_step*.pt"))
    assert len(rolling) == 3
    # The oldest two should be gone; latest three should be present.
    surviving_steps = {int(p.stem.split("step")[-1]) for p in rolling}
    assert surviving_steps == {300, 400, 500}


def test_milestones_are_never_pruned(tmp_path: Path) -> None:
    """Milestones survive any number of rolling saves."""
    ckpt = _make(
        tmp_path,
        rolling_interval=100,
        rolling_keep=2,
        milestones=[200, 400],
    )
    model = _Tiny()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in (100, 200, 300, 400, 500, 600):
        ckpt.maybe_save(step=step, model=model, optimizer=opt, config={})

    milestones = sorted(tmp_path.glob("milestone_step*.pt"))
    assert len(milestones) == 2


def test_roundtrip_preserves_state(tmp_path: Path) -> None:
    ckpt = _make(tmp_path, rolling_interval=10, milestones=[])
    model_a = _Tiny()
    opt_a = torch.optim.AdamW(model_a.parameters(), lr=1e-3)

    # Run a step to give optimizer some state.
    x = torch.randn(2, 4)
    model_a(x).sum().backward()
    opt_a.step()

    saved_path = ckpt.maybe_save(
        step=10, model=model_a, optimizer=opt_a, config={"foo": "bar"}
    )
    assert saved_path is not None

    model_b = _Tiny()
    opt_b = torch.optim.AdamW(model_b.parameters(), lr=1e-3)
    state = ckpt.load(saved_path, model_b, opt_b)

    # Weights match exactly.
    for (n1, p1), (n2, p2) in zip(
        model_a.named_parameters(), model_b.named_parameters()
    ):
        assert n1 == n2
        assert torch.allclose(p1, p2)

    # Config and step round-tripped.
    assert state["step"] == 10
    assert state["config"] == {"foo": "bar"}


def test_find_latest_returns_highest_step(tmp_path: Path) -> None:
    ckpt = _make(tmp_path, rolling_interval=100, milestones=[200])
    model = _Tiny()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in (100, 200, 300):
        ckpt.maybe_save(step=step, model=model, optimizer=opt, config={})

    latest = ckpt.find_latest()
    assert latest is not None
    assert "step00000300" in latest.name


def test_find_latest_returns_none_when_empty(tmp_path: Path) -> None:
    ckpt = _make(tmp_path)
    assert ckpt.find_latest() is None


def test_is_main_false_suppresses_writes(tmp_path: Path) -> None:
    ckpt = _make(
        tmp_path,
        rolling_interval=100,
        milestones=[],
        is_main=False,
    )
    model = _Tiny()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    assert ckpt.maybe_save(step=100, model=model, optimizer=opt, config={}) is None
    assert not any(tmp_path.glob("rolling*.pt"))


def test_validates_constructor_args(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        _make(tmp_path, rolling_interval=0)
    with pytest.raises(ValueError):
        _make(tmp_path, rolling_keep=0)


def test_unwraps_ddp_module_prefix(tmp_path: Path) -> None:
    """A state dict with 'module.' prefix should load correctly into a plain model."""
    ckpt = _make(tmp_path, rolling_interval=10, milestones=[])
    model = _Tiny()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Synthesize a "DDP-style" state by saving a model that has been
    # given a fake module.* prefix in its keys.
    path = ckpt.maybe_save(step=10, model=model, optimizer=opt, config={})
    assert path is not None
    state = torch.load(path)
    state["model"] = {f"module.{k}": v for k, v in state["model"].items()}
    torch.save(state, path)

    fresh = _Tiny()
    ckpt.load(path, fresh)  # should not raise
    for p1, p2 in zip(model.parameters(), fresh.parameters()):
        assert torch.allclose(p1, p2)
