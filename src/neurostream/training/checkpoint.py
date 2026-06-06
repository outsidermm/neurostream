"""Checkpoint manager for MAE pretraining.

Implements the spec from Days 7-9:

* **Rolling**: save every ``rolling_interval`` steps, keep the latest
  ``rolling_keep`` checkpoints, delete older ones.
* **Milestone**: at fixed step counts (50k, 100k, ...), save a permanent
  checkpoint that is never automatically deleted. These are the points
  the ablation study (Days 15-19) evaluates.

Checkpoints are torch ``.pt`` files containing model, optimizer, RNG,
step, and the resolved Hydra config (as a dict).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Save and load pretraining checkpoints with rolling + milestone retention.

    Args:
        ckpt_dir: Directory where checkpoints live. Created if missing.
        rolling_interval: Step interval for rolling saves.
        rolling_keep: Number of most-recent rolling checkpoints to retain.
        milestones: Step counts at which a permanent checkpoint is saved.
        is_main: Only the main process writes files (true by default for
            single-GPU; pass ``DistEnv.is_main`` in distributed runs).
    """

    ROLLING_PREFIX = "rolling"
    MILESTONE_PREFIX = "milestone"
    LATEST_NAME = "latest.pt"

    def __init__(
        self,
        ckpt_dir: Path | str,
        rolling_interval: int = 10_000,
        rolling_keep: int = 3,
        milestones: Iterable[int] = (50_000, 100_000, 150_000, 200_000, 300_000),
        is_main: bool = True,
    ) -> None:
        if rolling_interval <= 0:
            raise ValueError(
                f"rolling_interval must be positive, got {rolling_interval}"
            )
        if rolling_keep <= 0:
            raise ValueError(f"rolling_keep must be positive, got {rolling_keep}")

        self.ckpt_dir = Path(ckpt_dir)
        self.rolling_interval = rolling_interval
        self.rolling_keep = rolling_keep
        self.milestones = sorted(set(milestones))
        self.is_main = is_main

        if self.is_main:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def maybe_save(
        self,
        step: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: dict[str, Any],
        extra: dict[str, Any] | None = None,
    ) -> Path | None:
        """Save a checkpoint if ``step`` is a rolling or milestone trigger.

        Returns the path written, or ``None`` if no save occurred.
        """
        if not self.is_main:
            return None

        is_rolling = step > 0 and step % self.rolling_interval == 0
        is_milestone = step in self.milestones
        if not (is_rolling or is_milestone):
            return None

        state = self._build_state(step, model, optimizer, config, extra)

        # Milestone takes precedence (kept forever).
        if is_milestone:
            path = self.ckpt_dir / f"{self.MILESTONE_PREFIX}_step{step:08d}.pt"
            torch.save(state, path)
            logger.info("Saved milestone checkpoint: %s", path)
        else:
            path = self.ckpt_dir / f"{self.ROLLING_PREFIX}_step{step:08d}.pt"
            torch.save(state, path)
            logger.info("Saved rolling checkpoint: %s", path)
            self._prune_rolling()

        # Always update the convenience "latest.pt" symlink/copy.
        self._update_latest(path)
        return path

    def load(
        self,
        path: Path | str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        map_location: str | torch.device | None = None,
    ) -> dict[str, Any]:
        """Load a checkpoint into ``model`` (and optionally ``optimizer``).

        Returns the full state dict for callers that need extras (step,
        config, RNG state).
        """
        path = Path(path)
        state: dict[str, Any] = torch.load(path, map_location=map_location)

        # Strip DDP "module." prefix if present.
        msd = state["model"]
        msd = {k.removeprefix("module."): v for k, v in msd.items()}
        model.load_state_dict(msd)

        if optimizer is not None and "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])

        logger.info("Loaded checkpoint from step %d: %s", state.get("step", -1), path)
        return state

    def find_latest(self) -> Path | None:
        """Locate the most recent checkpoint (rolling or milestone)."""
        candidates = sorted(self.ckpt_dir.glob(f"{self.ROLLING_PREFIX}_step*.pt"))
        candidates += sorted(self.ckpt_dir.glob(f"{self.MILESTONE_PREFIX}_step*.pt"))
        if not candidates:
            return None
        # File names embed step as zero-padded ints, so lexicographic sort
        # over the combined list still orders by step count overall.
        return max(candidates, key=lambda p: int(p.stem.split("step")[-1]))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _build_state(
        self,
        step: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: dict[str, Any],
        extra: dict[str, Any] | None,
    ) -> dict[str, Any]:
        # Unwrap DDP to keep the saved state portable across single/multi-GPU.
        msd = (
            model.module.state_dict()
            if hasattr(model, "module")
            else model.state_dict()
        )
        state: dict[str, Any] = {
            "step": step,
            "model": msd,
            "optimizer": optimizer.state_dict(),
            "config": config,
            "torch_rng_state": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        if extra:
            state.update(extra)
        return state

    def _prune_rolling(self) -> None:
        rolling = sorted(self.ckpt_dir.glob(f"{self.ROLLING_PREFIX}_step*.pt"))
        excess = len(rolling) - self.rolling_keep
        for old in rolling[: max(0, excess)]:
            try:
                old.unlink()
                logger.debug("Pruned rolling checkpoint: %s", old)
            except OSError as e:
                logger.warning("Failed to prune %s: %s", old, e)

    def _update_latest(self, path: Path) -> None:
        latest = self.ckpt_dir / self.LATEST_NAME
        # Use a relative symlink so the directory is portable across moves.
        try:
            if latest.is_symlink() or latest.exists():
                latest.unlink()
            latest.symlink_to(path.name)
        except OSError:
            # Some filesystems (Windows-mounted shares, etc.) don't permit
            # symlinks. Fall back to a hard copy.
            import shutil

            shutil.copy2(path, latest)


__all__ = ["CheckpointManager"]
