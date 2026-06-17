"""Distributed training initialisation, safe for world size 1.

The MAE pretraining script is written to be distributed-correct from
day one even on a single GPU. ``torchrun --standalone --nproc_per_node=1``
will set the env vars this module reads, so single-GPU and multi-GPU
training go through identical code paths.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistEnv:
    """Resolved view of the distributed environment."""

    rank: int
    local_rank: int
    world_size: int
    backend: str

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def setup_distributed() -> DistEnv:
    """Initialise ``torch.distributed`` from environment variables.

    Reads ``RANK``, ``LOCAL_RANK``, ``WORLD_SIZE`` (all set by ``torchrun``).
    Falls back to single-process defaults if they're unset.

    Selects the NCCL backend on CUDA, Gloo on CPU. Sets the CUDA device to
    ``local_rank`` so subsequent ``.cuda()`` calls land on the right device.

    Returns:
        :class:`DistEnv` describing the resolved environment.
    """
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if torch.cuda.is_available():
        backend = "nccl"
        torch.cuda.set_device(local_rank)
    else:
        backend = "gloo"

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend)

    return DistEnv(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        backend=backend,
    )


def cleanup_distributed() -> None:
    """Destroy the distributed process group if it was initialised."""
    if dist.is_initialized():
        dist.destroy_process_group()


def reduce_mean(value: torch.Tensor) -> torch.Tensor:
    """All-reduce a scalar tensor to its global mean.

    No-op when ``torch.distributed`` is not initialised. Useful for
    aggregating per-rank loss values for logging.
    """
    if not dist.is_initialized():
        return value
    out = value.detach().clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    out /= dist.get_world_size()
    return out


__all__ = [
    "DistEnv",
    "setup_distributed",
    "cleanup_distributed",
    "reduce_mean",
]
