"""MAE pretraining entry point.

Run via torchrun, even at world size 1:

    torchrun --standalone --nproc_per_node=1 -m neurostream.training.pretrain

Hydra composes the config from ``configs/pretrain.yaml`` plus its
``defaults`` (model, data, train). All overrides are positional Hydra
overrides as usual:

    torchrun ... -m neurostream.training.pretrain \\
        model.mask_ratio=0.75 train.total_steps=150000

This script:

  * sets up distributed (world size 1 works the same way),
  * instantiates the MAE model and wraps in DDP if world_size > 1,
  * builds the streaming dataset + DataLoader with correct worker seeding,
  * configures AdamW with He et al. 2022's MAE hyperparameters,
  * runs warmup-cosine LR with bf16 mixed precision,
  * logs every step to MLflow (rank 0 only),
  * saves rolling and milestone checkpoints,
  * is resumable from any saved checkpoint.

It does NOT include the linear-probe evaluation hook — that's Days 10-11
and lives in a separate module (``neurostream.training.linear_probe``)
which the pretrain script calls every ``probe_interval`` steps.
"""

from __future__ import annotations

import logging
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, cast

import hydra
import mlflow
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from neurostream.data.window_dataset import EEGWindowDataset, worker_init_fn
from neurostream.training.checkpoint import CheckpointManager
from neurostream.training.distributed import (
    DistEnv,
    cleanup_distributed,
    reduce_mean,
    setup_distributed,
)
from neurostream.training.optim import split_weight_decay_param_groups
from neurostream.training.scheduler import apply_lr, warmup_cosine_lr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------
def _build_model(cfg: DictConfig, device: torch.device) -> nn.Module:
    model = hydra.utils.instantiate(cfg.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %s | params=%.2fM", type(model).__name__, n_params / 1e6)
    return model


def _build_dataloader(cfg: DictConfig, env: DistEnv) -> DataLoader:
    dataset = EEGWindowDataset(
        index_path=cfg.data.index_path,
        window_samples=cfg.data.window_samples,
        source_weights=cast(
            Mapping[str, float],
            OmegaConf.to_container(cfg.data.source_weights),
        )
        if cfg.data.get("source_weights")
        else None,
        seed=cfg.data.seed + env.rank,
        normalize=cfg.data.normalize,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.train.num_workers > 0,
        prefetch_factor=cfg.train.get("prefetch_factor", 2)
        if cfg.train.num_workers > 0
        else None,
    )


def _build_optimizer(model: nn.Module, cfg: DictConfig) -> torch.optim.Optimizer:
    param_groups = split_weight_decay_param_groups(
        model, weight_decay=cfg.train.weight_decay
    )
    return torch.optim.AdamW(
        param_groups,
        lr=cfg.train.base_lr,
        betas=tuple(cfg.train.betas),
        eps=cfg.train.get("eps", 1e-8),
    )


def _resolve_amp_dtype(name: str) -> torch.dtype | None:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[name]


# ---------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------
def _train_step(
    model: nn.Module,
    batch: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    amp_dtype: torch.dtype | None,
    device: torch.device,
    grad_clip: float,
) -> dict[str, float]:
    """Run one optimisation step. Returns scalar metrics for logging."""
    batch = batch.to(device, non_blocking=True)

    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=amp_dtype)
        if amp_dtype is not None
        else nullcontext()
    )

    optimizer.zero_grad(set_to_none=True)
    with autocast_ctx:
        loss, _, _ = model(batch)

    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    return {"loss": loss.detach().float().item(), "grad_norm": float(grad_norm)}


def _maybe_resume(
    cfg: DictConfig,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ckpt: CheckpointManager,
    device: torch.device,
) -> int:
    """Resume from the latest checkpoint if one exists. Returns the resume step."""
    if not cfg.train.get("auto_resume", True):
        return 0
    latest = ckpt.find_latest()
    if latest is None:
        return 0
    state = ckpt.load(latest, model, optimizer, map_location=device)
    return int(state.get("step", 0))


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
@hydra.main(version_base=None, config_path="../../../configs", config_name="pretrain")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))

    env = setup_distributed()
    device = torch.device(
        f"cuda:{env.local_rank}" if torch.cuda.is_available() else "cpu"
    )
    torch.manual_seed(cfg.train.seed + env.rank)

    # ---- Build everything --------------------------------------------
    model = _build_model(cfg, device)
    if env.is_distributed:
        model = DDP(
            model,
            device_ids=[env.local_rank] if torch.cuda.is_available() else None,
            output_device=env.local_rank if torch.cuda.is_available() else None,
        )

    loader = _build_dataloader(cfg, env)
    optimizer = _build_optimizer(model.module if isinstance(model, DDP) else model, cfg)

    amp_dtype = _resolve_amp_dtype(cfg.train.amp)
    # GradScaler is only needed for fp16, never for bf16.
    scaler = torch.cuda.amp.GradScaler() if amp_dtype == torch.float16 else None

    ckpt = CheckpointManager(
        ckpt_dir=Path(cfg.train.ckpt_dir),
        rolling_interval=cfg.train.rolling_interval,
        rolling_keep=cfg.train.rolling_keep,
        milestones=cfg.train.milestones,
        is_main=env.is_main,
    )
    start_step = _maybe_resume(cfg, model, optimizer, ckpt, device)
    if start_step > 0 and env.is_main:
        logger.info("Resuming from step %d", start_step)

    # ---- MLflow setup (rank 0 only) ----------------------------------
    if env.is_main:
        mlflow.set_tracking_uri(cfg.log.tracking_uri)
        mlflow.set_experiment(cfg.log.experiment)
        mlflow.start_run(run_name=cfg.log.get("run_name"))
        mlflow.log_params(_flatten_config(OmegaConf.to_container(cfg, resolve=True)))

    # ---- Training loop ------------------------------------------------
    model.train()
    step = start_step
    last_log = time.time()
    try:
        for batch in loader:
            if step >= cfg.train.total_steps:
                break

            lr = warmup_cosine_lr(
                step,
                base_lr=cfg.train.base_lr,
                warmup_steps=cfg.train.warmup_steps,
                total_steps=cfg.train.total_steps,
                min_lr=cfg.train.get("min_lr", 0.0),
            )
            apply_lr(optimizer, lr)

            metrics = _train_step(
                model,
                batch,
                optimizer,
                scaler,
                amp_dtype,
                device,
                grad_clip=cfg.train.grad_clip,
            )

            # Aggregate loss across ranks for accurate logging.
            loss_tensor = torch.tensor(metrics["loss"], device=device)
            global_loss = reduce_mean(loss_tensor).item()

            # Log every step on rank 0; flush throughput stats periodically.
            if env.is_main:
                step_time = time.time() - last_log
                last_log = time.time()
                mlflow.log_metrics(
                    {
                        "train/loss": global_loss,
                        "train/lr": lr,
                        "train/grad_norm": metrics["grad_norm"],
                        "train/step_time_s": step_time,
                    },
                    step=step,
                )
                if step % cfg.log.console_every == 0:
                    logger.info(
                        "step=%d loss=%.4f lr=%.2e grad_norm=%.3f t=%.2fs",
                        step,
                        global_loss,
                        lr,
                        metrics["grad_norm"],
                        step_time,
                    )

            # Checkpoint hooks.
            ckpt.maybe_save(
                step=step + 1,
                model=model.module if isinstance(model, DDP) else model,
                optimizer=optimizer,
                config=OmegaConf.to_container(cfg, resolve=True),  # type: ignore[arg-type]
            )

            step += 1

        if env.is_main:
            logger.info("Pretraining complete at step %d", step)
    finally:
        if env.is_main and mlflow.active_run():
            mlflow.end_run()
        cleanup_distributed()


def _flatten_config(cfg: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten a nested config dict into ``a.b.c -> value`` form for MLflow."""
    out: dict[str, Any] = {}
    if isinstance(cfg, dict):
        for k, v in cfg.items():
            out.update(_flatten_config(v, f"{prefix}{k}."))
    elif isinstance(cfg, (list, tuple)):
        out[prefix.rstrip(".")] = str(cfg)
    else:
        out[prefix.rstrip(".")] = cfg
    return out


if __name__ == "__main__":
    main()
