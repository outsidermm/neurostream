"""
Within-subject EEGNet training loop.

Design constraints (Phase 1):
- No frameworks (Lightning, Accelerate) — every step must be inspectable
- Single seed controls all stochasticity
- Hydra config drives all hyperparameters
- MLflow logs every run artefact needed to reproduce or compare results
"""

from __future__ import annotations

import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import flatdict
import hydra
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from neurostream.data.loader import load_subject
from neurostream.models.eegnet import EEGNet
from neurostream.preprocessing.data_split import load_split
from neurostream.preprocessing.filters import BandpassParams
from neurostream.preprocessing.pipeline import (
    PipelineConfig,
    fit_pipeline,
    save_pipeline,
)


# ── Reproducibility ──────────────────────────────────────────────────────────


def set_deterministic_seed(seed: int) -> None:
    """Lock down all RNG sources. Call before any tensor allocation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic cudnn ops — slight perf cost, required for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Data helpers ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FittedShape:
    n_channels: int
    n_samples: int


def make_loaders(
    subject_id: int,
    split_path: Path,
    bandpass: BandpassParams,
    batch_size: int,
    pipelines_dir: Path,
) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any], FittedShape]:
    """
    Build train/val/test DataLoaders for one subject.

      Session T  →  train + val (committed within-subject split, shared across subjects)
      Session E  →  test (held-out, never touched during training)

    The fitted preprocessing pipeline (bandpass + per-channel z-score) is fit
    on train only and persisted to disk so the test set is reproducible from
    artefacts alone.
    """
    epochs_train, labels_train = load_subject(subject_id, "T")
    epochs_test, labels_test = load_subject(subject_id, "E")

    split = load_split(split_path)
    train_idx, val_idx = split.train, split.val

    # ── Preprocessing ────────────────────────────────────────────────────────
    pipeline = fit_pipeline(
        epochs_train[train_idx],
        config=PipelineConfig(bandpass=bandpass),
    )
    save_pipeline(pipeline, pipelines_dir / f"subject_{subject_id:02d}")

    X_train = pipeline.transform(epochs_train[train_idx])
    X_val = pipeline.transform(epochs_train[val_idx])
    X_test = pipeline.transform(epochs_test)

    y_train = labels_train[train_idx]
    y_val = labels_train[val_idx]
    y_test = labels_test

    def to_loader(X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader[Any]:
        dataset = TensorDataset(
            torch.from_numpy(X).float(),
            torch.from_numpy(y).long(),
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,  # cached npz loads are fast; multiproc adds overhead
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    return (
        to_loader(X_train, y_train, shuffle=True),
        to_loader(X_val, y_val, shuffle=False),
        to_loader(X_test, y_test, shuffle=False),
        FittedShape(n_channels=X_train.shape[1], n_samples=X_train.shape[2]),
    )


# ── Training ─────────────────────────────────────────────────────────────────


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float


def run_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> EpochMetrics:
    """
    Single pass through `loader`. If optimizer is None, runs in eval mode.
    Returns loss and accuracy averaged over all samples.
    """
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(training):
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            if training and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(y_batch)
            correct += (logits.argmax(dim=1) == y_batch).sum().item()
            total += len(y_batch)

    return EpochMetrics(
        loss=total_loss / total,
        accuracy=correct / total,
    )


def train_subject(
    subject_id: int,
    cfg: DictConfig,
    device: torch.device,
    split_path: Path,
    pipelines_dir: Path,
    checkpoints_dir: Path,
) -> float:
    """Train EEGNet on one subject. Returns test accuracy."""

    bandpass = BandpassParams(
        low_hz=cfg.preprocessing.bandpass.low_hz,
        high_hz=cfg.preprocessing.bandpass.high_hz,
        fs_hz=cfg.preprocessing.bandpass.fs_hz,
        order=cfg.preprocessing.bandpass.order,
    )

    train_loader, val_loader, test_loader, shape = make_loaders(
        subject_id=subject_id,
        split_path=split_path,
        bandpass=bandpass,
        batch_size=cfg.training.batch_size,
        pipelines_dir=pipelines_dir,
    )

    # n_samples comes from the data, not the config — single source of truth
    model = EEGNet(
        n_classes=cfg.model.n_classes,
        n_channels=shape.n_channels,
        n_samples=shape.n_samples,
        fs=cfg.model.fs,
        f1=cfg.model.f1,
        d=cfg.model.d,
        dropout=cfg.model.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = checkpoints_dir / f"subject_{subject_id:02d}_best.pt"

    for epoch in range(1, cfg.training.epochs + 1):
        train_m = run_epoch(model, train_loader, criterion, optimizer, device)
        val_m = run_epoch(model, val_loader, criterion, None, device)

        if val_m.loss < best_val_loss:
            best_val_loss = val_m.loss
            torch.save(model.state_dict(), best_ckpt_path)

        mlflow.log_metrics(
            {
                f"s{subject_id:02d}/train_loss": train_m.loss,
                f"s{subject_id:02d}/train_acc": train_m.accuracy,
                f"s{subject_id:02d}/val_loss": val_m.loss,
                f"s{subject_id:02d}/val_acc": val_m.accuracy,
            },
            step=epoch,
        )

    # ── Test evaluation ──────────────────────────────────────────────────────
    # Use best-val checkpoint, not final weights (which may have overfit)
    model.load_state_dict(
        torch.load(best_ckpt_path, map_location=device, weights_only=True)
    )
    test_m = run_epoch(model, test_loader, criterion, None, device)

    mlflow.log_metric(f"s{subject_id:02d}/test_acc", test_m.accuracy)
    mlflow.log_artifact(str(best_ckpt_path))
    mlflow.log_artifacts(
        str(pipelines_dir / f"subject_{subject_id:02d}"),
        artifact_path=f"pipelines/subject_{subject_id:02d}",
    )

    return test_m.accuracy


# ── Entry point ──────────────────────────────────────────────────────────────


def _get_git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _flatten_for_mlflow(d: dict[Any, Any]) -> dict[str, Any]:
    """MLflow rejects nested dicts — flatten to dotted string keys."""
    return {str(k): v for k, v in flatdict.FlatDict(d, delimiter=".").items()}


@hydra.main(version_base=None, config_path="../../../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    set_deterministic_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split_path = Path(to_absolute_path(cfg.split.path))
    checkpoints_dir = Path(to_absolute_path(cfg.paths.checkpoints_dir))
    pipelines_dir = Path(to_absolute_path(cfg.paths.pipelines_dir))

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run(run_name=f"eegnet-baseline-seed{cfg.seed}"):
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(cfg_dict, dict)
        mlflow.log_params(_flatten_for_mlflow(cfg_dict))
        mlflow.set_tag("git_sha", _get_git_sha())
        mlflow.set_tag("device", str(device))

        per_subject_acc: dict[int, float] = {}
        for subject_id in cfg.subjects:
            acc = train_subject(
                subject_id=subject_id,
                cfg=cfg,
                device=device,
                split_path=split_path,
                pipelines_dir=pipelines_dir,
                checkpoints_dir=checkpoints_dir,
            )
            per_subject_acc[subject_id] = acc
            print(f"Subject {subject_id:02d}: test_acc={acc:.4f}")

        mean_acc = float(np.mean(list(per_subject_acc.values())))
        mlflow.log_metric("mean_test_acc", mean_acc)
        print(f"\nMean test accuracy: {mean_acc:.4f}")


if __name__ == "__main__":
    main()
