"""
Within-subject EEGNet training loop with 4-fold cross-validation on session T.

Design constraints (Phase 1):
- No frameworks (Lightning, Accelerate) — every step must be inspectable
- Single seed controls all stochasticity
- Hydra config drives all hyperparameters
- MLflow logs every run artefact needed to reproduce or compare results

Protocol (Lawhern 2018):
- Stratified 4-fold CV on session T per subject
- Per-fold pipeline fit on train portion only (no leakage)
- Subject score = mean of 4 best-val accuracies across folds
- Session E is not used (T→E cross-session is a different benchmark)
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
import numpy as np
import torch
import torch.nn as nn
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

from neurostream.data.loader import load_subject
from neurostream.models.eegnet import EEGNet
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


def _to_loader(
    X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool
) -> DataLoader[Any]:
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

    After each optimiser step, applies the model's max_norm constraints if
    defined (Keras-style; required for faithful EEGNet reproduction).
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
                apply = getattr(model, "apply_max_norm", None)
                if callable(apply):
                    apply()

            total_loss += loss.item() * len(y_batch)
            correct += (logits.argmax(dim=1) == y_batch).sum().item()
            total += len(y_batch)

    return EpochMetrics(
        loss=total_loss / total,
        accuracy=correct / total,
    )


def _train_fold(
    *,
    fold_idx: int,
    subject_id: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: DictConfig,
    device: torch.device,
    checkpoints_dir: Path,
) -> float:
    """Train one fold, return best val accuracy (at min val loss)."""

    train_loader = _to_loader(X_train, y_train, cfg.training.batch_size, shuffle=True)
    val_loader = _to_loader(X_val, y_val, cfg.training.batch_size, shuffle=False)

    model = EEGNet(
        n_classes=cfg.model.n_classes,
        n_channels=X_train.shape[1],
        n_samples=X_train.shape[2],
        fs=cfg.model.fs,
        f1=cfg.model.f1,
        d=cfg.model.d,
        dropout=cfg.model.dropout,
        kernel_length=cfg.model.kernel_length,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_val_acc = 0.0
    ckpt_path = checkpoints_dir / f"subject_{subject_id:02d}_fold{fold_idx}_best.pt"

    for epoch in range(1, cfg.training.epochs + 1):
        train_m = run_epoch(model, train_loader, criterion, optimizer, device)
        val_m = run_epoch(model, val_loader, criterion, None, device)

        if val_m.loss < best_val_loss:
            best_val_loss = val_m.loss
            best_val_acc = val_m.accuracy
            torch.save(model.state_dict(), ckpt_path)

        mlflow.log_metrics(
            {
                f"s{subject_id:02d}/fold{fold_idx}/train_loss": train_m.loss,
                f"s{subject_id:02d}/fold{fold_idx}/train_acc": train_m.accuracy,
                f"s{subject_id:02d}/fold{fold_idx}/val_loss": val_m.loss,
                f"s{subject_id:02d}/fold{fold_idx}/val_acc": val_m.accuracy,
            },
            step=epoch,
        )

    mlflow.log_metric(f"s{subject_id:02d}/fold{fold_idx}/best_val_acc", best_val_acc)
    mlflow.log_artifact(str(ckpt_path))
    return best_val_acc


def train_subject(
    subject_id: int,
    cfg: DictConfig,
    device: torch.device,
    pipelines_dir: Path,
    checkpoints_dir: Path,
) -> float:
    """Stratified 4-fold CV on session T. Returns mean best-val accuracy."""
    epochs_train, labels_train = load_subject(subject_id, "T")

    bandpass = BandpassParams(
        low_hz=cfg.preprocessing.bandpass.low_hz,
        high_hz=cfg.preprocessing.bandpass.high_hz,
        fs_hz=cfg.preprocessing.bandpass.fs_hz,
        order=cfg.preprocessing.bandpass.order,
    )

    skf = StratifiedKFold(
        n_splits=cfg.training.n_folds, shuffle=True, random_state=cfg.seed
    )
    fold_accs: list[float] = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(epochs_train, labels_train)
    ):
        # Per-fold pipeline fit — train statistics never see val
        pipeline = fit_pipeline(
            epochs_train[train_idx], config=PipelineConfig(bandpass=bandpass)
        )
        # Persist fold 0 only — others are recoverable from seed + fold index
        if fold_idx == 0:
            save_pipeline(pipeline, pipelines_dir / f"subject_{subject_id:02d}")

        X_train = pipeline.transform(epochs_train[train_idx])
        X_val = pipeline.transform(epochs_train[val_idx])
        y_train = labels_train[train_idx]
        y_val = labels_train[val_idx]

        acc = _train_fold(
            fold_idx=fold_idx,
            subject_id=subject_id,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            cfg=cfg,
            device=device,
            checkpoints_dir=checkpoints_dir,
        )
        fold_accs.append(acc)
        print(f"  Subject {subject_id:02d} fold {fold_idx}: best_val_acc={acc:.4f}")

    mean_acc = float(np.mean(fold_accs))
    mlflow.log_metric(f"s{subject_id:02d}/mean_fold_val_acc", mean_acc)
    return mean_acc


# ── Entry point ──────────────────────────────────────────────────────────────


def _get_git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _flatten_for_mlflow(d: dict[Any, Any]) -> dict[str, Any]:
    """MLflow rejects nested dicts — flatten to dotted string keys."""
    return {str(k): v for k, v in flatdict.FlatDict(d, delimiter=".").items()}


def _pick_device() -> torch.device:
    """CUDA if present, else Apple MPS, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@hydra.main(version_base=None, config_path="../../../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    set_deterministic_seed(cfg.seed)
    device = _pick_device()

    checkpoints_dir = Path(to_absolute_path(cfg.paths.checkpoints_dir))
    pipelines_dir = Path(to_absolute_path(cfg.paths.pipelines_dir))
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    pipelines_dir.mkdir(parents=True, exist_ok=True)

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
                pipelines_dir=pipelines_dir,
                checkpoints_dir=checkpoints_dir,
            )
            per_subject_acc[subject_id] = acc
            print(f"Subject {subject_id:02d}: mean_fold_val_acc={acc:.4f}")

        mean_acc = float(np.mean(list(per_subject_acc.values())))
        mlflow.log_metric("mean_val_acc", mean_acc)
        print(f"\nMean 4-fold val accuracy across subjects: {mean_acc:.4f}")


if __name__ == "__main__":
    main()
