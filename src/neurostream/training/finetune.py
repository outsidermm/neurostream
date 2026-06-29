"""End-to-end fine-tuning of a pretrained MAE on BCI Competition IV 2a.

Mirrors ``training/train.py`` (raw PyTorch — no Lightning) but for the
self-supervised path: load the pretrained encoder, attach a classification
head (:class:`MAEClassifier`), and fine-tune end-to-end per subject with
layer-wise LR decay, warmup→cosine scheduling, mixup and early stopping.

Protocol (Phase 2, Days 12-14): within-subject. Train on session **T**
(with an internal stratified train/val split for early stopping), evaluate
on session **E**. The session-E accuracy is the headline number, directly
comparable to the linear probe and the ≥71% Phase 2 target.

The core ``train_one_subject`` is framework-free and takes pre-split numpy
arrays, so it is unit-testable on synthetic data without real GDF files or an
MLflow server.
"""

import copy
import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from neurostream.models.mae_classifier import MAEClassifier
from neurostream.training.feature_extract import (
    load_encoder_from_checkpoint,
    make_random_init_encoder,
)
from neurostream.training.mixup import mixup_batch, mixup_loss
from neurostream.training.optim import param_groups_llrd
from neurostream.training.scheduler import apply_lr, warmup_cosine_lr

MetricLogger = Callable[[dict[str, float], int], None]


@dataclass
class FinetuneConfig:
    """Fine-tune recipe (Phase 2 spec, Days 12-14)."""

    base_lr: float = 1e-3
    llrd_decay: float = 0.70
    weight_decay: float = 0.05
    batch_size: int = 32
    epochs: int = 100
    warmup_epochs: int = 10
    patience: int = 15
    mixup_alpha: float = 0.2
    dropout: float = 0.5
    val_fraction: float = 0.2
    seed: int = 42
    n_classes: int = 4
    pool: str = "mean"
    head_hidden_dim: int = 0
    label_smoothing: float = 0.0
    freeze_encoder_epochs: int = 0


@dataclass
class SubjectFinetuneResult:
    """Per-subject fine-tune outcome (session-E accuracy is the headline)."""

    subject_id: int
    test_accuracy: float
    best_val_loss: float
    best_val_acc: float
    epochs_run: int


# ── Early stopping ───────────────────────────────────────────────────────────


class EarlyStopping:
    """Stop after ``patience`` epochs without validation-loss improvement."""

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best = math.inf
        self.num_bad = 0

    def update(self, val_loss: float) -> bool:
        """Record a validation loss. Returns True if it improved on the best."""
        if val_loss < self.best:
            self.best = val_loss
            self.num_bad = 0
            return True
        self.num_bad += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.num_bad > self.patience


# ── Model construction ───────────────────────────────────────────────────────


def build_classifier(
    checkpoint_path: str | None,
    *,
    n_classes: int,
    dropout: float,
    device: torch.device | str = "cpu",
    random_init: bool = False,
    seed: int = 0,
    pool: str = "mean",
    head_hidden_dim: int = 0,
) -> MAEClassifier:
    """Build an :class:`MAEClassifier` from a pretrained (or random) encoder.

    ``random_init=True`` builds the architecture with untrained weights — the
    control that fine-tuning must beat, mirroring the linear-probe control.
    Requires ``checkpoint_path`` either way (it carries the architecture).
    """
    if checkpoint_path is None:
        raise ValueError("checkpoint_path is required (it carries the architecture)")
    if random_init:
        encoder = make_random_init_encoder(checkpoint_path, seed=seed)
    else:
        # strict=False: the checkpoint holds the full MAE; we keep the decoder
        # weights but never use them (frozen below).
        encoder = load_encoder_from_checkpoint(
            checkpoint_path, map_location=device, strict=True
        )
    clf = MAEClassifier(encoder, n_classes=n_classes, dropout=dropout, pool=pool, head_hidden_dim=head_hidden_dim)
    freeze_unused_decoder(clf)
    return clf.to(device)


def freeze_unused_decoder(clf: MAEClassifier) -> None:
    """Freeze the MAE decoder + mask token — unused on the ``encode`` path."""
    for name, p in clf.named_parameters():
        if "decoder" in name or name.endswith("mask_token"):
            p.requires_grad = False


def stratified_train_val_split(
    x: np.ndarray,
    y: np.ndarray,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified split of session-T data into train / val for early stopping."""
    x_tr, x_val, y_tr, y_val = train_test_split(
        x, y, test_size=val_fraction, stratify=y, random_state=seed
    )
    return x_tr, y_tr, x_val, y_val


# ── Data plumbing ────────────────────────────────────────────────────────────


def _loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> DataLoader[Any]:
    ds = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        total_loss += criterion(logits, yb).item() * len(yb)
        correct += (logits.argmax(1) == yb).sum().item()
        total += len(yb)
    return total_loss / total, correct / total


# ── Core training ────────────────────────────────────────────────────────────


def train_one_subject(
    clf: MAEClassifier,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    cfg: FinetuneConfig,
    device: torch.device,
    *,
    subject_id: int = 0,
    log_metric: MetricLogger | None = None,
) -> SubjectFinetuneResult:
    """Fine-tune one subject end-to-end; evaluate on session E.

    Returns the best (min-val-loss) checkpoint's session-E accuracy. The model
    is left holding the best weights so callers can persist it.
    """
    gen = torch.Generator().manual_seed(cfg.seed + subject_id)
    mix_gen = torch.Generator().manual_seed(cfg.seed * 7919 + subject_id)

    train_loader = _loader(train_x, train_y, cfg.batch_size, True, gen)
    val_loader = _loader(val_x, val_y, cfg.batch_size, False)
    test_loader = _loader(test_x, test_y, cfg.batch_size, False)

    groups = param_groups_llrd(
        clf, base_lr=cfg.base_lr, decay=cfg.llrd_decay, weight_decay=cfg.weight_decay
    )
    optimizer = torch.optim.AdamW(groups, lr=cfg.base_lr, betas=(0.9, 0.95))
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    steps_per_epoch = max(1, len(train_loader))
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = min(cfg.warmup_epochs * steps_per_epoch, total_steps - 1)

    freeze_epochs = cfg.freeze_encoder_epochs
    if freeze_epochs > 0:
        for p in clf.encoder.parameters():
            p.requires_grad = False

    early = EarlyStopping(cfg.patience)
    best_state = copy.deepcopy(clf.state_dict())
    best_val_loss = math.inf
    best_val_acc = 0.0
    global_step = 0
    epochs_run = 0

    for epoch in range(1, cfg.epochs + 1):
        if freeze_epochs > 0 and epoch == freeze_epochs + 1:
            for p in clf.encoder.parameters():
                p.requires_grad = True
            early = EarlyStopping(cfg.patience)

        epochs_run = epoch
        clf.train()
        train_loss, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            apply_lr(
                optimizer,
                warmup_cosine_lr(global_step, cfg.base_lr, warmup_steps, total_steps),
            )
            if cfg.mixup_alpha > 0.0:
                xb, ya, yb2, lam = mixup_batch(
                    xb, yb, alpha=cfg.mixup_alpha, generator=mix_gen
                )
                logits = clf(xb)
                loss = mixup_loss(criterion, logits, ya, yb2, lam)
            else:
                logits = clf(xb)
                loss = criterion(logits, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            global_step += 1
            train_loss += loss.item() * len(yb)
            n += len(yb)

        val_loss, val_acc = _evaluate(clf, val_loader, criterion, device)
        if early.update(val_loss):
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_state = copy.deepcopy(clf.state_dict())

        if log_metric is not None:
            log_metric(
                {
                    f"s{subject_id:02d}/train_loss": train_loss / n,
                    f"s{subject_id:02d}/val_loss": val_loss,
                    f"s{subject_id:02d}/val_acc": val_acc,
                },
                epoch,
            )

        if early.should_stop:
            break

    # Restore the best checkpoint and report its session-E accuracy.
    clf.load_state_dict(best_state)
    _, test_acc = _evaluate(clf, test_loader, criterion, device)

    return SubjectFinetuneResult(
        subject_id=subject_id,
        test_accuracy=test_acc,
        best_val_loss=best_val_loss,
        best_val_acc=best_val_acc,
        epochs_run=epochs_run,
    )


__all__ = [
    "FinetuneConfig",
    "SubjectFinetuneResult",
    "EarlyStopping",
    "build_classifier",
    "freeze_unused_decoder",
    "stratified_train_val_split",
    "train_one_subject",
]
