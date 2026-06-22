"""Linear-probe evaluation of a frozen MAE encoder on BCI Competition IV 2a.

Protocol (per the Phase 2 doc, Days 10-11):

  For each subject 1..9:
      1. Load Phase 1 train/eval splits unchanged.
      2. Extract features from the frozen encoder.
      3. Fit a logistic regression on the train features.
      4. Evaluate accuracy on the eval features.
  Report per-subject and mean accuracy.

Run with ``--pretrained-checkpoint`` for the actual MAE, or with
``--random-init`` for the **critical control**: random-init linear probe.
Pretraining is only validated as useful if it exceeds the random-init
baseline by ≥15pp (per the Phase 2 spec).

This module is BOTH:

  * importable (the ``run_probe`` function is called by the Days 7-9
    training loop every K steps to monitor downstream transfer during
    pretraining), and
  * runnable as a standalone CLI for the formal v0.2.0 evaluation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from neurostream.eval.reports import ProbeReport, SubjectResult, per_subject_metrics
from neurostream.models.mae import EEGMaskedAutoencoder
from neurostream.training.feature_extract import (
    PoolMode,
    extract_features,
    load_encoder_from_checkpoint,
    make_random_init_encoder,
)

logger = logging.getLogger(__name__)


class _Phase1DataLoader(Protocol):
    """Structural type for Phase 1's BCI IV 2a data loader.

    Phase 1 must provide a callable returning ``(epochs, labels)`` for a given
    subject and session. The exact module location may vary; the probe code
    accepts any callable matching this protocol.
    """

    def __call__(
        self, subject_id: int, session: Literal["T", "E"]
    ) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass
class ProbeConfig:
    """All knobs for one linear-probe evaluation pass."""

    pool: PoolMode = "mean"
    batch_size: int = 64
    standardize: bool = True
    logreg_c: float = 1.0
    logreg_max_iter: int = 5000
    seed: int = 0
    subjects: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9)


def run_probe(
    encoder: EEGMaskedAutoencoder,
    load_subject: _Phase1DataLoader,
    cfg: ProbeConfig,
    *,
    device: str | torch.device = "cpu",
    label: str = "pretrained",
    checkpoint_path: str | None = None,
) -> ProbeReport:
    """Run the per-subject linear-probe protocol against a (frozen) encoder.

    Args:
        encoder: A pretrained or random-init MAE; will be set to eval mode.
        load_subject: Phase 1 data loader returning ``(epochs, labels)`` for
            a given ``(subject_id, session)``.
        cfg: Probe hyperparameters.
        device: Where to run the encoder forward passes.
        label: Identifier for the report — typically ``"pretrained"`` or
            ``"random"``.
        checkpoint_path: Optional path string recorded in the report metadata.

    Returns:
        A populated :class:`ProbeReport`.
    """
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False

    subject_results: list[SubjectResult] = []
    feature_dim: int | None = None

    for subject_id in cfg.subjects:
        train_x, train_y = load_subject(subject_id, "T")
        test_x, test_y = load_subject(subject_id, "E")

        train_feat = extract_features(
            encoder,
            torch.as_tensor(train_x, dtype=torch.float32),
            pool=cfg.pool,
            batch_size=cfg.batch_size,
            device=device,
        )
        test_feat = extract_features(
            encoder,
            torch.as_tensor(test_x, dtype=torch.float32),
            pool=cfg.pool,
            batch_size=cfg.batch_size,
            device=device,
        )

        feature_dim = train_feat.shape[1]

        if cfg.standardize:
            scaler = StandardScaler().fit(train_feat)
            train_feat = scaler.transform(train_feat)
            test_feat = scaler.transform(test_feat)

        clf = LogisticRegression(
            C=cfg.logreg_c,
            max_iter=cfg.logreg_max_iter,
            random_state=cfg.seed,
        )
        clf.fit(train_feat, train_y)

        preds = clf.predict(test_feat)
        acc, cm = per_subject_metrics(test_y, preds, n_classes=4)
        subject_results.append(
            SubjectResult(
                subject_id=subject_id,
                accuracy=acc,
                n_train=len(train_y),
                n_test=len(test_y),
                confusion=cm,
            )
        )
        logger.info(
            "Subject A%02d  acc=%.4f  (n_train=%d, n_test=%d)",
            subject_id, acc, len(train_y), len(test_y),
        )

    assert feature_dim is not None  # set in loop; subjects is non-empty
    return ProbeReport(
        pretrained_or_random=label,
        pool_mode=cfg.pool,
        feature_dim=feature_dim,
        checkpoint_path=checkpoint_path,
        subjects=subject_results,
    )


def run_pretrained_vs_random(
    pretrained_ckpt_path: str,
    load_subject: _Phase1DataLoader,
    cfg: ProbeConfig | None = None,
    *,
    device: str | torch.device = "cpu",
) -> tuple[ProbeReport, ProbeReport]:
    """Evaluate the pretrained encoder AND its random-init control.

    Returns ``(pretrained_report, random_report)``. The pretrained mean
    accuracy must exceed the random-init mean by ≥15pp for the pretraining
    to be considered useful (per the Phase 2 spec).
    """
    cfg = cfg or ProbeConfig()

    logger.info("Linear probe: PRETRAINED  (%s)", pretrained_ckpt_path)
    pretrained = load_encoder_from_checkpoint(pretrained_ckpt_path, map_location=device)
    pretrained_report = run_probe(
        pretrained,
        load_subject,
        cfg,
        device=device,
        label="pretrained",
        checkpoint_path=pretrained_ckpt_path,
    )

    logger.info("Linear probe: RANDOM-INIT (control)")
    random_encoder = make_random_init_encoder(pretrained_ckpt_path, seed=cfg.seed)
    random_report = run_probe(
        random_encoder,
        load_subject,
        cfg,
        device=device,
        label="random",
        checkpoint_path=None,
    )

    diff = pretrained_report.mean_accuracy - random_report.mean_accuracy
    logger.info(
        "Mean acc: pretrained=%.4f random=%.4f (Δ=%+.4f)",
        pretrained_report.mean_accuracy, random_report.mean_accuracy, diff,
    )
    if diff < 0.15:
        logger.warning(
            "Pretrained-vs-random gap is %.2fpp, below the 15pp threshold "
            "the Phase 2 spec requires for pretraining to be considered useful.",
            diff * 100,
        )

    return pretrained_report, random_report


__all__ = [
    "ProbeConfig",
    "run_probe",
    "run_pretrained_vs_random",
]
