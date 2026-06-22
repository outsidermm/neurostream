"""Linear-probe evaluation entry point.

Run via Hydra:
    python -m scripts.linear_probe \\
        probe.pretrained_checkpoint=checkpoints/phase2_batch64_1.2m/milestone_step01200000.pt

To sweep across pretraining milestones, loop over them in a shell script;
each invocation produces an independent MLflow run that can be compared
in the UI.

Phase 1 integration:
    The probe needs a function with signature
        load_subject(subject_id: int, session: Literal["T", "E"])
            -> (epochs: np.ndarray, labels: np.ndarray)
    Phase 1 should expose this from ``neurostream.data.bci_iv_2a``. If your
    module path differs, update the import below.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import hydra
import mlflow
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from neurostream.training.linear_probe import (
    ProbeConfig,
    run_pretrained_vs_random,
    run_probe,
)
from neurostream.training.feature_extract import load_encoder_from_checkpoint

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Phase 1 data loader adapter
# ---------------------------------------------------------------------
def _phase1_load_subject(
    subject_id: int, session: Literal["T", "E"]
) -> tuple[np.ndarray, np.ndarray]:
    """Adapter that calls Phase 1's BCI IV 2a loader and returns preprocessed data.

    This wraps Phase 1's ``load_subject`` so the output matches what the
    encoder expects: ``(n_trials, 22, 1000)`` float32 epochs and integer labels.

    Phase 1's loader is expected to live in ``neurostream.data.bci_iv_2a`` with
    a ``load_subject(subject_id, session) -> (epochs, labels)`` signature
    returning ``(n_trials, 22, 750)`` at 250 Hz across 3 seconds.

    The MAE was pretrained on 1000-sample windows. We pad/extend the 750-sample
    BCI IV 2a windows to 1000 samples by sampling a longer epoch window from
    the same trial. The simplest correct approach: re-extract a 4-second window
    instead of 3 seconds via the Phase 1 loader if it supports this; otherwise
    zero-pad symmetrically. We assume Phase 1's loader accepts an optional
    ``window_samples`` argument; if not, this adapter needs updating.
    """
    try:
        from neurostream.data.bci_iv_loader import load_subject as phase1_loader
    except ImportError as e:
        raise ImportError(
            "Could not import Phase 1's BCI IV 2a loader from "
            "neurostream.data.bci_iv_loader. Update the import path in "
            "scripts/linear_probe.py to match your Phase 1 module layout."
        ) from e

    epochs, labels = phase1_loader(subject_id, session)

    # Ensure shape (n_trials, 22, 1000) — pad if necessary.
    if epochs.shape[-1] < 1000:
        pad = 1000 - epochs.shape[-1]
        before = pad // 2
        after = pad - before
        epochs = np.pad(
            epochs.astype(np.float32),
            ((0, 0), (0, 0), (before, after)),
            mode="constant",
            constant_values=0.0,
        )
    elif epochs.shape[-1] > 1000:
        # Take centred 1000-sample slice.
        start = (epochs.shape[-1] - 1000) // 2
        epochs = epochs[..., start : start + 1000]

    # Per-window z-score per channel (mirrors the pretraining normalisation).
    mean = epochs.mean(axis=-1, keepdims=True)
    std = epochs.std(axis=-1, keepdims=True) + 1e-6
    epochs = ((epochs - mean) / std).astype(np.float32)

    labels = labels.astype(np.int64)
    return epochs, labels


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
@hydra.main(version_base=None, config_path="../configs", config_name="linear_probe")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    probe_cfg = ProbeConfig(
        pool=cfg.probe.pool,
        batch_size=cfg.probe.batch_size,
        standardize=cfg.probe.standardize,
        logreg_c=cfg.probe.logreg_c,
        logreg_max_iter=cfg.probe.logreg_max_iter,
        seed=cfg.probe.seed,
        subjects=tuple(cfg.probe.subjects),
    )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- MLflow setup -------------------------------------------------
    mlflow.set_tracking_uri(cfg.log.tracking_uri)
    mlflow.set_experiment(cfg.log.experiment)
    with mlflow.start_run(run_name=cfg.log.get("run_name")):
        mlflow.log_params({
            "pretrained_checkpoint": cfg.probe.pretrained_checkpoint,
            "pool": cfg.probe.pool,
            "standardize": cfg.probe.standardize,
            "logreg_c": cfg.probe.logreg_c,
            "subjects": str(list(cfg.probe.subjects)),
        })

        # --- Run probe(s) -------------------------------------------
        if cfg.probe.run_random_control:
            pretrained_report, random_report = run_pretrained_vs_random(
                pretrained_ckpt_path=cfg.probe.pretrained_checkpoint,
                load_subject=_phase1_load_subject,
                cfg=probe_cfg,
                device=device,
            )
            reports = [
                ("pretrained", pretrained_report),
                ("random", random_report),
            ]
        else:
            encoder = load_encoder_from_checkpoint(
                cfg.probe.pretrained_checkpoint, map_location=device
            )
            pretrained_report = run_probe(
                encoder,
                _phase1_load_subject,
                probe_cfg,
                device=device,
                label="pretrained",
                checkpoint_path=cfg.probe.pretrained_checkpoint,
            )
            reports = [("pretrained", pretrained_report)]

        # --- Log + save reports --------------------------------------
        for label, report in reports:
            logger.info("\n%s probe results:\n%s", label.upper(), report.summary_table())

            # MLflow metrics: mean accuracy + per-subject.
            mlflow.log_metric(f"{label}/mean_accuracy", report.mean_accuracy)
            mlflow.log_metric(f"{label}/std_accuracy", report.std_accuracy)
            for sid, acc in report.per_subject_accuracy.items():
                mlflow.log_metric(f"{label}/subject_{sid:02d}/accuracy", acc)

            # Persist the full report as JSON.
            json_path = output_dir / f"{label}_report.json"
            report.save(json_path)
            mlflow.log_artifact(str(json_path))
            logger.info("Saved %s report to %s", label, json_path)

        # --- Gap check ---------------------------------------------
        if cfg.probe.run_random_control:
            gap = pretrained_report.mean_accuracy - random_report.mean_accuracy
            mlflow.log_metric("pretrained_minus_random_gap", gap)
            logger.info("\nPretrained-vs-random gap: %.2fpp", gap * 100)
            if gap >= 0.15:
                logger.info(
                    "✓ Gap meets the 15pp threshold — pretraining is validated."
                )
            else:
                logger.warning(
                    "✗ Gap is below 15pp — pretraining may not be doing useful work. "
                    "Investigate before proceeding to fine-tuning."
                )


if __name__ == "__main__":
    main()
