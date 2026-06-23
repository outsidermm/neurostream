"""Linear-probe evaluation entry point.

Run via Hydra:
    python -m scripts.linear_probe \\
        probe.pretrained_checkpoint=checkpoints/phase2_batch64_1.2m/milestone_step01200000.pt

To sweep across pretraining milestones, loop over them in a shell script;
each invocation produces an independent MLflow run that can be compared
in the UI.

The probe feeds corpus-harmonised BCI IV 2a windows to the frozen encoder
(see ``data/bci_iv_harmonised``). For the preprocessing/window ablation behind
this choice, see ``scripts/probe_ablation`` and the Phase 2 notes.
"""

import logging
from pathlib import Path

import hydra
import mlflow
import torch
from omegaconf import DictConfig, OmegaConf

from neurostream.data.bci_iv_harmonised import make_probe_adapter
from neurostream.training.linear_probe import (
    ProbeConfig,
    run_pretrained_vs_random,
    run_probe,
)
from neurostream.training.feature_extract import load_encoder_from_checkpoint

logger = logging.getLogger(__name__)


# Validated probe config (docs/phase-2-notes/probe-data-mismatch.md): the encoder
# was pretrained on corpus-harmonised EEG, so probe windows must be harmonised the
# same way (128 Hz + 0.5-45 Hz band-pass + common-average reference) or they are
# out-of-distribution and the probe gap collapses. A 2 s motor-imagery window
# (padded to 1000) beats both the 4 s and the full continuous window.
_load_subject = make_probe_adapter(harmonise=True, window="pad2s")


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
                load_subject=_load_subject,
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
                _load_subject,
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
