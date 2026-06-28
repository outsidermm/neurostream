"""Fine-tune the pretrained MAE on BCI Competition IV 2a (Phase 2, Days 12-14).

Run via Hydra:
    python -m scripts.finetune \\
        finetune.pretrained_checkpoint=checkpoints/phase2_batch64_1.2m/milestone_step01200000.pt

Per subject, within-subject: fine-tune end-to-end on session T (internal
stratified train/val split for early stopping), evaluate on session E. The
session-E mean accuracy across the 9 subjects is the Phase 2 headline number
(target ≥71%, beating both the EEGNet baseline and the random-init control).

Data is harmonised to the validated probe distribution (128 Hz + 0.5-45 Hz
band-pass + CAR + per-window z-score, 2 s window padded to 1000) so the encoder
sees in-distribution inputs — see ``data/bci_iv_harmonised`` and the Phase 2
notes for why this parity is load-bearing.
"""

import json
import logging
from pathlib import Path

import hydra
import mlflow
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch

from neurostream.data.bci_iv_harmonised import make_probe_adapter
from neurostream.training.finetune import (
    FinetuneConfig,
    SubjectFinetuneResult,
    build_classifier,
    stratified_train_val_split,
    train_one_subject,
)
from neurostream.training.train import set_deterministic_seed
from neurostream.utils.git import git_sha

logger = logging.getLogger(__name__)


def _finetune_variant(
    *,
    label: str,
    random_init: bool,
    cfg: DictConfig,
    ft_cfg: FinetuneConfig,
    load_subject,
    device: torch.device,
) -> list[SubjectFinetuneResult]:
    """Fine-tune every subject for one variant ("pretrained" or "random")."""
    results: list[SubjectFinetuneResult] = []
    for subject_id in cfg.finetune.subjects:
        train_x, train_y = load_subject(subject_id, "T")
        test_x, test_y = load_subject(subject_id, "E")
        x_tr, y_tr, x_val, y_val = stratified_train_val_split(
            train_x, train_y, ft_cfg.val_fraction, ft_cfg.seed
        )

        clf = build_classifier(
            cfg.finetune.pretrained_checkpoint,
            n_classes=ft_cfg.n_classes,
            dropout=ft_cfg.dropout,
            device=device,
            random_init=random_init,
            seed=ft_cfg.seed,
        )

        def _log(metrics: dict[str, float], step: int, _l: str = label) -> None:
            mlflow.log_metrics({f"{_l}/{k}": v for k, v in metrics.items()}, step=step)

        result = train_one_subject(
            clf,
            x_tr,
            y_tr,
            x_val,
            y_val,
            test_x,
            test_y,
            ft_cfg,
            device,
            subject_id=subject_id,
            log_metric=_log,
        )
        results.append(result)
        logger.info(
            "[%s] A%02d  test_acc=%.4f  best_val_acc=%.4f  epochs=%d",
            label,
            subject_id,
            result.test_accuracy,
            result.best_val_acc,
            result.epochs_run,
        )
    return results


def _summarise(label: str, results: list[SubjectFinetuneResult]) -> float:
    accs = [r.test_accuracy for r in results]
    mean, std = float(np.mean(accs)), float(np.std(accs))
    mlflow.log_metric(f"{label}/mean_test_acc", mean)
    mlflow.log_metric(f"{label}/std_test_acc", std)
    for r in results:
        mlflow.log_metric(
            f"{label}/subject_{r.subject_id:02d}/test_acc", r.test_accuracy
        )
    logger.info(
        "\n[%s] session-E accuracy: mean=%.4f ± %.4f over %d subjects",
        label,
        mean,
        std,
        len(results),
    )
    return mean


@hydra.main(version_base=None, config_path="../configs", config_name="finetune")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))

    set_deterministic_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    ft_cfg = FinetuneConfig(
        base_lr=cfg.train.base_lr,
        llrd_decay=cfg.train.llrd_decay,
        weight_decay=cfg.train.weight_decay,
        batch_size=cfg.train.batch_size,
        epochs=cfg.train.epochs,
        warmup_epochs=cfg.train.warmup_epochs,
        patience=cfg.train.patience,
        mixup_alpha=cfg.train.mixup_alpha,
        dropout=cfg.train.dropout,
        val_fraction=cfg.train.val_fraction,
        seed=cfg.seed,
        n_classes=cfg.finetune.n_classes,
    )

    load_subject = make_probe_adapter(
        harmonise=cfg.finetune.harmonise, window=cfg.finetune.window
    )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(cfg.log.tracking_uri)
    mlflow.set_experiment(cfg.log.experiment)
    with mlflow.start_run(run_name=cfg.log.get("run_name")):
        mlflow.log_params(
            {
                "pretrained_checkpoint": cfg.finetune.pretrained_checkpoint,
                "window": cfg.finetune.window,
                "harmonise": cfg.finetune.harmonise,
                "base_lr": ft_cfg.base_lr,
                "llrd_decay": ft_cfg.llrd_decay,
                "mixup_alpha": ft_cfg.mixup_alpha,
                "subjects": str(list(cfg.finetune.subjects)),
            }
        )
        mlflow.set_tag("git_sha", git_sha())
        mlflow.set_tag("device", str(device))

        variants = [("pretrained", False)]
        if cfg.finetune.run_random_control:
            variants.append(("random", True))

        means: dict[str, float] = {}
        all_results: dict[str, list[dict]] = {}
        for label, random_init in variants:
            results = _finetune_variant(
                label=label,
                random_init=random_init,
                cfg=cfg,
                ft_cfg=ft_cfg,
                load_subject=load_subject,
                device=device,
            )
            means[label] = _summarise(label, results)
            all_results[label] = [
                {
                    "subject_id": r.subject_id,
                    "test_accuracy": r.test_accuracy,
                    "best_val_acc": r.best_val_acc,
                    "best_val_loss": r.best_val_loss,
                    "epochs_run": r.epochs_run,
                }
                for r in results
            ]

        json_path = output_dir / "finetune_results.json"
        json_path.write_text(json.dumps(all_results, indent=2))
        mlflow.log_artifact(str(json_path))

        if "random" in means:
            gap = means["pretrained"] - means["random"]
            mlflow.log_metric("pretrained_minus_random_gap", gap)
            logger.info("Fine-tune gain over random init: %+.2fpp", gap * 100)
        if means["pretrained"] >= 0.71:
            logger.info("✓ Mean session-E accuracy meets the ≥71%% Phase 2 target.")
        else:
            logger.warning(
                "Mean session-E accuracy %.2f%% is below the 71%% target.",
                means["pretrained"] * 100,
            )


if __name__ == "__main__":
    main()
