"""Linear-probe ablation: isolate preprocessing vs windowing on the 1.2M MAE.

The original probe fed BCI IV 2a windows that were (1) not band-pass/CAR
harmonised like the pretraining corpus and (2) zero-padded from 256 -> 1000
samples. Both are distribution mismatches. This script runs three configs on a
single checkpoint so the two effects can be attributed:

    (a) raw + 2 s zero-padded   -> reproduces the ~38% baseline (harness check)
    (b) harmonised + 2 s padded -> isolates band-pass + CAR  [(a)->(b)]
    (c) harmonised + continuous -> isolates real 1000-sample window [(b)->(c)]

Each config logs a pretrained-vs-random pair to a fresh MLflow experiment so the
gap (the metric that validates pretraining) is directly comparable.

Run (no Hydra; logs straight to the sqlite store, no MLflow server needed):
    uv run python -m scripts.probe_ablation
"""

import argparse
import logging
from typing import Literal

import mlflow
import torch

from neurostream.data.bci_iv_harmonised import make_probe_adapter
from neurostream.training.linear_probe import ProbeConfig, run_pretrained_vs_random

logger = logging.getLogger("probe_ablation")

WindowMode = Literal["pad2s", "pad4s", "continuous"]

DEFAULT_CKPT = (
    r"C:\Users\xjmao\Desktop\neurostream\checkpoints"
    r"\phase2_batch64_1.2m\milestone_step01200000.pt"
)

# (run_name, harmonise, window, human description)
CONFIGS: list[tuple[str, bool, WindowMode, str]] = [
    ("a_raw_pad2s", False, "pad2s", "raw preprocessing + 2s zero-padded (baseline)"),
    ("b_harmonised_pad2s", True, "pad2s", "+bandpass+CAR, still 2s padded"),
    ("c_harmonised_continuous", True, "continuous", "+continuous 1000-sample window"),
    ("d_harmonised_pad4s", True, "pad4s", "+bandpass+CAR, 4s padded (wider MI window)"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument(
        "--experiment", default="neurostream-phase2-probe-continuous-window"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s  checkpoint=%s", device, args.checkpoint)

    cfg = ProbeConfig()  # defaults match configs/probe/default.yaml

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    summary = []
    for run_name, harmonise, window, desc in CONFIGS:
        logger.info("=== CONFIG %s: %s ===", run_name, desc)
        adapter = make_probe_adapter(harmonise=harmonise, window=window)
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(
                {
                    "config": run_name,
                    "description": desc,
                    "harmonise": harmonise,
                    "window": window,
                    "checkpoint": args.checkpoint,
                    "pool": cfg.pool,
                    "logreg_c": cfg.logreg_c,
                }
            )
            pre, rand = run_pretrained_vs_random(
                args.checkpoint, adapter, cfg, device=device
            )
            gap = pre.mean_accuracy - rand.mean_accuracy
            mlflow.log_metric("pretrained_mean_accuracy", pre.mean_accuracy)
            mlflow.log_metric("random_mean_accuracy", rand.mean_accuracy)
            mlflow.log_metric("gap", gap)
            for sid, acc in pre.per_subject_accuracy.items():
                mlflow.log_metric(f"pretrained/subject_{sid:02d}", acc)
        summary.append((run_name, pre.mean_accuracy, rand.mean_accuracy, gap))
        logger.info(
            "%s -> pretrained=%.4f random=%.4f gap=%+.2fpp",
            run_name, pre.mean_accuracy, rand.mean_accuracy, gap * 100,
        )

    # Plain-ASCII summary (avoid Windows GBK console crashes on glyphs).
    print("\n" + "=" * 68)
    print(f"{'config':28} {'pretrained':>11} {'random':>9} {'gap(pp)':>9}")
    print("-" * 68)
    for run_name, pre_acc, rand_acc, gap in summary:
        print(f"{run_name:28} {pre_acc:>11.4f} {rand_acc:>9.4f} {gap * 100:>+9.2f}")
    print("=" * 68)
    best = max(summary, key=lambda r: r[3])
    if best[3] >= 0.15:
        print(f"VALIDATED: '{best[0]}' gap {best[3] * 100:.2f}pp >= 15pp threshold.")
    else:
        print(
            f"BELOW THRESHOLD: best '{best[0]}' gap {best[3] * 100:.2f}pp < 15pp. "
            "Window length is not the sole bottleneck -- flag for fresh diagnostic."
        )


if __name__ == "__main__":
    main()
