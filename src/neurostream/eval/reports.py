"""Evaluation reporting helpers for linear-probe and fine-tuning workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


@dataclass
class SubjectResult:
    """Per-subject evaluation outcome."""

    subject_id: int
    accuracy: float
    n_train: int
    n_test: int
    confusion: np.ndarray = field(repr=False)


@dataclass
class ProbeReport:
    """Aggregated linear-probe evaluation outcome across all subjects."""

    pretrained_or_random: str    # "pretrained" or "random"
    pool_mode: str
    feature_dim: int
    checkpoint_path: str | None
    subjects: list[SubjectResult]

    @property
    def mean_accuracy(self) -> float:
        return float(np.mean([s.accuracy for s in self.subjects]))

    @property
    def std_accuracy(self) -> float:
        return float(np.std([s.accuracy for s in self.subjects]))

    @property
    def per_subject_accuracy(self) -> dict[int, float]:
        return {s.subject_id: s.accuracy for s in self.subjects}

    def summary_table(self) -> str:
        """Render a markdown table of per-subject accuracies."""
        lines = [
            "| Subject | Train N | Test N | Accuracy |",
            "|---------|---------|--------|----------|",
        ]
        for s in self.subjects:
            lines.append(
                f"| A{s.subject_id:02d}    | {s.n_train:7d} | {s.n_test:6d} | "
                f"{s.accuracy:6.2%} |"
            )
        lines.append(
            f"| **Mean** | — | — | **{self.mean_accuracy:6.2%} "
            f"± {self.std_accuracy:5.2%}** |"
        )
        return "\n".join(lines)

    def save(self, path: Path | str) -> None:
        """Persist the report as JSON (for MLflow artifact upload, etc.)."""
        import json

        out = {
            "pretrained_or_random": self.pretrained_or_random,
            "pool_mode": self.pool_mode,
            "feature_dim": self.feature_dim,
            "checkpoint_path": self.checkpoint_path,
            "mean_accuracy": self.mean_accuracy,
            "std_accuracy": self.std_accuracy,
            "subjects": [
                {
                    "subject_id": s.subject_id,
                    "accuracy": s.accuracy,
                    "n_train": s.n_train,
                    "n_test": s.n_test,
                    "confusion": s.confusion.tolist(),
                }
                for s in self.subjects
            ],
        }
        Path(path).write_text(json.dumps(out, indent=2))


def per_subject_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 4
) -> tuple[float, np.ndarray]:
    """Accuracy + confusion matrix for one subject's predictions."""
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )
    acc = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    return acc, cm


__all__ = [
    "ProbeReport",
    "SubjectResult",
    "per_subject_metrics",
]
