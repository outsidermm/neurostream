"""Tests for the ProbeReport dataclass and metrics helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from neurostream.eval.reports import ProbeReport, SubjectResult, per_subject_metrics


def _fake_subject(subject_id: int, accuracy: float, n: int = 100) -> SubjectResult:
    cm = np.zeros((4, 4), dtype=np.int64)
    correct = int(accuracy * n)
    np.fill_diagonal(cm[:correct % 4 + 1], correct // 4)
    return SubjectResult(
        subject_id=subject_id,
        accuracy=accuracy,
        n_train=n,
        n_test=n,
        confusion=cm,
    )


def test_mean_and_std_accuracy() -> None:
    subjects = [
        _fake_subject(1, 0.60),
        _fake_subject(2, 0.70),
        _fake_subject(3, 0.65),
    ]
    report = ProbeReport(
        pretrained_or_random="pretrained",
        pool_mode="mean",
        feature_dim=256,
        checkpoint_path="test.pt",
        subjects=subjects,
    )
    assert report.mean_accuracy == pytest.approx(0.65, abs=1e-6)
    assert report.std_accuracy == pytest.approx(np.std([0.60, 0.70, 0.65]), abs=1e-6)


def test_per_subject_accuracy_mapping() -> None:
    subjects = [_fake_subject(i, 0.5 + i * 0.05) for i in range(1, 4)]
    report = ProbeReport(
        pretrained_or_random="pretrained",
        pool_mode="mean",
        feature_dim=256,
        checkpoint_path=None,
        subjects=subjects,
    )
    mapping = report.per_subject_accuracy
    assert mapping == {1: 0.55, 2: 0.60, 3: 0.65}


def test_summary_table_renders_markdown() -> None:
    subjects = [_fake_subject(1, 0.68)]
    report = ProbeReport("pretrained", "mean", 256, None, subjects)
    table = report.summary_table()
    assert "| Subject |" in table
    assert "A01" in table
    assert "68.00%" in table
    assert "Mean" in table


def test_save_writes_valid_json(tmp_path: Path) -> None:
    subjects = [_fake_subject(1, 0.60), _fake_subject(2, 0.70)]
    report = ProbeReport(
        pretrained_or_random="pretrained",
        pool_mode="cls",
        feature_dim=256,
        checkpoint_path="/path/to/ckpt.pt",
        subjects=subjects,
    )
    out_path = tmp_path / "report.json"
    report.save(out_path)

    loaded = json.loads(out_path.read_text())
    assert loaded["pretrained_or_random"] == "pretrained"
    assert loaded["pool_mode"] == "cls"
    assert loaded["mean_accuracy"] == pytest.approx(0.65)
    assert len(loaded["subjects"]) == 2
    assert loaded["subjects"][0]["subject_id"] == 1


def test_per_subject_metrics_accuracy() -> None:
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 2, 0, 1, 0, 3])  # 6/8 correct
    acc, cm = per_subject_metrics(y_true, y_pred, n_classes=4)
    assert acc == pytest.approx(6 / 8)
    assert cm.shape == (4, 4)
    assert cm.sum() == 8


def test_per_subject_metrics_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        per_subject_metrics(np.array([1, 2]), np.array([1, 2, 3]))
