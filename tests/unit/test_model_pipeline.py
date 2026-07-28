# Copyright (C) 2026 RiverSort AI contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

from collections import Counter
from pathlib import Path

import pytest

from riversort.model_pipeline import (
    GroundTruth,
    Prediction,
    _average_precision,
    _class_counts_at_threshold,
    _confusion_matrix,
    _iou,
    _metrics_from_counts,
    _percentile,
    load_contract,
)


def test_baseline_contract_is_locked_to_snapshot_and_requested_hyperparameters() -> None:
    contract = load_contract(Path("configs/training_yolo11n_baseline.yaml"))
    assert contract["snapshot_id"] == "riversort_fpwr4-20260722-r2"
    assert contract["model"] == "yolo11n.pt"
    assert contract["epochs"] == 100
    assert contract["batch"] == 8
    assert contract["seed"] == 42
    assert contract["cache"] is False
    assert contract["evaluation"]["confidence_grid"] == pytest.approx([index / 100 for index in range(5, 51, 5)])


def test_iou_and_fixed_threshold_matching_are_one_to_one() -> None:
    assert _iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
    truths = {"image.jpg": [GroundTruth(0, (0, 0, 10, 10))]}
    predictions = {
        "image.jpg": [
            Prediction(0, 0.9, (0, 0, 10, 10)),
            Prediction(0, 0.8, (0, 0, 10, 10)),
            Prediction(1, 0.1, (0, 0, 10, 10)),
        ]
    }
    counts = _class_counts_at_threshold(predictions, truths, 0.5, 0.5)
    assert counts[0] == Counter(tp=1, fp=1, fn=0)
    assert counts[1] == Counter(tp=0, fp=0, fn=0)
    metrics = _metrics_from_counts(counts)
    assert metrics["per_class"]["0"]["precision"] == pytest.approx(0.5)
    assert metrics["per_class"]["0"]["recall"] == 1.0


def test_ap50_and_confusion_matrix_have_explicit_background() -> None:
    truths = {
        "a.jpg": [GroundTruth(0, (0, 0, 10, 10))],
        "b.jpg": [GroundTruth(0, (0, 0, 10, 10))],
    }
    predictions = {
        "a.jpg": [Prediction(0, 0.9, (0, 0, 10, 10))],
        "b.jpg": [Prediction(1, 0.8, (0, 0, 10, 10))],
    }
    assert _average_precision(0, predictions, truths, 0.5) == pytest.approx(0.5)
    matrix = _confusion_matrix(predictions, truths, 0.5, 0.5)
    assert len(matrix) == 5
    assert matrix[0][0] == 1
    assert matrix[0][1] == 1


def test_type7_percentile_for_latency() -> None:
    assert _percentile([0, 1, 2, 3, 4], 0.9) == pytest.approx(3.6)
