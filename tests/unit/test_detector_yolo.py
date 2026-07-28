# Copyright (C) 2026 RiverSort AI contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from riversort.adapters.detector_yolo import YoloDetectorAdapter, sha256_file
from riversort.application import AnalysisError
from riversort.domain import AnalysisErrorCode
from riversort.infrastructure.config import capability_profile, load_project_config


class FakeModel:
    names = {
        0: "plastic_bottle",
        1: "plastic_bag",
        2: "plastic_packaging_sachet",
        3: "other_plastic",
    }

    def __init__(self):
        self.last_kwargs = None

    def predict(self, **kwargs):
        self.last_kwargs = kwargs
        boxes = SimpleNamespace(
            xyxy=torch.tensor([[-5.0, 2.0, 12.0, 20.0], [4.0, 4.0, 4.0, 8.0]]),
            conf=torch.tensor([0.9, 0.8]),
            cls=torch.tensor([0.0, 1.0]),
        )
        return [SimpleNamespace(boxes=boxes)]


def write_package(weight: Path, config) -> None:
    (weight.parent / config.package_manifest_filename).write_text(
        json.dumps(
            {
                "model_version": config.model_version,
                "snapshot_id": config.dataset_snapshot_id,
                "snapshot_tree_sha256_v1": config.dataset_snapshot_sha256,
                "compatibility": {
                    "compatible": True,
                    "names": config.names,
                    "weight_sha256": sha256_file(weight),
                },
            }
        ),
        encoding="utf-8",
    )


def test_detector_verifies_artifact_and_canonicalizes_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_project_config()
    weight = tmp_path / "best.pt"
    weight.write_bytes(b"fake-weight")
    monkeypatch.setenv("RIVERSORT_MODEL_PATH", str(weight))
    detector_config = config.detector.model_copy(
        update={"model_sha256": sha256_file(weight)}
    )
    write_package(weight, detector_config)
    model = FakeModel()
    detector = YoloDetectorAdapter(
        detector_config,
        capability_profile(config),
        model_factory=lambda _: model,
    )
    pixels = torch.zeros((24, 16, 3), dtype=torch.uint8).numpy()
    detections = detector.detect_image(pixels)
    assert len(detections) == 1
    assert detections[0].class_name == "plastic_bottle"
    assert detections[0].bbox_xyxy == (0.0, 2.0, 12.0, 20.0)
    assert model.last_kwargs["imgsz"] == 960


def test_detector_rejects_checksum_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_project_config()
    weight = tmp_path / "best.pt"
    weight.write_bytes(b"wrong")
    monkeypatch.setenv("RIVERSORT_MODEL_PATH", str(weight))
    with pytest.raises(AnalysisError) as error:
        YoloDetectorAdapter(
            config.detector,
            capability_profile(config),
            model_factory=lambda _: FakeModel(),
        )
    assert error.value.code is AnalysisErrorCode.MODEL_CHECKSUM_MISMATCH


def test_detector_rejects_snapshot_manifest_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_project_config()
    weight = tmp_path / "best.pt"
    weight.write_bytes(b"fake-weight")
    detector_config = config.detector.model_copy(
        update={"model_sha256": sha256_file(weight)}
    )
    write_package(weight, detector_config)
    manifest = weight.parent / detector_config.package_manifest_filename
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["snapshot_id"] = "wrong-snapshot"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("RIVERSORT_MODEL_PATH", str(weight))

    with pytest.raises(AnalysisError) as error:
        YoloDetectorAdapter(
            detector_config,
            capability_profile(config),
            model_factory=lambda _: FakeModel(),
        )
    assert error.value.code is AnalysisErrorCode.MODEL_PACKAGE_MISMATCH
