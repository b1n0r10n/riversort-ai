# Copyright (C) 2026 RiverSort AI contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import json
import csv
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from riversort.adapters import OpenCvAnnotator, OpenCvMediaAdapter, ResultExporter
from riversort.application import UploadedMedia
from riversort.application.analyze_image import AnalyzeImage
from riversort.domain import Detection, QualityFlagCode, SupportStatus
from riversort.infrastructure import load_project_config
from riversort.infrastructure.runtime import runtime_context
from riversort.infrastructure.workspace import AnalysisWorkspace


class FakeDetector:
    def __init__(self, detections):
        self.detections = detections

    def detect_image(self, pixels):
        return self.detections


class FailingDetector:
    def detect_image(self, pixels):
        raise RuntimeError("intentional inference failure")


class FailingExporter:
    def to_json(self, result):
        raise RuntimeError("intentional export failure")

    def to_csv(self, result):
        raise AssertionError("must not reach CSV after JSON failure")


def upload() -> UploadedMedia:
    stream = BytesIO()
    Image.new("RGB", (100, 80), (30, 80, 100)).save(stream, format="JPEG")
    return UploadedMedia("river.jpg", stream.getvalue(), "image/jpeg")


def use_case(tmp_path: Path, detections) -> AnalyzeImage:
    config = load_project_config()
    return AnalyzeImage(
        runtime=runtime_context(config),
        media=OpenCvMediaAdapter(config.media, config.tracker),
        detector=FakeDetector(detections),
        annotator=OpenCvAnnotator(),
        exporter=ResultExporter(),
        workspace_factory=lambda analysis_id: AnalysisWorkspace(analysis_id, tmp_path),
    )


def test_image_vertical_slice_returns_traceable_bundle_and_cleans_workspace(
    tmp_path: Path,
) -> None:
    detections = (
        Detection(0, "plastic_bottle", 0.9, (10.0, 10.0, 40.0, 50.0)),
        Detection(1, "plastic_bag", 0.8, (45.0, 10.0, 75.0, 40.0)),
    )
    bundle = use_case(tmp_path, detections).execute(
        upload(), {"location": "Titik demo", "notes": "fixture"}
    )
    assert bundle.annotated_mime == "image/jpeg"
    assert bundle.annotated_media.startswith(b"\xff\xd8")
    wire = json.loads(bundle.json_bytes)
    assert wire["unique_object_count"] == 2
    assert wire["model_version"] == "0.1.0-baseline.1"
    assert wire["coverage_p90"] > 0
    assert wire["metadata"]["location"] == "Titik demo"
    unsupported = next(
        item for item in wire["compositions"] if item["class_id"] == 4
    )
    assert unsupported["support_status"] == SupportStatus.NOT_SUPPORTED.value
    assert unsupported["object_count"] is None
    assert b"plastic_bottle" in bundle.csv_bytes
    rows = list(
        csv.DictReader(BytesIO(bundle.csv_bytes).read().decode("utf-8").splitlines())
    )
    assert len(rows) == 5
    assert rows[4]["support_status"] == "not_supported"
    assert rows[4]["object_count"] == ""
    assert rows[0]["original_width"] == "100"
    assert rows[0]["analysis_width"] == "100"
    assert rows[0]["resolution_normalized"] == "False"
    assert "/home/" not in bundle.json_bytes.decode("utf-8")
    assert list(tmp_path.iterdir()) == []


def test_no_detection_is_not_assessable_and_not_clean_claim(tmp_path: Path) -> None:
    bundle = use_case(tmp_path, ()).execute(upload())
    assert bundle.result.priority.status == "not_assessable"
    assert bundle.result.priority.score is None
    assert any(
        item.code is QualityFlagCode.NO_VALID_OBJECT
        for item in bundle.result.quality_flags
    )
    assert any(item.rule_id == "R_NO_ASSESSMENT" for item in bundle.result.recommendations)
    assert "jangan menyimpulkan sungai bersih" in bundle.json_bytes.decode("utf-8").lower()


def test_workspace_is_cleaned_when_inference_raises(tmp_path: Path) -> None:
    config = load_project_config()
    pipeline = AnalyzeImage(
        runtime=runtime_context(config),
        media=OpenCvMediaAdapter(config.media, config.tracker),
        detector=FailingDetector(),
        annotator=OpenCvAnnotator(),
        exporter=ResultExporter(),
        workspace_factory=lambda analysis_id: AnalysisWorkspace(analysis_id, tmp_path),
    )
    with pytest.raises(RuntimeError, match="intentional"):
        pipeline.execute(upload())
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("class_id", "class_name"),
    [
        (0, "plastic_bottle"),
        (1, "plastic_bag"),
        (2, "plastic_packaging_sachet"),
        (3, "other_plastic"),
    ],
)
def test_each_supported_class_reaches_result(
    tmp_path: Path, class_id: int, class_name: str
) -> None:
    detection = Detection(
        class_id, class_name, 0.9, (10.0, 10.0, 40.0, 50.0)
    )
    result = use_case(tmp_path, (detection,)).execute(upload()).result
    metric = next(item for item in result.compositions if item.class_id == class_id)
    assert metric.object_count == 1
    assert metric.object_count_share == 1.0


def test_workspace_is_cleaned_when_export_raises(tmp_path: Path) -> None:
    config = load_project_config()
    pipeline = AnalyzeImage(
        runtime=runtime_context(config),
        media=OpenCvMediaAdapter(config.media, config.tracker),
        detector=FakeDetector(()),
        annotator=OpenCvAnnotator(),
        exporter=FailingExporter(),
        workspace_factory=lambda analysis_id: AnalysisWorkspace(analysis_id, tmp_path),
    )
    with pytest.raises(RuntimeError, match="intentional export"):
        pipeline.execute(upload())
    assert list(tmp_path.iterdir()) == []
