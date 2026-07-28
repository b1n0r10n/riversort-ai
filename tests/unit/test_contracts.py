# Copyright (C) 2026 RiverSort AI contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path

import pytest

from riversort.domain import (
    AnalysisRequest,
    AnalysisResult,
    CompositionMetric,
    MediaType,
    ModelCapabilityProfile,
    PriorityAssessment,
    QualityFlag,
    QualityFlagCode,
    SupportStatus,
)


def capability() -> ModelCapabilityProfile:
    return ModelCapabilityProfile(
        profile_id="riversort_plastic4_mvp",
        profile_version="1.0.0",
        target_taxonomy_version="1.0.0",
        supported_classes={0: "plastic_bottle", 1: "plastic_bag", 2: "plastic_packaging_sachet", 3: "other_plastic"},
        unsupported_target_classes={4: "other_floating_debris"},
    )


def test_request_omits_local_temporary_path() -> None:
    request = AnalysisRequest("a-1", MediaType.IMAGE, "river.jpg", local_temporary_path=Path("/tmp/private.jpg"))
    from riversort.domain.contracts import to_wire

    assert to_wire(request) == {
        "analysis_id": "a-1",
        "media_type": "image",
        "original_filename": "river.jpg",
        "metadata": {},
    }


def test_analysis_result_serializes_stable_schema_and_unsupported_nulls() -> None:
    result = AnalysisResult(
        analysis_id="a-1",
        media_type=MediaType.IMAGE,
        capability_profile=capability(),
        dataset_snapshot_id=None,
        dataset_snapshot_sha256=None,
        model_version=None,
        model_sha256=None,
        detector_config_version="1.0.0",
        priority_config_version="1.1.0",
        recommendation_config_version="1.1.0",
        compositions=[CompositionMetric(4, "other_floating_debris", SupportStatus.NOT_SUPPORTED, None, None, None)],
        priority=PriorityAssessment("not_assessable", None, None, "supported_classes_only", "1.1.0"),
        recommendations=[],
        quality_flags=[QualityFlag(QualityFlagCode.UNSUPPORTED_TARGET_CLASS, "not supported")],
        limitations=[],
    )
    wire = result.to_dict()
    assert wire["schema_version"] == "1.0.0"
    assert wire["compositions"][0]["support_status"] == "not_supported"
    assert wire["compositions"][0]["object_count"] is None


def test_arbitrary_path_cannot_leak_through_serialization() -> None:
    from riversort.domain.contracts import to_wire

    with pytest.raises(ValueError, match="must not be serialized"):
        to_wire(Path("/private/media.jpg"))

