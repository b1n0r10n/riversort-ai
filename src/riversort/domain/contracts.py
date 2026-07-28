# Copyright (C) 2026 RiverSort AI contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Stable schema 1.0.0 contracts; only the Python standard library is used."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "1.0.0"


class StableEnum(str, Enum):
    """Enum whose wire representation is its stable string value."""


class MediaType(StableEnum):
    IMAGE = "image"
    VIDEO = "video"


class SupportStatus(StableEnum):
    SUPPORTED = "supported"
    NOT_SUPPORTED = "not_supported"


class QualityFlagCode(StableEnum):
    NO_VALID_OBJECT = "NO_VALID_OBJECT"
    UNSUPPORTED_TARGET_CLASS = "UNSUPPORTED_TARGET_CLASS"
    CONFIG_MISMATCH = "CONFIG_MISMATCH"
    LOW_ANALYSIS_QUALITY = "LOW_ANALYSIS_QUALITY"
    CLASSIFICATION_UNCERTAIN = "CLASSIFICATION_UNCERTAIN"
    TRACKING_UNSTABLE = "TRACKING_UNSTABLE"


class AnalysisErrorCode(StableEnum):
    INVALID_MEDIA_TYPE = "INVALID_MEDIA_TYPE"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    IMAGE_TOO_LARGE = "IMAGE_TOO_LARGE"
    DECODE_FAILED = "DECODE_FAILED"
    UNSUPPORTED_CODEC = "UNSUPPORTED_CODEC"
    VIDEO_TOO_LONG = "VIDEO_TOO_LONG"
    MODEL_NOT_READY = "MODEL_NOT_READY"
    MODEL_CHECKSUM_MISMATCH = "MODEL_CHECKSUM_MISMATCH"
    MODEL_CLASS_MAP_MISMATCH = "MODEL_CLASS_MAP_MISMATCH"
    MODEL_PACKAGE_MISMATCH = "MODEL_PACKAGE_MISMATCH"
    INFERENCE_FAILED = "INFERENCE_FAILED"
    TRACKING_FAILED = "TRACKING_FAILED"
    ENCODER_UNAVAILABLE = "ENCODER_UNAVAILABLE"
    EXPORT_FAILED = "EXPORT_FAILED"


@dataclass(frozen=True)
class MediaInfo:
    display_filename: str
    media_type: MediaType
    mime_type: str
    byte_size: int
    width: int
    height: int
    original_width: int | None = None
    original_height: int | None = None
    resolution_normalized: bool = False
    duration_seconds: float | None = None
    source_fps: float | None = None
    frame_count: int | None = None
    codec: str | None = None
    analysis_fps: float | None = None


@dataclass(frozen=True)
class AnalysisRequest:
    analysis_id: str
    media_type: MediaType
    original_filename: str
    metadata: Mapping[str, str] = field(default_factory=dict)
    local_temporary_path: Path | None = field(default=None, metadata={"serialize": False})


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    frame_index: int | None = None
    timestamp_ms: int | None = None


@dataclass(frozen=True)
class TrackSummary:
    track_id: int
    class_id: int
    class_name: str
    confirmed: bool
    confidence_weighted_votes: Mapping[int, float]
    normalized_bbox_areas: tuple[float, ...]
    first_frame: int
    last_frame: int
    observation_count: int | None = None
    class_vote_share: float | None = None
    class_vote_margin: float | None = None
    classification_status: str = "stable"


@dataclass(frozen=True)
class ModelCapabilityProfile:
    profile_id: str
    profile_version: str
    target_taxonomy_version: str
    supported_classes: Mapping[int, str]
    unsupported_target_classes: Mapping[int, str]


@dataclass(frozen=True)
class CompositionMetric:
    class_id: int
    class_name: str
    support_status: SupportStatus
    object_count: int | None
    object_count_share: float | None
    detected_area_share: float | None


@dataclass(frozen=True)
class PriorityAssessment:
    status: str
    score: int | None
    tier: str | None
    scope: str
    config_version: str
    contributions: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Recommendation:
    rule_id: str
    rule_version: str
    observed_metric: str
    observed_value: float | int | None
    operator: str
    threshold: float | int
    reason: str
    action: str
    evidence_fields: tuple[str, ...]


@dataclass(frozen=True)
class QualityFlag:
    code: QualityFlagCode
    message: str
    evidence: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnalysisResult:
    analysis_id: str
    media_type: MediaType
    capability_profile: ModelCapabilityProfile
    dataset_snapshot_id: str | None
    dataset_snapshot_sha256: str | None
    model_version: str | None
    model_sha256: str | None
    detector_config_version: str
    priority_config_version: str
    recommendation_config_version: str
    compositions: Sequence[CompositionMetric]
    priority: PriorityAssessment
    recommendations: Sequence[Recommendation]
    quality_flags: Sequence[QualityFlag]
    limitations: Sequence[str]
    status: str = "completed"
    created_at: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)
    media_info: MediaInfo | None = None
    detections: Sequence[Detection] = field(default_factory=tuple)
    track_summaries: Sequence[TrackSummary] = field(default_factory=tuple)
    unique_object_count: int = 0
    coverage_p90: float = 0.0
    processing_summary: Mapping[str, Any] = field(default_factory=dict)
    timings_ms: Mapping[str, float] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_wire(self)


def to_wire(value: Any) -> Any:
    """Convert contracts to JSON-safe data and omit local-only dataclass fields."""

    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: to_wire(getattr(value, item.name))
            for item in fields(value)
            if item.metadata.get("serialize", True)
        }
    if isinstance(value, Mapping):
        return {str(key): to_wire(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_wire(item) for item in value]
    if isinstance(value, Path):
        raise ValueError("Local filesystem paths must not be serialized")
    return value
