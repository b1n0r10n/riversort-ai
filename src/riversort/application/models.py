# Copyright (C) 2026 RiverSort AI contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Framework-neutral application DTOs used by media analysis use cases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from riversort.domain import (
    AnalysisResult,
    Detection,
    MediaInfo,
    ModelCapabilityProfile,
    PriorityPolicy,
    RecommendationPolicy,
)


@dataclass(frozen=True)
class UploadedMedia:
    filename: str
    content: bytes
    declared_mime: str | None = None


@dataclass(frozen=True)
class FramePacket:
    frame_index: int
    timestamp_ms: int
    pixels: Any
    width: int
    height: int


@dataclass(frozen=True)
class TrackObservation:
    track_id: int
    detection: Detection
    class_vote_share: float | None = None
    classification_status: str | None = None


@dataclass(frozen=True)
class ValidatedImage:
    media_info: MediaInfo
    pixels: Any


@dataclass(frozen=True)
class AnalysisBundle:
    result: AnalysisResult
    annotated_media: bytes
    annotated_mime: str
    json_bytes: bytes
    csv_bytes: bytes


@dataclass(frozen=True)
class RuntimeContext:
    capability: ModelCapabilityProfile
    priority_policy: PriorityPolicy
    recommendation_policy: RecommendationPolicy
    dataset_snapshot_id: str | None
    dataset_snapshot_sha256: str | None
    model_version: str | None
    model_sha256: str | None
    detector_config_version: str
    priority_config_version: str
    recommendation_config_version: str


Metadata = Mapping[str, str]
