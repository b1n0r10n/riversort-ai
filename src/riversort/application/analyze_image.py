# Copyright (C) 2026 RiverSort AI contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""End-to-end image analysis use case."""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Callable, Mapping

from riversort.domain import (
    AnalysisRequest,
    AnalysisResult,
    MediaType,
    assess_priority,
    build_quality_flags,
    compose_image,
    evaluate_recommendations,
    standard_limitations,
    union_bbox_coverage,
)
from .models import AnalysisBundle, RuntimeContext, UploadedMedia
from .ports import AnnotatorPort, DetectorPort, ExporterPort, MediaPort, WorkspacePort


ProgressCallback = Callable[[str], None]
WorkspaceFactory = Callable[[str], WorkspacePort]
LOGGER = logging.getLogger(__name__)


def _clean_metadata(metadata: Mapping[str, str] | None) -> dict[str, str]:
    cleaned: dict[str, str] = {}
    for key in ("location", "observed_at", "weather", "notes"):
        value = str((metadata or {}).get(key, "")).strip()
        if value:
            cleaned[key] = value[:500]
    return cleaned


class AnalyzeImage:
    def __init__(
        self,
        *,
        runtime: RuntimeContext,
        media: MediaPort,
        detector: DetectorPort,
        annotator: AnnotatorPort,
        exporter: ExporterPort,
        workspace_factory: WorkspaceFactory,
    ) -> None:
        self.runtime = runtime
        self.media = media
        self.detector = detector
        self.annotator = annotator
        self.exporter = exporter
        self.workspace_factory = workspace_factory

    def execute(
        self,
        upload: UploadedMedia,
        metadata: Mapping[str, str] | None = None,
        *,
        progress: ProgressCallback | None = None,
    ) -> AnalysisBundle:
        analysis_id = f"RIVER-{uuid.uuid4().hex[:12].upper()}"
        created_at = datetime.now(timezone.utc).isoformat()
        timings: dict[str, float] = {}
        callback = progress or (lambda _: None)
        cleaned_metadata = _clean_metadata(metadata)

        with self.workspace_factory(analysis_id) as workspace:
            media_path, display_name = workspace.save_upload(upload)
            request = AnalysisRequest(
                analysis_id=analysis_id,
                media_type=MediaType.IMAGE,
                original_filename=display_name,
                metadata=cleaned_metadata,
                local_temporary_path=media_path,
            )

            callback("validation")
            started = time.perf_counter()
            validated = self.media.validate_image(media_path, upload)
            timings["validation"] = self._elapsed_ms(started)

            callback("inference")
            started = time.perf_counter()
            detections = tuple(self.detector.detect_image(validated.pixels))
            timings["inference"] = self._elapsed_ms(started)

            callback("aggregation")
            started = time.perf_counter()
            compositions = compose_image(detections, self.runtime.capability)
            coverage = union_bbox_coverage(
                (item.bbox_xyxy for item in detections),
                (validated.media_info.width, validated.media_info.height),
            )
            priority = assess_priority(
                compositions,
                coverage,
                MediaType.IMAGE,
                self.runtime.capability,
                self.runtime.priority_policy,
            )
            recommendations = evaluate_recommendations(
                compositions,
                self.runtime.capability,
                self.runtime.recommendation_policy,
            )
            quality_flags = build_quality_flags(compositions, self.runtime.capability)
            unique_count = sum(item.object_count or 0 for item in compositions)
            timings["aggregation"] = self._elapsed_ms(started)

            callback("export")
            started = time.perf_counter()
            annotated = self.annotator.annotate_image(validated.pixels, detections)
            timings["annotation"] = self._elapsed_ms(started)
            result = AnalysisResult(
                analysis_id=request.analysis_id,
                media_type=MediaType.IMAGE,
                capability_profile=self.runtime.capability,
                dataset_snapshot_id=self.runtime.dataset_snapshot_id,
                dataset_snapshot_sha256=self.runtime.dataset_snapshot_sha256,
                model_version=self.runtime.model_version,
                model_sha256=self.runtime.model_sha256,
                detector_config_version=self.runtime.detector_config_version,
                priority_config_version=self.runtime.priority_config_version,
                recommendation_config_version=self.runtime.recommendation_config_version,
                compositions=compositions,
                priority=priority,
                recommendations=recommendations,
                quality_flags=quality_flags,
                limitations=standard_limitations(self.runtime.capability),
                status="completed",
                created_at=created_at,
                metadata=cleaned_metadata,
                media_info=validated.media_info,
                detections=detections,
                unique_object_count=unique_count,
                coverage_p90=coverage,
                processing_summary={
                    "raw_detection_count": len(detections),
                    "valid_detection_count": unique_count,
                    "resolution_normalized": (
                        validated.media_info.resolution_normalized
                    ),
                    "original_resolution": {
                        "width": validated.media_info.original_width,
                        "height": validated.media_info.original_height,
                    },
                    "analysis_resolution": {
                        "width": validated.media_info.width,
                        "height": validated.media_info.height,
                    },
                },
                timings_ms=timings,
            )
            json_bytes = self.exporter.to_json(result)
            csv_bytes = self.exporter.to_csv(result)
            LOGGER.info(
                "image analysis completed",
                extra={
                    "analysis_id": analysis_id,
                    "stage": "completed",
                    "duration_ms": sum(timings.values()),
                },
            )
            return AnalysisBundle(
                result=result,
                annotated_media=annotated,
                annotated_mime="image/jpeg",
                json_bytes=json_bytes,
                csv_bytes=csv_bytes,
            )

    @staticmethod
    def _elapsed_ms(started: float) -> float:
        return round((time.perf_counter() - started) * 1000.0, 3)
