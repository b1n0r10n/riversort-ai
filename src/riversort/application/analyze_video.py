# Copyright (C) 2026 RiverSort AI contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""End-to-end video analysis with chronological sampling and unique-track counting."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping

from riversort.domain import (
    AnalysisRequest,
    AnalysisResult,
    MediaType,
    TrackSummary,
    assess_priority,
    build_quality_flags,
    classification_uncertainty_evidence,
    compose_video,
    evaluate_recommendations,
    percentile90,
    standard_limitations,
    tracking_instability_evidence,
    union_bbox_coverage,
)

from .analyze_image import _clean_metadata
from .errors import AnalysisError
from .models import AnalysisBundle, RuntimeContext, TrackObservation, UploadedMedia
from .ports import (
    AnnotatorPort,
    DetectorPort,
    ExporterPort,
    MediaPort,
    TrackerPort,
    WorkspacePort,
)


ProgressCallback = Callable[[str], None]
EncoderFactory = Callable[..., object]
WorkspaceFactory = Callable[[str], WorkspacePort]
LOGGER = logging.getLogger(__name__)


class AnalyzeVideo:
    def __init__(
        self,
        *,
        runtime: RuntimeContext,
        media: MediaPort,
        detector: DetectorPort,
        tracker: TrackerPort,
        annotator: AnnotatorPort,
        exporter: ExporterPort,
        encoder_factory: EncoderFactory,
        workspace_factory: WorkspaceFactory,
        tracker_low_threshold: float,
        analysis_fps: int,
        max_decode_failure_ratio: float,
        class_vote_margin_threshold: float,
        stability_minimum_tracks: int,
        short_track_seconds: float,
        short_track_ratio_threshold: float,
    ) -> None:
        self.runtime = runtime
        self.media = media
        self.detector = detector
        self.tracker = tracker
        self.annotator = annotator
        self.exporter = exporter
        self.encoder_factory = encoder_factory
        self.workspace_factory = workspace_factory
        self.tracker_low_threshold = tracker_low_threshold
        self.analysis_fps = analysis_fps
        self.max_decode_failure_ratio = max_decode_failure_ratio
        self.class_vote_margin_threshold = class_vote_margin_threshold
        self.stability_minimum_tracks = stability_minimum_tracks
        self.short_track_seconds = short_track_seconds
        self.short_track_ratio_threshold = short_track_ratio_threshold

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
        encoder = None

        with self.workspace_factory(analysis_id) as workspace:
            media_path, display_name = workspace.save_upload(upload)
            request = AnalysisRequest(
                analysis_id=analysis_id,
                media_type=MediaType.VIDEO,
                original_filename=display_name,
                metadata=cleaned_metadata,
                local_temporary_path=media_path,
            )
            try:
                callback("validation")
                started = time.perf_counter()
                source_media_info = self.media.probe_video(media_path, upload)
                timings["validation"] = self._elapsed_ms(started)
                callback("normalization")
                started = time.perf_counter()
                media_path, media_info = self.media.normalize_video(
                    media_path,
                    source_media_info,
                    workspace.allocate("normalized.mp4"),
                )
                timings["normalization"] = self._elapsed_ms(started)
                self.tracker.reset()

                sampled_frames = 0
                raw_detections = 0
                last_source_frame = -1
                observations_by_frame: list[
                    tuple[int, tuple[TrackObservation, ...]]
                ] = []
                inference_ms = 0.0
                tracking_ms = 0.0

                callback("inference")
                for packet in self.media.iter_video_frames(media_path, media_info):
                    last_source_frame = packet.frame_index
                    sampled_frames += 1
                    started = time.perf_counter()
                    detections = tuple(
                        self.detector.detect_frame(
                            packet.pixels,
                            confidence_floor=self.tracker_low_threshold,
                            frame_index=packet.frame_index,
                            timestamp_ms=packet.timestamp_ms,
                        )
                    )
                    inference_ms += self._elapsed_ms(started)
                    raw_detections += len(detections)

                    callback("tracking")
                    started = time.perf_counter()
                    observations = tuple(self.tracker.update(detections, packet.pixels))
                    tracking_ms += self._elapsed_ms(started)
                    observations_by_frame.append((packet.frame_index, observations))

                if sampled_frames == 0:
                    raise AnalysisError(
                        code=self._decode_error_code(),
                        public_message="Tidak ada frame video yang berhasil diproses.",
                        recovery_action="Encode ulang video menjadi MP4 H.264.",
                    )
                timings["inference"] = round(inference_ms, 3)
                timings["tracking"] = round(tracking_ms, 3)

                callback("aggregation")
                started = time.perf_counter()
                track_summaries = tuple(self.tracker.finalize())
                summary_by_id = {
                    track.track_id: track
                    for track in track_summaries
                    if track.confirmed
                }
                coverages = [
                    union_bbox_coverage(
                        (
                            observation.detection.bbox_xyxy
                            for observation in observations
                            if observation.track_id in summary_by_id
                        ),
                        (media_info.width, media_info.height),
                    )
                    for _, observations in observations_by_frame
                ]
                compositions = compose_video(track_summaries, self.runtime.capability)
                coverage_p90 = percentile90(coverages)
                priority = assess_priority(
                    compositions,
                    coverage_p90,
                    MediaType.VIDEO,
                    self.runtime.capability,
                    self.runtime.priority_policy,
                    duration_seconds=media_info.duration_seconds,
                )
                recommendations = evaluate_recommendations(
                    compositions,
                    self.runtime.capability,
                    self.runtime.recommendation_policy,
                )
                expected_frames = max(media_info.frame_count or 0, 1)
                source_span = last_source_frame + 1
                decode_failure_ratio = max(
                    0.0, min(1.0, (expected_frames - source_span) / expected_frames)
                )
                low_quality = (
                    {"decode_failure_ratio": decode_failure_ratio}
                    if decode_failure_ratio
                    > self.max_decode_failure_ratio
                    else None
                )
                class_uncertainty = classification_uncertainty_evidence(
                    track_summaries,
                    margin_threshold=self.class_vote_margin_threshold,
                )
                tracking_instability = tracking_instability_evidence(
                    track_summaries,
                    analysis_fps=float(media_info.analysis_fps or self.analysis_fps),
                    minimum_track_count=self.stability_minimum_tracks,
                    short_track_seconds=self.short_track_seconds,
                    short_track_ratio_threshold=self.short_track_ratio_threshold,
                )
                quality_flags = build_quality_flags(
                    compositions,
                    self.runtime.capability,
                    low_quality_evidence=low_quality,
                    classification_uncertainty_evidence=class_uncertainty,
                    tracking_instability_evidence=tracking_instability,
                )
                unique_count = sum(item.object_count or 0 for item in compositions)
                timings["aggregation"] = self._elapsed_ms(started)

                callback("encoding")
                started = time.perf_counter()
                output_path = workspace.allocate("annotated.mp4")
                encoder = self.encoder_factory(
                    output_path,
                    width=media_info.width,
                    height=media_info.height,
                    fps=float(media_info.analysis_fps or self.analysis_fps),
                )
                encoder.open()
                second_pass_frames = 0
                for packet in self.media.iter_video_frames(media_path, media_info):
                    if second_pass_frames >= len(observations_by_frame):
                        raise AnalysisError(
                            code=self._decode_error_code(),
                            public_message=(
                                "Frame preview tidak konsisten dengan pass tracking."
                            ),
                            recovery_action="Encode ulang video dan coba kembali.",
                        )
                    expected_frame, raw_observations = observations_by_frame[
                        second_pass_frames
                    ]
                    if packet.frame_index != expected_frame:
                        raise AnalysisError(
                            code=self._decode_error_code(),
                            public_message=(
                                "Urutan frame berubah saat preview dibuat."
                            ),
                            recovery_action="Encode ulang video dan coba kembali.",
                        )
                    stabilized = self._stabilize_observations(
                        raw_observations, summary_by_id
                    )
                    annotated_frame = self.annotator.annotate_video_frame(
                        packet.pixels, stabilized
                    )
                    encoder.write(annotated_frame)
                    second_pass_frames += 1
                if second_pass_frames != len(observations_by_frame):
                    raise AnalysisError(
                        code=self._decode_error_code(),
                        public_message="Sebagian frame preview gagal dibaca ulang.",
                        recovery_action="Encode ulang video dan coba kembali.",
                    )
                annotated_bytes = encoder.close()
                encoder = None
                timings["encoding"] = self._elapsed_ms(started)

                callback("export")
                result = AnalysisResult(
                    analysis_id=request.analysis_id,
                    media_type=MediaType.VIDEO,
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
                    media_info=media_info,
                    track_summaries=track_summaries,
                    unique_object_count=unique_count,
                    coverage_p90=coverage_p90,
                    processing_summary={
                        "sampled_frame_count": sampled_frames,
                        "raw_detection_count": raw_detections,
                        "confirmed_track_count": unique_count,
                        "confirmed_only_coverage": True,
                        "decode_failure_ratio": decode_failure_ratio,
                        "interpretation_status": (
                            "review_required"
                            if class_uncertainty or tracking_instability
                            else "standard"
                        ),
                        "classification_uncertain_track_count": (
                            class_uncertainty["uncertain_track_count"]
                            if class_uncertainty
                            else 0
                        ),
                        "short_track_count": (
                            tracking_instability["short_track_count"]
                            if tracking_instability
                            else 0
                        ),
                        "short_track_ratio": (
                            tracking_instability["short_track_ratio"]
                            if tracking_instability
                            else 0.0
                        ),
                        "resolution_normalized": media_info.resolution_normalized,
                        "original_resolution": {
                            "width": media_info.original_width,
                            "height": media_info.original_height,
                        },
                        "analysis_resolution": {
                            "width": media_info.width,
                            "height": media_info.height,
                        },
                    },
                    timings_ms=timings,
                )
                json_bytes = self.exporter.to_json(result)
                csv_bytes = self.exporter.to_csv(result)
                LOGGER.info(
                    "video analysis completed",
                    extra={
                        "analysis_id": analysis_id,
                        "stage": "completed",
                        "duration_ms": sum(timings.values()),
                    },
                )
                return AnalysisBundle(
                    result=result,
                    annotated_media=annotated_bytes,
                    annotated_mime="video/mp4",
                    json_bytes=json_bytes,
                    csv_bytes=csv_bytes,
                )
            finally:
                if encoder is not None:
                    encoder.abort()

    @staticmethod
    def _stabilize_observations(
        observations: tuple[TrackObservation, ...],
        summary_by_id: Mapping[int, TrackSummary],
    ) -> tuple[TrackObservation, ...]:
        stabilized = []
        for observation in observations:
            summary = summary_by_id.get(observation.track_id)
            if summary is None:
                continue
            vote_share = summary.class_vote_share or 0.0
            stabilized.append(
                TrackObservation(
                    track_id=observation.track_id,
                    detection=replace(
                        observation.detection,
                        class_id=summary.class_id,
                        class_name=summary.class_name,
                        confidence=vote_share,
                    ),
                    class_vote_share=vote_share,
                    classification_status=summary.classification_status,
                )
            )
        return tuple(stabilized)

    @staticmethod
    def _elapsed_ms(started: float) -> float:
        return round((time.perf_counter() - started) * 1000.0, 3)

    @staticmethod
    def _decode_error_code():
        from riversort.domain import AnalysisErrorCode

        return AnalysisErrorCode.DECODE_FAILED
