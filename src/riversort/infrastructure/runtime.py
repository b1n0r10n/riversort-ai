# Copyright (C) 2026 RiverSort AI contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Composition root for local image/video analysis services."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from riversort.adapters import (
    ByteTrackAdapter,
    FfmpegVideoEncoder,
    OpenCvAnnotator,
    OpenCvMediaAdapter,
    ResultExporter,
    build_detector_runtime,
)
from riversort.application.analyze_image import AnalyzeImage
from riversort.application.analyze_video import AnalyzeVideo
from riversort.application.models import RuntimeContext

from .config import (
    ProjectConfig,
    capability_profile,
    priority_policy,
    recommendation_policy,
)
from .workspace import AnalysisWorkspace


@dataclass(frozen=True)
class RuntimeServices:
    image: AnalyzeImage
    video: AnalyzeVideo


def runtime_context(config: ProjectConfig) -> RuntimeContext:
    return RuntimeContext(
        capability=capability_profile(config),
        priority_policy=priority_policy(config),
        recommendation_policy=recommendation_policy(config),
        dataset_snapshot_id=config.detector.dataset_snapshot_id,
        dataset_snapshot_sha256=config.detector.dataset_snapshot_sha256,
        model_version=config.detector.model_version,
        model_sha256=config.detector.model_sha256,
        detector_config_version=config.detector.config_version,
        priority_config_version=config.priority.config_version,
        recommendation_config_version=config.recommendations.config_version,
    )


def build_runtime_services(
    config: ProjectConfig,
    *,
    detector: Any | None = None,
    workspace_root: Path | None = None,
) -> RuntimeServices:
    runtime = runtime_context(config)
    detector_adapter = detector or build_detector_runtime(config)
    media = OpenCvMediaAdapter(config.media, config.tracker)
    annotator = OpenCvAnnotator()
    exporter = ResultExporter()
    workspace_factory = partial(AnalysisWorkspace, root=workspace_root)
    image = AnalyzeImage(
        runtime=runtime,
        media=media,
        detector=detector_adapter,
        annotator=annotator,
        exporter=exporter,
        workspace_factory=workspace_factory,
    )
    video = AnalyzeVideo(
        runtime=runtime,
        media=media,
        detector=detector_adapter,
        tracker=ByteTrackAdapter(config.tracker),
        annotator=annotator,
        exporter=exporter,
        encoder_factory=FfmpegVideoEncoder,
        workspace_factory=workspace_factory,
        tracker_low_threshold=config.tracker.track_low_thresh,
        analysis_fps=config.tracker.analysis_fps,
        max_decode_failure_ratio=config.media.video.max_decode_failure_ratio,
        class_vote_margin_threshold=config.tracker.class_vote_margin_threshold,
        stability_minimum_tracks=config.tracker.stability_minimum_tracks,
        short_track_seconds=config.tracker.short_track_seconds,
        short_track_ratio_threshold=config.tracker.short_track_ratio_threshold,
    )
    return RuntimeServices(image=image, video=video)
