# Copyright (C) 2026 RiverSort AI contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Detector runtime readiness gate; inference is intentionally deferred."""

from __future__ import annotations

from typing import Any

from riversort.application import ModelNotReadyError
from riversort.infrastructure.config import capability_profile

from .detector_yolo import YoloDetectorAdapter


def build_detector_runtime(project_config: Any) -> YoloDetectorAdapter:
    detector_config = getattr(project_config, "detector", project_config)
    if detector_config.status != "configured":
        raise ModelNotReadyError(
            "MODEL_NOT_READY: detector configuration is not ready for inference"
        )
    if not hasattr(project_config, "classes"):
        raise ModelNotReadyError("MODEL_NOT_READY: full project config is required")
    return YoloDetectorAdapter(detector_config, capability_profile(project_config))
