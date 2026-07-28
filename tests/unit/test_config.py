# Copyright (C) 2026 RiverSort AI contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from riversort.infrastructure.config import ProjectConfig, load_project_config


CONFIG_DIR = Path(__file__).parents[2] / "configs"


def raw_project() -> dict:
    payload = {}
    for name in ("classes", "detector", "tracker", "media", "priority", "recommendations"):
        payload[name] = yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text(encoding="utf-8"))
    return payload


def test_project_config_loads() -> None:
    config = load_project_config(CONFIG_DIR)
    assert config.detector.status == "configured"
    assert config.detector.model_version == "0.1.0-baseline.1"
    assert config.detector.confidence_threshold == 0.40
    assert config.detector.config_version == "1.2.0"
    assert config.detector.inference_imgsz == 960
    assert config.detector.model_sha256 == "cc235703c6b8b9f6244e68145ca82fea115c95526b91e5716468c3f2d2c31235"
    assert config.detector.dataset_snapshot_id == "riversort_fpwr4-20260722-r2"
    assert config.tracker.config_version == "1.2.0"
    assert config.tracker.minimum_observations == 2
    assert config.tracker.class_vote_margin_threshold == 0.20
    assert config.tracker.short_track_ratio_threshold == 0.50
    assert config.media.config_version == "1.1.0"
    assert config.media.image.max_bytes == 20 * 1024 * 1024
    assert config.media.image.max_pixels == 25_000_000
    assert config.media.image.max_source_pixels == 80_000_000
    assert config.media.video.max_source_pixels == 33_177_600
    assert config.priority.config_version == "1.1.0"


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_name",
        "bad_weight",
        "unknown_rule",
        "active_residual",
        "model_mismatch",
        "image_source_below_target",
        "video_source_below_target",
        "detector_imgsz_not_multiple",
        "detector_imgsz_too_large",
    ],
)
def test_invalid_cross_contracts_fail(mutation: str) -> None:
    payload = raw_project()
    if mutation == "duplicate_name":
        payload["classes"]["target_classes"][4]["name"] = "other_plastic"
    elif mutation == "bad_weight":
        payload["priority"]["weights"]["coverage"] = 0.40
    elif mutation == "unknown_rule":
        payload["recommendations"]["rules"][0]["class_id"] = 99
    elif mutation == "active_residual":
        payload["priority"]["residual_override"]["enabled"] = True
    elif mutation == "model_mismatch":
        payload["detector"]["names"][3] = "wrong_name"
    elif mutation == "image_source_below_target":
        payload["media"]["image"]["max_source_pixels"] = 1
    elif mutation == "video_source_below_target":
        payload["media"]["video"]["max_source_pixels"] = 1
    elif mutation == "detector_imgsz_not_multiple":
        payload["detector"]["inference_imgsz"] = 950
    elif mutation == "detector_imgsz_too_large":
        payload["detector"]["inference_imgsz"] = 1312
    with pytest.raises((ValidationError, ValueError)):
        ProjectConfig.model_validate(payload)
