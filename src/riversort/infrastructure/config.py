# Copyright (C) 2026 RiverSort AI contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Versioned YAML configuration loader and cross-contract validation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClassEntry(StrictModel):
    id: int
    name: str


class CapabilityProfileConfig(StrictModel):
    profile_id: str
    profile_version: str
    supported_class_ids: list[int]
    unsupported_target_class_ids: list[int]


class ClassesConfig(StrictModel):
    config_version: str
    target_taxonomy_version: str
    target_classes: list[ClassEntry]
    profiles: list[CapabilityProfileConfig]

    @model_validator(mode="after")
    def validate_taxonomy(self) -> "ClassesConfig":
        ids = [item.id for item in self.target_classes]
        names = [item.name for item in self.target_classes]
        if ids != list(range(len(ids))):
            raise ValueError("target class IDs must be unique and contiguous from zero")
        if len(set(names)) != len(names):
            raise ValueError("target class names must be unique")
        known = set(ids)
        for profile in self.profiles:
            supported = set(profile.supported_class_ids)
            unsupported = set(profile.unsupported_target_class_ids)
            if supported & unsupported:
                raise ValueError("supported and unsupported classes must not overlap")
            if supported | unsupported != known:
                raise ValueError("profile must classify every target class")
        return self


class DetectorConfig(StrictModel):
    config_version: str
    status: Literal["unconfigured", "configured"]
    capability_profile_id: str
    model_version: str | None
    model_sha256: str | None
    dataset_snapshot_id: str | None
    dataset_snapshot_sha256: str | None
    model_path_env: str
    package_manifest_filename: str
    names: dict[int, str]
    confidence_threshold: float | None = Field(default=None, ge=0, le=1)
    iou_threshold: float = Field(ge=0, le=1)
    inference_imgsz: int = Field(ge=640, le=1280)

    @model_validator(mode="after")
    def configured_values_are_complete(self) -> "DetectorConfig":
        values = (
            self.model_version,
            self.model_sha256,
            self.dataset_snapshot_id,
            self.dataset_snapshot_sha256,
            self.confidence_threshold,
        )
        if self.status == "configured" and any(value is None for value in values):
            raise ValueError("configured detector requires model version, checksum, and threshold")
        if Path(self.package_manifest_filename).name != self.package_manifest_filename:
            raise ValueError("model package manifest must be a plain filename")
        if self.inference_imgsz % 32 != 0:
            raise ValueError("detector inference_imgsz must be a multiple of 32")
        return self


class TrackerConfig(StrictModel):
    config_version: str
    tracker: Literal["bytetrack"]
    analysis_fps: int = Field(gt=0)
    track_high_thresh: float = Field(ge=0, le=1)
    track_low_thresh: float = Field(ge=0, le=1)
    new_track_thresh: float = Field(ge=0, le=1)
    track_buffer: int = Field(gt=0)
    match_thresh: float = Field(ge=0, le=1)
    fuse_score: bool
    minimum_observations: int = Field(gt=0)
    class_resolution: Literal["confidence_weighted_vote"]
    counting_unit: Literal["confirmed_unique_track"]
    class_vote_margin_threshold: float = Field(ge=0, le=1)
    stability_minimum_tracks: int = Field(gt=0)
    short_track_seconds: float = Field(gt=0)
    short_track_ratio_threshold: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_thresholds(self) -> "TrackerConfig":
        if self.track_low_thresh > self.track_high_thresh:
            raise ValueError("track low threshold cannot exceed high threshold")
        if self.new_track_thresh < self.track_high_thresh:
            raise ValueError("new track threshold cannot be below high threshold")
        return self


class ImageMediaConfig(StrictModel):
    allowed_extensions: list[str]
    allowed_mime_types: list[str]
    max_bytes: int = Field(gt=0)
    max_pixels: int = Field(gt=0)
    max_source_pixels: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_pixel_limits(self) -> "ImageMediaConfig":
        if self.max_source_pixels < self.max_pixels:
            raise ValueError("image source safety limit cannot be below processing target")
        return self


class VideoMediaConfig(StrictModel):
    allowed_extensions: list[str]
    allowed_mime_types: list[str]
    allowed_codecs: list[str]
    max_bytes: int = Field(gt=0)
    max_duration_seconds: float = Field(gt=0)
    max_width: int = Field(gt=0)
    max_height: int = Field(gt=0)
    max_source_pixels: int = Field(gt=0)
    max_source_dimension: int = Field(gt=0)
    max_decode_failure_ratio: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_resolution_limits(self) -> "VideoMediaConfig":
        target_pixels = self.max_width * self.max_height
        if self.max_source_pixels < target_pixels:
            raise ValueError("video source safety limit cannot be below processing target")
        if self.max_source_dimension < max(self.max_width, self.max_height):
            raise ValueError("video source dimension cannot be below processing target")
        return self


class MediaConfig(StrictModel):
    config_version: str
    image: ImageMediaConfig
    video: VideoMediaConfig


class WeightConfig(StrictModel):
    coverage: float = Field(ge=0, le=1)
    count: float = Field(ge=0, le=1)
    mobility: float = Field(ge=0, le=1)


class NormalizationConfig(StrictModel):
    coverage_p90_cap: float = Field(gt=0)
    image_object_count_cap: int = Field(gt=0)
    video_tracks_per_minute_cap: int = Field(gt=0)


class TierConfig(StrictModel):
    name: str
    minimum: int = Field(ge=0, le=100)
    maximum: int = Field(ge=0, le=100)


class ResidualOverrideConfig(StrictModel):
    enabled: bool
    class_id: int
    threshold: float = Field(ge=0, le=1)


class PriorityConfig(StrictModel):
    config_version: str
    applicable_profiles: list[str]
    scope: Literal["supported_classes_only"]
    weights: WeightConfig
    normalization: NormalizationConfig
    mobility_class_ids: list[int]
    tiers: list[TierConfig]
    residual_override: ResidualOverrideConfig

    @model_validator(mode="after")
    def validate_formula(self) -> "PriorityConfig":
        total = self.weights.coverage + self.weights.count + self.weights.mobility
        if abs(total - 1.0) > 1e-9:
            raise ValueError("priority weights must sum to 1")
        expected = 0
        for tier in self.tiers:
            if tier.minimum != expected or tier.minimum > tier.maximum:
                raise ValueError("priority tiers must be ordered and contiguous")
            expected = tier.maximum + 1
        if expected != 101:
            raise ValueError("priority tiers must cover 0 through 100")
        return self


class RecommendationRuleConfig(StrictModel):
    rule_id: str
    priority: int
    class_id: int | None
    metric: str
    operator: Literal[">", ">=", "=="]
    threshold: float
    applicable_profiles: list[str]
    reason: str
    action: str


class RecommendationsConfig(StrictModel):
    config_version: str
    rules: list[RecommendationRuleConfig]

    @model_validator(mode="after")
    def validate_rule_ids(self) -> "RecommendationsConfig":
        rule_ids = [rule.rule_id for rule in self.rules]
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("recommendation rule IDs must be unique")
        return self


class ProjectConfig(StrictModel):
    classes: ClassesConfig
    detector: DetectorConfig
    tracker: TrackerConfig
    media: MediaConfig
    priority: PriorityConfig
    recommendations: RecommendationsConfig

    @model_validator(mode="after")
    def validate_cross_contracts(self) -> "ProjectConfig":
        profiles = {item.profile_id: item for item in self.classes.profiles}
        if self.detector.capability_profile_id not in profiles:
            raise ValueError("detector refers to an unknown capability profile")
        profile = profiles[self.detector.capability_profile_id]
        class_by_id = {item.id: item.name for item in self.classes.target_classes}
        expected_names = {class_id: class_by_id[class_id] for class_id in profile.supported_class_ids}
        if self.detector.names != expected_names:
            raise ValueError("detector names do not exactly match supported profile classes")
        known = set(class_by_id)
        if len(self.priority.mobility_class_ids) != len(set(self.priority.mobility_class_ids)):
            raise ValueError("mobility class IDs must be unique")
        if not set(self.priority.mobility_class_ids) <= set(profile.supported_class_ids):
            raise ValueError("mobility classes must be supported by the active profile")
        for rule in self.recommendations.rules:
            if rule.class_id is not None and rule.class_id not in known:
                raise ValueError(f"rule {rule.rule_id} refers to an unknown class")
            if (
                rule.class_id in profile.unsupported_target_class_ids
                and profile.profile_id in rule.applicable_profiles
            ):
                raise ValueError(f"rule {rule.rule_id} activates an unsupported class")
        if self.priority.residual_override.enabled and self.priority.residual_override.class_id in profile.unsupported_target_class_ids:
            raise ValueError("residual override cannot be active for the four-class profile")
        return self


def _read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path.name}")
    return data


def load_project_config(config_dir: Path | str = "configs") -> ProjectConfig:
    directory = Path(config_dir)
    return ProjectConfig(
        classes=ClassesConfig.model_validate(_read_yaml(directory / "classes.yaml")),
        detector=DetectorConfig.model_validate(_read_yaml(directory / "detector.yaml")),
        tracker=TrackerConfig.model_validate(_read_yaml(directory / "tracker.yaml")),
        media=MediaConfig.model_validate(_read_yaml(directory / "media.yaml")),
        priority=PriorityConfig.model_validate(_read_yaml(directory / "priority.yaml")),
        recommendations=RecommendationsConfig.model_validate(_read_yaml(directory / "recommendations.yaml")),
    )


def capability_profile(config: ProjectConfig):
    """Translate validated infrastructure config into a standard-library contract."""

    from riversort.domain import ModelCapabilityProfile

    profile = next(
        item
        for item in config.classes.profiles
        if item.profile_id == config.detector.capability_profile_id
    )
    class_by_id = {item.id: item.name for item in config.classes.target_classes}
    return ModelCapabilityProfile(
        profile_id=profile.profile_id,
        profile_version=profile.profile_version,
        target_taxonomy_version=config.classes.target_taxonomy_version,
        supported_classes={class_id: class_by_id[class_id] for class_id in profile.supported_class_ids},
        unsupported_target_classes={
            class_id: class_by_id[class_id]
            for class_id in profile.unsupported_target_class_ids
        },
    )


def priority_policy(config: ProjectConfig):
    from riversort.domain import PriorityPolicy, PriorityTier

    priority = config.priority
    return PriorityPolicy(
        config_version=priority.config_version,
        applicable_profiles=tuple(priority.applicable_profiles),
        scope=priority.scope,
        coverage_weight=priority.weights.coverage,
        count_weight=priority.weights.count,
        mobility_weight=priority.weights.mobility,
        coverage_p90_cap=priority.normalization.coverage_p90_cap,
        image_object_count_cap=priority.normalization.image_object_count_cap,
        video_tracks_per_minute_cap=priority.normalization.video_tracks_per_minute_cap,
        mobility_class_ids=tuple(priority.mobility_class_ids),
        tiers=tuple(
            PriorityTier(tier.name, tier.minimum, tier.maximum)
            for tier in priority.tiers
        ),
        residual_override_enabled=priority.residual_override.enabled,
    )


def recommendation_policy(config: ProjectConfig):
    from riversort.domain import RecommendationPolicy, RecommendationRule

    return RecommendationPolicy(
        config_version=config.recommendations.config_version,
        rules=tuple(
            RecommendationRule(
                rule_id=rule.rule_id,
                priority=rule.priority,
                class_id=rule.class_id,
                metric=rule.metric,
                operator=rule.operator,
                threshold=rule.threshold,
                applicable_profiles=tuple(rule.applicable_profiles),
                reason=rule.reason,
                action=rule.action,
            )
            for rule in config.recommendations.rules
        ),
    )
