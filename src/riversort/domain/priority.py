# Copyright (C) 2026 RiverSort AI contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Transparent cleanup-priority index for the four-class MVP."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Sequence

from .contracts import CompositionMetric, MediaType, ModelCapabilityProfile, PriorityAssessment, SupportStatus
from .errors import DomainInputError


@dataclass(frozen=True)
class PriorityTier:
    name: str
    minimum: int
    maximum: int


@dataclass(frozen=True)
class PriorityPolicy:
    config_version: str
    applicable_profiles: tuple[str, ...]
    scope: str
    coverage_weight: float
    count_weight: float
    mobility_weight: float
    coverage_p90_cap: float
    image_object_count_cap: int
    video_tracks_per_minute_cap: int
    mobility_class_ids: tuple[int, ...]
    tiers: tuple[PriorityTier, ...]
    residual_override_enabled: bool


def _decimal(value: float | int) -> Decimal:
    return Decimal(str(value))


def tier_for_score(score: int, tiers: Sequence[PriorityTier]) -> str:
    if not 0 <= score <= 100:
        raise DomainInputError("Priority score must be in range 0..100")
    for tier in tiers:
        if tier.minimum <= score <= tier.maximum:
            return tier.name
    raise DomainInputError("Priority tiers do not cover the score")


def assess_priority(
    compositions: Sequence[CompositionMetric],
    coverage_p90: float,
    media_type: MediaType,
    capability: ModelCapabilityProfile,
    policy: PriorityPolicy,
    *,
    duration_seconds: float | None = None,
) -> PriorityAssessment:
    """Assess the visual index from supported objects only."""

    if capability.profile_id not in policy.applicable_profiles:
        raise DomainInputError("Priority policy is not applicable to capability profile")
    if policy.scope != "supported_classes_only":
        raise DomainInputError("MVP priority scope must be supported_classes_only")
    if policy.residual_override_enabled:
        raise DomainInputError("Residual override must be disabled for the four-class MVP")
    if not math.isfinite(coverage_p90) or not 0 <= coverage_p90 <= 1:
        raise DomainInputError("Coverage p90 must be a finite ratio in range 0..1")

    supported = {
        metric.class_id: metric
        for metric in compositions
        if metric.support_status is SupportStatus.SUPPORTED
    }
    if set(supported) != set(capability.supported_classes):
        raise DomainInputError("Composition rows do not match supported capability classes")
    object_count = sum(metric.object_count or 0 for metric in supported.values())
    if object_count == 0:
        return PriorityAssessment(
            status="not_assessable",
            score=None,
            tier=None,
            scope=policy.scope,
            config_version=policy.config_version,
            contributions={},
        )

    if media_type is MediaType.IMAGE:
        count_value = float(object_count)
        count_norm = min(count_value / policy.image_object_count_cap, 1.0)
    elif media_type is MediaType.VIDEO:
        if duration_seconds is None or not math.isfinite(duration_seconds) or duration_seconds <= 0:
            raise DomainInputError("Positive video duration is required when confirmed tracks exist")
        count_value = object_count * 60.0 / duration_seconds
        count_norm = min(count_value / policy.video_tracks_per_minute_cap, 1.0)
    else:
        raise DomainInputError(f"Unsupported media type {media_type}")

    coverage_norm = min(coverage_p90 / policy.coverage_p90_cap, 1.0)
    mobility = sum((supported[class_id].object_count_share or 0.0) for class_id in policy.mobility_class_ids)
    weighted_coverage = policy.coverage_weight * coverage_norm
    weighted_count = policy.count_weight * count_norm
    weighted_mobility = policy.mobility_weight * mobility
    raw_score = 100.0 * (weighted_coverage + weighted_count + weighted_mobility)
    score = int(_decimal(raw_score).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    score = min(max(score, 0), 100)
    return PriorityAssessment(
        status="assessed",
        score=score,
        tier=tier_for_score(score, policy.tiers),
        scope=policy.scope,
        config_version=policy.config_version,
        contributions={
            "coverage_p90": coverage_p90,
            "coverage_norm": coverage_norm,
            "object_count_or_tracks_per_minute": count_value,
            "count_norm": count_norm,
            "mobility_share": mobility,
            "weighted_coverage": weighted_coverage,
            "weighted_count": weighted_count,
            "weighted_mobility": weighted_mobility,
            "raw_score": raw_score,
        },
    )
