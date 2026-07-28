# Copyright (C) 2026 RiverSort AI contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest

from riversort.domain import (
    CompositionMetric,
    DomainInputError,
    MediaType,
    SupportStatus,
    assess_priority,
    tier_for_score,
)
from riversort.infrastructure.config import capability_profile, load_project_config, priority_policy


def policies():
    config = load_project_config()
    return capability_profile(config), priority_policy(config)


def composition(counts: tuple[int, int, int, int]) -> tuple[CompositionMetric, ...]:
    names = ("plastic_bottle", "plastic_bag", "plastic_packaging_sachet", "other_plastic")
    total = sum(counts)
    rows = tuple(
        CompositionMetric(
            class_id,
            name,
            SupportStatus.SUPPORTED,
            count,
            count / total if total else None,
            count / total if total else None,
        )
        for class_id, (name, count) in enumerate(zip(names, counts))
    )
    return rows + (
        CompositionMetric(4, "other_floating_debris", SupportStatus.NOT_SUPPORTED, None, None, None),
    )


def test_empty_composition_is_not_assessable() -> None:
    capability, policy = policies()
    result = assess_priority(composition((0, 0, 0, 0)), 0.9, MediaType.IMAGE, capability, policy)
    assert result.status == "not_assessable"
    assert result.score is None
    assert result.tier is None


def test_formula_uses_half_up_rounding_and_exposes_contributions() -> None:
    capability, policy = policies()
    result = assess_priority(composition((1, 0, 0, 0)), 0.12, MediaType.IMAGE, capability, policy)
    assert result.contributions["raw_score"] == pytest.approx(34.5)
    assert result.score == 35
    assert result.tier == "medium"
    assert result.scope == "supported_classes_only"
    reconstructed = 100 * (
        result.contributions["weighted_coverage"]
        + result.contributions["weighted_count"]
        + result.contributions["weighted_mobility"]
    )
    assert reconstructed == pytest.approx(result.contributions["raw_score"])


def test_priority_tier_boundaries() -> None:
    _, policy = policies()
    assert tier_for_score(34, policy.tiers) == "low"
    assert tier_for_score(35, policy.tiers) == "medium"
    assert tier_for_score(69, policy.tiers) == "medium"
    assert tier_for_score(70, policy.tiers) == "high"


def test_video_requires_duration_and_uses_tracks_per_minute() -> None:
    capability, policy = policies()
    with pytest.raises(DomainInputError, match="Positive video duration"):
        assess_priority(composition((1, 0, 0, 0)), 0.1, MediaType.VIDEO, capability, policy)
    result = assess_priority(
        composition((3, 0, 0, 0)),
        0.0,
        MediaType.VIDEO,
        capability,
        policy,
        duration_seconds=60,
    )
    assert result.contributions["object_count_or_tracks_per_minute"] == 3
    assert result.contributions["count_norm"] == pytest.approx(0.1)
