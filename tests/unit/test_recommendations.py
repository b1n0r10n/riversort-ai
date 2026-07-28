# Copyright (C) 2026 RiverSort AI contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest

from riversort.domain import CompositionMetric, SupportStatus, evaluate_recommendations
from riversort.infrastructure.config import capability_profile, load_project_config, recommendation_policy


def rows(shares: tuple[float, float, float, float] | None):
    names = ("plastic_bottle", "plastic_bag", "plastic_packaging_sachet", "other_plastic")
    active = tuple(
        CompositionMetric(
            class_id,
            name,
            SupportStatus.SUPPORTED,
            0 if shares is None else 1,
            None if shares is None else shares[class_id],
            None if shares is None else shares[class_id],
        )
        for class_id, name in enumerate(names)
    )
    return active + (
        CompositionMetric(4, "other_floating_debris", SupportStatus.NOT_SUPPORTED, None, None, None),
    )


def engine(compositions):
    config = load_project_config()
    return evaluate_recommendations(
        compositions,
        capability_profile(config),
        recommendation_policy(config),
    )


@pytest.mark.parametrize(
    ("shares", "expected"),
    [
        ((0.30, 0.35, 0.20, 0.15), ()),
        ((0.29, 0.36, 0.20, 0.15), ("R_BAG_FINE_NET",)),
        ((0.31, 0.36, 0.32, 0.31), (
            "R_BAG_FINE_NET",
            "R_BOTTLE_SORT_PET",
            "R_SACHET_SMALL_MESH",
            "R_OTHER_PLASTIC_MANUAL_SORT",
        )),
    ],
)
def test_strict_thresholds_and_deterministic_multi_rule_order(shares, expected) -> None:
    assert tuple(item.rule_id for item in engine(rows(shares))) == expected


def test_no_assessment_rule_has_traceable_evidence() -> None:
    result = engine(rows(None))
    assert len(result) == 1
    recommendation = result[0]
    assert recommendation.rule_id == "R_NO_ASSESSMENT"
    assert recommendation.observed_value == 0
    assert recommendation.operator == "=="
    assert recommendation.threshold == 0
    assert recommendation.rule_version == "1.1.0"
    assert recommendation.evidence_fields == ("composition.supported_object_count",)


def test_residual_rule_is_not_applicable_to_mvp() -> None:
    assert "R_DEBRIS_MANUAL_REVIEW" not in {item.rule_id for item in engine(rows((0.25,) * 4))}


@pytest.mark.parametrize(
    ("class_id", "threshold", "rule_id"),
    [
        (0, 0.30, "R_BOTTLE_SORT_PET"),
        (1, 0.35, "R_BAG_FINE_NET"),
        (2, 0.30, "R_SACHET_SMALL_MESH"),
        (3, 0.30, "R_OTHER_PLASTIC_MANUAL_SORT"),
    ],
)
def test_every_active_class_rule_is_strictly_greater_than_threshold(
    class_id: int,
    threshold: float,
    rule_id: str,
) -> None:
    for observed, expected_trigger in (
        (threshold - 0.01, False),
        (threshold, False),
        (threshold + 0.01, True),
    ):
        shares = [0.0, 0.0, 0.0, 0.0]
        shares[class_id] = observed
        shares[(class_id + 1) % 4] = 1.0 - observed
        triggered = {item.rule_id for item in engine(rows(tuple(shares)))}
        assert (rule_id in triggered) is expected_trigger
