# Copyright (C) 2026 RiverSort AI contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import math

import pytest

from riversort.domain import (
    Detection,
    DomainInputError,
    ModelCapabilityProfile,
    SupportStatus,
    TrackSummary,
    compose_image,
    compose_video,
    percentile90,
    union_bbox_coverage,
)


def capability() -> ModelCapabilityProfile:
    return ModelCapabilityProfile(
        "riversort_plastic4_mvp",
        "1.0.0",
        "1.0.0",
        {0: "plastic_bottle", 1: "plastic_bag", 2: "plastic_packaging_sachet", 3: "other_plastic"},
        {4: "other_floating_debris"},
    )


def test_image_composition_separates_count_and_area_share() -> None:
    metrics = compose_image(
        [
            Detection(0, "plastic_bottle", 0.9, (0, 0, 10, 10)),
            Detection(0, "plastic_bottle", 0.8, (10, 0, 20, 10)),
            Detection(1, "plastic_bag", 0.7, (0, 0, 20, 10)),
            Detection(3, "other_plastic", 0.6, (2, 2, 2, 3)),
            Detection(4, "other_floating_debris", 0.9, (0, 0, 100, 100)),
        ],
        capability(),
    )
    by_id = {metric.class_id: metric for metric in metrics}
    assert len(metrics) == 5
    assert by_id[0].object_count == 2
    assert by_id[0].object_count_share == pytest.approx(2 / 3)
    assert by_id[0].detected_area_share == pytest.approx(0.5)
    assert by_id[1].object_count_share == pytest.approx(1 / 3)
    assert by_id[1].detected_area_share == pytest.approx(0.5)
    assert by_id[4].support_status is SupportStatus.NOT_SUPPORTED
    assert by_id[4].object_count is None
    assert by_id[4].object_count_share is None
    assert by_id[4].detected_area_share is None
    assert sum(item.object_count_share or 0 for item in metrics) == pytest.approx(1.0)
    assert sum(item.detected_area_share or 0 for item in metrics) == pytest.approx(1.0)


def test_empty_image_has_supported_zero_counts_and_null_shares() -> None:
    metrics = compose_image([], capability())
    for metric in metrics[:4]:
        assert metric.object_count == 0
        assert metric.object_count_share is None
        assert metric.detected_area_share is None


def test_video_uses_confirmed_unique_track_and_median_area() -> None:
    metrics = compose_video(
        [
            TrackSummary(10, 0, "plastic_bottle", True, {0: 2.0}, (0.1, 0.3, 0.2), 0, 8),
            TrackSummary(11, 1, "plastic_bag", True, {1: 1.0}, (0.1,), 1, 3),
            TrackSummary(12, 1, "plastic_bag", False, {1: 0.5}, (0.9,), 1, 2),
        ],
        capability(),
    )
    by_id = {metric.class_id: metric for metric in metrics}
    assert by_id[0].object_count == 1
    assert by_id[1].object_count == 1
    assert by_id[0].detected_area_share == pytest.approx(2 / 3)
    assert by_id[1].detected_area_share == pytest.approx(1 / 3)


def test_video_rejects_duplicate_confirmed_track_id() -> None:
    track = TrackSummary(10, 0, "plastic_bottle", True, {0: 1.0}, (0.1,), 0, 1)
    with pytest.raises(DomainInputError, match="Duplicate confirmed track ID"):
        compose_video([track, track], capability())


def test_union_coverage_clamps_edges_and_does_not_double_count_overlap() -> None:
    assert union_bbox_coverage([(0, 0, 6, 10), (4, 0, 10, 10)], (10, 10)) == 1.0
    assert union_bbox_coverage([(-5, -5, 5, 5)], (10, 10)) == pytest.approx(0.25)
    assert union_bbox_coverage([], (10, 10)) == 0.0


def test_percentile90_uses_type7_linear_interpolation() -> None:
    assert percentile90([]) == 0.0
    assert percentile90([0, 0.25, 0.5, 0.75, 1.0]) == pytest.approx(0.9)
    assert math.isclose(percentile90([0.4]), 0.4)
