# Copyright (C) 2026 RiverSort AI contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Deterministic composition and coverage calculations."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from statistics import median
from typing import Iterable, Sequence

from .contracts import (
    CompositionMetric,
    Detection,
    ModelCapabilityProfile,
    SupportStatus,
    TrackSummary,
)
from .errors import DomainInputError


FLOAT_TOLERANCE = 1e-9


def _taxonomy(capability: ModelCapabilityProfile) -> list[tuple[int, str, SupportStatus]]:
    overlap = set(capability.supported_classes) & set(capability.unsupported_target_classes)
    if overlap:
        raise DomainInputError(f"Capability classes overlap: {sorted(overlap)}")
    rows = [
        (class_id, name, SupportStatus.SUPPORTED)
        for class_id, name in capability.supported_classes.items()
    ] + [
        (class_id, name, SupportStatus.NOT_SUPPORTED)
        for class_id, name in capability.unsupported_target_classes.items()
    ]
    return sorted(rows)


def _bbox_area(box: tuple[float, float, float, float]) -> float | None:
    if len(box) != 4 or not all(math.isfinite(value) for value in box):
        raise DomainInputError("Bounding box coordinates must be finite xyxy values")
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return None
    return (x2 - x1) * (y2 - y1)


def _build_metrics(
    capability: ModelCapabilityProfile,
    counts: Counter[int],
    areas: dict[int, float],
) -> tuple[CompositionMetric, ...]:
    total_count = sum(counts.values())
    total_area = sum(areas.values())
    metrics = []
    for class_id, class_name, support_status in _taxonomy(capability):
        if support_status is SupportStatus.NOT_SUPPORTED:
            metrics.append(
                CompositionMetric(class_id, class_name, support_status, None, None, None)
            )
            continue
        count = counts[class_id]
        metrics.append(
            CompositionMetric(
                class_id=class_id,
                class_name=class_name,
                support_status=support_status,
                object_count=count,
                object_count_share=(count / total_count) if total_count else None,
                detected_area_share=(areas[class_id] / total_area) if total_area else None,
            )
        )
    _assert_share_sum(metrics, "object_count_share")
    _assert_share_sum(metrics, "detected_area_share")
    return tuple(metrics)


def _assert_share_sum(metrics: Sequence[CompositionMetric], field: str) -> None:
    values = [
        getattr(metric, field)
        for metric in metrics
        if metric.support_status is SupportStatus.SUPPORTED and getattr(metric, field) is not None
    ]
    if values and not math.isclose(sum(values), 1.0, abs_tol=FLOAT_TOLERANCE):
        raise ArithmeticError(f"{field} does not sum to one")


def compose_image(
    detections: Iterable[Detection],
    capability: ModelCapabilityProfile,
) -> tuple[CompositionMetric, ...]:
    """Calculate supported-class count share and bbox-area share for one image."""

    counts: Counter[int] = Counter()
    areas: dict[int, float] = defaultdict(float)
    for detection in detections:
        if detection.class_id in capability.unsupported_target_classes:
            continue
        expected_name = capability.supported_classes.get(detection.class_id)
        if expected_name is None:
            raise DomainInputError(f"Unknown detection class ID {detection.class_id}")
        if detection.class_name != expected_name:
            raise DomainInputError(f"Class name mismatch for ID {detection.class_id}")
        if not math.isfinite(detection.confidence) or not 0 <= detection.confidence <= 1:
            raise DomainInputError("Detection confidence must be between zero and one")
        area = _bbox_area(detection.bbox_xyxy)
        if area is None:
            continue
        counts[detection.class_id] += 1
        areas[detection.class_id] += area
    return _build_metrics(capability, counts, areas)


def compose_video(
    track_summaries: Iterable[TrackSummary],
    capability: ModelCapabilityProfile,
) -> tuple[CompositionMetric, ...]:
    """Calculate composition from confirmed unique tracks, once per track ID."""

    counts: Counter[int] = Counter()
    areas: dict[int, float] = defaultdict(float)
    seen_track_ids: set[int] = set()
    for track in track_summaries:
        if not track.confirmed:
            continue
        if track.track_id in seen_track_ids:
            raise DomainInputError(f"Duplicate confirmed track ID {track.track_id}")
        seen_track_ids.add(track.track_id)
        if track.class_id in capability.unsupported_target_classes:
            continue
        expected_name = capability.supported_classes.get(track.class_id)
        if expected_name is None:
            raise DomainInputError(f"Unknown track class ID {track.class_id}")
        if track.class_name != expected_name:
            raise DomainInputError(f"Class name mismatch for ID {track.class_id}")
        if track.last_frame < track.first_frame:
            raise DomainInputError("Track frame range is reversed")
        if any(not math.isfinite(value) or value < 0 or value > 1 for value in track.normalized_bbox_areas):
            raise DomainInputError("Normalized track bbox area must be finite in range 0..1")
        counts[track.class_id] += 1
        valid_areas = [value for value in track.normalized_bbox_areas if value > 0]
        if valid_areas:
            areas[track.class_id] += median(valid_areas)
    return _build_metrics(capability, counts, areas)


def union_bbox_coverage(
    boxes: Iterable[tuple[float, float, float, float]],
    frame_size: tuple[int, int],
) -> float:
    """Return exact rectangle-union coverage after clamping boxes to the frame."""

    width, height = frame_size
    if width <= 0 or height <= 0:
        raise DomainInputError("Frame width and height must be positive")
    rectangles = []
    for box in boxes:
        if len(box) != 4 or not all(math.isfinite(value) for value in box):
            raise DomainInputError("Bounding box coordinates must be finite xyxy values")
        x1, y1, x2, y2 = box
        clamped = (
            max(0.0, min(float(width), x1)),
            max(0.0, min(float(height), y1)),
            max(0.0, min(float(width), x2)),
            max(0.0, min(float(height), y2)),
        )
        if clamped[2] > clamped[0] and clamped[3] > clamped[1]:
            rectangles.append(clamped)
    if not rectangles:
        return 0.0

    x_coordinates = sorted({value for rectangle in rectangles for value in (rectangle[0], rectangle[2])})
    union_area = 0.0
    for left, right in zip(x_coordinates, x_coordinates[1:]):
        if right <= left:
            continue
        intervals = sorted(
            (y1, y2)
            for x1, y1, x2, y2 in rectangles
            if x1 < right and x2 > left
        )
        covered_y = 0.0
        if intervals:
            start, end = intervals[0]
            for next_start, next_end in intervals[1:]:
                if next_start <= end:
                    end = max(end, next_end)
                else:
                    covered_y += end - start
                    start, end = next_start, next_end
            covered_y += end - start
        union_area += (right - left) * covered_y
    return min(union_area / (width * height), 1.0)


def percentile90(values: Iterable[float]) -> float:
    """Compute the Type-7 90th percentile without NumPy."""

    ordered = sorted(values)
    if not ordered:
        return 0.0
    if any(not math.isfinite(value) or value < 0 or value > 1 for value in ordered):
        raise DomainInputError("Coverage values must be finite ratios in range 0..1")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * 0.90
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction
