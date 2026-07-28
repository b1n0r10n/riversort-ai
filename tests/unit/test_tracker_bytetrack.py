# Copyright (C) 2026 RiverSort AI contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np

from riversort.adapters import ByteTrackAdapter
from riversort.domain import Detection
from riversort.infrastructure import load_project_config


def detection(frame: int, class_id: int = 0, confidence: float = 0.9) -> Detection:
    return Detection(
        class_id=class_id,
        class_name=(
            "plastic_bottle" if class_id == 0 else "plastic_packaging_sachet"
        ),
        confidence=confidence,
        bbox_xyxy=(10.0 + frame, 10.0, 30.0 + frame, 30.0),
        frame_index=frame,
        timestamp_ms=frame * 200,
    )


def test_one_object_across_frames_becomes_one_confirmed_track() -> None:
    config = load_project_config()
    tracker = ByteTrackAdapter(config.tracker)
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    observations = []
    for index in range(5):
        observations.extend(tracker.update([detection(index)], frame))
    summaries = tracker.finalize()
    assert len(summaries) == 1
    assert summaries[0].class_id == 0
    assert summaries[0].observation_count >= 2
    assert summaries[0].class_vote_share == 1.0
    assert summaries[0].class_vote_margin == 1.0
    assert summaries[0].classification_status == "stable"
    assert len({item.track_id for item in observations}) == 1


def test_confidence_weighted_vote_and_tie_break_are_deterministic() -> None:
    config = load_project_config()
    tracker = ByteTrackAdapter(config.tracker)
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    tracker.update([detection(0, 0, 0.9)], frame)
    tracker.update([detection(1, 2, 0.6)], frame)
    tracker.update([detection(2, 0, 0.4)], frame)
    summaries = tracker.finalize()
    assert len(summaries) == 1
    assert summaries[0].class_id == 0
    assert summaries[0].confidence_weighted_votes[0] > summaries[0].confidence_weighted_votes[2]


def test_exact_vote_tie_uses_smallest_class_id() -> None:
    config = load_project_config()
    tracker = ByteTrackAdapter(config.tracker)
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    tracker.update([detection(0, 2, 0.8)], frame)
    tracker.update([detection(1, 0, 0.8)], frame)
    summary = tracker.finalize()[0]
    assert summary.confidence_weighted_votes[0] == summary.confidence_weighted_votes[2]
    assert summary.class_id == 0
    assert summary.class_vote_share == 0.5
    assert summary.class_vote_margin == 0.0
    assert summary.classification_status == "uncertain"


def test_two_crossing_objects_are_not_counted_per_frame() -> None:
    config = load_project_config()
    tracker = ByteTrackAdapter(config.tracker)
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    for index in range(6):
        left = Detection(
            0,
            "plastic_bottle",
            0.9,
            (10.0 + index * 5, 10.0, 25.0 + index * 5, 30.0),
            index,
            index * 200,
        )
        right = Detection(
            1,
            "plastic_bag",
            0.9,
            (65.0 - index * 5, 45.0, 80.0 - index * 5, 65.0),
            index,
            index * 200,
        )
        tracker.update((left, right), frame)
    summaries = tracker.finalize()
    assert len(summaries) == 2
    assert sum(item.observation_count or 0 for item in summaries) >= 4


def test_single_observation_is_not_counted() -> None:
    config = load_project_config()
    tracker = ByteTrackAdapter(config.tracker)
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    tracker.update([detection(0)], frame)
    assert tracker.finalize() == ()
