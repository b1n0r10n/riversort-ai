# Copyright (C) 2026 RiverSort AI contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ByteTrack adapter with deterministic class voting and unique-track summaries."""

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker

from riversort.application import AnalysisError, TrackObservation
from riversort.domain import AnalysisErrorCode, Detection, TrackSummary
from riversort.infrastructure.config import TrackerConfig


class ByteTrackAdapter:
    def __init__(self, config: TrackerConfig) -> None:
        self.config = config
        self._tracker = self._new_tracker()
        self._votes: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self._names: dict[int, dict[int, str]] = defaultdict(dict)
        self._areas: dict[int, list[float]] = defaultdict(list)
        self._first: dict[int, int] = {}
        self._last: dict[int, int] = {}
        self._observations: dict[int, int] = defaultdict(int)

    def _new_tracker(self) -> BYTETracker:
        args = SimpleNamespace(
            track_high_thresh=self.config.track_high_thresh,
            track_low_thresh=self.config.track_low_thresh,
            new_track_thresh=self.config.new_track_thresh,
            track_buffer=self.config.track_buffer,
            match_thresh=self.config.match_thresh,
            fuse_score=self.config.fuse_score,
        )
        return BYTETracker(args, frame_rate=self.config.analysis_fps)

    def reset(self) -> None:
        self._tracker.reset()
        self._votes.clear()
        self._names.clear()
        self._areas.clear()
        self._first.clear()
        self._last.clear()
        self._observations.clear()

    def update(
        self, detections: Sequence[Detection], pixels: Any
    ) -> tuple[TrackObservation, ...]:
        height, width = pixels.shape[:2]
        rows = np.asarray(
            [
                [*item.bbox_xyxy, item.confidence, float(item.class_id)]
                for item in detections
            ],
            dtype=np.float32,
        )
        if rows.size == 0:
            rows = np.empty((0, 6), dtype=np.float32)
        boxes = Boxes(rows, (height, width))
        try:
            tracked = self._tracker.update(boxes, pixels)
        except Exception as error:
            raise AnalysisError(
                AnalysisErrorCode.TRACKING_FAILED,
                "ByteTrack gagal memperbarui track.",
                "Coba video lain atau periksa tracker config.",
            ) from error
        observations = []
        for row in tracked:
            if len(row) < 8:
                continue
            x1, y1, x2, y2, raw_track_id, score, raw_class_id, raw_index = row[:8]
            index = int(raw_index)
            if not 0 <= index < len(detections):
                continue
            matched = detections[index]
            track_id = int(raw_track_id)
            class_id = matched.class_id
            self._votes[track_id][class_id] += matched.confidence
            self._names[track_id][class_id] = matched.class_name
            normalized_area = max(0.0, float(x2 - x1)) * max(
                0.0, float(y2 - y1)
            ) / float(width * height)
            self._areas[track_id].append(min(normalized_area, 1.0))
            frame_index = matched.frame_index if matched.frame_index is not None else 0
            self._first.setdefault(track_id, frame_index)
            self._last[track_id] = frame_index
            self._observations[track_id] += 1
            observations.append(
                TrackObservation(
                    track_id=track_id,
                    detection=Detection(
                        class_id=class_id,
                        class_name=matched.class_name,
                        confidence=float(score),
                        bbox_xyxy=(
                            float(x1),
                            float(y1),
                            float(x2),
                            float(y2),
                        ),
                        frame_index=matched.frame_index,
                        timestamp_ms=matched.timestamp_ms,
                    ),
                )
            )
        return tuple(observations)

    def finalize(self) -> tuple[TrackSummary, ...]:
        summaries = []
        for track_id in sorted(self._votes):
            observation_count = self._observations[track_id]
            if observation_count < self.config.minimum_observations:
                continue
            votes = dict(self._votes[track_id])
            class_id = min(votes, key=lambda item: (-votes[item], item))
            ordered_votes = sorted(votes.values(), reverse=True)
            total_vote = sum(ordered_votes)
            winning_vote = ordered_votes[0]
            runner_up_vote = ordered_votes[1] if len(ordered_votes) > 1 else 0.0
            vote_share = winning_vote / total_vote
            vote_margin = (winning_vote - runner_up_vote) / total_vote
            summaries.append(
                TrackSummary(
                    track_id=track_id,
                    class_id=class_id,
                    class_name=self._names[track_id][class_id],
                    confirmed=True,
                    confidence_weighted_votes=votes,
                    normalized_bbox_areas=tuple(self._areas[track_id]),
                    first_frame=self._first[track_id],
                    last_frame=self._last[track_id],
                    observation_count=observation_count,
                    class_vote_share=vote_share,
                    class_vote_margin=vote_margin,
                    classification_status=(
                        "stable"
                        if vote_margin >= self.config.class_vote_margin_threshold
                        else "uncertain"
                    ),
                )
            )
        return tuple(summaries)
