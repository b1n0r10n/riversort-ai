# Copyright (C) 2026 RiverSort AI contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import json
from pathlib import Path

import numpy as np

from riversort.adapters import ByteTrackAdapter, OpenCvAnnotator, ResultExporter
from riversort.application import FramePacket, UploadedMedia
from riversort.application.analyze_video import AnalyzeVideo
from riversort.domain import Detection, MediaInfo, MediaType
from riversort.infrastructure import load_project_config
from riversort.infrastructure.runtime import runtime_context
from riversort.infrastructure.workspace import AnalysisWorkspace


class FakeMedia:
    def probe_video(self, path, upload):
        return MediaInfo(
            display_filename="demo.mp4",
            media_type=MediaType.VIDEO,
            mime_type="video/mp4",
            byte_size=len(upload.content),
            width=100,
            height=80,
            duration_seconds=1.0,
            source_fps=5.0,
            frame_count=5,
            codec="h264",
            analysis_fps=5.0,
        )

    def iter_video_frames(self, path, media_info):
        for index in range(5):
            yield FramePacket(
                frame_index=index,
                timestamp_ms=index * 200,
                pixels=np.zeros((80, 100, 3), dtype=np.uint8),
                width=100,
                height=80,
            )

    def normalize_video(self, path, media_info, output_path):
        return path, media_info


class PartialDecodeMedia(FakeMedia):
    def probe_video(self, path, upload):
        info = super().probe_video(path, upload)
        return MediaInfo(
            **{
                **info.__dict__,
                "frame_count": 100,
            }
        )

    def iter_video_frames(self, path, media_info):
        yield next(iter(super().iter_video_frames(path, media_info)))


class FakeDetector:
    def detect_frame(self, pixels, *, confidence_floor, frame_index, timestamp_ms):
        return (
            Detection(
                0,
                "plastic_bottle",
                0.9,
                (10.0 + frame_index, 10.0, 30.0 + frame_index, 30.0),
                frame_index,
                timestamp_ms,
            ),
        )


class JitterDetector(FakeDetector):
    def detect_frame(self, pixels, *, confidence_floor, frame_index, timestamp_ms):
        class_id = 2 if frame_index == 0 else 0
        class_name = (
            "plastic_packaging_sachet"
            if class_id == 2
            else "plastic_bottle"
        )
        return (
            Detection(
                class_id,
                class_name,
                0.9,
                (10.0 + frame_index, 10.0, 30.0 + frame_index, 30.0),
                frame_index,
                timestamp_ms,
            ),
        )


class FirstFrameOnlyDetector(FakeDetector):
    def detect_frame(self, pixels, *, confidence_floor, frame_index, timestamp_ms):
        if frame_index > 0:
            return ()
        return super().detect_frame(
            pixels,
            confidence_floor=confidence_floor,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
        )


class RecordingAnnotator:
    def __init__(self):
        self.frames = []

    def annotate_video_frame(self, pixels, observations):
        self.frames.append(tuple(observations))
        return pixels


class FakeEncoder:
    def __init__(self, output_path, *, width, height, fps):
        self.frames = 0

    def open(self):
        pass

    def write(self, pixels):
        self.frames += 1

    def close(self):
        return f"video-with-{self.frames}-frames".encode()

    def abort(self):
        pass


def build_use_case(
    tmp_path: Path,
    *,
    media=None,
    detector=None,
    annotator=None,
) -> AnalyzeVideo:
    config = load_project_config()
    return AnalyzeVideo(
        runtime=runtime_context(config),
        media=media or FakeMedia(),
        detector=detector or FakeDetector(),
        tracker=ByteTrackAdapter(config.tracker),
        annotator=annotator or OpenCvAnnotator(),
        exporter=ResultExporter(),
        encoder_factory=FakeEncoder,
        workspace_factory=lambda analysis_id: AnalysisWorkspace(analysis_id, tmp_path),
        tracker_low_threshold=config.tracker.track_low_thresh,
        analysis_fps=config.tracker.analysis_fps,
        max_decode_failure_ratio=config.media.video.max_decode_failure_ratio,
        class_vote_margin_threshold=config.tracker.class_vote_margin_threshold,
        stability_minimum_tracks=config.tracker.stability_minimum_tracks,
        short_track_seconds=config.tracker.short_track_seconds,
        short_track_ratio_threshold=config.tracker.short_track_ratio_threshold,
    )


def test_video_counts_one_track_not_five_detections(tmp_path: Path) -> None:
    use_case = build_use_case(tmp_path)
    bundle = use_case.execute(
        UploadedMedia("demo.mp4", b"fake-video", "video/mp4")
    )
    wire = json.loads(bundle.json_bytes)
    assert wire["processing_summary"]["raw_detection_count"] == 5
    assert wire["unique_object_count"] == 1
    assert len(wire["track_summaries"]) == 1
    assert wire["track_summaries"][0]["class_vote_margin"] == 1.0
    assert wire["processing_summary"]["confirmed_only_coverage"] is True
    assert bundle.annotated_mime == "video/mp4"
    assert bundle.annotated_media == b"video-with-5-frames"
    assert list(tmp_path.iterdir()) == []


def test_partial_decode_adds_measured_quality_flag(tmp_path: Path) -> None:
    use_case = build_use_case(tmp_path, media=PartialDecodeMedia())
    bundle = use_case.execute(
        UploadedMedia("demo.mp4", b"fake-video", "video/mp4")
    )
    assert any(
        flag.code.value == "LOW_ANALYSIS_QUALITY"
        for flag in bundle.result.quality_flags
    )
    assert bundle.result.processing_summary["decode_failure_ratio"] > 0.05
    assert list(tmp_path.iterdir()) == []


def test_second_pass_preview_uses_final_track_class_for_every_frame(
    tmp_path: Path,
) -> None:
    annotator = RecordingAnnotator()
    use_case = build_use_case(
        tmp_path,
        detector=JitterDetector(),
        annotator=annotator,
    )
    bundle = use_case.execute(UploadedMedia("demo.mp4", b"video", "video/mp4"))
    assert bundle.result.track_summaries[0].class_name == "plastic_bottle"
    rendered = [
        observation
        for frame in annotator.frames
        for observation in frame
    ]
    assert rendered
    assert {item.detection.class_name for item in rendered} == {"plastic_bottle"}
    assert {item.classification_status for item in rendered} == {"stable"}


def test_unconfirmed_track_is_absent_from_preview_count_and_coverage(
    tmp_path: Path,
) -> None:
    annotator = RecordingAnnotator()
    use_case = build_use_case(
        tmp_path,
        detector=FirstFrameOnlyDetector(),
        annotator=annotator,
    )
    bundle = use_case.execute(UploadedMedia("demo.mp4", b"video", "video/mp4"))
    assert bundle.result.unique_object_count == 0
    assert bundle.result.coverage_p90 == 0.0
    assert all(not observations for observations in annotator.frames)
