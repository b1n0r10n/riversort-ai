# Copyright (C) 2026 RiverSort AI contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Framework-neutral ports implemented by the active runtime adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

from riversort.domain.contracts import Detection, MediaInfo, TrackSummary

from .models import FramePacket, TrackObservation, UploadedMedia, ValidatedImage


class DetectorPort(Protocol):
    def detect_image(self, pixels: Any) -> Sequence[Detection]: ...

    def detect_frame(
        self,
        pixels: Any,
        *,
        confidence_floor: float,
        frame_index: int,
        timestamp_ms: int,
    ) -> Sequence[Detection]: ...


class MediaPort(Protocol):
    def validate_image(self, path: Path, upload: UploadedMedia) -> ValidatedImage: ...

    def probe_video(self, path: Path, upload: UploadedMedia) -> MediaInfo: ...

    def normalize_video(
        self, path: Path, media_info: MediaInfo, output_path: Path
    ) -> tuple[Path, MediaInfo]: ...

    def iter_video_frames(self, path: Path, media_info: MediaInfo) -> Iterable[FramePacket]: ...


class TrackerPort(Protocol):
    def reset(self) -> None: ...

    def update(
        self, detections: Sequence[Detection], pixels: Any
    ) -> Sequence[TrackObservation]: ...

    def finalize(self) -> Sequence[TrackSummary]: ...


class AnnotatorPort(Protocol):
    def annotate_image(self, pixels: Any, detections: Sequence[Detection]) -> bytes: ...

    def annotate_video_frame(
        self, pixels: Any, observations: Sequence[TrackObservation]
    ) -> Any: ...


class ExporterPort(Protocol):
    def to_json(self, result: Any) -> bytes: ...

    def to_csv(self, result: Any) -> bytes: ...


class WorkspacePort(Protocol):
    path: Path

    def __enter__(self) -> "WorkspacePort": ...

    def save_upload(self, upload: UploadedMedia) -> tuple[Path, str]: ...

    def allocate(self, filename: str) -> Path: ...

    def __exit__(self, exc_type, exc_value, traceback) -> None: ...
