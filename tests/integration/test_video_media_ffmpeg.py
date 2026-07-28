# Copyright (C) 2026 RiverSort AI contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import subprocess
from pathlib import Path

import numpy as np
import pytest

from riversort.adapters import FfmpegVideoEncoder, OpenCvMediaAdapter
from riversort.application import AnalysisError, UploadedMedia
from riversort.domain import AnalysisErrorCode
from riversort.infrastructure import load_project_config


FFMPEG = Path("/usr/bin/ffmpeg")


@pytest.mark.skipif(not FFMPEG.is_file(), reason="requires system ffmpeg")
def test_h264_probe_and_low_fps_sampling_do_not_duplicate_frames(
    tmp_path: Path,
) -> None:
    video = tmp_path / "source.mp4"
    subprocess.run(
        [
            str(FFMPEG),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=160x120:r=2:d=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
    )
    content = video.read_bytes()
    config = load_project_config()
    adapter = OpenCvMediaAdapter(config.media, config.tracker)
    info = adapter.probe_video(
        video, UploadedMedia("source.mp4", content, "video/mp4")
    )
    frames = list(adapter.iter_video_frames(video, info))
    assert info.codec == "h264"
    assert info.analysis_fps == 2.0
    assert len(frames) == 2
    assert [item.frame_index for item in frames] == [0, 1]


@pytest.mark.skipif(not FFMPEG.is_file(), reason="requires system ffmpeg")
def test_encoder_creates_browser_compatible_h264_preview(tmp_path: Path) -> None:
    output = tmp_path / "annotated.mp4"
    encoder = FfmpegVideoEncoder(output, width=160, height=120, fps=2.0)
    encoder.open()
    encoder.write(np.zeros((120, 160, 3), dtype=np.uint8))
    encoder.write(np.full((120, 160, 3), 127, dtype=np.uint8))
    content = encoder.close()

    assert content
    config = load_project_config()
    adapter = OpenCvMediaAdapter(config.media, config.tracker)
    info = adapter.probe_video(
        output, UploadedMedia("annotated.mp4", content, "video/mp4")
    )
    assert info.codec == "h264"
    assert info.width == 160
    assert info.height == 120


@pytest.mark.skipif(not FFMPEG.is_file(), reason="requires system ffmpeg")
def test_high_resolution_video_is_downscaled_before_frame_sampling(
    tmp_path: Path,
) -> None:
    video = tmp_path / "high-resolution.mp4"
    subprocess.run(
        [
            str(FFMPEG),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=yellow:s=320x240:r=2:d=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
    )
    content = video.read_bytes()
    config = load_project_config()
    media_config = config.media.model_copy(
        update={
            "video": config.media.video.model_copy(
                update={"max_width": 192, "max_height": 108}
            )
        }
    )
    adapter = OpenCvMediaAdapter(media_config, config.tracker)
    source_info = adapter.probe_video(
        video, UploadedMedia("high-resolution.mp4", content, "video/mp4")
    )
    normalized_path, normalized_info = adapter.normalize_video(
        video, source_info, tmp_path / "normalized.mp4"
    )

    assert normalized_path != video
    assert normalized_path.is_file()
    assert normalized_info.resolution_normalized is True
    assert normalized_info.original_width == 320
    assert normalized_info.original_height == 240
    assert normalized_info.width == 144
    assert normalized_info.height == 108
    frames = list(adapter.iter_video_frames(normalized_path, normalized_info))
    assert len(frames) == 2
    assert all(item.width == 144 and item.height == 108 for item in frames)


@pytest.mark.skipif(not FFMPEG.is_file(), reason="requires system ffmpeg")
def test_non_h264_video_is_rejected(tmp_path: Path) -> None:
    video = tmp_path / "mpeg4.mp4"
    subprocess.run(
        [
            str(FFMPEG),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=64x64:r=1:d=1",
            "-c:v",
            "mpeg4",
            str(video),
        ],
        check=True,
    )
    content = video.read_bytes()
    config = load_project_config()
    with pytest.raises(AnalysisError) as error:
        OpenCvMediaAdapter(config.media, config.tracker).probe_video(
            video, UploadedMedia("mpeg4.mp4", content, "video/mp4")
        )
    assert error.value.code is AnalysisErrorCode.UNSUPPORTED_CODEC


@pytest.mark.skipif(not FFMPEG.is_file(), reason="requires system ffmpeg")
def test_video_duration_and_byte_limits_are_rejected(tmp_path: Path) -> None:
    video = tmp_path / "long.mp4"
    subprocess.run(
        [
            str(FFMPEG),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=green:s=64x64:r=1:d=61",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
    )
    content = video.read_bytes()
    config = load_project_config()
    adapter = OpenCvMediaAdapter(config.media, config.tracker)
    with pytest.raises(AnalysisError) as duration_error:
        adapter.probe_video(
            video, UploadedMedia("long.mp4", content, "video/mp4")
        )
    assert duration_error.value.code is AnalysisErrorCode.VIDEO_TOO_LONG

    byte_config = config.media.model_copy(
        update={
            "video": config.media.video.model_copy(
                update={"max_bytes": len(content) - 1}
            )
        }
    )
    with pytest.raises(AnalysisError) as byte_error:
        OpenCvMediaAdapter(byte_config, config.tracker).probe_video(
            video, UploadedMedia("long.mp4", content, "video/mp4")
        )
    assert byte_error.value.code is AnalysisErrorCode.FILE_TOO_LARGE
