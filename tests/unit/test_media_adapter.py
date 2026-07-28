# Copyright (C) 2026 RiverSort AI contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from riversort.adapters import OpenCvMediaAdapter, resolve_binary
from riversort.application import AnalysisError, UploadedMedia
from riversort.domain import AnalysisErrorCode
from riversort.infrastructure import load_project_config


def png_bytes(width: int = 16, height: int = 12) -> bytes:
    stream = BytesIO()
    Image.new("RGB", (width, height), (20, 80, 120)).save(stream, format="PNG")
    return stream.getvalue()


def test_valid_png_is_decoded_and_reported(tmp_path: Path) -> None:
    config = load_project_config()
    adapter = OpenCvMediaAdapter(config.media, config.tracker)
    content = png_bytes()
    path = tmp_path / "media.png"
    path.write_bytes(content)
    validated = adapter.validate_image(
        path, UploadedMedia("river.png", content, "image/png")
    )
    assert validated.media_info.width == 16
    assert validated.media_info.height == 12
    assert validated.media_info.mime_type == "image/png"
    assert validated.pixels.shape == (12, 16, 3)


def test_spoofed_image_extension_is_rejected(tmp_path: Path) -> None:
    config = load_project_config()
    adapter = OpenCvMediaAdapter(config.media, config.tracker)
    content = png_bytes()
    path = tmp_path / "media.jpg"
    path.write_bytes(content)
    with pytest.raises(AnalysisError) as error:
        adapter.validate_image(
            path, UploadedMedia("river.jpg", content, "image/jpeg")
        )
    assert error.value.code is AnalysisErrorCode.INVALID_MEDIA_TYPE


def test_empty_or_corrupt_image_is_rejected(tmp_path: Path) -> None:
    config = load_project_config()
    adapter = OpenCvMediaAdapter(config.media, config.tracker)
    path = tmp_path / "media.png"
    path.write_bytes(b"not-an-image")
    with pytest.raises(AnalysisError) as error:
        adapter.validate_image(
            path, UploadedMedia("river.png", b"not-an-image", "image/png")
        )
    assert error.value.code is AnalysisErrorCode.DECODE_FAILED


def test_image_byte_limit_is_enforced_and_target_pixels_are_normalized(
    tmp_path: Path,
) -> None:
    config = load_project_config()
    content = png_bytes(16, 12)
    path = tmp_path / "media.png"
    path.write_bytes(content)

    byte_config = config.media.model_copy(
        update={
            "image": config.media.image.model_copy(
                update={"max_bytes": len(content) - 1}
            )
        }
    )
    with pytest.raises(AnalysisError) as byte_error:
        OpenCvMediaAdapter(byte_config, config.tracker).validate_image(
            path, UploadedMedia("river.png", content, "image/png")
        )
    assert byte_error.value.code is AnalysisErrorCode.FILE_TOO_LARGE

    pixel_config = config.media.model_copy(
        update={
            "image": config.media.image.model_copy(
                update={"max_pixels": 100, "max_source_pixels": 1_000}
            )
        }
    )
    validated = OpenCvMediaAdapter(pixel_config, config.tracker).validate_image(
        path, UploadedMedia("river.png", content, "image/png")
    )
    assert validated.media_info.resolution_normalized is True
    assert validated.media_info.original_width == 16
    assert validated.media_info.original_height == 12
    assert validated.media_info.width * validated.media_info.height <= 100
    assert validated.pixels.shape[:2] == (
        validated.media_info.height,
        validated.media_info.width,
    )


def test_image_above_source_safety_limit_is_rejected(tmp_path: Path) -> None:
    config = load_project_config()
    content = png_bytes(16, 12)
    path = tmp_path / "media.png"
    path.write_bytes(content)
    safety_config = config.media.model_copy(
        update={
            "image": config.media.image.model_copy(
                update={"max_pixels": 100, "max_source_pixels": 100}
            )
        }
    )
    with pytest.raises(AnalysisError) as error:
        OpenCvMediaAdapter(safety_config, config.tracker).validate_image(
            path, UploadedMedia("river.png", content, "image/png")
        )
    assert error.value.code is AnalysisErrorCode.IMAGE_TOO_LARGE


def test_ffmpeg_resolver_rejects_empty_wrapper_and_finds_system_binary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RIVERSORT_FFMPEG_PATH", raising=False)
    assert resolve_binary(
        "RIVERSORT_FFMPEG_PATH", "ffmpeg", "ffmpeg version", "/usr/bin/ffmpeg"
    ) == "/usr/bin/ffmpeg"


def test_binary_resolver_fails_safely_when_no_candidate_is_valid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RIVERSORT_TEST_BINARY", "/usr/local/bin/ffmpeg")
    with pytest.raises(AnalysisError) as error:
        resolve_binary(
            "RIVERSORT_TEST_BINARY",
            "definitely-not-a-riversort-command",
            "ffmpeg version",
            "/missing/ffmpeg",
        )
    assert error.value.code is AnalysisErrorCode.ENCODER_UNAVAILABLE
