# Copyright (C) 2026 RiverSort AI contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Local image/video validation and chronological frame decoding."""

from __future__ import annotations

import json
import math
import os
import subprocess
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from shutil import which
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from riversort.application import (
    AnalysisError,
    FramePacket,
    UploadedMedia,
    ValidatedImage,
)
from riversort.domain import AnalysisErrorCode, MediaInfo, MediaType
from riversort.infrastructure.config import MediaConfig, TrackerConfig
from riversort.infrastructure.workspace import sanitize_filename


def resolve_binary(
    environment_name: str,
    command_name: str,
    expected_marker: str,
    fallback: str,
) -> str:
    candidates = [os.environ.get(environment_name), which(command_name), fallback]
    checked: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in checked:
            continue
        checked.add(candidate)
        path = Path(candidate)
        if not path.is_file() or not os.access(path, os.X_OK):
            continue
        try:
            process = subprocess.run(
                [str(path), "-version"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        output = f"{process.stdout}\n{process.stderr}".lower()
        if process.returncode == 0 and expected_marker.lower() in output:
            return str(path)
    raise AnalysisError(
        AnalysisErrorCode.ENCODER_UNAVAILABLE,
        f"{command_name} lokal tidak tersedia atau tidak valid.",
        f"Atur {environment_name} ke binary {command_name} yang dapat dijalankan.",
    )


class OpenCvMediaAdapter:
    def __init__(self, media_config: MediaConfig, tracker_config: TrackerConfig) -> None:
        self.config = media_config
        self.tracker = tracker_config

    def validate_image(self, path: Path, upload: UploadedMedia) -> ValidatedImage:
        display_name = sanitize_filename(upload.filename)
        extension = Path(display_name).suffix.lower()
        image_config = self.config.image
        if extension not in image_config.allowed_extensions:
            self._invalid_type("Format gambar harus JPEG atau PNG.")
        if len(upload.content) > image_config.max_bytes:
            raise AnalysisError(
                AnalysisErrorCode.FILE_TOO_LARGE,
                "Ukuran gambar melewati batas 20 MB.",
                "Kompres atau perkecil gambar lalu unggah kembali.",
            )
        if upload.declared_mime and upload.declared_mime not in image_config.allowed_mime_types:
            self._invalid_type("MIME gambar tidak sesuai JPEG/PNG.")
        try:
            with Image.open(path) as source:
                detected_mime = Image.MIME.get(source.format or "")
                source_width, source_height = source.size
                if detected_mime not in image_config.allowed_mime_types:
                    self._invalid_type("Isi file bukan JPEG/PNG yang didukung.")
                if upload.declared_mime and upload.declared_mime != detected_mime:
                    self._invalid_type("Extension, MIME, dan isi gambar tidak cocok.")
                source_pixels = source_width * source_height
                if source_width <= 0 or source_height <= 0:
                    raise AnalysisError(
                        AnalysisErrorCode.DECODE_FAILED,
                        "Dimensi gambar tidak valid.",
                        "Gunakan JPEG/PNG valid dan coba kembali.",
                    )
                if source_pixels > image_config.max_source_pixels:
                    raise AnalysisError(
                        AnalysisErrorCode.IMAGE_TOO_LARGE,
                        "Resolusi sumber gambar melewati batas keamanan 80 megapixel.",
                        "Gunakan gambar maksimal 80 megapixel lalu unggah kembali.",
                    )
                oriented = ImageOps.exif_transpose(source).convert("RGB")
                original_width, original_height = oriented.size
                resolution_normalized = (
                    original_width * original_height > image_config.max_pixels
                )
                if resolution_normalized:
                    target_width, target_height = self._fit_image_dimensions(
                        original_width,
                        original_height,
                        image_config.max_pixels,
                    )
                    oriented = oriented.resize(
                        (target_width, target_height),
                        Image.Resampling.LANCZOS,
                    )
                pixels = cv2.cvtColor(np.asarray(oriented), cv2.COLOR_RGB2BGR)
        except AnalysisError:
            raise
        except Image.DecompressionBombError as error:
            raise AnalysisError(
                AnalysisErrorCode.IMAGE_TOO_LARGE,
                "Resolusi sumber gambar melewati batas keamanan decoder.",
                "Gunakan gambar maksimal 80 megapixel lalu unggah kembali.",
            ) from error
        except (UnidentifiedImageError, OSError, ValueError) as error:
            raise AnalysisError(
                AnalysisErrorCode.DECODE_FAILED,
                "Gambar rusak atau tidak dapat dibaca.",
                "Gunakan JPEG/PNG valid dan coba kembali.",
            ) from error
        height, width = pixels.shape[:2]
        return ValidatedImage(
            MediaInfo(
                display_filename=display_name,
                media_type=MediaType.IMAGE,
                mime_type=detected_mime,
                byte_size=len(upload.content),
                width=width,
                height=height,
                original_width=original_width,
                original_height=original_height,
                resolution_normalized=resolution_normalized,
            ),
            pixels,
        )

    def probe_video(self, path: Path, upload: UploadedMedia) -> MediaInfo:
        display_name = sanitize_filename(upload.filename)
        extension = Path(display_name).suffix.lower()
        video_config = self.config.video
        if extension not in video_config.allowed_extensions:
            self._invalid_type("Format video harus MP4.")
        if len(upload.content) > video_config.max_bytes:
            raise AnalysisError(
                AnalysisErrorCode.FILE_TOO_LARGE,
                "Ukuran video melewati batas 200 MB.",
                "Kompres atau potong video lalu unggah kembali.",
            )
        if upload.declared_mime and upload.declared_mime not in video_config.allowed_mime_types:
            self._invalid_type("MIME video harus video/mp4.")
        ffprobe = resolve_binary(
            "RIVERSORT_FFPROBE_PATH", "ffprobe", "ffprobe version", "/usr/bin/ffprobe"
        )
        command = [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,r_frame_rate,nb_frames,duration:format=duration",
            "-of",
            "json",
            str(path),
        ]
        try:
            process = subprocess.run(
                command, capture_output=True, text=True, timeout=20, check=False
            )
            payload = json.loads(process.stdout or "{}")
            stream = payload["streams"][0]
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError, KeyError, IndexError) as error:
            raise AnalysisError(
                AnalysisErrorCode.DECODE_FAILED,
                "Video rusak atau metadata videonya tidak dapat dibaca.",
                "Gunakan MP4 H.264 valid dan coba kembali.",
            ) from error
        if process.returncode != 0:
            raise AnalysisError(
                AnalysisErrorCode.DECODE_FAILED,
                "Video tidak dapat diperiksa oleh decoder lokal.",
                "Encode ulang video menjadi MP4 H.264.",
            )
        codec = str(stream.get("codec_name") or "").lower()
        if codec not in video_config.allowed_codecs:
            raise AnalysisError(
                AnalysisErrorCode.UNSUPPORTED_CODEC,
                "Codec video belum didukung; MVP memerlukan H.264.",
                "Encode ulang video menjadi MP4 H.264.",
            )
        width = int(stream.get("width") or 0)
        height = int(stream.get("height") or 0)
        if width <= 0 or height <= 0:
            raise AnalysisError(
                AnalysisErrorCode.DECODE_FAILED,
                "Dimensi video tidak valid.",
                "Encode ulang video menjadi MP4 H.264.",
            )
        if (
            width * height > video_config.max_source_pixels
            or max(width, height) > video_config.max_source_dimension
        ):
            raise AnalysisError(
                AnalysisErrorCode.IMAGE_TOO_LARGE,
                "Resolusi sumber video melewati batas keamanan 8K.",
                "Gunakan video hingga resolusi 8K lalu unggah kembali.",
            )
        duration = float(
            stream.get("duration")
            or payload.get("format", {}).get("duration")
            or 0
        )
        if duration <= 0:
            raise AnalysisError(
                AnalysisErrorCode.DECODE_FAILED,
                "Durasi video tidak valid.",
                "Encode ulang video menjadi MP4 H.264.",
            )
        if duration > video_config.max_duration_seconds + 1e-6:
            raise AnalysisError(
                AnalysisErrorCode.VIDEO_TOO_LONG,
                "Durasi video melewati batas 60 detik.",
                "Potong video menjadi maksimal 60 detik.",
            )
        try:
            source_fps = float(Fraction(str(stream.get("r_frame_rate") or "0/1")))
        except (ValueError, ZeroDivisionError):
            source_fps = 0.0
        if source_fps <= 0:
            raise AnalysisError(
                AnalysisErrorCode.DECODE_FAILED,
                "FPS sumber video tidak valid.",
                "Encode ulang video dengan frame rate tetap.",
            )
        raw_frame_count = stream.get("nb_frames")
        frame_count = int(raw_frame_count) if str(raw_frame_count).isdigit() else round(duration * source_fps)
        return MediaInfo(
            display_filename=display_name,
            media_type=MediaType.VIDEO,
            mime_type="video/mp4",
            byte_size=len(upload.content),
            width=width,
            height=height,
            original_width=width,
            original_height=height,
            resolution_normalized=False,
            duration_seconds=duration,
            source_fps=source_fps,
            frame_count=frame_count,
            codec=codec,
            analysis_fps=min(source_fps, float(self.tracker.analysis_fps)),
        )

    def normalize_video(
        self, path: Path, media_info: MediaInfo, output_path: Path
    ) -> tuple[Path, MediaInfo]:
        video_config = self.config.video
        if (
            media_info.width <= video_config.max_width
            and media_info.height <= video_config.max_height
        ):
            return path, media_info

        target_width, target_height = self._fit_video_dimensions(
            media_info.width,
            media_info.height,
            video_config.max_width,
            video_config.max_height,
        )
        ffmpeg = resolve_binary(
            "RIVERSORT_FFMPEG_PATH", "ffmpeg", "ffmpeg version", "/usr/bin/ffmpeg"
        )
        command = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            f"scale={target_width}:{target_height}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-map_metadata",
            "-1",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        try:
            process = subprocess.run(
                command,
                capture_output=True,
                timeout=300,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise AnalysisError(
                AnalysisErrorCode.DECODE_FAILED,
                "Video gagal diperkecil ke resolusi pemrosesan.",
                "Coba video lain atau perkecil resolusinya secara manual.",
            ) from error
        if (
            process.returncode != 0
            or not output_path.is_file()
            or output_path.stat().st_size <= 0
        ):
            raise AnalysisError(
                AnalysisErrorCode.DECODE_FAILED,
                "Video gagal diperkecil ke resolusi pemrosesan.",
                "Coba video lain atau perkecil resolusinya secara manual.",
            )
        return output_path, replace(
            media_info,
            width=target_width,
            height=target_height,
            resolution_normalized=True,
        )

    def iter_video_frames(
        self, path: Path, media_info: MediaInfo
    ) -> Iterable[FramePacket]:
        source_fps = media_info.source_fps or 0.0
        analysis_fps = media_info.analysis_fps or 0.0
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened() or source_fps <= 0 or analysis_fps <= 0:
            capture.release()
            raise AnalysisError(
                AnalysisErrorCode.DECODE_FAILED,
                "Video tidak dapat dibuka oleh OpenCV.",
                "Encode ulang video menjadi MP4 H.264.",
            )
        next_sample_time = 0.0
        frame_index = 0
        try:
            while True:
                ok, pixels = capture.read()
                if not ok:
                    break
                timestamp = frame_index / source_fps
                if timestamp + 1e-9 >= next_sample_time:
                    height, width = pixels.shape[:2]
                    yield FramePacket(
                        frame_index=frame_index,
                        timestamp_ms=round(timestamp * 1000),
                        pixels=pixels,
                        width=width,
                        height=height,
                    )
                    next_sample_time += 1.0 / analysis_fps
                frame_index += 1
        finally:
            capture.release()

    @staticmethod
    def _fit_image_dimensions(
        width: int, height: int, max_pixels: int
    ) -> tuple[int, int]:
        scale = min(1.0, math.sqrt(max_pixels / float(width * height)))
        target_width = max(1, math.floor(width * scale))
        target_height = max(1, math.floor(height * scale))
        while target_width * target_height > max_pixels:
            if target_width >= target_height:
                target_width -= 1
            else:
                target_height -= 1
        return target_width, target_height

    @staticmethod
    def _fit_video_dimensions(
        width: int,
        height: int,
        max_width: int,
        max_height: int,
    ) -> tuple[int, int]:
        scale = min(1.0, max_width / float(width), max_height / float(height))
        target_width = max(2, math.floor(width * scale))
        target_height = max(2, math.floor(height * scale))
        target_width -= target_width % 2
        target_height -= target_height % 2
        return target_width, target_height

    @staticmethod
    def _invalid_type(message: str) -> None:
        raise AnalysisError(
            AnalysisErrorCode.INVALID_MEDIA_TYPE,
            message,
            "Pilih file dengan format yang didukung dan unggah kembali.",
        )
