# Copyright (C) 2026 RiverSort AI contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Validated local FFmpeg encoder for browser-compatible annotated MP4."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from riversort.application import AnalysisError
from riversort.domain import AnalysisErrorCode

from .media_opencv import resolve_binary


class FfmpegVideoEncoder:
    def __init__(
        self,
        output_path: Path,
        *,
        width: int,
        height: int,
        fps: float,
    ) -> None:
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.process: subprocess.Popen[bytes] | None = None

    def open(self) -> None:
        binary = resolve_binary(
            "RIVERSORT_FFMPEG_PATH", "ffmpeg", "ffmpeg version", "/usr/bin/ffmpeg"
        )
        command = [
            binary,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            f"{self.fps:.6f}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self.output_path),
        ]
        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except OSError as error:
            raise AnalysisError(
                AnalysisErrorCode.ENCODER_UNAVAILABLE,
                "Encoder video gagal dimulai.",
                "Periksa FFmpeg lokal dan ulangi analisis.",
            ) from error

    def write(self, pixels: Any) -> None:
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("encoder is not open")
        if pixels.shape[1] != self.width or pixels.shape[0] != self.height:
            raise AnalysisError(
                AnalysisErrorCode.EXPORT_FAILED,
                "Ukuran frame video berubah saat encoding.",
                "Gunakan video dengan resolusi tetap.",
            )
        try:
            self.process.stdin.write(pixels.tobytes())
        except (BrokenPipeError, OSError) as error:
            raise AnalysisError(
                AnalysisErrorCode.EXPORT_FAILED,
                "Encoder berhenti saat membuat preview video.",
                "Periksa codec lokal dan ulangi analisis.",
            ) from error

    def close(self) -> bytes:
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("encoder is not open")
        try:
            self.process.stdin.close()
            return_code = self.process.wait(timeout=60)
        except (BrokenPipeError, OSError, subprocess.TimeoutExpired) as error:
            self.abort()
            raise AnalysisError(
                AnalysisErrorCode.EXPORT_FAILED,
                "Preview H.264 gagal diselesaikan.",
                "Periksa FFmpeg/libx264 lalu ulangi analisis.",
            ) from error
        stderr = self.process.stderr.read() if self.process.stderr else b""
        if return_code != 0 or not self.output_path.is_file():
            raise AnalysisError(
                AnalysisErrorCode.EXPORT_FAILED,
                "Preview H.264 gagal dibuat.",
                "Periksa FFmpeg/libx264 lalu ulangi analisis.",
            )
        if stderr:
            # FFmpeg can emit non-fatal diagnostics; never expose them to the UI.
            pass
        return self.output_path.read_bytes()

    def abort(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.kill()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
