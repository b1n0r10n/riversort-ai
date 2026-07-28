# Copyright (C) 2026 RiverSort AI contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Request-scoped temporary workspace with deterministic cleanup."""

from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path

from riversort.application.models import UploadedMedia


SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_filename(filename: str) -> str:
    name = Path(filename).name
    safe = SAFE_NAME.sub("_", name).strip("._")
    return safe[:120] or "uploaded_media"


class AnalysisWorkspace:
    def __init__(self, analysis_id: str, root: Path | None = None) -> None:
        base = root or (Path(tempfile.gettempdir()) / "riversort")
        self.path = base / analysis_id

    def __enter__(self) -> "AnalysisWorkspace":
        self.path.mkdir(parents=True, exist_ok=False)
        return self

    def save_upload(self, upload: UploadedMedia) -> tuple[Path, str]:
        display_name = sanitize_filename(upload.filename)
        suffix = Path(display_name).suffix.lower()
        target = self.path / f"media{suffix}"
        target.write_bytes(upload.content)
        return target, display_name

    def allocate(self, filename: str) -> Path:
        return self.path / sanitize_filename(filename)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        shutil.rmtree(self.path, ignore_errors=True)
