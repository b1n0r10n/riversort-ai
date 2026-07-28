# Copyright (C) 2026 RiverSort AI contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path

from riversort.application import UploadedMedia
from riversort.infrastructure.workspace import AnalysisWorkspace, sanitize_filename


def test_filename_is_sanitized_and_workspace_is_cleaned(tmp_path: Path) -> None:
    assert sanitize_filename("../../private river?.jpg") == "private_river_.jpg"
    workspace_path = tmp_path / "analysis-1"
    with AnalysisWorkspace("analysis-1", tmp_path) as workspace:
        path, display = workspace.save_upload(
            UploadedMedia("../../private river?.jpg", b"content", "image/jpeg")
        )
        assert path.parent == workspace_path
        assert display == "private_river_.jpg"
        assert path.read_bytes() == b"content"
    assert not workspace_path.exists()
