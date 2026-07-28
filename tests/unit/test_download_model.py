# Copyright (C) 2026 RiverSort AI contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.download_model import ModelDownloadError, download_model, sha256_file


def test_download_model_from_file_url(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    source.write_bytes(b"verified-model-bytes")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    output = tmp_path / "models" / "model.pt"

    result = download_model(
        url=source.as_uri(),
        output=output,
        expected_sha256=expected,
        verify_model=False,
    )

    assert result == output
    assert output.read_bytes() == source.read_bytes()
    assert sha256_file(output) == expected
    assert not output.with_name("model.part.pt").exists()


def test_download_model_rejects_wrong_checksum(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    source.write_bytes(b"unexpected")
    output = tmp_path / "model.pt"

    with pytest.raises(ModelDownloadError, match="Downloaded file has SHA-256"):
        download_model(
            url=source.as_uri(),
            output=output,
            expected_sha256="0" * 64,
            verify_model=False,
        )

    assert not output.exists()
    assert not output.with_name("model.part.pt").exists()


def test_existing_model_with_wrong_checksum_is_not_overwritten(
    tmp_path: Path,
) -> None:
    output = tmp_path / "model.pt"
    output.write_bytes(b"local-file")

    with pytest.raises(ModelDownloadError, match="Existing file has SHA-256"):
        download_model(
            url="https://example.invalid/model.pt",
            output=output,
            expected_sha256="0" * 64,
            verify_model=False,
        )

    assert output.read_bytes() == b"local-file"
