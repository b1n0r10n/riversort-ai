# Copyright (C) 2026 RiverSort AI contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import hashlib
from pathlib import Path
from zipfile import ZipFile, ZipInfo

import pytest

from riversort.data_pipeline import (
    DataPipelineError,
    SOURCE_TO_TARGET,
    clamp_box,
    parse_yolo_label,
    tree_sha256_v1,
    validate_snapshot,
    validate_zip,
)


def test_tree_digest_v1_includes_path_size_and_content(tmp_path: Path) -> None:
    (tmp_path / "b.txt").write_text("second", encoding="utf-8")
    (tmp_path / "a.txt").write_text("first", encoding="utf-8")
    digest, files, logical_bytes = tree_sha256_v1(tmp_path)
    assert digest == "dbb7a6d71bd9216e925970f8a26e09e0c6a86f03d38ce31929d69ff8a1879a7e"
    assert files == 2
    assert logical_bytes == 11


@pytest.mark.parametrize("unsafe", ["../escape.txt", "/absolute.txt"])
def test_zip_path_traversal_is_rejected(tmp_path: Path, unsafe: str) -> None:
    archive = tmp_path / "unsafe.zip"
    with ZipFile(archive, "w") as stream:
        stream.writestr(unsafe, "bad")
    with pytest.raises(DataPipelineError, match="Unsafe archive path"):
        validate_zip(archive)


def test_zip_symlink_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "symlink.zip"
    info = ZipInfo("link")
    info.create_system = 3
    info.external_attr = 0o120777 << 16
    with ZipFile(archive, "w") as stream:
        stream.writestr(info, "target")
    with pytest.raises(DataPipelineError, match="symlink"):
        validate_zip(archive)


def test_source_class_mapping_and_clamp_are_deterministic(tmp_path: Path) -> None:
    label = tmp_path / "sample.txt"
    label.write_text("2 0.95 0.5 0.2 0.4\n", encoding="utf-8")
    source = parse_yolo_label(label)
    assert source[0].target_class_id == SOURCE_TO_TARGET[2] == 0
    target, changed = clamp_box(source[0])
    assert changed is True
    assert target.x == pytest.approx(0.925)
    assert target.width == pytest.approx(0.15)


def test_invalid_yolo_rows_fail_closed(tmp_path: Path) -> None:
    label = tmp_path / "bad.txt"
    label.write_text("9 nan 0.5 0.1 0.1\n", encoding="utf-8")
    with pytest.raises(DataPipelineError):
        parse_yolo_label(label)


def test_snapshot_validation_rejects_generated_cache_files(tmp_path: Path) -> None:
    snapshot = tmp_path / "datasets" / "snapshot"
    snapshot.mkdir(parents=True)
    payload = snapshot / "payload.txt"
    payload.write_text("immutable", encoding="utf-8")
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    (snapshot / "checksums.sha256").write_text(f"{digest}  payload.txt\n", encoding="utf-8")
    (snapshot / "labels.cache").write_text("generated", encoding="utf-8")

    with pytest.raises(DataPipelineError, match="file set mismatch"):
        validate_snapshot(tmp_path, "snapshot")
