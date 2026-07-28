# Copyright (C) 2026 RiverSort AI contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
from pathlib import Path

import pytest

from riversort.data_pipeline import EXPECTED, inventory, resolve_data_root, validate_snapshot


@pytest.mark.skipif("RIVERSORT_DATA_ROOT" not in os.environ, reason="requires local data root")
def test_local_source_inventory_and_promoted_snapshot() -> None:
    root = resolve_data_root()
    actual = inventory(root)
    assert actual["sources"]["fpwr"]["tree_sha256_v1"] == EXPECTED["fpwr"]["tree_sha256_v1"]
    assert actual["sources"]["river_trash"]["tree_sha256_v1"] == EXPECTED["river_trash"]["tree_sha256_v1"]
    snapshot = validate_snapshot(root, "riversort_fpwr4-20260722-r2")
    assert snapshot["images"] == 2824
    assert snapshot["group_leakage"] == 0
    assert snapshot["tree_sha256_v1"] == "74badb41175aacef724f6b2c4db722ce46dcd6875546f9035472d60e46690b2c"
