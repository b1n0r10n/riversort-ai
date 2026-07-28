# Copyright (C) 2026 RiverSort AI contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import json
import logging

from riversort.infrastructure.logging import JsonFormatter


def test_structured_log_only_emits_allowlisted_context() -> None:
    record = logging.LogRecord("riversort", logging.INFO, __file__, 1, "done", (), None)
    record.analysis_id = "a-1"
    record.stage = "config"
    record.invalid_bbox_count = 2
    record.absolute_path = "/private/media.jpg"
    payload = json.loads(JsonFormatter().format(record))
    assert payload["analysis_id"] == "a-1"
    assert payload["invalid_bbox_count"] == 2
    assert "absolute_path" not in payload
