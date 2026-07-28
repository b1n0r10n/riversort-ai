# Copyright (C) 2026 RiverSort AI contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_dashboard_runs_without_model_and_explains_recovery() -> None:
    app_path = Path(__file__).parents[2] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=15).run()
    assert not app.exception
    assert any("MODEL_NOT_READY" in item.value for item in app.error)
    assert any("not_supported" in item.value for item in app.warning)
    assert app.button[0].disabled
