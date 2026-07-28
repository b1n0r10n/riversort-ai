# Copyright (C) 2026 RiverSort AI contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Thin Streamlit entry point."""

from riversort.infrastructure.logging import configure_logging
from riversort.presentation import render_app


configure_logging()
render_app()
