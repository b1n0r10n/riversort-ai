# Copyright (C) 2026 RiverSort AI contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import ast
from pathlib import Path


FORBIDDEN = {"streamlit", "cv2", "ultralytics", "pandas", "plotly", "pydantic", "yaml"}


def test_domain_has_no_framework_imports() -> None:
    domain_dir = Path(__file__).parents[2] / "src" / "riversort" / "domain"
    violations = []
    for path in domain_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]]
            for name in names:
                if name in FORBIDDEN:
                    violations.append(f"{path.name}:{node.lineno}:{name}")
    assert not violations, violations


def test_application_has_no_framework_imports() -> None:
    application_dir = Path(__file__).parents[2] / "src" / "riversort" / "application"
    violations = []
    for path in application_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]]
            for name in names:
                if name in FORBIDDEN:
                    violations.append(f"{path.name}:{node.lineno}:{name}")
    assert not violations, violations
