# Copyright (C) 2026 RiverSort AI contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Audit the tracked public repository for internal or unsafe artifacts."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[2]
MAX_TRACKED_BYTES = 10 * 1024 * 1024
REQUIRED_PATHS = {
    "README.md",
    "LICENSE",
    "NOTICE.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "pyproject.toml",
    "app.py",
    "docs/PRODUCT.md",
    "docs/ARCHITECTURE.md",
    "docs/DATA_AND_MODEL.md",
    "docs/EVALUATION.md",
    "docs/USAGE.md",
    "docs/ROADMAP.md",
    "docs/DECISIONS.md",
    "models/MODEL_CARD.md",
    "reports/model_evaluation.json",
}
FORBIDDEN_ROOTS = {
    "presentation",
    "dataset",
    "datasets",
    "raw",
    "staging",
    "runs",
    "artifacts",
}
FORBIDDEN_NAMES = {
    "AGENTS.md",
    "RiverSort_AI_BRIN_AIDeaNation_2026.pdf",
    "RiverSort_AI_Dataset_Strategy.md",
}
FORBIDDEN_SUFFIXES = {
    ".pdf",
    ".pptx",
    ".pt",
    ".pth",
    ".onnx",
    ".engine",
    ".zip",
    ".tar",
    ".gz",
    ".7z",
    ".mp4",
}
TEXT_SUFFIXES = {
    ".cff",
    ".csv",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".svg",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
FORBIDDEN_TEXT = {
    "product-specific assistant name": re.compile(
        r"\bco" + r"dex\b", re.IGNORECASE
    ),
    "chat product name": re.compile(r"\bchat" + r"gpt\b", re.IGNORECASE),
    "provider process reference": re.compile(r"\bopen" + r"ai\b", re.IGNORECASE),
    "automated authoring role": re.compile(
        r"\b(?:ai|coding)[ -]?(?:software )?ag" + r"ent\b", re.IGNORECASE
    ),
    "internal planning label": re.compile(
        r"\bplan" + r" mode\b", re.IGNORECASE
    ),
    "internal milestone label": re.compile(r"\bPM-\d{2}\b"),
    "local home path": re.compile(r"/home/" + r"binorion(?:/|\b)"),
    "unfinished marker": re.compile(r"\bT" + r"BD\b"),
    "private email": re.compile(r"\b[\w.+-]+@(?:gmail|yahoo|outlook)\.com\b", re.I),
}
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def tracked_files() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT, text=False
    )
    return [
        ROOT / item.decode("utf-8")
        for item in output.split(b"\0")
        if item
    ]


def audit_links(path: Path, text: str, issues: list[str]) -> None:
    for raw_target in MARKDOWN_LINK.findall(text):
        target = raw_target.strip().strip("<>")
        if not target or target.startswith(("#", "http://", "https://", "mailto:")):
            continue
        target = unquote(target.split("#", 1)[0])
        resolved = (path.parent / target).resolve()
        try:
            resolved.relative_to(ROOT)
        except ValueError:
            issues.append(f"{path.relative_to(ROOT)}: link escapes repository: {target}")
            continue
        if not resolved.exists():
            issues.append(f"{path.relative_to(ROOT)}: broken link: {target}")


def audit() -> dict[str, object]:
    issues: list[str] = []
    files = tracked_files()
    relative_files = {path.relative_to(ROOT).as_posix() for path in files}

    missing = sorted(REQUIRED_PATHS - relative_files)
    issues.extend(f"missing required file: {item}" for item in missing)

    for path in files:
        relative = path.relative_to(ROOT)
        relative_text = relative.as_posix()
        if relative.parts[0] in FORBIDDEN_ROOTS:
            issues.append(f"forbidden public path: {relative_text}")
        if path.name in FORBIDDEN_NAMES:
            issues.append(f"forbidden public file: {relative_text}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            issues.append(f"forbidden tracked artifact: {relative_text}")
        if path.stat().st_size > MAX_TRACKED_BYTES:
            issues.append(
                f"tracked file exceeds 10 MiB: {relative_text} ({path.stat().st_size})"
            )
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {
            "LICENSE",
            ".gitignore",
            ".gitattributes",
            ".env.example",
        }:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            issues.append(f"expected text file is not UTF-8: {relative_text}")
            continue
        for label, pattern in FORBIDDEN_TEXT.items():
            if pattern.search(text):
                issues.append(f"{relative_text}: contains {label}")
        if path.suffix.lower() == ".md":
            audit_links(path, text, issues)

    return {
        "status": "passed" if not issues else "failed",
        "issues": sorted(set(issues)),
        "tracked_files": len(files),
        "markdown_files": sum(path.suffix.lower() == ".md" for path in files),
    }


def main() -> int:
    result = audit()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
