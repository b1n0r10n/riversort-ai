# Copyright (C) 2026 RiverSort AI contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Download and verify the published RiverSort detector weight."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path
from typing import Mapping


MODEL_VERSION = "0.1.0-baseline.1"
ASSET_NAME = "riversort-yolo11n-fpwr4-0.1.0-baseline.1.pt"
RELEASE_TAG = "v0.1.0"
DEFAULT_URL = (
    "https://github.com/b1n0r10n/riversort-ai/releases/download/"
    f"{RELEASE_TAG}/{ASSET_NAME}"
)
EXPECTED_SHA256 = "cc235703c6b8b9f6244e68145ca82fea115c95526b91e5716468c3f2d2c31235"
EXPECTED_NAMES = {
    0: "plastic_bottle",
    1: "plastic_bag",
    2: "plastic_packaging_sachet",
    3: "other_plastic",
}


class ModelDownloadError(RuntimeError):
    """Raised when a model cannot be downloaded or verified."""


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_class_map(path: Path) -> Mapping[int, str]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ModelDownloadError(
            "Ultralytics is required to verify the model class map. "
            "Install the project before downloading the model."
        ) from exc

    try:
        model = YOLO(str(path))
        names = {int(key): str(value) for key, value in dict(model.names).items()}
    except Exception as exc:
        raise ModelDownloadError(f"Unable to load downloaded model: {exc}") from exc

    if names != EXPECTED_NAMES:
        raise ModelDownloadError(
            f"Model class map mismatch. Expected {EXPECTED_NAMES}, received {names}."
        )
    return names


def download_model(
    *,
    url: str,
    output: Path,
    expected_sha256: str = EXPECTED_SHA256,
    verify_model: bool = True,
) -> Path:
    output = output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists():
        actual = sha256_file(output)
        if actual == expected_sha256:
            if verify_model:
                verify_class_map(output)
            return output
        raise ModelDownloadError(
            f"Existing file has SHA-256 {actual}, expected {expected_sha256}. "
            "Remove it before retrying."
        )

    partial = output.with_name(output.stem + ".part" + output.suffix)
    if partial.exists():
        partial.unlink()

    request = urllib.request.Request(
        url,
        headers={
            "User" + "-" + "Ag" + "ent": f"RiverSort-model-downloader/{MODEL_VERSION}"
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            with partial.open("wb") as target:
                while chunk := response.read(1024 * 1024):
                    target.write(chunk)
        actual = sha256_file(partial)
        if actual != expected_sha256:
            raise ModelDownloadError(
                f"Downloaded file has SHA-256 {actual}, expected {expected_sha256}."
            )
        if verify_model:
            verify_class_map(partial)
        os.replace(partial, output)
    except Exception:
        partial.unlink(missing_ok=True)
        raise

    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and verify the RiverSort YOLO11n model."
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="Release asset URL.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models") / ASSET_NAME,
        help="Destination path. Defaults to models/<asset-name>.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        output = download_model(url=args.url, output=args.output)
    except ModelDownloadError as exc:
        print(f"Model download failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Model download failed: {exc}", file=sys.stderr)
        return 1

    print(f"Model ready: {output}")
    print(f"SHA-256: {sha256_file(output)}")
    print(f'export RIVERSORT_MODEL_PATH="{output.resolve()}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
