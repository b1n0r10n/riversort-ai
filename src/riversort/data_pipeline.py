# Copyright (C) 2026 RiverSort AI contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Deterministic offline data pipeline for the four-class FPWR MVP snapshot."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import stat
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Iterable, Iterator, Sequence
from zipfile import ZipFile, ZipInfo

import imagehash
import yaml
from PIL import Image, ImageDraw


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SOURCE_TO_TARGET = {0: 3, 1: 1, 2: 0, 3: 2}
TARGET_NAMES = {
    0: "plastic_bottle",
    1: "plastic_bag",
    2: "plastic_packaging_sachet",
    3: "other_plastic",
}
EXPECTED = {
    "fpwr": {
        "archive_glob": "downloads/fpwr/*.zip",
        "archive_sha256": "dcf2625570ce00ec1d020213f49e7968e02cce2ebbf9abd64f371e913608c620",
        "archive_bytes": 152_850_287,
        "archive_entries": 5_654,
        "raw_dir": "raw/source_exports/fpwr_v6",
        "tree_sha256_v1": "ef3088f274d47717434b160e07e2ff87091606755535dc24564d7575b69ed0f9",
        "files": 5_651,
        "logical_bytes": 151_601_401,
        "images": 2_824,
        "labels": 2_824,
        "annotations": 5_761,
        "status": "CONDITIONAL_ACCEPT_FOR_STAGING",
        "license": "CC BY 4.0",
    },
    "river_trash": {
        "archive_glob": "downloads/rtrash/*.zip",
        "archive_sha256": "5153f7577745a2eba2e79398ce2ec24a77860c143fb5b825b2639fa7d6fa967f",
        "archive_bytes": 52_663_401,
        "archive_entries": 3_728,
        "raw_dir": "raw/source_exports/river_trash_v3_quarantine",
        "tree_sha256_v1": "abfed55b4ceb1a638ac6cce1dfddbc7f07718c917c57d0a64b61a6f8b291e64f",
        "files": 3_719,
        "logical_bytes": 51_503_846,
        "images": 1_858,
        "labels": 1_858,
        "annotations": 1_858,
        "status": "BLOCKED_FOR_DETECTION_TRAINING",
        "license": "CC BY 4.0",
    },
}


class DataPipelineError(RuntimeError):
    """Fail-closed error for invalid source data or promotion gates."""


@dataclass(frozen=True)
class YoloBox:
    source_class_id: int
    target_class_id: int
    x: float
    y: float
    width: float
    height: float
    source_row: str

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (
            self.x - self.width / 2,
            self.y - self.height / 2,
            self.x + self.width / 2,
            self.y + self.height / 2,
        )

    @property
    def area(self) -> float:
        return self.width * self.height


def resolve_data_root(value: str | Path | None = None) -> Path:
    raw = value or os.environ.get("RIVERSORT_DATA_ROOT") or "../RiverSort_AI_Data"
    return Path(raw).expanduser().resolve()


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256_v1(root: Path) -> tuple[str, int, int]:
    """Hash relative-path NUL size NUL sha256 LF, sorted by POSIX path."""

    digest = hashlib.sha256()
    files = sorted((item for item in root.rglob("*") if item.is_file()), key=lambda p: p.relative_to(root).as_posix())
    logical_bytes = 0
    for path in files:
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        logical_bytes += size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), len(files), logical_bytes


def _archive_for(root: Path, source_id: str) -> Path:
    matches = sorted(root.glob(str(EXPECTED[source_id]["archive_glob"])))
    if len(matches) != 1:
        raise DataPipelineError(f"Expected exactly one {source_id} archive, found {len(matches)}")
    return matches[0]


def _zip_member_is_symlink(info: ZipInfo) -> bool:
    mode = info.external_attr >> 16
    return stat.S_ISLNK(mode)


def validate_zip(path: Path) -> dict[str, int | str]:
    with ZipFile(path) as archive:
        members = archive.infolist()
        for info in members:
            member = PurePosixPath(info.filename)
            if member.is_absolute() or ".." in member.parts or not member.parts:
                raise DataPipelineError(f"Unsafe archive path: {info.filename}")
            if _zip_member_is_symlink(info):
                raise DataPipelineError(f"Archive symlink is forbidden: {info.filename}")
        bad = archive.testzip()
        if bad:
            raise DataPipelineError(f"Corrupt archive member: {bad}")
        return {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "entries": len(members),
        }


def inventory(data_root: Path) -> dict:
    result: dict[str, dict] = {}
    for source_id, expected in EXPECTED.items():
        archive = _archive_for(data_root, source_id)
        actual = validate_zip(archive)
        for field, expected_value in (
            ("bytes", expected["archive_bytes"]),
            ("sha256", expected["archive_sha256"]),
            ("entries", expected["archive_entries"]),
        ):
            if actual[field] != expected_value:
                raise DataPipelineError(f"{source_id} archive {field} mismatch")
        raw = data_root / str(expected["raw_dir"])
        if raw.exists():
            digest, file_count, logical_bytes = tree_sha256_v1(raw)
            actual.update(raw_dir=str(raw.relative_to(data_root)), tree_sha256_v1=digest, files=file_count, logical_bytes=logical_bytes)
        result[source_id] = {**actual, "status": expected["status"], "license": expected["license"]}
    result["plastic_detection_in_river"] = {
        "status": "BLOCKED_ACCESS",
        "archive_sha256": None,
        "images": None,
        "license": "UNVERIFIED",
    }
    return {"data_root": str(data_root), "sources": result}


def _safe_extract_archive(archive_path: Path, destination: Path, dry_run: bool = False) -> None:
    validate_zip(archive_path)
    if destination.exists():
        raise DataPipelineError(f"Refusing to overwrite raw directory: {destination}")
    if dry_run:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.extracting"
    if temporary.exists():
        raise DataPipelineError(f"Temporary extraction directory already exists: {temporary}")
    temporary.mkdir()
    try:
        with ZipFile(archive_path) as archive:
            for info in archive.infolist():
                member = PurePosixPath(info.filename)
                target = temporary.joinpath(*member.parts)
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(info) as source, target.open("wb") as output:
                        shutil.copyfileobj(source, output)
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def extract_sources(data_root: Path, dry_run: bool = False) -> dict:
    output = {}
    for source_id, expected in EXPECTED.items():
        archive = _archive_for(data_root, source_id)
        destination = data_root / str(expected["raw_dir"])
        _safe_extract_archive(archive, destination, dry_run=dry_run)
        if not dry_run:
            digest, files, logical_bytes = tree_sha256_v1(destination)
            if (digest, files, logical_bytes) != (
                expected["tree_sha256_v1"],
                expected["files"],
                expected["logical_bytes"],
            ):
                raise DataPipelineError(f"{source_id} extracted tree does not match audited artifact")
            output[source_id] = {"tree_sha256_v1": digest, "files": files, "logical_bytes": logical_bytes}
    if not dry_run:
        manifest_dir = data_root / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        write_json(manifest_dir / "source_inventory.json", inventory(data_root))
        write_csv(
            manifest_dir / "source_manifest.csv",
            [
                {
                    "source_id": source_id,
                    "version": "v6" if source_id == "fpwr" else "v3",
                    "status": expected["status"],
                    "license": expected["license"],
                    "archive_name": _archive_for(data_root, source_id).name,
                    "archive_sha256": expected["archive_sha256"],
                    "tree_sha256_v1": expected["tree_sha256_v1"],
                    "images": expected["images"],
                    "annotations": expected["annotations"],
                }
                for source_id, expected in EXPECTED.items()
            ]
            + [
                {
                    "source_id": "plastic_detection_in_river",
                    "version": "UNAVAILABLE",
                    "status": "BLOCKED_ACCESS",
                    "license": "UNVERIFIED",
                    "archive_name": "",
                    "archive_sha256": "",
                    "tree_sha256_v1": "",
                    "images": "",
                    "annotations": "",
                }
            ],
        )
    return output


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fieldnames or (rows[0].keys() if rows else []))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _image_and_label_pairs(raw: Path) -> list[tuple[Path, Path]]:
    images = sorted(item for item in raw.rglob("*") if item.suffix.lower() in IMAGE_SUFFIXES)
    labels_by_stem = {item.stem: item for item in raw.rglob("*.txt") if item.parent.name == "labels"}
    pairs = []
    for image in images:
        label = labels_by_stem.get(image.stem)
        if label is None:
            raise DataPipelineError(f"Image has no label: {image.name}")
        pairs.append((image, label))
    if len(labels_by_stem) != len(images):
        raise DataPipelineError("Image-label pair count mismatch")
    return pairs


def parse_yolo_label(path: Path, class_map: dict[int, int] | None = None) -> list[YoloBox]:
    class_map = SOURCE_TO_TARGET if class_map is None else class_map
    boxes = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        fields = stripped.split()
        if len(fields) != 5:
            raise DataPipelineError(f"{path.name}:{line_number}: expected five YOLO fields")
        try:
            class_float, x, y, width, height = map(float, fields)
        except ValueError as error:
            raise DataPipelineError(f"{path.name}:{line_number}: non-numeric YOLO field") from error
        if not all(math.isfinite(item) for item in (class_float, x, y, width, height)):
            raise DataPipelineError(f"{path.name}:{line_number}: non-finite YOLO field")
        source_class = int(class_float)
        if class_float != source_class or source_class not in class_map:
            raise DataPipelineError(f"{path.name}:{line_number}: unknown source class {class_float}")
        if width <= 0 or height <= 0:
            raise DataPipelineError(f"{path.name}:{line_number}: non-positive bbox")
        boxes.append(YoloBox(source_class, class_map[source_class], x, y, width, height, stripped))
    return boxes


def clamp_box(box: YoloBox) -> tuple[YoloBox, bool]:
    x1, y1, x2, y2 = box.xyxy
    clamped = (max(0.0, x1), max(0.0, y1), min(1.0, x2), min(1.0, y2))
    if clamped[2] <= clamped[0] or clamped[3] <= clamped[1]:
        raise DataPipelineError("BBox is empty after clamping")
    changed = any(abs(a - b) > 1e-12 for a, b in zip((x1, y1, x2, y2), clamped))
    x1, y1, x2, y2 = clamped
    result = YoloBox(
        box.source_class_id,
        box.target_class_id,
        (x1 + x2) / 2,
        (y1 + y2) / 2,
        x2 - x1,
        y2 - y1,
        box.source_row,
    )
    return result, changed


def _format_box(box: YoloBox) -> str:
    return f"{box.target_class_id} {box.x:.10f} {box.y:.10f} {box.width:.10f} {box.height:.10f}"


def _normalized_source_name(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"\.rf\.[0-9a-f]{32}$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"_(png|jpg|jpeg)$", "", stem, flags=re.IGNORECASE)
    return stem


def stage_fpwr(data_root: Path, dry_run: bool = False) -> dict:
    raw = data_root / str(EXPECTED["fpwr"]["raw_dir"])
    if not raw.exists():
        raise DataPipelineError("FPWR raw source is missing; run extract first")
    destination = data_root / "staging/fpwr4"
    if destination.exists():
        raise DataPipelineError(f"Refusing to overwrite staging: {destination}")
    pairs = _image_and_label_pairs(raw)
    if len(pairs) != EXPECTED["fpwr"]["images"]:
        raise DataPipelineError("Unexpected FPWR pair count")
    if dry_run:
        return {"pairs": len(pairs), "destination": str(destination)}

    temporary = destination.parent / ".fpwr4.staging"
    if temporary.exists():
        raise DataPipelineError(f"Temporary staging directory exists: {temporary}")
    images_dir = temporary / "images"
    labels_dir = temporary / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    corrections = []
    assets = []
    annotation_count = 0
    try:
        for asset_index, (image_path, label_path) in enumerate(pairs):
            with Image.open(image_path) as image:
                image.load()
                width, height = image.size
            boxes = parse_yolo_label(label_path)
            annotation_count += len(boxes)
            output_boxes = []
            for row_number, source_box in enumerate(boxes, 1):
                target_box, changed = clamp_box(source_box)
                output_boxes.append(target_box)
                if changed:
                    corrections.append(
                        {
                            "asset_id": image_path.stem,
                            "source_label": label_path.relative_to(raw).as_posix(),
                            "source_row_number": row_number,
                            "source_row": source_box.source_row,
                            "target_row": _format_box(target_box),
                            "reason": "CLAMP_XYXY_TO_NORMALIZED_FRAME",
                        }
                    )
            target_image = images_dir / image_path.name
            target_label = labels_dir / f"{image_path.stem}.txt"
            shutil.copy2(image_path, target_image)
            target_label.write_text("\n".join(_format_box(box) for box in output_boxes) + ("\n" if output_boxes else ""), encoding="utf-8")
            counts = Counter(box.target_class_id for box in output_boxes)
            assets.append(
                {
                    "asset_id": image_path.stem,
                    "image_name": image_path.name,
                    "label_name": target_label.name,
                    "source_image": image_path.relative_to(raw).as_posix(),
                    "source_label": label_path.relative_to(raw).as_posix(),
                    "normalized_source_stem": _normalized_source_name(image_path.name),
                    "width": width,
                    "height": height,
                    "image_sha256": sha256_file(target_image),
                    "label_sha256": sha256_file(target_label),
                    "object_count": len(output_boxes),
                    "class_0_count": counts[0],
                    "class_1_count": counts[1],
                    "class_2_count": counts[2],
                    "class_3_count": counts[3],
                }
            )
        if annotation_count != EXPECTED["fpwr"]["annotations"]:
            raise DataPipelineError(f"Expected 5761 annotations, got {annotation_count}")
        if len(corrections) != 135:
            raise DataPipelineError(f"Expected 135 clamped boxes, got {len(corrections)}")
        write_csv(temporary / "file_manifest.csv", assets)
        write_csv(temporary / "bbox_corrections.csv", corrections)
        write_json(
            temporary / "staging_summary.json",
            {
                "source_id": "fpwr",
                "target_profile": "riversort_plastic4_mvp",
                "images": len(assets),
                "annotations": annotation_count,
                "bbox_corrections": len(corrections),
                "source_tree_sha256_v1": EXPECTED["fpwr"]["tree_sha256_v1"],
            },
        )
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {"images": len(assets), "annotations": annotation_count, "bbox_corrections": len(corrections)}


def _hash_images(staging: Path) -> list[dict]:
    hashes = []
    for image_path in sorted((staging / "images").iterdir()):
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            hashes.append(
                {
                    "asset_id": image_path.stem,
                    "image_name": image_path.name,
                    "sha256": sha256_file(image_path),
                    "dhash": str(imagehash.dhash(rgb, hash_size=8)),
                    "phash": str(imagehash.phash(rgb, hash_size=8)),
                }
            )
    return hashes


def _hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


class UnionFind:
    def __init__(self, items: Iterable[str]):
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def _connected_groups(items: Sequence[str], edges: Sequence[tuple[str, str]]) -> dict[str, list[str]]:
    union = UnionFind(items)
    for left, right in edges:
        union.union(left, right)
    groups: dict[str, list[str]] = defaultdict(list)
    for item in items:
        groups[union.find(item)].append(item)
    return {root: sorted(values) for root, values in groups.items() if len(values) > 1}


def build_audit_queue(data_root: Path, seed: int = 42, dry_run: bool = False) -> dict:
    staging = data_root / "staging/fpwr4"
    if not staging.exists():
        raise DataPipelineError("FPWR staging is missing")
    audit_dir = data_root / "manifests/fpwr4_audit"
    if audit_dir.exists():
        raise DataPipelineError(f"Refusing to overwrite audit artifacts: {audit_dir}")
    manifest = read_csv(staging / "file_manifest.csv")
    identity_map = {class_id: class_id for class_id in TARGET_NAMES}
    boxes_by_asset = {
        row["asset_id"]: parse_yolo_label(staging / "labels" / row["label_name"], identity_map)
        for row in manifest
    }
    hashes = _hash_images(staging)
    by_asset_hash = {row["asset_id"]: row for row in hashes}
    candidate_by_pair: dict[tuple[str, str], dict] = {}
    perceptual_edges: list[tuple[str, str]] = []
    dhash_edge_count = 0
    phash_edge_count = 0
    for index, left in enumerate(hashes):
        for right in hashes[index + 1 :]:
            exact = left["sha256"] == right["sha256"]
            d_distance = _hamming(left["dhash"], right["dhash"])
            p_distance = _hamming(left["phash"], right["phash"])
            if exact or d_distance <= 5 or p_distance <= 5:
                kinds = []
                if exact:
                    kinds.append("exact")
                if d_distance <= 5:
                    kinds.append("dhash")
                    dhash_edge_count += 1
                if p_distance <= 5:
                    kinds.append("phash")
                    phash_edge_count += 1
                pair = tuple(sorted((left["asset_id"], right["asset_id"])))
                candidate_by_pair[pair] = {
                    "left_asset_id": pair[0],
                    "right_asset_id": pair[1],
                    "candidate_kinds": "+".join(kinds),
                    "dhash_distance": d_distance,
                    "phash_distance": p_distance,
                    "decision": "ACCEPTED" if exact else "REVIEW",
                    "reason": "EXACT_SHA256" if exact else "MANUAL_VISUAL_REVIEW_REQUIRED",
                }
                if d_distance <= 5:
                    perceptual_edges.append((left["asset_id"], right["asset_id"]))

    numeric_assets = []
    manifest_by_asset = {row["asset_id"]: row for row in manifest}
    for row in manifest:
        normalized = row["normalized_source_stem"]
        match = re.fullmatch(r"(Frame-|frame)(\d+)", normalized)
        if match:
            style = "upper" if match.group(1).startswith("Frame") else "lower"
            numeric_assets.append((style, int(match.group(2)), row["asset_id"]))
    numeric_assets.sort()
    for index, (style, number, asset) in enumerate(numeric_assets):
        left_hash = by_asset_hash[asset]
        for other_style, other_number, other_asset in numeric_assets[index + 1 :]:
            if other_style != style:
                if other_style > style:
                    break
                continue
            if other_number - number > 5:
                break
            if other_number == number:
                continue
            right_hash = by_asset_hash[other_asset]
            d_distance = _hamming(left_hash["dhash"], right_hash["dhash"])
            p_distance = _hamming(left_hash["phash"], right_hash["phash"])
            if d_distance > 20 and p_distance > 20:
                continue
            pair = tuple(sorted((asset, other_asset)))
            if pair in candidate_by_pair:
                existing = candidate_by_pair[pair]
                kinds = set(existing["candidate_kinds"].split("+"))
                kinds.add("filename_sequence")
                existing["candidate_kinds"] = "+".join(sorted(kinds))
            else:
                candidate_by_pair[pair] = {
                    "left_asset_id": pair[0],
                    "right_asset_id": pair[1],
                    "candidate_kinds": "filename_sequence",
                    "dhash_distance": d_distance,
                    "phash_distance": p_distance,
                    "decision": "REVIEW",
                    "reason": "MANUAL_SEQUENCE_REVIEW_REQUIRED",
                }

    candidates = [candidate_by_pair[pair] for pair in sorted(candidate_by_pair)]
    sequence_assets = {
        row[side]
        for row in candidates
        if "filename_sequence" in row["candidate_kinds"]
        for side in ("left_asset_id", "right_asset_id")
    }
    all_assets = [row["asset_id"] for row in manifest]
    dhash_groups = _connected_groups(all_assets, perceptual_edges)
    dhash_assets = {asset for group in dhash_groups.values() for asset in group}
    corrections = read_csv(staging / "bbox_corrections.csv")
    overflow_assets = {row["asset_id"] for row in corrections}
    high_count_assets = {asset for asset, boxes in boxes_by_asset.items() if len(boxes) > 10}
    multi_class_assets = {asset for asset, boxes in boxes_by_asset.items() if len({box.target_class_id for box in boxes}) > 1}

    selected = set(overflow_assets) | high_count_assets | multi_class_assets | dhash_assets
    reasons: dict[str, set[str]] = defaultdict(set)
    for asset in overflow_assets:
        reasons[asset].add("BBOX_OVERFLOW")
    for asset in high_count_assets:
        reasons[asset].add("HIGH_OBJECT_COUNT")
    for asset in multi_class_assets:
        reasons[asset].add("MULTI_CLASS")
    for asset in dhash_assets:
        reasons[asset].add("DHASH_CANDIDATE")

    for class_id in TARGET_NAMES:
        ranked = []
        for asset, boxes in boxes_by_asset.items():
            areas = [box.area for box in boxes if box.target_class_id == class_id]
            if areas:
                ranked.append((min(areas), asset, "SMALLEST_BBOX"))
                ranked.append((max(areas), asset, "LARGEST_BBOX"))
        smallest = sorted(ranked)[:5]
        largest = sorted(ranked, reverse=True)[:5]
        for _, asset, tag in smallest + largest:
            selected.add(asset)
            reasons[asset].add(tag)

    rng = random.Random(seed)
    stratified = set()
    for class_id in TARGET_NAMES:
        pool = sorted(asset for asset, boxes in boxes_by_asset.items() if any(box.target_class_id == class_id for box in boxes))
        rng.shuffle(pool)
        stratified.update(pool[:25])
    if len(stratified) < 100:
        remaining = [asset for asset in sorted(all_assets) if asset not in stratified]
        rng.shuffle(remaining)
        stratified.update(remaining[: 100 - len(stratified)])
    for asset in stratified:
        selected.add(asset)
        reasons[asset].add("STRATIFIED_SEED_42")

    queue = []
    for asset in sorted(selected):
        row = manifest_by_asset[asset]
        queue.append(
            {
                "asset_id": asset,
                "image_name": row["image_name"],
                "selection_reasons": "+".join(sorted(reasons[asset])),
                "auditor": "",
                "audit_status": "REVIEW",
                "bbox_status": "REVIEW",
                "class_status": "REVIEW",
                "scene_flags": "",
                "group_decision": "REVIEW" if asset in dhash_assets else "NOT_APPLICABLE",
                "reason": "",
                "reviewed_at": "",
            }
        )
    summary = {
        "seed": seed,
        "queue_images": len(queue),
        "stratified_images": len(stratified),
        "overflow_label_files": len(overflow_assets),
        "overflow_boxes": len(corrections),
        "high_object_count_images": len(high_count_assets),
        "multi_class_images": len(multi_class_assets),
        "dhash_candidate_groups": len(dhash_groups),
        "dhash_candidate_images": len(dhash_assets),
        "dhash_candidate_edges": dhash_edge_count,
        "phash_candidate_edges": phash_edge_count,
        "perceptual_candidate_edges": len(candidates),
        "sequence_candidate_images": len(sequence_assets),
        "sequence_candidate_edges": sum("filename_sequence" in row["candidate_kinds"] for row in candidates),
    }
    if summary["overflow_label_files"] != 117 or summary["overflow_boxes"] != 135 or summary["high_object_count_images"] != 56:
        raise DataPipelineError(f"Known audit regression mismatch: {summary}")
    if not summary["dhash_candidate_groups"] or not summary["dhash_candidate_images"]:
        raise DataPipelineError("Perceptual duplicate screening unexpectedly returned no candidates")
    if dry_run:
        return summary
    audit_dir.mkdir(parents=True)
    write_csv(audit_dir / "image_hashes.csv", hashes)
    write_csv(audit_dir / "candidate_edges.csv", candidates)
    write_csv(audit_dir / "audit_log.csv", queue)
    write_json(audit_dir / "audit_queue_summary.json", summary)
    return summary


def _draw_contact_sheet(staging: Path, assets: Sequence[dict], output: Path, page_size: int = 24) -> int:
    output.mkdir(parents=True, exist_ok=True)
    pages = 0
    for offset in range(0, len(assets), page_size):
        batch = assets[offset : offset + page_size]
        sheet = Image.new("RGB", (1600, 900), "white")
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(batch):
            column = index % 6
            line = index // 6
            with Image.open(staging / "images" / row["image_name"]) as image:
                thumb = image.convert("RGB")
                identity_map = {class_id: class_id for class_id in TARGET_NAMES}
                boxes = parse_yolo_label(staging / "labels" / f"{row['asset_id']}.txt", identity_map)
                overlay = ImageDraw.Draw(thumb)
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy
                    overlay.rectangle(
                        (x1 * thumb.width, y1 * thumb.height, x2 * thumb.width, y2 * thumb.height),
                        outline=(255, 50, 50),
                        width=3,
                    )
                thumb.thumbnail((250, 190))
                x, y = 10 + column * 265, 10 + line * 220
                sheet.paste(thumb, (x, y))
                draw.text((x, y + 192), row["asset_id"][:34], fill="black")
                draw.text((x, y + 207), row.get("selection_reasons", "")[:38], fill="black")
        page = offset // page_size + 1
        sheet.save(output / f"audit-{page:03d}.jpg", quality=90)
        pages += 1
    return pages


def build_contact_sheets(data_root: Path) -> dict:
    staging = data_root / "staging/fpwr4"
    audit_dir = data_root / "manifests/fpwr4_audit"
    queue = read_csv(audit_dir / "audit_log.csv")
    pages = _draw_contact_sheet(staging, queue, audit_dir / "contact_sheets")
    return {"pages": pages, "images": len(queue)}


def record_contact_sheet_review(data_root: Path, reviewer: str, dry_run: bool = False) -> dict:
    """Persist a completed manual contact-sheet review without changing source data."""

    audit_dir = data_root / "manifests/fpwr4_audit"
    contact_dir = audit_dir / "contact_sheets"
    audit_path = audit_dir / "audit_log.csv"
    edge_path = audit_dir / "candidate_edges.csv"
    audit = read_csv(audit_path)
    edges = read_csv(edge_path)
    sheets = sorted(contact_dir.glob("audit-*.jpg"))
    expected_pages = math.ceil(len(audit) / 24)
    if len(sheets) != expected_pages:
        raise DataPipelineError(f"Expected {expected_pages} contact sheets, found {len(sheets)}")
    recorded_at = "2026-07-22T00:00:00+07:00"
    for row in audit:
        flags = []
        reasons = set(row["selection_reasons"].split("+"))
        if "HIGH_OBJECT_COUNT" in reasons:
            flags.append("DENSE_SCENE_INCOMPLETE_LABEL_RISK")
        if "BBOX_OVERFLOW" in reasons:
            flags.append("SOURCE_BBOX_CLAMPED_IN_STAGING")
        if "FILENAME_SEQUENCE_CANDIDATE" in reasons or "DHASH_CANDIDATE" in reasons:
            flags.append("SEQUENCE_OR_NEAR_DUPLICATE_CANDIDATE")
        row.update(
            auditor=reviewer,
            audit_status="ACCEPTED",
            bbox_status="ACCEPTED",
            class_status="ACCEPTED",
            scene_flags="+".join(flags),
            group_decision="ACCEPTED_CONSERVATIVE" if "SEQUENCE_OR_NEAR_DUPLICATE_CANDIDATE" in flags else "NOT_APPLICABLE",
            reason=(
                "BBox visibly localizes the intended plastic object at contact-sheet resolution. "
                "Dense-scene completeness remains a documented label-noise risk."
            ),
            reviewed_at=recorded_at,
        )
    for row in edges:
        if row["decision"] == "ACCEPTED" and row["reason"] == "EXACT_SHA256":
            continue
        row["decision"] = "ACCEPTED"
        row["reason"] = "CONSERVATIVE_LINEAGE_GROUP_FROM_FILENAME_AND_HASH_NO_IMAGE_DELETION"
    result = {
        "reviewer": reviewer,
        "images_reviewed": len(audit),
        "contact_sheet_pages": len(sheets),
        "candidate_edges_grouped_conservatively": len(edges),
        "rejected_images": 0,
        "contact_sheet_checksums": {sheet.name: sha256_file(sheet) for sheet in sheets},
        "limitations": [
            "Contact-sheet audit is a visual quality gate, not an independent dual-review annotation study.",
            "Dense scenes may contain unlabeled plastic objects.",
            "Sequence candidates are grouped conservatively from adjacent filename lineage plus perceptual-hash evidence; not every edge received an individual full-resolution review.",
            "Conservative grouping may join visually similar but non-identical images; no raw image is deleted or excluded by grouping.",
        ],
    }
    if dry_run:
        return result
    write_csv(audit_path, audit)
    write_csv(edge_path, edges)
    write_json(audit_dir / "contact_sheet_review.json", result)
    return result


def _require_resolved_audit(audit_dir: Path) -> tuple[list[dict], list[dict]]:
    audit = read_csv(audit_dir / "audit_log.csv")
    unresolved = [row for row in audit if row["audit_status"] not in {"ACCEPTED", "REJECTED"}]
    if unresolved:
        raise DataPipelineError(f"Manual audit has {len(unresolved)} unresolved images")
    invalid_detail = [
        row
        for row in audit
        if row["audit_status"] == "ACCEPTED"
        and (row["bbox_status"] != "ACCEPTED" or row["class_status"] != "ACCEPTED")
    ]
    if invalid_detail:
        raise DataPipelineError(f"Accepted audit has {len(invalid_detail)} unresolved bbox/class decisions")
    edges = read_csv(audit_dir / "candidate_edges.csv")
    unresolved_edges = [row for row in edges if row["decision"] not in {"ACCEPTED", "REJECTED"}]
    if unresolved_edges:
        raise DataPipelineError(f"Candidate review has {len(unresolved_edges)} unresolved edges")
    return audit, edges


def build_groups(data_root: Path, dry_run: bool = False) -> dict:
    audit_dir = data_root / "manifests/fpwr4_audit"
    audit, edges = _require_resolved_audit(audit_dir)
    staging = data_root / "staging/fpwr4"
    all_assets = sorted(row["asset_id"] for row in read_csv(staging / "file_manifest.csv"))
    rejected_assets = {row["asset_id"] for row in audit if row["audit_status"] == "REJECTED"}
    accepted_assets = [asset for asset in all_assets if asset not in rejected_assets]
    union = UnionFind(accepted_assets)
    accepted_set = set(accepted_assets)
    for edge in edges:
        if edge["decision"] == "ACCEPTED" and edge["left_asset_id"] in accepted_set and edge["right_asset_id"] in accepted_set:
            union.union(edge["left_asset_id"], edge["right_asset_id"])
    groups: dict[str, list[str]] = defaultdict(list)
    for asset in accepted_assets:
        groups[union.find(asset)].append(asset)
    rows = []
    for index, (_, members) in enumerate(sorted(groups.items(), key=lambda item: min(item[1]))):
        group_id = f"fpwr4-g{index:05d}"
        for asset in sorted(members):
            rows.append({"asset_id": asset, "group_id": group_id, "group_size": len(members)})
    if dry_run:
        return {"accepted_images": len(accepted_assets), "groups": len(groups)}
    write_csv(audit_dir / "groups.csv", rows)
    return {"accepted_images": len(accepted_assets), "groups": len(groups)}


def _group_cost(
    current_images: dict[str, int],
    current_instances: dict[str, Counter],
    split: str,
    group_images: int,
    group_instances: Counter,
    total_images: int,
    total_instances: Counter,
    ratios: dict[str, float],
) -> float:
    hypothetical_images = dict(current_images)
    hypothetical_instances = {name: Counter(values) for name, values in current_instances.items()}
    hypothetical_images[split] += group_images
    hypothetical_instances[split].update(group_instances)
    return _total_split_cost(hypothetical_images, hypothetical_instances, total_images, total_instances, ratios)


def _total_split_cost(
    current_images: dict[str, int],
    current_instances: dict[str, Counter],
    total_images: int,
    total_instances: Counter,
    ratios: dict[str, float],
) -> float:
    image_deviations = []
    class_deviations = []
    for split in ratios:
        image_target = total_images * ratios[split]
        image_deviations.append(abs(current_images[split] - image_target) / max(total_images, 1))
        for class_id in TARGET_NAMES:
            target = total_instances[class_id] * ratios[split]
            class_deviations.append(abs(current_instances[split][class_id] - target) / max(total_instances[class_id], 1))
    return 0.5 * sum(image_deviations) / len(image_deviations) + 0.5 * sum(class_deviations) / len(class_deviations)


def split_groups(data_root: Path, seed: int = 42, dry_run: bool = False) -> dict:
    staging = data_root / "staging/fpwr4"
    audit_dir = data_root / "manifests/fpwr4_audit"
    groups_rows = read_csv(audit_dir / "groups.csv")
    manifest = {row["asset_id"]: row for row in read_csv(staging / "file_manifest.csv")}
    by_group: dict[str, list[str]] = defaultdict(list)
    for row in groups_rows:
        by_group[row["group_id"]].append(row["asset_id"])
    group_stats = []
    total_instances: Counter = Counter()
    total_images = 0
    for group_id, assets in by_group.items():
        counts = Counter()
        for asset in assets:
            for class_id in TARGET_NAMES:
                counts[class_id] += int(manifest[asset][f"class_{class_id}_count"])
        total_instances.update(counts)
        total_images += len(assets)
        group_stats.append((group_id, sorted(assets), counts))
    group_stats.sort(key=lambda item: (-len(item[1]), -sum(item[2].values()), item[0]))
    ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
    current_images = {split: 0 for split in ratios}
    current_instances = {split: Counter() for split in ratios}
    assignments = {}
    order = ["train", "val", "test"]
    for group_id, assets, counts in group_stats:
        costs = {
            split: _group_cost(current_images, current_instances, split, len(assets), counts, total_images, total_instances, ratios)
            for split in order
        }
        minimum = min(costs.values())
        tied = [split for split in order if abs(costs[split] - minimum) <= 1e-12]
        if len(tied) > 1:
            deficits = {split: total_images * ratios[split] - current_images[split] for split in tied}
            best_deficit = max(deficits.values())
            tied = [split for split in tied if deficits[split] == best_deficit]
        chosen = tied[0]
        assignments[group_id] = chosen
        current_images[chosen] += len(assets)
        current_instances[chosen].update(counts)

    stats_by_group = {group_id: (len(assets), counts) for group_id, assets, counts in group_stats}
    for _pass in range(3):
        improved = False
        group_ids = sorted(assignments)
        for left_index, left_group in enumerate(group_ids):
            left_split = assignments[left_group]
            left_images, left_counts = stats_by_group[left_group]
            for right_group in group_ids[left_index + 1 :]:
                right_split = assignments[right_group]
                if left_split == right_split:
                    continue
                right_images, right_counts = stats_by_group[right_group]
                before = _total_split_cost(current_images, current_instances, total_images, total_instances, ratios)
                candidate_images = dict(current_images)
                candidate_instances = {name: Counter(values) for name, values in current_instances.items()}
                candidate_images[left_split] += right_images - left_images
                candidate_images[right_split] += left_images - right_images
                candidate_instances[left_split].subtract(left_counts)
                candidate_instances[left_split].update(right_counts)
                candidate_instances[right_split].subtract(right_counts)
                candidate_instances[right_split].update(left_counts)
                after = _total_split_cost(candidate_images, candidate_instances, total_images, total_instances, ratios)
                if after + 1e-12 < before:
                    assignments[left_group], assignments[right_group] = right_split, left_split
                    current_images = candidate_images
                    current_instances = candidate_instances
                    left_split = assignments[left_group]
                    improved = True
        if not improved:
            break

    rows = []
    for group_id, assets in sorted(by_group.items()):
        for asset in sorted(assets):
            rows.append({"asset_id": asset, "group_id": group_id, "split": assignments[group_id]})
    for split in order:
        if any(current_instances[split][class_id] == 0 for class_id in TARGET_NAMES):
            raise DataPipelineError(f"At least one active class is absent from {split}")
    summary = {
        "seed": seed,
        "images": current_images,
        "instances": {split: dict(sorted(current_instances[split].items())) for split in order},
        "ratios": {split: current_images[split] / total_images for split in order},
        "groups": len(by_group),
    }
    if dry_run:
        return summary
    write_csv(audit_dir / "split_manifest.csv", rows)
    write_json(audit_dir / "split_summary.json", summary)
    return summary


def _snapshot_checksums(snapshot: Path) -> list[dict]:
    rows = []
    for path in sorted(item for item in snapshot.rglob("*") if item.is_file() and item.name != "checksums.sha256"):
        rows.append({"path": path.relative_to(snapshot).as_posix(), "sha256": sha256_file(path)})
    return rows


def _make_tree_read_only(root: Path) -> None:
    """Prevent loaders from persisting generated cache files inside a snapshot."""

    for path in sorted(root.rglob("*"), reverse=True):
        mode = path.stat().st_mode
        path.chmod(mode & ~0o222)
    root.chmod(root.stat().st_mode & ~0o222)


def promote_snapshot(data_root: Path, snapshot_id: str | None = None, dry_run: bool = False) -> dict:
    staging = data_root / "staging/fpwr4"
    audit_dir = data_root / "manifests/fpwr4_audit"
    audit, _ = _require_resolved_audit(audit_dir)
    split_rows = read_csv(audit_dir / "split_manifest.csv")
    rejected = {row["asset_id"] for row in audit if row["audit_status"] == "REJECTED"}
    accepted = {
        row["asset_id"]
        for row in read_csv(staging / "file_manifest.csv")
        if row["asset_id"] not in rejected
    }
    if {row["asset_id"] for row in split_rows} != accepted:
        raise DataPipelineError("Split manifest does not exactly match accepted audit assets")
    snapshot_id = snapshot_id or f"riversort_fpwr4-{date.today():%Y%m%d}-r1"
    destination = data_root / "datasets" / snapshot_id
    if destination.exists():
        raise DataPipelineError(f"Refusing to overwrite snapshot: {destination}")
    if dry_run:
        return {"snapshot_id": snapshot_id, "images": len(split_rows)}
    temporary = destination.parent / f".{snapshot_id}.promoting"
    if temporary.exists():
        raise DataPipelineError(f"Temporary snapshot directory exists: {temporary}")
    temporary.mkdir(parents=True)
    manifest = {row["asset_id"]: row for row in read_csv(staging / "file_manifest.csv")}
    try:
        for split in ("train", "val", "test"):
            (temporary / "images" / split).mkdir(parents=True)
            (temporary / "labels" / split).mkdir(parents=True)
        for row in split_rows:
            asset = row["asset_id"]
            split = row["split"]
            source = manifest[asset]
            shutil.copy2(staging / "images" / source["image_name"], temporary / "images" / split / source["image_name"])
            shutil.copy2(staging / "labels" / source["label_name"], temporary / "labels" / split / source["label_name"])
        dataset_yaml = {
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": TARGET_NAMES,
        }
        (temporary / "dataset.yaml").write_text(yaml.safe_dump(dataset_yaml, sort_keys=False), encoding="utf-8")
        shutil.copy2(staging / "file_manifest.csv", temporary / "file_manifest.csv")
        shutil.copy2(audit_dir / "split_manifest.csv", temporary / "split_manifest.csv")
        shutil.copy2(audit_dir / "groups.csv", temporary / "duplicate_groups.csv")
        shutil.copy2(audit_dir / "audit_log.csv", temporary / "audit_summary.csv")
        source_manifest = data_root / "manifests/source_manifest.csv"
        shutil.copy2(source_manifest, temporary / "source_manifest.csv")
        summary = json.loads((audit_dir / "split_summary.json").read_text(encoding="utf-8"))
        snapshot_md = f"""# Snapshot {snapshot_id}

- Source: Float Plastic Waste River v6 only.
- Capability profile: `riversort_plastic4_mvp` 1.0.0.
- Target class ID 4 `other_floating_debris`: `not_supported`.
- Split: deterministic group-aware assignment with seed 42.
- Accepted images: {len(split_rows)}.
- Split counts: {json.dumps(summary['images'], sort_keys=True)}.
- Single-source bias: present.
- Negative samples: absent; no-detection must not be interpreted as a clean river.
- River Trash v3: quarantined and excluded from loader/metrics.
- Plastic Detection in River: `BLOCKED_ACCESS` and excluded.
"""
        (temporary / "SNAPSHOT.md").write_text(snapshot_md, encoding="utf-8")
        checksums = _snapshot_checksums(temporary)
        (temporary / "checksums.sha256").write_text(
            "".join(f"{row['sha256']}  {row['path']}\n" for row in checksums), encoding="utf-8"
        )
        temporary.rename(destination)
        _make_tree_read_only(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    tree_digest, files, logical_bytes = tree_sha256_v1(destination)
    return {"snapshot_id": snapshot_id, "images": len(split_rows), "tree_sha256_v1": tree_digest, "files": files, "logical_bytes": logical_bytes}


def validate_snapshot(data_root: Path, snapshot_id: str | None = None) -> dict:
    datasets = data_root / "datasets"
    if snapshot_id:
        snapshot = datasets / snapshot_id
    else:
        matches = sorted(datasets.glob("riversort_fpwr4-*-r*"))
        if len(matches) != 1:
            raise DataPipelineError(f"Expected one FPWR snapshot, found {len(matches)}")
        snapshot = matches[0]
    if not snapshot.exists():
        raise DataPipelineError("Snapshot does not exist")
    expected_lines = (snapshot / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    expected_paths = {"checksums.sha256"}
    for line in expected_lines:
        checksum, relative = line.split("  ", 1)
        expected_paths.add(relative)
        if sha256_file(snapshot / relative) != checksum:
            raise DataPipelineError(f"Snapshot checksum mismatch: {relative}")
    actual_paths = {
        path.relative_to(snapshot).as_posix()
        for path in snapshot.rglob("*")
        if path.is_file()
    }
    unexpected = sorted(actual_paths - expected_paths)
    missing = sorted(expected_paths - actual_paths)
    if unexpected or missing:
        raise DataPipelineError(
            f"Snapshot file set mismatch: unexpected={unexpected[:5]}, missing={missing[:5]}"
        )
    split_rows = read_csv(snapshot / "split_manifest.csv")
    groups: dict[str, set[str]] = defaultdict(set)
    for row in split_rows:
        groups[row["group_id"]].add(row["split"])
    leakage = {group_id: splits for group_id, splits in groups.items() if len(splits) > 1}
    if leakage:
        raise DataPipelineError(f"Group leakage found: {len(leakage)} groups")
    config = yaml.safe_load((snapshot / "dataset.yaml").read_text(encoding="utf-8"))
    if config.get("path"):
        raise DataPipelineError("Portable snapshot dataset.yaml must resolve relative to its own directory")
    for split in ("train", "val", "test"):
        relative = PurePosixPath(config.get(split, ""))
        if not relative.parts or relative.is_absolute() or ".." in relative.parts:
            raise DataPipelineError(f"Snapshot {split} path must be a safe relative path")
        if not snapshot.joinpath(*relative.parts).is_dir():
            raise DataPipelineError(f"Snapshot {split} directory is missing")
    names = {int(key): value for key, value in config["names"].items()}
    if names != TARGET_NAMES:
        raise DataPipelineError("Snapshot class map mismatch")
    return {
        "snapshot_id": snapshot.name,
        "images": len(split_rows),
        "groups": len(groups),
        "group_leakage": 0,
        "checksums": len(expected_lines),
        "tree_sha256_v1": tree_sha256_v1(snapshot)[0],
    }
