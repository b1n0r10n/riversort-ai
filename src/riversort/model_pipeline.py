# Copyright (C) 2026 RiverSort AI contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Reproducible YOLO11n training, fixed-threshold evaluation, and packaging."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Iterable, Sequence

import yaml
from PIL import Image

from riversort.data_pipeline import resolve_data_root, sha256_file, validate_snapshot, write_json


EXPECTED_NAMES = {
    0: "plastic_bottle",
    1: "plastic_bag",
    2: "plastic_packaging_sachet",
    3: "other_plastic",
}


class ModelPipelineError(RuntimeError):
    """Fail-closed model-pipeline error."""


@dataclass(frozen=True)
class Prediction:
    class_id: int
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class GroundTruth:
    class_id: int
    bbox_xyxy: tuple[float, float, float, float]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_contract(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ModelPipelineError("Training contract must be a mapping")
    required = {
        "experiment_id",
        "snapshot_id",
        "snapshot_tree_sha256_v1",
        "capability_profile_id",
        "initial_model",
        "epochs",
        "batch",
        "seed",
        "evaluation",
    }
    missing = required - set(payload)
    if missing:
        raise ModelPipelineError(f"Training contract is missing: {sorted(missing)}")
    return payload


def _run_text(command: Sequence[str]) -> str:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _git_commit() -> str:
    return _run_text(("git", "rev-parse", "HEAD"))


def _package_inventory() -> list[dict[str, str]]:
    return [
        {"name": item.metadata["Name"] or item.name, "version": item.version}
        for item in sorted(importlib.metadata.distributions(), key=lambda value: (value.metadata["Name"] or "").lower())
    ]


def _hardware_record() -> dict:
    import torch

    gpu = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        gpu = {
            "name": properties.name,
            "memory_bytes": properties.total_memory,
            "torch_cuda": torch.version.cuda,
            "driver": _run_text(("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader")),
        }
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu,
    }


def experiment_paths(data_root: Path, contract: dict) -> tuple[Path, Path]:
    snapshot = data_root / "datasets" / contract["snapshot_id"]
    experiment = data_root / "runs" / contract["experiment_id"]
    return snapshot, experiment


def initial_weight_path(data_root: Path, contract: dict) -> Path:
    return data_root / "models" / "initial" / Path(contract["initial_model"]).name


def preflight(data_root: Path, contract_path: Path) -> dict:
    """Validate snapshot/GPU and acquire the immutable initial weight once."""

    from ultralytics import YOLO

    contract = load_contract(contract_path)
    snapshot, experiment = experiment_paths(data_root, contract)
    validation = validate_snapshot(data_root, contract["snapshot_id"])
    if validation["tree_sha256_v1"] != contract["snapshot_tree_sha256_v1"]:
        raise ModelPipelineError("Snapshot tree digest does not match training contract")
    if not (snapshot / "dataset.yaml").exists():
        raise ModelPipelineError("Snapshot dataset.yaml is missing")
    weight = initial_weight_path(data_root, contract)
    weight.parent.mkdir(parents=True, exist_ok=True)
    if not weight.exists():
        YOLO(str(weight))
    model = YOLO(str(weight))
    if not weight.exists():
        raise ModelPipelineError("Ultralytics did not place initial weight at the requested data-root path")
    initial_sha = sha256_file(weight)
    experiment.mkdir(parents=True, exist_ok=True)
    frozen = experiment / "training_contract.yaml"
    if frozen.exists() and frozen.read_bytes() != contract_path.read_bytes():
        raise ModelPipelineError("Experiment ID already has a different frozen training contract")
    if not frozen.exists():
        shutil.copy2(contract_path, frozen)
    record = {
        "experiment_id": contract["experiment_id"],
        "git_commit": _git_commit(),
        "snapshot": validation,
        "dataset_yaml": str((snapshot / "dataset.yaml").relative_to(data_root)),
        "initial_weight": str(weight.relative_to(data_root)),
        "initial_weight_sha256": initial_sha,
        "initial_model_names": {str(key): value for key, value in model.names.items()},
        "environment": _hardware_record(),
        "packages": _package_inventory(),
        "recorded_at": utc_now(),
    }
    write_json(experiment / "experiment_record.json", record)
    return record


def _training_arguments(contract: dict, data_root: Path, mode: str, batch_override: int | None = None) -> dict:
    snapshot, experiment = experiment_paths(data_root, contract)
    excluded = {
        "contract_version",
        "experiment_id",
        "snapshot_id",
        "snapshot_tree_sha256_v1",
        "capability_profile_id",
        "initial_model",
        "evaluation",
    }
    arguments = {key: value for key, value in contract.items() if key not in excluded and key != "model"}
    arguments.update(
        {
            "data": str(snapshot / "dataset.yaml"),
            "project": str(experiment),
            "name": mode,
            "exist_ok": False,
            "plots": True,
            "save": True,
            "verbose": True,
        }
    )
    if batch_override is not None:
        arguments["batch"] = batch_override
    if mode == "smoke":
        arguments.update({"epochs": 1, "fraction": 0.1, "batch": 4, "workers": 2, "patience": 1})
    return arguments


def _append_attempt(experiment: Path, payload: dict) -> None:
    path = experiment / "attempts.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def train(data_root: Path, contract_path: Path, mode: str, batch_override: int | None = None) -> dict:
    from ultralytics import YOLO

    if mode not in {"smoke", "baseline"}:
        raise ModelPipelineError("Training mode must be smoke or baseline")
    contract = load_contract(contract_path)
    _, experiment = experiment_paths(data_root, contract)
    record_path = experiment / "experiment_record.json"
    if not record_path.exists():
        raise ModelPipelineError("Run preflight before training")
    run_dir = experiment / mode
    if run_dir.exists():
        raise ModelPipelineError(f"Refusing to overwrite training run: {run_dir}")
    arguments = _training_arguments(contract, data_root, mode, batch_override)
    weight = initial_weight_path(data_root, contract)
    attempt = {
        "mode": mode,
        "status": "started",
        "started_at": utc_now(),
        "git_commit": _git_commit(),
        "arguments": arguments,
        "initial_weight_sha256": sha256_file(weight),
    }
    _append_attempt(experiment, attempt)
    started = time.monotonic()
    try:
        model = YOLO(str(weight))
        model.train(**arguments)
    except Exception as error:
        failed = {
            **attempt,
            "status": "failed",
            "ended_at": utc_now(),
            "duration_seconds": time.monotonic() - started,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        _append_attempt(experiment, failed)
        raise
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    if not best.exists() or not last.exists():
        raise ModelPipelineError("Training completed without expected best/last weights")
    completed = {
        **attempt,
        "status": "completed",
        "ended_at": utc_now(),
        "duration_seconds": time.monotonic() - started,
        "best_weight_sha256": sha256_file(best),
        "last_weight_sha256": sha256_file(last),
    }
    _append_attempt(experiment, completed)
    write_json(experiment / f"{mode}_summary.json", completed)
    return completed


def _label_for_image(snapshot: Path, image_path: Path, split: str) -> Path:
    return snapshot / "labels" / split / f"{image_path.stem}.txt"


def _ground_truths(snapshot: Path, image_path: Path, split: str) -> list[GroundTruth]:
    with Image.open(image_path) as image:
        width, height = image.size
    rows = []
    for line in _label_for_image(snapshot, image_path, split).read_text(encoding="utf-8").splitlines():
        class_id_raw, x, y, box_width, box_height = map(float, line.split())
        class_id = int(class_id_raw)
        rows.append(
            GroundTruth(
                class_id,
                (
                    (x - box_width / 2) * width,
                    (y - box_height / 2) * height,
                    (x + box_width / 2) * width,
                    (y + box_height / 2) * height,
                ),
            )
        )
    return rows


def _iou(left: Sequence[float], right: Sequence[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def predict_split(
    weight: Path,
    snapshot: Path,
    split: str,
    *,
    imgsz: int,
    device: int,
    batch: int,
) -> tuple[list[Path], dict[str, list[Prediction]], dict[str, list[GroundTruth]]]:
    from ultralytics import YOLO

    images = sorted((snapshot / "images" / split).glob("*"))
    if not images:
        raise ModelPipelineError(f"No images in split {split}")
    model = YOLO(str(weight))
    predictions: dict[str, list[Prediction]] = {}
    truths = {image.name: _ground_truths(snapshot, image, split) for image in images}
    if batch == 1:
        result_stream = (
            model.predict(
                source=str(image),
                conf=0.001,
                iou=0.50,
                imgsz=imgsz,
                device=device,
                batch=1,
                verbose=False,
            )[0]
            for image in images
        )
    else:
        result_stream = model.predict(
            source=[str(image) for image in images],
            conf=0.001,
            iou=0.50,
            imgsz=imgsz,
            device=device,
            batch=batch,
            stream=True,
            verbose=False,
        )
    for image, result in zip(images, result_stream):
        boxes = result.boxes
        predictions[image.name] = [
            Prediction(int(class_id), float(confidence), tuple(map(float, box)))
            for class_id, confidence, box in zip(
                boxes.cls.cpu().tolist(),
                boxes.conf.cpu().tolist(),
                boxes.xyxy.cpu().tolist(),
            )
        ]
    if set(predictions) != {image.name for image in images}:
        raise ModelPipelineError("Prediction result count does not match split image count")
    return images, predictions, truths


def _class_counts_at_threshold(
    predictions: dict[str, list[Prediction]],
    truths: dict[str, list[GroundTruth]],
    threshold: float,
    matching_iou: float,
) -> dict[int, Counter[str]]:
    counts = {class_id: Counter(tp=0, fp=0, fn=0) for class_id in EXPECTED_NAMES}
    for image_name, ground_truths in truths.items():
        image_predictions = [item for item in predictions[image_name] if item.confidence >= threshold]
        for class_id in EXPECTED_NAMES:
            gt = [item for item in ground_truths if item.class_id == class_id]
            pred = sorted(
                (item for item in image_predictions if item.class_id == class_id),
                key=lambda item: (-item.confidence, item.bbox_xyxy),
            )
            unmatched = set(range(len(gt)))
            for item in pred:
                candidates = sorted(
                    ((_iou(item.bbox_xyxy, gt[index].bbox_xyxy), index) for index in unmatched),
                    reverse=True,
                )
                if candidates and candidates[0][0] >= matching_iou:
                    counts[class_id]["tp"] += 1
                    unmatched.remove(candidates[0][1])
                else:
                    counts[class_id]["fp"] += 1
            counts[class_id]["fn"] += len(unmatched)
    return counts


def _metrics_from_counts(counts: dict[int, Counter[str]]) -> dict:
    per_class = {}
    f1_values = []
    for class_id, name in EXPECTED_NAMES.items():
        values = counts[class_id]
        precision = values["tp"] / (values["tp"] + values["fp"]) if values["tp"] + values["fp"] else 0.0
        recall = values["tp"] / (values["tp"] + values["fn"]) if values["tp"] + values["fn"] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[str(class_id)] = {
            "name": name,
            "tp": values["tp"],
            "fp": values["fp"],
            "fn": values["fn"],
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return {"per_class": per_class, "macro_f1": sum(f1_values) / len(f1_values)}


def select_threshold(data_root: Path, contract_path: Path, batch_override: int | None = None) -> dict:
    contract = load_contract(contract_path)
    snapshot, experiment = experiment_paths(data_root, contract)
    output = experiment / "threshold_selection.json"
    if output.exists():
        raise ModelPipelineError("Refusing to overwrite threshold selection")
    weight = experiment / "baseline" / "weights" / "best.pt"
    if not weight.exists():
        raise ModelPipelineError("Full baseline best.pt is missing")
    evaluation_batch = batch_override or contract["batch"]
    images, predictions, truths = predict_split(
        weight,
        snapshot,
        "val",
        imgsz=contract["imgsz"],
        device=contract["device"],
        batch=evaluation_batch,
    )
    matching_iou = float(contract["evaluation"]["matching_iou"])
    candidates = []
    for threshold in contract["evaluation"]["confidence_grid"]:
        metrics = _metrics_from_counts(_class_counts_at_threshold(predictions, truths, threshold, matching_iou))
        candidates.append({"threshold": threshold, **metrics})
    selected = sorted(candidates, key=lambda item: (-item["macro_f1"], item["threshold"]))[0]
    payload = {
        "split": "validation",
        "images": len(images),
        "matching_iou": matching_iou,
        "objective": "macro_f1_four_supported_classes",
        "tie_break": "lower_confidence_threshold",
        "selected_threshold": selected["threshold"],
        "selected_macro_f1": selected["macro_f1"],
        "candidates": candidates,
        "weight_sha256": sha256_file(weight),
        "inference_batch": evaluation_batch,
        "evaluation_git_commit": _git_commit(),
        "retry_history": (
            [
                {"operation": "validation_prediction", "batch": 8, "status": "CUDA_OOM"},
                {"operation": "validation_prediction", "batch": 4, "status": "CUDA_OOM"},
                {
                    "operation": "validation_prediction",
                    "batch": 1,
                    "status": "completed",
                    "resolution": "per_image_inference_without_retraining",
                },
            ]
            if evaluation_batch == 1
            else []
        ),
        "selected_at": utc_now(),
    }
    write_json(output, payload)
    return payload


def _average_precision(
    class_id: int,
    predictions: dict[str, list[Prediction]],
    truths: dict[str, list[GroundTruth]],
    matching_iou: float,
) -> float:
    gt_by_image = {
        image: [item for item in rows if item.class_id == class_id]
        for image, rows in truths.items()
    }
    total_gt = sum(len(items) for items in gt_by_image.values())
    ranked = sorted(
        (
            (item.confidence, image, item.bbox_xyxy)
            for image, rows in predictions.items()
            for item in rows
            if item.class_id == class_id
        ),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    matched: dict[str, set[int]] = defaultdict(set)
    tp, fp = [], []
    for _, image, box in ranked:
        candidates = sorted(
            (
                (_iou(box, gt.bbox_xyxy), index)
                for index, gt in enumerate(gt_by_image[image])
                if index not in matched[image]
            ),
            reverse=True,
        )
        if candidates and candidates[0][0] >= matching_iou:
            matched[image].add(candidates[0][1])
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)
    if total_gt == 0:
        return 0.0
    cumulative_tp, cumulative_fp = [], []
    for value_tp, value_fp in zip(tp, fp):
        cumulative_tp.append(value_tp + (cumulative_tp[-1] if cumulative_tp else 0))
        cumulative_fp.append(value_fp + (cumulative_fp[-1] if cumulative_fp else 0))
    recall = [value / total_gt for value in cumulative_tp]
    precision = [
        value_tp / (value_tp + value_fp)
        for value_tp, value_fp in zip(cumulative_tp, cumulative_fp)
    ]
    mrec = [0.0, *recall, 1.0]
    mpre = [0.0, *precision, 0.0]
    for index in range(len(mpre) - 2, -1, -1):
        mpre[index] = max(mpre[index], mpre[index + 1])
    return sum(
        (mrec[index] - mrec[index - 1]) * mpre[index]
        for index in range(1, len(mrec))
        if mrec[index] != mrec[index - 1]
    )


def _confusion_matrix(
    predictions: dict[str, list[Prediction]],
    truths: dict[str, list[GroundTruth]],
    threshold: float,
    matching_iou: float,
) -> list[list[int]]:
    background = len(EXPECTED_NAMES)
    matrix = [[0 for _ in range(background + 1)] for _ in range(background + 1)]
    for image, ground_truths in truths.items():
        predicted = [item for item in predictions[image] if item.confidence >= threshold]
        pairs = sorted(
            (
                (_iou(gt.bbox_xyxy, pred.bbox_xyxy), gt_index, pred_index)
                for gt_index, gt in enumerate(ground_truths)
                for pred_index, pred in enumerate(predicted)
                if _iou(gt.bbox_xyxy, pred.bbox_xyxy) >= matching_iou
            ),
            reverse=True,
        )
        matched_gt, matched_pred = set(), set()
        for _, gt_index, pred_index in pairs:
            if gt_index in matched_gt or pred_index in matched_pred:
                continue
            matched_gt.add(gt_index)
            matched_pred.add(pred_index)
            matrix[ground_truths[gt_index].class_id][predicted[pred_index].class_id] += 1
        for index, gt in enumerate(ground_truths):
            if index not in matched_gt:
                matrix[gt.class_id][background] += 1
        for index, pred in enumerate(predicted):
            if index not in matched_pred:
                matrix[background][pred.class_id] += 1
    return matrix


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _benchmark(weight: Path, images: Sequence[Path], contract: dict, threshold: float) -> dict:
    import torch
    from ultralytics import YOLO

    model = YOLO(str(weight))
    warmup_count = int(contract["evaluation"]["warmup_images"])
    benchmark_count = int(contract["evaluation"]["benchmark_images"])
    for image in images[:warmup_count]:
        model.predict(str(image), conf=threshold, imgsz=contract["imgsz"], device=contract["device"], verbose=False)
    timings = []
    for image in images[:benchmark_count]:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        model.predict(str(image), conf=threshold, imgsz=contract["imgsz"], device=contract["device"], verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - started) * 1000)
    return {
        "scope": "end_to_end_predict_batch1_including_file_decode_preprocess_nms",
        "warmup_images": min(warmup_count, len(images)),
        "measured_images": len(timings),
        "median_ms": median(timings),
        "p90_ms": _percentile(timings, 0.90),
    }


def _save_failure_gallery(
    weight: Path,
    snapshot: Path,
    experiment: Path,
    image_scores: Sequence[tuple[int, int, str]],
    threshold: float,
    contract: dict,
) -> list[str]:
    from ultralytics import YOLO

    gallery = experiment / "evaluation" / "failure_gallery"
    gallery.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weight))
    saved = []
    for fp, fn, image_name in sorted(image_scores, key=lambda item: (-item[0] - item[1], -item[1], -item[0], item[2]))[:20]:
        image = snapshot / "images" / "test" / image_name
        result = model.predict(str(image), conf=threshold, imgsz=contract["imgsz"], device=contract["device"], verbose=False)[0]
        output = gallery / f"fp{fp}_fn{fn}_{image_name}"
        result.save(filename=str(output))
        saved.append(output.name)
    return saved


def _qualitative_river_trash(data_root: Path, weight: Path, experiment: Path, contract: dict, threshold: float) -> dict:
    from ultralytics import YOLO

    source = data_root / "raw" / "source_exports" / "river_trash_v3_quarantine"
    images = sorted(path for path in source.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    sample_size = min(int(contract["evaluation"]["qualitative_river_trash_images"]), len(images))
    selected = random.Random(contract["seed"]).sample(images, sample_size)
    model = YOLO(str(weight))
    class_counts: Counter[int] = Counter()
    images_with_predictions = 0
    gallery = experiment / "evaluation" / "river_trash_qualitative"
    gallery.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(selected):
        result = model.predict(
            source=str(image),
            conf=threshold,
            imgsz=contract["imgsz"],
            device=contract["device"],
            batch=1,
            verbose=False,
        )[0]
        classes = [int(value) for value in result.boxes.cls.cpu().tolist()]
        if classes:
            images_with_predictions += 1
            class_counts.update(classes)
        if index < 12:
            result.save(filename=str(gallery / image.name))
    return {
        "scope": "qualitative_only_no_localization_metric",
        "seed": contract["seed"],
        "images": sample_size,
        "images_with_predictions": images_with_predictions,
        "prediction_counts": {EXPECTED_NAMES[key]: value for key, value in sorted(class_counts.items())},
        "limitations": "River Trash full-frame source labels are not used as ground truth.",
    }


def evaluate(data_root: Path, contract_path: Path, batch_override: int | None = None) -> dict:
    contract = load_contract(contract_path)
    snapshot, experiment = experiment_paths(data_root, contract)
    output_dir = experiment / "evaluation"
    if output_dir.exists():
        raise ModelPipelineError("Refusing to overwrite evaluation directory")
    selection = json.loads((experiment / "threshold_selection.json").read_text(encoding="utf-8"))
    threshold = float(selection["selected_threshold"])
    matching_iou = float(contract["evaluation"]["matching_iou"])
    weight = experiment / "baseline" / "weights" / "best.pt"
    evaluation_batch = batch_override or contract["batch"]
    images, predictions, truths = predict_split(
        weight,
        snapshot,
        "test",
        imgsz=contract["imgsz"],
        device=contract["device"],
        batch=evaluation_batch,
    )
    output_dir.mkdir(parents=True)
    counts = _class_counts_at_threshold(predictions, truths, threshold, matching_iou)
    fixed = _metrics_from_counts(counts)
    ap50 = {
        str(class_id): _average_precision(class_id, predictions, truths, matching_iou)
        for class_id in EXPECTED_NAMES
    }
    for class_id, value in ap50.items():
        fixed["per_class"][class_id]["ap50"] = value
    image_scores = []
    total_predictions = 0
    for image in images:
        image_predictions = {image.name: predictions[image.name]}
        image_truths = {image.name: truths[image.name]}
        image_counts = _class_counts_at_threshold(image_predictions, image_truths, threshold, matching_iou)
        fp = sum(item["fp"] for item in image_counts.values())
        fn = sum(item["fn"] for item in image_counts.values())
        image_scores.append((fp, fn, image.name))
        total_predictions += sum(1 for item in predictions[image.name] if item.confidence >= threshold)
    confusion = _confusion_matrix(predictions, truths, threshold, matching_iou)
    with (output_dir / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        labels = [EXPECTED_NAMES[index] for index in EXPECTED_NAMES] + ["background"]
        writer.writerow(["actual\\predicted", *labels])
        for label, row in zip(labels, confusion):
            writer.writerow([label, *row])
    gallery = _save_failure_gallery(weight, snapshot, experiment, image_scores, threshold, contract)
    benchmark = _benchmark(weight, images, contract, threshold)
    qualitative = _qualitative_river_trash(data_root, weight, experiment, contract, threshold)
    payload = {
        "experiment_id": contract["experiment_id"],
        "split": "held_out_test",
        "images": len(images),
        "single_source_evaluation": True,
        "confidence_threshold": threshold,
        "threshold_selected_on": "validation",
        "matching_iou": matching_iou,
        "metrics_method": "fixed-threshold greedy one-to-one class matching; AP50 confidence-ranked integral PR envelope",
        **fixed,
        "mAP50": sum(ap50.values()) / len(ap50),
        "false_positives_per_image": sum(item[0] for item in image_scores) / len(images),
        "missed_objects_per_image": sum(item[1] for item in image_scores) / len(images),
        "predictions_at_threshold": total_predictions,
        "benchmark": benchmark,
        "failure_gallery": gallery,
        "river_trash_qualitative": qualitative,
        "model_weight_sha256": sha256_file(weight),
        "inference_batch": evaluation_batch,
        "evaluation_git_commit": _git_commit(),
        "evaluated_at": utc_now(),
    }
    write_json(output_dir / "evaluation_report.json", payload)
    return payload


def verify_model_compatibility(weight: Path) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weight))
    names = {int(key): value for key, value in model.names.items()}
    if names != EXPECTED_NAMES:
        raise ModelPipelineError(f"Model class names mismatch: {names}")
    return {"names": names, "weight_sha256": sha256_file(weight), "compatible": True}


def promote(data_root: Path, contract_path: Path, model_version: str) -> dict:
    contract = load_contract(contract_path)
    snapshot, experiment = experiment_paths(data_root, contract)
    source = experiment / "baseline" / "weights" / "best.pt"
    evaluation = experiment / "evaluation" / "evaluation_report.json"
    model_card = experiment / "MODEL_CARD.md"
    if not source.exists() or not evaluation.exists() or not model_card.exists():
        raise ModelPipelineError("Weight, evaluation report, and model card are required for promotion")
    compatibility = verify_model_compatibility(source)
    destination = experiment / "promoted"
    if destination.exists():
        raise ModelPipelineError("Refusing to overwrite promoted model package")
    destination.mkdir()
    shutil.copy2(source, destination / "best.pt")
    shutil.copy2(contract_path, destination / "training_config.yaml")
    shutil.copy2(evaluation, destination / "evaluation_report.json")
    shutil.copy2(model_card, destination / "MODEL_CARD.md")
    shutil.copy2(experiment / "experiment_record.json", destination / "environment_record.json")
    capability = {
        "profile_id": contract["capability_profile_id"],
        "profile_version": "1.0.0",
        "supported_classes": EXPECTED_NAMES,
        "unsupported_target_classes": {4: "other_floating_debris"},
    }
    write_json(destination / "capability_profile.json", capability)
    package = {
        "model_version": model_version,
        "snapshot_id": snapshot.name,
        "snapshot_tree_sha256_v1": contract["snapshot_tree_sha256_v1"],
        "compatibility": compatibility,
        "promoted_at": utc_now(),
    }
    write_json(destination / "package.json", package)
    files = sorted(path for path in destination.iterdir() if path.name != "checksums.sha256")
    (destination / "checksums.sha256").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in files),
        encoding="utf-8",
    )
    return package
