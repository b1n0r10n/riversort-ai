# Copyright (C) 2026 RiverSort AI contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Ultralytics YOLO adapter with artifact and capability verification."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Sequence

from riversort.application import AnalysisError
from riversort.domain import AnalysisErrorCode, Detection, ModelCapabilityProfile
from riversort.infrastructure.config import DetectorConfig


LOGGER = logging.getLogger(__name__)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class YoloDetectorAdapter:
    def __init__(
        self,
        config: DetectorConfig,
        capability: ModelCapabilityProfile,
        *,
        model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        if config.status != "configured":
            raise AnalysisError(
                AnalysisErrorCode.MODEL_NOT_READY,
                "Detector belum dikonfigurasi.",
                "Gunakan artifact model yang telah dipromosikan.",
            )
        model_value = os.environ.get(config.model_path_env)
        if not model_value:
            raise AnalysisError(
                AnalysisErrorCode.MODEL_NOT_READY,
                f"Environment {config.model_path_env} belum diatur.",
                "Arahkan environment ke promoted best.pt.",
            )
        model_path = Path(model_value).expanduser().resolve()
        if not model_path.is_file():
            raise AnalysisError(
                AnalysisErrorCode.MODEL_NOT_READY,
                "Weight model lokal tidak ditemukan.",
                "Periksa RIVERSORT_MODEL_PATH dan artifact promoted.",
            )
        actual_hash = sha256_file(model_path)
        if actual_hash != config.model_sha256:
            raise AnalysisError(
                AnalysisErrorCode.MODEL_CHECKSUM_MISMATCH,
                "Checksum weight tidak cocok dengan detector config.",
                "Pulihkan weight promoted yang benar; jangan menjalankan inference.",
            )
        self._verify_package_manifest(model_path, config)
        expected_names = dict(capability.supported_classes)
        if config.names != expected_names:
            raise AnalysisError(
                AnalysisErrorCode.MODEL_CLASS_MAP_MISMATCH,
                "Class map detector tidak cocok dengan capability profile.",
                "Perbaiki config/artifact sebelum inference.",
            )
        if model_factory is None:
            from ultralytics import YOLO

            model_factory = YOLO
        try:
            self.model = model_factory(str(model_path))
        except Exception as error:
            raise AnalysisError(
                AnalysisErrorCode.MODEL_NOT_READY,
                "Weight model gagal dimuat.",
                "Periksa kompatibilitas environment dan artifact.",
            ) from error
        model_names = {int(key): str(value) for key, value in dict(self.model.names).items()}
        if model_names != expected_names:
            raise AnalysisError(
                AnalysisErrorCode.MODEL_CLASS_MAP_MISMATCH,
                "Nama kelas di dalam weight tidak cocok dengan profile aktif.",
                "Gunakan weight baseline RiverSort yang dipromosikan.",
            )
        self.config = config
        self.capability = capability

    @staticmethod
    def _verify_package_manifest(model_path: Path, config: DetectorConfig) -> None:
        manifest_path = model_path.parent / config.package_manifest_filename
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            compatibility = payload["compatibility"]
            names = {
                int(key): str(value)
                for key, value in dict(compatibility["names"]).items()
            }
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise AnalysisError(
                AnalysisErrorCode.MODEL_PACKAGE_MISMATCH,
                "Manifest bundle model tidak tersedia atau tidak valid.",
                "Gunakan folder promoted lengkap, bukan hanya menyalin best.pt.",
            ) from error
        expected = {
            "model_version": config.model_version,
            "snapshot_id": config.dataset_snapshot_id,
            "snapshot_tree_sha256_v1": config.dataset_snapshot_sha256,
        }
        actual = {key: payload.get(key) for key in expected}
        if (
            actual != expected
            or compatibility.get("weight_sha256") != config.model_sha256
            or names != config.names
            or compatibility.get("compatible") is not True
        ):
            raise AnalysisError(
                AnalysisErrorCode.MODEL_PACKAGE_MISMATCH,
                "Manifest bundle model tidak cocok dengan detector config.",
                "Pulihkan bundle promoted yang sesuai snapshot dan versi model.",
            )

    def detect_image(self, pixels: Any) -> Sequence[Detection]:
        return self._predict(
            pixels,
            confidence=float(self.config.confidence_threshold),
            frame_index=None,
            timestamp_ms=None,
        )

    def detect_frame(
        self,
        pixels: Any,
        *,
        confidence_floor: float,
        frame_index: int,
        timestamp_ms: int,
    ) -> Sequence[Detection]:
        return self._predict(
            pixels,
            confidence=confidence_floor,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
        )

    def _predict(
        self,
        pixels: Any,
        *,
        confidence: float,
        frame_index: int | None,
        timestamp_ms: int | None,
    ) -> tuple[Detection, ...]:
        try:
            result = self.model.predict(
                source=pixels,
                conf=confidence,
                iou=self.config.iou_threshold,
                imgsz=self.config.inference_imgsz,
                verbose=False,
            )[0]
            boxes = result.boxes
            height, width = pixels.shape[:2]
            detections = []
            invalid_bbox_count = 0
            if boxes is None:
                return ()
            xyxy_rows = boxes.xyxy.detach().cpu().tolist()
            confidences = boxes.conf.detach().cpu().tolist()
            class_ids = boxes.cls.detach().cpu().tolist()
            for raw_box, raw_confidence, raw_class_id in zip(
                xyxy_rows, confidences, class_ids
            ):
                class_id = int(raw_class_id)
                class_name = self.capability.supported_classes.get(class_id)
                if class_name is None:
                    continue
                x1, y1, x2, y2 = (
                    max(0.0, min(float(width), float(raw_box[0]))),
                    max(0.0, min(float(height), float(raw_box[1]))),
                    max(0.0, min(float(width), float(raw_box[2]))),
                    max(0.0, min(float(height), float(raw_box[3]))),
                )
                if x2 <= x1 or y2 <= y1:
                    invalid_bbox_count += 1
                    continue
                detections.append(
                    Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=float(raw_confidence),
                        bbox_xyxy=(x1, y1, x2, y2),
                        frame_index=frame_index,
                        timestamp_ms=timestamp_ms,
                    )
                )
            if invalid_bbox_count:
                LOGGER.warning(
                    "invalid detector boxes discarded",
                    extra={
                        "stage": "inference",
                        "error_code": "INVALID_BBOX_DISCARDED",
                        "invalid_bbox_count": invalid_bbox_count,
                    },
                )
            return tuple(detections)
        except AnalysisError:
            raise
        except Exception as error:
            raise AnalysisError(
                AnalysisErrorCode.INFERENCE_FAILED,
                "Inference model gagal.",
                "Coba media lain atau muat ulang model lokal.",
            ) from error
