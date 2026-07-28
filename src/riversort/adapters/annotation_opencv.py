# Copyright (C) 2026 RiverSort AI contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""OpenCV annotation renderer for canonical detections and tracks."""

from __future__ import annotations

from typing import Any, Sequence

import cv2

from riversort.application import AnalysisError, TrackObservation
from riversort.domain import AnalysisErrorCode, Detection


COLORS = {
    0: (255, 120, 0),
    1: (0, 180, 255),
    2: (160, 80, 255),
    3: (80, 220, 80),
}


class OpenCvAnnotator:
    def annotate_image(self, pixels: Any, detections: Sequence[Detection]) -> bytes:
        annotated = pixels.copy()
        for detection in detections:
            self._draw(annotated, detection, None)
        ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise AnalysisError(
                AnalysisErrorCode.EXPORT_FAILED,
                "Preview gambar gagal dibuat.",
                "Coba media lain atau ulangi analisis.",
            )
        return encoded.tobytes()

    def annotate_video_frame(
        self, pixels: Any, observations: Sequence[TrackObservation]
    ) -> Any:
        annotated = pixels.copy()
        for observation in observations:
            self._draw(
                annotated,
                observation.detection,
                observation.track_id,
                classification_status=observation.classification_status,
            )
        return annotated

    @staticmethod
    def _draw(
        frame: Any,
        detection: Detection,
        track_id: int | None,
        *,
        classification_status: str | None = None,
    ) -> None:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = (round(value) for value in detection.bbox_xyxy)
        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width - 1, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(0, min(height - 1, y2))
        uncertain = track_id is not None and classification_status == "uncertain"
        color = (
            (120, 120, 120)
            if uncertain
            else COLORS.get(detection.class_id, (255, 255, 255))
        )
        scale_basis = min(height, width)
        font_scale = min(1.0, max(0.55, scale_basis / 1440.0))
        thickness = max(2, min(3, round(scale_basis / 720.0)))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        if track_id is None:
            label = f"{detection.class_name} {detection.confidence:.2f}"
        elif uncertain:
            label = (
                f"ID {track_id} | kelas belum pasti | "
                f"dukungan {detection.confidence:.0%}"
            )
        else:
            label = (
                f"ID {track_id} | {detection.class_name} | "
                f"dukungan {detection.confidence:.0%}"
            )
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness,
        )
        text_x = max(0, min(x1, width - text_width - 8))
        if y1 - text_height - baseline - 8 >= 0:
            background_top = y1 - text_height - baseline - 8
            background_bottom = y1
            text_y = y1 - baseline - 4
        else:
            background_top = y1
            background_bottom = min(height - 1, y1 + text_height + baseline + 8)
            text_y = min(height - baseline - 1, y1 + text_height + 4)
        cv2.rectangle(
            frame,
            (text_x, background_top),
            (min(width - 1, text_x + text_width + 8), background_bottom),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (text_x + 4, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
