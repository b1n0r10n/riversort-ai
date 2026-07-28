# Copyright (C) 2026 RiverSort AI contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Stable JSON and single-table CSV export from AnalysisResult."""

from __future__ import annotations

import csv
import io
import json

from riversort.application import AnalysisError
from riversort.domain import AnalysisErrorCode, AnalysisResult


class ResultExporter:
    def to_json(self, result: AnalysisResult) -> bytes:
        try:
            return json.dumps(
                result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True
            ).encode("utf-8")
        except (TypeError, ValueError) as error:
            raise AnalysisError(
                AnalysisErrorCode.EXPORT_FAILED,
                "Hasil JSON gagal dibuat.",
                "Ulangi analisis atau periksa kontrak hasil.",
            ) from error

    def to_csv(self, result: AnalysisResult) -> bytes:
        try:
            stream = io.StringIO(newline="")
            fieldnames = [
                "schema_version",
                "analysis_id",
                "media_type",
                "original_width",
                "original_height",
                "analysis_width",
                "analysis_height",
                "resolution_normalized",
                "class_id",
                "class_name",
                "support_status",
                "object_count",
                "object_count_share",
                "detected_area_share",
                "priority_status",
                "priority_score",
                "priority_tier",
                "recommendation_rule_ids",
                "model_version",
                "detector_config_version",
                "priority_config_version",
                "recommendation_config_version",
            ]
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            rule_ids = "|".join(item.rule_id for item in result.recommendations)
            for metric in result.compositions:
                writer.writerow(
                    {
                        "schema_version": result.schema_version,
                        "analysis_id": result.analysis_id,
                        "media_type": result.media_type.value,
                        "original_width": (
                            result.media_info.original_width
                            if result.media_info
                            else None
                        ),
                        "original_height": (
                            result.media_info.original_height
                            if result.media_info
                            else None
                        ),
                        "analysis_width": (
                            result.media_info.width if result.media_info else None
                        ),
                        "analysis_height": (
                            result.media_info.height if result.media_info else None
                        ),
                        "resolution_normalized": (
                            result.media_info.resolution_normalized
                            if result.media_info
                            else False
                        ),
                        "class_id": metric.class_id,
                        "class_name": metric.class_name,
                        "support_status": metric.support_status.value,
                        "object_count": metric.object_count,
                        "object_count_share": metric.object_count_share,
                        "detected_area_share": metric.detected_area_share,
                        "priority_status": result.priority.status,
                        "priority_score": result.priority.score,
                        "priority_tier": result.priority.tier,
                        "recommendation_rule_ids": rule_ids,
                        "model_version": result.model_version,
                        "detector_config_version": result.detector_config_version,
                        "priority_config_version": result.priority_config_version,
                        "recommendation_config_version": result.recommendation_config_version,
                    }
                )
            return stream.getvalue().encode("utf-8")
        except (csv.Error, TypeError, ValueError) as error:
            raise AnalysisError(
                AnalysisErrorCode.EXPORT_FAILED,
                "Hasil CSV gagal dibuat.",
                "Ulangi analisis atau periksa kontrak hasil.",
            ) from error
