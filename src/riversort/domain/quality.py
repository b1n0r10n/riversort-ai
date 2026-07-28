# Copyright (C) 2026 RiverSort AI contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Quality flags and mandatory limitations for explainable results."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .contracts import (
    CompositionMetric,
    ModelCapabilityProfile,
    QualityFlag,
    QualityFlagCode,
    SupportStatus,
    TrackSummary,
)


CANONICAL_COMPOSITION_CLAIM = (
    "Estimasi komposisi visual berdasarkan jumlah objek unik dan proporsi area deteksi."
)


def build_quality_flags(
    compositions: Sequence[CompositionMetric],
    capability: ModelCapabilityProfile,
    *,
    config_mismatch: Mapping[str, Any] | None = None,
    low_quality_evidence: Mapping[str, Any] | None = None,
    classification_uncertainty_evidence: Mapping[str, Any] | None = None,
    tracking_instability_evidence: Mapping[str, Any] | None = None,
) -> tuple[QualityFlag, ...]:
    flags = []
    valid_count = sum(
        metric.object_count or 0
        for metric in compositions
        if metric.support_status is SupportStatus.SUPPORTED
    )
    if valid_count == 0:
        flags.append(
            QualityFlag(
                QualityFlagCode.NO_VALID_OBJECT,
                "Tidak ada objek valid dari kelas yang didukung; kondisi sungai tidak dapat disimpulkan.",
                {"supported_object_count": 0},
            )
        )
    if capability.unsupported_target_classes:
        flags.append(
            QualityFlag(
                QualityFlagCode.UNSUPPORTED_TARGET_CLASS,
                "Model MVP belum mendukung seluruh kelas pada taksonomi target.",
                {"class_ids": tuple(sorted(capability.unsupported_target_classes))},
            )
        )
    if config_mismatch:
        flags.append(
            QualityFlag(
                QualityFlagCode.CONFIG_MISMATCH,
                "Konfigurasi model dan capability profile tidak cocok.",
                dict(config_mismatch),
            )
        )
    if low_quality_evidence:
        flags.append(
            QualityFlag(
                QualityFlagCode.LOW_ANALYSIS_QUALITY,
                "Kualitas analisis rendah berdasarkan evidence terukur.",
                dict(low_quality_evidence),
            )
        )
    if classification_uncertainty_evidence:
        flags.append(
            QualityFlag(
                QualityFlagCode.CLASSIFICATION_UNCERTAIN,
                "Satu atau lebih track belum memiliki dukungan kelas yang stabil.",
                dict(classification_uncertainty_evidence),
            )
        )
    if tracking_instability_evidence:
        flags.append(
            QualityFlag(
                QualityFlagCode.TRACKING_UNSTABLE,
                "Banyak track hanya bertahan singkat; jumlah objek dapat terfragmentasi.",
                dict(tracking_instability_evidence),
            )
        )
    return tuple(flags)


def classification_uncertainty_evidence(
    tracks: Sequence[TrackSummary],
    *,
    margin_threshold: float,
) -> Mapping[str, Any] | None:
    uncertain = [
        {
            "track_id": track.track_id,
            "class_vote_margin": track.class_vote_margin,
        }
        for track in tracks
        if track.confirmed
        and track.class_vote_margin is not None
        and track.class_vote_margin < margin_threshold
    ]
    if not uncertain:
        return None
    return {
        "margin_threshold": margin_threshold,
        "uncertain_track_count": len(uncertain),
        "tracks": tuple(uncertain),
    }


def tracking_instability_evidence(
    tracks: Sequence[TrackSummary],
    *,
    analysis_fps: float,
    minimum_track_count: int,
    short_track_seconds: float,
    short_track_ratio_threshold: float,
) -> Mapping[str, Any] | None:
    confirmed = [track for track in tracks if track.confirmed]
    if len(confirmed) < minimum_track_count:
        return None
    observation_limit = max(1, math.ceil(analysis_fps * short_track_seconds))
    short_tracks = [
        track
        for track in confirmed
        if (track.observation_count or 0) < observation_limit
    ]
    ratio = len(short_tracks) / len(confirmed)
    if ratio <= short_track_ratio_threshold:
        return None
    return {
        "confirmed_track_count": len(confirmed),
        "short_track_count": len(short_tracks),
        "short_track_ratio": ratio,
        "short_track_observation_limit": observation_limit,
        "short_track_seconds": short_track_seconds,
        "ratio_threshold": short_track_ratio_threshold,
        "track_ids": tuple(track.track_id for track in short_tracks),
    }


def standard_limitations(capability: ModelCapabilityProfile) -> tuple[str, ...]:
    limitations = [
        CANONICAL_COMPOSITION_CLAIM,
        "Hasil bukan estimasi berat, massa, tonase, atau volume fisik.",
        "Hasil dapat mengandung false positive dan false negative.",
    ]
    if capability.unsupported_target_classes:
        limitations.extend(
            [
                "Model MVP belum mendukung debris nonplastik.",
                "Indeks prioritas dapat lebih rendah daripada kondisi lapangan sebenarnya.",
            ]
        )
    limitations.append("Verifikasi manusia tetap diperlukan sebelum tindakan pembersihan.")
    return tuple(limitations)
