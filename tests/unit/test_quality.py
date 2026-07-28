# Copyright (C) 2026 RiverSort AI contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

from riversort.domain import (
    CANONICAL_COMPOSITION_CLAIM,
    CompositionMetric,
    QualityFlagCode,
    SupportStatus,
    TrackSummary,
    build_quality_flags,
    classification_uncertainty_evidence,
    standard_limitations,
    tracking_instability_evidence,
)
from riversort.infrastructure.config import capability_profile, load_project_config


def empty_rows():
    config = load_project_config()
    capability = capability_profile(config)
    rows = tuple(
        CompositionMetric(class_id, name, SupportStatus.SUPPORTED, 0, None, None)
        for class_id, name in capability.supported_classes.items()
    ) + tuple(
        CompositionMetric(class_id, name, SupportStatus.NOT_SUPPORTED, None, None, None)
        for class_id, name in capability.unsupported_target_classes.items()
    )
    return capability, rows


def test_quality_flags_require_evidence_for_low_quality() -> None:
    capability, rows = empty_rows()
    base_codes = {flag.code for flag in build_quality_flags(rows, capability)}
    assert base_codes == {
        QualityFlagCode.NO_VALID_OBJECT,
        QualityFlagCode.UNSUPPORTED_TARGET_CLASS,
    }
    flags = build_quality_flags(rows, capability, low_quality_evidence={"mean_confidence": 0.2})
    low = next(flag for flag in flags if flag.code is QualityFlagCode.LOW_ANALYSIS_QUALITY)
    assert low.evidence == {"mean_confidence": 0.2}
    mismatch = build_quality_flags(rows, capability, config_mismatch={"model_names": "mismatch"})
    config_flag = next(flag for flag in mismatch if flag.code is QualityFlagCode.CONFIG_MISMATCH)
    assert config_flag.evidence == {"model_names": "mismatch"}


def test_standard_limitations_are_safe_and_complete() -> None:
    capability, _ = empty_rows()
    limitations = standard_limitations(capability)
    assert limitations[0] == CANONICAL_COMPOSITION_CLAIM
    assert any("bukan estimasi berat" in item for item in limitations)
    assert any("false positive" in item for item in limitations)
    assert any("debris nonplastik" in item for item in limitations)
    assert any("lebih rendah" in item for item in limitations)
    assert any("Verifikasi manusia" in item for item in limitations)


def track(
    track_id: int,
    observations: int,
    *,
    margin: float = 1.0,
) -> TrackSummary:
    return TrackSummary(
        track_id,
        0,
        "plastic_bottle",
        True,
        {0: 1.0},
        (0.1,),
        0,
        observations,
        observations,
        1.0,
        margin,
        "stable" if margin >= 0.20 else "uncertain",
    )


def test_classification_uncertainty_uses_strict_margin_boundary() -> None:
    assert (
        classification_uncertainty_evidence(
            [track(1, 5, margin=0.20)],
            margin_threshold=0.20,
        )
        is None
    )
    evidence = classification_uncertainty_evidence(
        [track(1, 5, margin=0.19)],
        margin_threshold=0.20,
    )
    assert evidence["uncertain_track_count"] == 1
    assert evidence["tracks"][0]["track_id"] == 1


def test_tracking_instability_requires_majority_of_short_tracks() -> None:
    boundary = tracking_instability_evidence(
        [track(1, 4), track(2, 4), track(3, 8), track(4, 8)],
        analysis_fps=5,
        minimum_track_count=3,
        short_track_seconds=1.0,
        short_track_ratio_threshold=0.50,
    )
    assert boundary is None
    evidence = tracking_instability_evidence(
        [track(1, 4), track(2, 4), track(3, 4), track(4, 8)],
        analysis_fps=5,
        minimum_track_count=3,
        short_track_seconds=1.0,
        short_track_ratio_threshold=0.50,
    )
    assert evidence["short_track_observation_limit"] == 5
    assert evidence["short_track_ratio"] == 0.75


def test_quality_flags_include_track_evidence() -> None:
    capability, rows = empty_rows()
    flags = build_quality_flags(
        rows,
        capability,
        classification_uncertainty_evidence={"uncertain_track_count": 1},
        tracking_instability_evidence={"short_track_ratio": 0.75},
    )
    codes = {flag.code for flag in flags}
    assert QualityFlagCode.CLASSIFICATION_UNCERTAIN in codes
    assert QualityFlagCode.TRACKING_UNSTABLE in codes
