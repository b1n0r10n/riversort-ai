# Copyright (C) 2026 RiverSort AI contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Framework-independent contracts and deterministic business rules."""

from .contracts import (
    AnalysisErrorCode,
    AnalysisRequest,
    AnalysisResult,
    CompositionMetric,
    Detection,
    MediaInfo,
    MediaType,
    ModelCapabilityProfile,
    PriorityAssessment,
    QualityFlag,
    QualityFlagCode,
    Recommendation,
    SupportStatus,
    TrackSummary,
)
from .composition import compose_image, compose_video, percentile90, union_bbox_coverage
from .errors import DomainInputError
from .priority import PriorityPolicy, PriorityTier, assess_priority, tier_for_score
from .quality import (
    CANONICAL_COMPOSITION_CLAIM,
    build_quality_flags,
    classification_uncertainty_evidence,
    standard_limitations,
    tracking_instability_evidence,
)
from .recommendations import (
    RecommendationPolicy,
    RecommendationRule,
    evaluate_recommendations,
)

__all__ = [
    "AnalysisRequest",
    "AnalysisResult",
    "AnalysisErrorCode",
    "CompositionMetric",
    "Detection",
    "MediaInfo",
    "MediaType",
    "ModelCapabilityProfile",
    "PriorityAssessment",
    "QualityFlag",
    "QualityFlagCode",
    "Recommendation",
    "SupportStatus",
    "TrackSummary",
    "CANONICAL_COMPOSITION_CLAIM",
    "DomainInputError",
    "PriorityPolicy",
    "PriorityTier",
    "RecommendationPolicy",
    "RecommendationRule",
    "assess_priority",
    "build_quality_flags",
    "classification_uncertainty_evidence",
    "compose_image",
    "compose_video",
    "evaluate_recommendations",
    "percentile90",
    "standard_limitations",
    "tier_for_score",
    "tracking_instability_evidence",
    "union_bbox_coverage",
]
