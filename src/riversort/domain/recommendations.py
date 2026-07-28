# Copyright (C) 2026 RiverSort AI contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Deterministic and traceable recommendation rule engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .contracts import CompositionMetric, ModelCapabilityProfile, Recommendation, SupportStatus
from .errors import DomainInputError


@dataclass(frozen=True)
class RecommendationRule:
    rule_id: str
    priority: int
    class_id: int | None
    metric: str
    operator: str
    threshold: float
    applicable_profiles: tuple[str, ...]
    reason: str
    action: str


@dataclass(frozen=True)
class RecommendationPolicy:
    config_version: str
    rules: tuple[RecommendationRule, ...]


def _matches(observed: float | int, operator: str, threshold: float) -> bool:
    if operator == ">":
        return observed > threshold
    if operator == ">=":
        return observed >= threshold
    if operator == "==":
        return observed == threshold
    raise DomainInputError(f"Unsupported rule operator {operator}")


def evaluate_recommendations(
    compositions: Sequence[CompositionMetric],
    capability: ModelCapabilityProfile,
    policy: RecommendationPolicy,
) -> tuple[Recommendation, ...]:
    """Evaluate all applicable rules and preserve deterministic priority order."""

    by_class = {metric.class_id: metric for metric in compositions}
    valid_object_count = sum(
        metric.object_count or 0
        for metric in compositions
        if metric.support_status is SupportStatus.SUPPORTED
    )
    triggered: list[tuple[int, Recommendation]] = []
    for rule in policy.rules:
        if capability.profile_id not in rule.applicable_profiles:
            continue
        if rule.class_id is None:
            if rule.metric != "valid_object_count":
                raise DomainInputError(f"Rule {rule.rule_id} lacks a class for metric {rule.metric}")
            observed: float | int = valid_object_count
            evidence_fields = ("composition.supported_object_count",)
        else:
            if rule.class_id in capability.unsupported_target_classes:
                raise DomainInputError(f"Rule {rule.rule_id} activates an unsupported class")
            metric = by_class.get(rule.class_id)
            if metric is None or metric.support_status is not SupportStatus.SUPPORTED:
                raise DomainInputError(f"Missing supported composition row for rule {rule.rule_id}")
            if rule.metric != "object_count_share":
                raise DomainInputError(f"Unsupported class rule metric {rule.metric}")
            if metric.object_count_share is None:
                continue
            observed = metric.object_count_share
            evidence_fields = (
                f"composition.{metric.class_name}.object_count_share",
                "capability_profile.profile_id",
            )
        if _matches(observed, rule.operator, rule.threshold):
            triggered.append(
                (
                    rule.priority,
                    Recommendation(
                        rule_id=rule.rule_id,
                        rule_version=policy.config_version,
                        observed_metric=rule.metric,
                        observed_value=observed,
                        operator=rule.operator,
                        threshold=rule.threshold,
                        reason=rule.reason,
                        action=rule.action,
                        evidence_fields=evidence_fields,
                    ),
                )
            )
    return tuple(item for _, item in sorted(triggered, key=lambda pair: (pair[0], pair[1].rule_id)))
