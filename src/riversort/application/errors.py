# Copyright (C) 2026 RiverSort AI contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Stable application failures that presentation can safely render."""

from __future__ import annotations

from riversort.domain import AnalysisErrorCode


class AnalysisError(RuntimeError):
    def __init__(
        self,
        code: AnalysisErrorCode,
        public_message: str,
        recovery_action: str,
    ) -> None:
        super().__init__(f"{code.value}: {public_message}")
        self.code = code
        self.public_message = public_message
        self.recovery_action = recovery_action


class ModelNotReadyError(AnalysisError):
    error_code = AnalysisErrorCode.MODEL_NOT_READY.value

    def __init__(self, message: str = "Model lokal belum siap digunakan.") -> None:
        super().__init__(
            AnalysisErrorCode.MODEL_NOT_READY,
            message,
            "Atur RIVERSORT_MODEL_PATH ke weight yang telah dipromosikan.",
        )
