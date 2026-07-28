# Copyright (C) 2026 RiverSort AI contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest

from riversort.adapters import build_detector_runtime
from riversort.application import ModelNotReadyError
from riversort.infrastructure import load_project_config


def test_unconfigured_detector_fails_with_stable_error_code() -> None:
    config = load_project_config()
    unconfigured = config.detector.model_copy(update={"status": "unconfigured"})
    with pytest.raises(ModelNotReadyError, match="MODEL_NOT_READY") as error:
        build_detector_runtime(unconfigured)
    assert error.value.error_code == "MODEL_NOT_READY"


def test_detector_builder_requires_full_project_config() -> None:
    config = load_project_config()
    with pytest.raises(ModelNotReadyError, match="full project config"):
        build_detector_runtime(config.detector)
