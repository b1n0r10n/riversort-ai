# Copyright (C) 2026 RiverSort AI contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Framework-independent domain errors."""


class DomainInputError(ValueError):
    """Raised when an input violates a deterministic domain contract."""
