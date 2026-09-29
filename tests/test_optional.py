"""Optional tests (release / workflow_dispatch)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.optional


def test_optional_placeholder() -> None:
    """Optional suite is collected. Replace with slow or extra checks."""
    pass
