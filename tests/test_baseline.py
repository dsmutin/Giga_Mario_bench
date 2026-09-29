"""Mandatory: baseline pipeline contract."""

from __future__ import annotations

import pytest

from giga_mario_bench.baseline import run_pipeline

pytestmark = pytest.mark.mandatory


def test_run_pipeline_keys() -> None:
    """Baseline result has status, ok, and input_path."""
    result = run_pipeline()
    assert set(result) >= {"status", "ok", "input_path"}
    assert result["status"] == "baseline"
    assert result["ok"] is True
