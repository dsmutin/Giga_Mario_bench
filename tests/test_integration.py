"""Mandatory: integrative path (CLI → baseline)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from giga_mario_bench.cli import main

pytestmark = pytest.mark.mandatory


def test_cli_writes_output_file(tmp_path: Path) -> None:
    """CLI can write the baseline JSON to a file."""
    out = tmp_path / "result.json"
    assert main(["-o", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["status"] == "baseline"
