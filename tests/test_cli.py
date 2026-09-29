"""Mandatory: CLI entry works."""

from __future__ import annotations

import json

import pytest

from giga_mario_bench.cli import main

pytestmark = pytest.mark.mandatory


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    """--version prints the package version."""
    assert main(["--version"]) == 0
    out = capsys.readouterr().out.strip()
    assert out


def test_cli_baseline_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    """Default CLI run prints JSON with baseline status."""
    assert main([]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "baseline"
    assert payload["ok"] is True
