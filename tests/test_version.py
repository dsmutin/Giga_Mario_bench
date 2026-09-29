"""Mandatory: version is defined once in VERSION."""

from __future__ import annotations

from pathlib import Path

import pytest

from giga_mario_bench import __version__

pytestmark = pytest.mark.mandatory


def test_version_file_matches_package(repo_root: Path) -> None:
    """Package __version__ must equal the VERSION file."""
    written = (repo_root / "VERSION").read_text(encoding="utf-8").strip()
    assert written == __version__
    assert written == "0.0.1" or len(written.split(".")) == 3
