"""giga_mario_bench: Benchmarks of data-split methods for in silico ligand hypotheses, with shared models and scores."""

from __future__ import annotations

from pathlib import Path


def _version() -> str:
    """Return the package version from the repository VERSION file."""
    path = Path(__file__).resolve().parents[2] / "VERSION"
    return path.read_text(encoding="utf-8").strip()


__version__ = _version()
