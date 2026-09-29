"""Baseline implementations that hold contracts until real logic lands."""

from __future__ import annotations


def run_pipeline(input_path: str | None = None) -> dict:
    """Run the tool pipeline.

    Baseline: succeed with a structured placeholder result. Replace this
    function body with the real implementation; keep the return keys.
    """
    return {
        "status": "baseline",
        "ok": True,
        "input_path": input_path,
    }
