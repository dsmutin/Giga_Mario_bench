"""Altair and cnsplots views of F1, R2, and ROC AUC."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd


def plot_scores(result: dict[str, Any], out_prefix: Path) -> dict[str, str | None]:
    """Write an Altair HTML chart and, when available, a cnsplots PNG."""
    rows = [
        {"metric": name, "value": float(result[name])}
        for name in ("f1", "r2", "rocauc")
    ]
    frame = pd.DataFrame(rows)
    chart = (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X("metric:N", sort=["f1", "r2", "rocauc"], title="metric"),
            y=alt.Y("value:Q", title="score"),
        )
        .properties(title="Benchmark scores", width=280, height=180)
    )
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    html_path = out_prefix.with_suffix(".html")
    chart.save(str(html_path))
    png_path = _cns_bar(frame, out_prefix.with_suffix(".png"))
    return {
        "html": str(html_path),
        "png": None if png_path is None else str(png_path),
    }


def _cns_bar(frame: pd.DataFrame, png_path: Path) -> Path | None:
    try:
        import matplotlib.pyplot as plt
        import cnsplots
    except ImportError:
        return None
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    cnsplots.barplot(frame, x="metric", y="value", ax=ax, add_tip=True)
    ax.set_title("Benchmark scores")
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    return png_path
