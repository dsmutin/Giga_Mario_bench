"""Load Giga_Mario split-predict without copying its assignment code."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    """Return the giga_mario_bench repository root."""
    return Path(__file__).resolve().parents[2]


def giga_mario_root() -> Path:
    """Resolve the Giga_Mario checkout.

    ``GIGA_MARIO_ROOT`` wins. Otherwise the sibling directory ``Giga_Mario``
    next to this repository is used.
    """
    env = os.environ.get("GIGA_MARIO_ROOT")
    if env:
        root = Path(env).expanduser().resolve()
    else:
        root = repo_root().parent / "Giga_Mario"
    marker = root / "src" / "pipeline" / "split_predict.py"
    if not marker.is_file():
        raise FileNotFoundError(
            "Giga_Mario split-predict was not found at "
            f"{marker}. Clone https://github.com/Sirius-Back/Giga_Mario "
            "beside this repo or set GIGA_MARIO_ROOT."
        )
    return root


def run_split_predict(**kwargs: Any) -> Path:
    """Call ``src.pipeline.split_predict.run_split_predict`` from Giga_Mario.

    Giga_Mario's pipeline lives in a top-level ``src`` package. This inserts
    that checkout ahead of any other ``src`` entry so the bench package does
    not shadow it.
    """
    root = giga_mario_root()
    root_s = str(root)
    if root_s in sys.path:
        sys.path.remove(root_s)
    sys.path.insert(0, root_s)
    for name in list(sys.modules):
        if name == "src" or name.startswith("src."):
            del sys.modules[name]
    from src.pipeline.split_predict import run_split_predict as _run

    out = _run(**kwargs)
    return Path(out)
