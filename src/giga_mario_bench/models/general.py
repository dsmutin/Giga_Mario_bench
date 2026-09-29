"""Model registry. ``get`` initializes a local baseline or records an existing one.

Genomic foundation models that Giga_Mario already downloads and adapts should
be added in that repository. These two baselines exist only for the
reverse-complement toy and have no remote checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MODELS: dict[str, dict[str, Any]] = {
    "many_to_many_rnn": {
        "hidden": 32,
        "lr": 0.05,
        "family": "bidirectional-elman-rnn",
        "source": "in-process",
    },
    "encoder_decoder": {
        "hidden": 32,
        "lr": 0.05,
        "family": "encoder-decoder-rnn",
        "source": "in-process",
    },
}


def list_models() -> list[str]:
    """Return registered model names."""
    return sorted(MODELS)


def model_params_slug(name: str, hidden: int | None = None) -> str:
    """Directory slug for one model hyperparameter set."""
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}")
    width = MODELS[name]["hidden"] if hidden is None else hidden
    return f"h{width}"


def initialize_model(name: str, model_out: Path, *, all_models: bool = False) -> dict[str, Any]:
    """Write an initialization marker. Skip models that are already present."""
    names = list_models() if all_models else [name]
    if not all_models and name not in MODELS:
        raise ValueError(f"unknown model {name!r}; known: {', '.join(list_models())}")
    results = []
    for model_name in names:
        dest = Path(model_out) / model_name / "initialized.json"
        if dest.is_file():
            print(f"model already initialized: {model_name}")
            results.append({"model": model_name, "status": "exists", "path": str(dest)})
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        payload = {"model": model_name, "status": "initialized", **MODELS[model_name]}
        dest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"initialized model: {model_name}")
        results.append({"model": model_name, "status": "initialized", "path": str(dest)})
    return {"results": results}
