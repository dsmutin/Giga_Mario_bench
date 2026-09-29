#!/usr/bin/env python3
"""Toy reverse-complement benchmark.

Ten random DNA strings, ten point-mutants (rate 0.1), two pairs held out as
zero-shot, and a Giga_Mario random split of the other sixteen sequences.
Both baseline models train until validation loss plateaus.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from giga_mario_bench.pipeline import run_cell  # noqa: E402

SPEC = {
    "n_pairs": 10,
    "length": 16,
    "rate": 0.1,
    "seed": 42,
    "zsv_pairs": 2,
}
MODELS = ("many_to_many_rnn", "encoder_decoder")


def run() -> int:
    """Train both toy models and require finite F1, R2, and ROC AUC."""
    out = Path(__file__).resolve().parent / "out"
    summaries = []
    for model in MODELS:
        result = run_cell(
            bench="reverse_complement",
            spec=SPEC,
            split="random",
            model=model,
            data_out=out / "data",
            model_out=out / "models",
            score_out=out / "scores",
            hidden=16,
            max_epochs=40,
            patience=5,
        )
        summaries.append(
            {
                "model": model,
                "f1": result["f1"],
                "r2": result["r2"],
                "rocauc": result["rocauc"],
                "scores": result["scores"],
            }
        )
        for key in ("f1", "r2", "rocauc"):
            if result[key] != result[key]:
                print(f"{model} produced a non-finite {key}", file=sys.stderr)
                return 1
    payload = {"status": "ok", "bench": "reverse_complement", "models": summaries}
    dest = Path(__file__).resolve().parent / "data" / "toy_result.json"
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
