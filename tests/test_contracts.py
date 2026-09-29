"""Contract tests for the reverse-complement toy.

Two contracts, both required:

1. raw sequences become model-ready integer rows
2. model-ready rows go through a Giga_Mario random split, a short train, and a score
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from giga_mario_bench.bench.build import build_one
from giga_mario_bench.bench.reverse_complement.adaptor import reverse_complement
from giga_mario_bench.io_utils import read_jsonl, read_pipe
from giga_mario_bench.models.data_prepare.encode import prepare_records
from giga_mario_bench.models.general import MODELS
from giga_mario_bench.pipeline import run_cell

pytestmark = [pytest.mark.mandatory, pytest.mark.contracts]

SPEC = {"n_pairs": 10, "length": 8, "rate": 0.1, "seed": 7, "zsv_pairs": 2}


def test_raw_to_model_ready(tmp_path: Path) -> None:
    """Generator output adapts to x/y integers the models consume."""
    built = build_one("reverse_complement", tmp_path / "bench", spec=SPEC)
    assert built["status"] == "built"
    records = read_jsonl(tmp_path / "bench" / "input" / "records.jsonl")
    answers = read_jsonl(tmp_path / "bench" / "output" / "answers.jsonl")
    assert len(records) == 20
    originals = [row for row in records if row["role"] == "original"]
    mutants = [row for row in records if row["role"] == "mutant"]
    assert len(originals) == 10 and len(mutants) == 10
    ready = prepare_records(records, answers)
    assert len(ready) == 20
    for row, answer in zip(ready, answers):
        assert len(row["x"]) == SPEC["length"]
        assert len(row["y"]) == SPEC["length"]
        assert set(row["x"]).issubset({0, 1, 2, 3})
        text = records[[item["id"] for item in records].index(row["id"])]["sequence"]
        assert answer["sequence"] == reverse_complement(text)


def test_model_ready_split_train_score(tmp_path: Path) -> None:
    """Random Giga_Mario split, short plateau train, then F1 / R2 / ROC AUC."""
    for model in MODELS:
        result = run_cell(
            bench="reverse_complement",
            spec=SPEC,
            split="random",
            model=model,
            data_out=tmp_path / "data",
            model_out=tmp_path / "models",
            score_out=tmp_path / "scores",
            hidden=8,
            max_epochs=3,
            patience=2,
        )
        assert result["status"] == "ok"
        for key in ("f1", "r2", "rocauc"):
            assert math.isfinite(result[key])
        split_rows = read_pipe(result["split_csv"])
        counts: dict[str, int] = {}
        for row in split_rows:
            counts[row["train_test"]] = counts.get(row["train_test"], 0) + 1
        assert counts.get("zsv") == 4
        assert counts.get("train", 0) >= 1
        assert counts.get("val", 0) >= 1
        assert counts.get("test", 0) >= 1
        assert sum(counts.values()) == 20
        assert Path(result["scores"]).is_file()
