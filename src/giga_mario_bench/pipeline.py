"""Start-to-end cell: build, split, train, test, score.

Generated scripts under ``src/giga_mario_bench/exec/`` call ``run_cell``.
Each step skips work that is already on disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from giga_mario_bench.bench.build import build_one, param_slug
from giga_mario_bench.bin.traintestsplit import split_from_bench
from giga_mario_bench.io_utils import read_jsonl, write_json, write_jsonl
from giga_mario_bench.models.data_prepare.encode import prepare_records
from giga_mario_bench.models.general import initialize_model, model_params_slug
from giga_mario_bench.models.reverse_complement.common import test_model, train_model
from giga_mario_bench.score.general import score_general
from giga_mario_bench.vizualisation.metrics import plot_scores


def run_cell(
    *,
    bench: str,
    spec: dict[str, Any],
    split: str,
    model: str,
    data_out: Path,
    model_out: Path,
    score_out: Path,
    datadir: Path | None = None,
    hidden: int | None = None,
    max_epochs: int = 20,
    patience: int = 5,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run one benchmark × parameter set × split × model, reusing artifacts."""
    spec = dict(spec)
    if seed is not None:
        spec["seed"] = seed
    slug = param_slug(spec)
    model_slug = model_params_slug(model, hidden)
    bench_dir = Path(data_out) / bench / slug
    build_one(bench, bench_dir, datadir=datadir, spec=spec)
    initialize_model(model, Path(model_out) / "registry")

    prepared = (
        Path(model_out) / "prepared" / bench / slug / f"{model}.jsonl"
    )
    if prepared.is_file():
        print(f"prepared data already present: {prepared}")
    else:
        records = read_jsonl(bench_dir / "input" / "records.jsonl")
        answers = read_jsonl(bench_dir / "output" / "answers.jsonl")
        write_jsonl(prepared, prepare_records(records, answers))
        print(f"prepared data: {prepared}")

    split_root = bench_dir / "splits"
    # A single method writes split.csv directly under the method folder when
    # we ask split_from_bench for one method: outdir/split.csv.
    split_dir = split_root / split
    split_csv = split_dir / "split.csv"
    if split_csv.is_file():
        print(f"split already present: {split_csv}")
    else:
        split_from_bench(
            bench_dir,
            split_dir,
            [split],
            seed=int(spec["seed"]),
            zsv_pairs=int(spec.get("zsv_pairs", 2)),
        )

    checkpoint_dir = (
        Path(model_out) / "checkpoints" / bench / slug / split / model / model_slug
    )
    train_info = train_model(
        model,
        prepared,
        split_csv,
        checkpoint_dir,
        hidden=hidden,
        max_epochs=max_epochs,
        patience=patience,
        seed=int(spec["seed"]),
    )
    predictions = (
        Path(score_out) / bench / slug / split / f"{model}_{model_slug}" / "predictions.jsonl"
    )
    test_model(
        model,
        prepared,
        split_csv,
        checkpoint_dir / "weights.npz",
        predictions,
    )
    score_path = predictions.parent / "scores.json"
    if score_path.is_file():
        print(f"scores already present: {score_path}")
        from giga_mario_bench.io_utils import read_json

        scores = read_json(score_path)
    else:
        scored_rows = [
            row
            for row in read_jsonl(predictions)
            if row["train_test"] in {"test", "zsv"}
        ]
        scores = score_general(scored_rows)
        scores.update(
            {
                "bench": bench,
                "params": slug,
                "split": split,
                "model": model,
                "model_params": model_slug,
                "stopped_reason": train_info.get("stopped_reason"),
            }
        )
        write_json(score_path, scores)
        plot_scores(scores, predictions.parent / "metrics")
        print(f"scores: {score_path}")
    return {
        "status": "ok",
        "bench_dir": str(bench_dir),
        "split_csv": str(split_csv),
        "predictions": str(predictions),
        "scores": str(score_path),
        "f1": scores["f1"],
        "r2": scores["r2"],
        "rocauc": scores["rocauc"],
    }
