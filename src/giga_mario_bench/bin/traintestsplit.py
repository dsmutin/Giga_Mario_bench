"""Train/test/val assignment through Giga_Mario ``run_split_predict``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from giga_mario_bench.giga_mario import run_split_predict
from giga_mario_bench.io_utils import read_json, read_jsonl, write_pipe


def zero_shot_ids(records: list[dict[str, Any]], zsv_pairs: int) -> set[str]:
    """Hold out the last ``zsv_pairs`` pairs (original and mutant)."""
    pair_ids = sorted({row["pair_id"] for row in records})
    if zsv_pairs < 1:
        raise ValueError("zsv_pairs must be >= 1")
    if zsv_pairs >= len(pair_ids):
        raise ValueError("zsv_pairs leaves no sequences for train/val/test")
    held = set(pair_ids[-zsv_pairs:])
    return {row["id"] for row in records if row["pair_id"] in held}


def split_methods(
    records: list[dict[str, Any]],
    outdir: Path,
    methods: list[str],
    *,
    seed: int = 42,
    zsv_pairs: int = 2,
    ratios: tuple[float, float, float] | None = None,
) -> dict[str, str]:
    """Write ``split.csv`` for each method.

    One method writes ``outdir/split.csv``. Several methods write
    ``outdir/<method>/split.csv``. Assignment is Giga_Mario split-predict.
    Zero-shot ids are passed in ``fold.csv`` with fold ``zsv``; a held-out
    pair contributes both the original and the mutant. ``ratios`` is
    train, test, val for the remaining sequences. A ``role`` column
    stratifies that assignment so originals and mutants share each split.
    """
    if not methods:
        raise ValueError("at least one split method is required")
    outdir = Path(outdir)
    held = zero_shot_ids(records, zsv_pairs)
    id_rows = [{"ID": row["id"]} for row in records]
    fold_rows = [
        {"ID": row["id"], "fold": "zsv" if row["id"] in held else "0"}
        for row in records
    ]
    written: dict[str, str] = {}
    multiple = len(methods) > 1
    for method in methods:
        dest = outdir / method if multiple else outdir
        dest.mkdir(parents=True, exist_ok=True)
        split_path = dest / "split.csv"
        if split_path.is_file():
            print(f"split already present: {split_path}")
            written[method] = str(split_path)
            continue
        id_csv = dest / "id.csv"
        fold_csv = dest / "fold.csv"
        strat_csv = dest / "stratification.csv"
        write_pipe(id_csv, id_rows, ["ID"])
        write_pipe(fold_csv, fold_rows, ["ID", "fold"])
        write_pipe(
            strat_csv,
            [{"ID": row["id"], "role": row["role"]} for row in records],
            ["ID", "role"],
        )
        produced = run_split_predict(
            outdir=dest,
            type=method,
            seed=seed,
            id_csv=id_csv,
            fold_csv=fold_csv,
            stratification_csv=strat_csv,
            ratios=ratios,
        )
        print(f"wrote split {method}: {produced}")
        written[method] = str(produced)
    return written


def split_from_bench(
    bench_dir: Path,
    outdir: Path | None,
    methods: list[str],
    *,
    seed: int = 42,
    zsv_pairs: int | None = None,
    ratios: tuple[float, float, float] | None = None,
) -> dict[str, str]:
    """Read ``input/records.jsonl`` and assign splits beside that benchmark."""
    bench_dir = Path(bench_dir)
    records = read_jsonl(bench_dir / "input" / "records.jsonl")
    panel_path = bench_dir / "input" / "panel.json"
    panel = read_json(panel_path) if panel_path.is_file() else {}
    if zsv_pairs is None:
        zsv_pairs = int(panel.get("zsv_pairs", 2))
    if ratios is None and panel.get("split_ratios"):
        ratios = tuple(float(item) for item in panel["split_ratios"])
    target = bench_dir / "splits" if outdir is None else Path(outdir)
    return split_methods(
        records,
        target,
        methods,
        seed=seed,
        zsv_pairs=zsv_pairs,
        ratios=ratios,
    )
