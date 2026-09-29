"""Materialize a benchmark into ``input/`` and ``output/``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from giga_mario_bench.bench.data.random_cds import generate_pairs, parse_rate
from giga_mario_bench.bench.reverse_complement.adaptor import answers_for
from giga_mario_bench.io_utils import read_jsonl, write_json, write_jsonl

BENCHES = ("reverse_complement",)

DEFAULT_SPEC: dict[str, Any] = {
    "n_pairs": 10,
    "length": 16,
    "rate": 0.1,
    "seed": 42,
    "zsv_pairs": 2,
}


def param_slug(spec: dict[str, Any]) -> str:
    """Stable directory name for one benchmark parameter set."""
    rate = spec["rate"]
    rate_text = rate if isinstance(rate, str) else str(rate)
    rate_text = rate_text.replace("/", "_").replace(" ", "")
    return (
        f"n{spec['n_pairs']}_len{spec['length']}_rate{rate_text}_seed{spec['seed']}"
    )


def _find_pregenerated(
    bench: str, outdir: Path, datadir: Path | None
) -> Path | None:
    candidates: list[Path] = []
    if datadir is not None:
        candidates.extend(
            [
                datadir / "records.jsonl",
                datadir / bench / "records.jsonl",
            ]
        )
    candidates.extend(
        [
            outdir / "datadir" / "records.jsonl",
            outdir / "datadir" / bench / "records.jsonl",
        ]
    )
    for path in candidates:
        if path.is_file():
            return path
    return None


def build_one(
    bench: str,
    outdir: Path,
    *,
    datadir: Path | None = None,
    spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one benchmark or reuse an existing tree.

    Order: existing ``input`` + ``output``, then pre-generated records, then
    a fresh ``randomCDS`` draw and the reverse-complement adaptor.
    """
    if bench not in BENCHES:
        known = ", ".join(BENCHES)
        raise ValueError(f"unknown benchmark {bench!r}; known: {known}")
    merged = dict(DEFAULT_SPEC)
    if spec:
        merged.update(spec)
    merged["rate"] = _rate_value(merged["rate"])
    outdir = Path(outdir)
    records_path = outdir / "input" / "records.jsonl"
    answers_path = outdir / "output" / "answers.jsonl"
    if records_path.is_file() and answers_path.is_file():
        print(f"benchmark already generated: {outdir}")
        return {"status": "exists", "bench": bench, "outdir": str(outdir), "spec": merged}

    pre = _find_pregenerated(bench, outdir, datadir)
    if pre is not None:
        print(f"using pre-generated data: {pre}")
        records = read_jsonl(pre)
    else:
        records = generate_pairs(
            n_pairs=int(merged["n_pairs"]),
            length=int(merged["length"]),
            rate=merged["rate"],
            seed=int(merged["seed"]),
        )
    answers = answers_for(records)
    write_jsonl(records_path, records)
    write_jsonl(answers_path, answers)
    write_json(outdir / "input" / "panel.json", {"bench": bench, **_json_spec(merged)})
    return {
        "status": "built" if pre is None else "from_pregenerated",
        "bench": bench,
        "outdir": str(outdir),
        "spec": _json_spec(merged),
        "n_records": len(records),
    }


def build_bench(
    bench: str | None,
    outdir: Path,
    *,
    datadir: Path | None = None,
    spec: dict[str, Any] | None = None,
    all_benches: bool = False,
) -> dict[str, Any]:
    """Build one benchmark, or every registered benchmark under ``outdir/<name>/``."""
    if all_benches:
        results = [
            build_one(name, Path(outdir) / name, datadir=datadir, spec=spec)
            for name in BENCHES
        ]
        return {"status": "all", "results": results}
    if not bench:
        raise ValueError("bench name is required unless --all is set")
    return build_one(bench, outdir, datadir=datadir, spec=spec)


def _rate_value(rate: Any) -> Any:
    if isinstance(rate, str):
        kind = parse_rate(rate)
        if kind[0] == "fixed":
            return kind[1]
        return rate
    return rate


def _json_spec(spec: dict[str, Any]) -> dict[str, Any]:
    rate = spec["rate"]
    if isinstance(rate, tuple):
        rate = list(rate)
    return {
        "n_pairs": int(spec["n_pairs"]),
        "length": int(spec["length"]),
        "rate": rate,
        "seed": int(spec["seed"]),
        "zsv_pairs": int(spec["zsv_pairs"]),
    }
