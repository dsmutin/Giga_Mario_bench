"""Materialize a benchmark into ``input/`` and ``output/``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from giga_mario_bench.bench.data.random_cds import generate_pairs, parse_rate
from giga_mario_bench.bench.reverse_complement.adaptor import answers_for
from giga_mario_bench.io_utils import read_jsonl, write_json, write_jsonl

BENCHES = ("reverse_complement",)

# Locked analysis panel. ``n_sequences`` counts originals and mutants together.
# Length is fixed at 100. Of 5_000 pairs, 1_000 (20%) are zero-shot as a whole
# pair: 1_000 originals and 1_000 mutants. The other 4_000 + 4_000 are split
# evenly across train, test, and val, stratified by role. Mutations are SNPs.
DEFAULT_SPEC: dict[str, Any] = {
    "n_sequences": 10_000,
    "min_length": 100,
    "max_length": 100,
    "rate": 0.1,
    "seed": 42,
    "split_ratios": (1.0, 1.0, 1.0),
    "mutation": "snp",
}
# 1_000 zero-shot pairs out of 5_000. The same fraction keeps a smaller
# override legal: 10 pairs hold out 2.
ZSV_PAIR_FRACTION = 0.2


def resolve_panel(spec: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fill the pipeline panel from ``DEFAULT_SPEC`` and explicit overrides.

    ``n_pairs`` sets the pair count and replaces ``n_sequences``. ``length``
    fixes every read at that length. ``zsv_pairs``, when omitted, is 20% of
    pairs (1_000 when the panel has 5_000), and at least 2. ``split_ratios``
    is train, test, val.
    """
    raw: dict[str, Any] = dict(DEFAULT_SPEC)
    explicit_pairs = False
    explicit_length = False
    explicit_zsv = False
    if spec:
        for key, value in spec.items():
            if value is None:
                continue
            raw[key] = value
            if key == "n_pairs":
                explicit_pairs = True
            elif key == "length":
                explicit_length = True
            elif key == "zsv_pairs":
                explicit_zsv = True
    if explicit_pairs:
        n_pairs = int(raw["n_pairs"])
        n_sequences = n_pairs * 2
    else:
        n_sequences = int(raw["n_sequences"])
        if n_sequences % 2 != 0:
            raise ValueError(f"n_sequences must be even, got {n_sequences}")
        n_pairs = n_sequences // 2
    if n_pairs < 3:
        raise ValueError("n_pairs must be >= 3 so a train/val/test split can drop two zero-shot pairs")
    if explicit_length:
        min_length = max_length = int(raw["length"])
    else:
        min_length = int(raw["min_length"])
        max_length = int(raw["max_length"])
    if min_length < 2 or max_length < min_length:
        raise ValueError(f"length range must satisfy 2 <= min_length <= max_length, got {min_length}..{max_length}")
    if explicit_zsv:
        zsv_pairs = int(raw["zsv_pairs"])
    else:
        zsv_pairs = max(2, int(round(n_pairs * ZSV_PAIR_FRACTION)))
    if zsv_pairs < 1 or zsv_pairs >= n_pairs:
        raise ValueError(f"zsv_pairs must be in [1, {n_pairs - 1}], got {zsv_pairs}")
    ratios = tuple(float(item) for item in raw["split_ratios"])
    if len(ratios) != 3 or any(item <= 0 for item in ratios):
        raise ValueError(f"split_ratios must be three positive train, test, val weights, got {ratios}")
    rate = _rate_value(raw["rate"])
    resolved: dict[str, Any] = {
        "n_sequences": n_sequences,
        "n_pairs": n_pairs,
        "min_length": min_length,
        "max_length": max_length,
        "rate": rate,
        "seed": int(raw["seed"]),
        "zsv_pairs": zsv_pairs,
        "split_ratios": ratios,
        "mutation": "snp",
    }
    if min_length == max_length:
        resolved["length"] = min_length
    return resolved


def param_slug(spec: dict[str, Any]) -> str:
    """Stable directory name for one benchmark parameter set."""
    resolved = resolve_panel(spec)
    rate = resolved["rate"]
    rate_text = rate if isinstance(rate, str) else str(rate)
    rate_text = rate_text.replace("/", "_").replace(" ", "")
    if resolved["min_length"] == resolved["max_length"]:
        length_text = str(resolved["min_length"])
    else:
        length_text = f"{resolved['min_length']}-{resolved['max_length']}"
    return (
        f"n{resolved['n_pairs']}_len{length_text}_rate{rate_text}_seed{resolved['seed']}"
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
    merged = resolve_panel(spec)
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
            min_length=int(merged["min_length"]),
            max_length=int(merged["max_length"]),
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
    payload = {
        "n_sequences": int(spec["n_sequences"]),
        "n_pairs": int(spec["n_pairs"]),
        "min_length": int(spec["min_length"]),
        "max_length": int(spec["max_length"]),
        "rate": rate,
        "seed": int(spec["seed"]),
        "zsv_pairs": int(spec["zsv_pairs"]),
        "split_ratios": [float(item) for item in spec["split_ratios"]],
        "mutation": spec.get("mutation", "snp"),
    }
    if spec.get("length") is not None:
        payload["length"] = int(spec["length"])
    return payload
