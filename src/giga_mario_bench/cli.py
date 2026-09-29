"""Command-line entry for giga_mario_bench."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from giga_mario_bench import __version__
from giga_mario_bench.baseline import run_pipeline


def _spec_from_args(args: argparse.Namespace) -> dict:
    return {
        "n_pairs": args.n_pairs,
        "length": args.length,
        "rate": args.rate,
        "seed": args.seed,
        "zsv_pairs": args.zsv_pairs,
    }


def _add_panel_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n-pairs", type=int, default=10)
    parser.add_argument("--length", type=int, default=16)
    parser.add_argument(
        "--rate",
        default="0.1",
        help="mutation rate, a low:high range, or comma-separated choices",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--zsv-pairs", type=int, default=2)


def _emit(payload: dict, output: str) -> None:
    text = json.dumps(payload, indent=2)
    if output in {"", "-"}:
        print(text)
    else:
        Path(output).write_text(text + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """Parser for the baseline entry and the bin subcommands."""
    parser = argparse.ArgumentParser(
        prog="giga_mario_bench",
        description="Benchmarks of data-split methods for in silico ligand hypotheses.",
    )
    parser.add_argument("--version", action="store_true", help="print version and exit")
    parser.add_argument("-o", "--output", default="-", help="JSON output path or - for stdout")
    sub = parser.add_subparsers(dest="command")

    build = sub.add_parser("build", help="materialize a benchmark into input/ and output/")
    build.add_argument("--bench", default="reverse_complement")
    build.add_argument("--outdir", required=True, type=Path)
    build.add_argument("--datadir", type=Path, default=None)
    build.add_argument("--all", action="store_true", help="write every bench under outdir/<name>/")
    _add_panel_args(build)

    model = sub.add_parser("model", help="initialize, prepare, train, or test a model")
    model_sub = model.add_subparsers(dest="model_cmd", required=True)
    get = model_sub.add_parser("get", help="initialize or download a model")
    get.add_argument("--model", default=None)
    get.add_argument("--outdir", required=True, type=Path)
    get.add_argument("--all", action="store_true")

    prepare = model_sub.add_parser("prepare", help="raw records to model-ready rows")
    prepare.add_argument("--model", required=True)
    prepare.add_argument("--bench-dir", required=True, type=Path)
    prepare.add_argument("--out", required=True, type=Path)

    train = model_sub.add_parser("train", help="train until validation loss plateaus")
    train.add_argument("--model", required=True)
    train.add_argument("--prepared", required=True, type=Path)
    train.add_argument("--split", required=True, type=Path)
    train.add_argument("--out", required=True, type=Path)
    train.add_argument("--hidden", type=int, default=None)
    train.add_argument("--max-epochs", type=int, default=20)
    train.add_argument("--patience", type=int, default=5)
    train.add_argument("--seed", type=int, default=42)

    test = model_sub.add_parser("test", help="run a checkpoint")
    test.add_argument("--model", required=True)
    test.add_argument("--prepared", required=True, type=Path)
    test.add_argument("--split", required=True, type=Path)
    test.add_argument("--checkpoint", required=True, type=Path)
    test.add_argument("--out", required=True, type=Path)

    split = sub.add_parser("traintestsplit", help="assign folds with a Giga_Mario method")
    split.add_argument("--bench-dir", required=True, type=Path)
    split.add_argument("--out", type=Path, default=None)
    split.add_argument("--method", default=None, help="one Giga_Mario split type")
    split.add_argument(
        "--methods",
        default=None,
        help="comma-separated methods; each is written to its own subdirectory",
    )
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--zsv-pairs", type=int, default=None)

    score = sub.add_parser("score", help="prepare or execute the benchmark grid")
    score_sub = score.add_subparsers(dest="score_cmd", required=True)
    prep = score_sub.add_parser("prepare", help="write start-to-end exec scripts")
    prep.add_argument("--data-out", type=Path, default=None)
    prep.add_argument("--model-out", type=Path, default=None)
    prep.add_argument("--score-out", type=Path, default=None)
    prep.add_argument("--spec", type=Path, default=None, help="JSON with data_out, model_out, score_out")
    prep.add_argument("--all", action="store_true")
    prep.add_argument("--bench", default="reverse_complement")
    prep.add_argument("--split", default="random")
    prep.add_argument("--model", default=None, help="default: every registered model")
    prep.add_argument("--exec-root", type=Path, default=None)
    prep.add_argument("--max-epochs", type=int, default=20)
    prep.add_argument("--patience", type=int, default=5)
    _add_panel_args(prep)

    exe = score_sub.add_parser("exec", help="run scripts from score prepare")
    exe.add_argument("--exec-root", type=Path, default=None)
    exe.add_argument(
        "--no-nextflow",
        action="store_true",
        help="run the Python cells directly",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and run the selected command.

    With no subcommand, print the baseline JSON contract.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        print(__version__)
        return 0
    if args.command is None:
        _emit(run_pipeline(), args.output)
        return 0
    if args.command == "build":
        from giga_mario_bench.bench.build import build_bench

        result = build_bench(
            None if args.all else args.bench,
            args.outdir,
            datadir=args.datadir,
            spec=_spec_from_args(args),
            all_benches=args.all,
        )
        _emit(result, args.output)
        return 0
    if args.command == "model":
        return _model(args)
    if args.command == "traintestsplit":
        from giga_mario_bench.bin.traintestsplit import split_from_bench

        if args.methods:
            methods = [part.strip() for part in args.methods.split(",") if part.strip()]
        elif args.method:
            methods = [args.method]
        else:
            methods = ["random"]
        written = split_from_bench(
            args.bench_dir,
            args.out,
            methods,
            seed=args.seed,
            zsv_pairs=args.zsv_pairs,
        )
        _emit({"status": "ok", "splits": written}, args.output)
        return 0
    if args.command == "score":
        return _score(args)
    parser.error(f"unknown command {args.command}")
    return 2


def _model(args: argparse.Namespace) -> int:
    if args.model_cmd == "get":
        from giga_mario_bench.models.general import initialize_model

        if not args.all and not args.model:
            raise SystemExit("model get needs --model or --all")
        result = initialize_model(
            args.model or "",
            args.outdir,
            all_models=args.all,
        )
        _emit(result, args.output)
        return 0
    if args.model_cmd == "prepare":
        from giga_mario_bench.io_utils import read_jsonl, write_jsonl
        from giga_mario_bench.models.data_prepare.encode import prepare_records

        records = read_jsonl(args.bench_dir / "input" / "records.jsonl")
        answers = read_jsonl(args.bench_dir / "output" / "answers.jsonl")
        write_jsonl(args.out, prepare_records(records, answers))
        _emit({"status": "prepared", "out": str(args.out), "n": len(records)}, args.output)
        return 0
    if args.model_cmd == "train":
        from giga_mario_bench.models.reverse_complement.common import train_model

        result = train_model(
            args.model,
            args.prepared,
            args.split,
            args.out,
            hidden=args.hidden,
            max_epochs=args.max_epochs,
            patience=args.patience,
            seed=args.seed,
        )
        _emit(result, args.output)
        return 0
    if args.model_cmd == "test":
        from giga_mario_bench.models.reverse_complement.common import test_model

        result = test_model(
            args.model,
            args.prepared,
            args.split,
            args.checkpoint,
            args.out,
        )
        _emit(result, args.output)
        return 0
    raise SystemExit(f"unknown model command {args.model_cmd}")


def _score(args: argparse.Namespace) -> int:
    if args.score_cmd == "prepare":
        from giga_mario_bench.score.prepare import prepare_grid

        scripts = prepare_grid(
            data_out=args.data_out,
            model_out=args.model_out,
            score_out=args.score_out,
            spec_json=args.spec,
            all_cells=args.all,
            bench=None if args.all else args.bench,
            split=None if args.all else args.split,
            model=None if args.all else args.model,
            spec=_spec_from_args(args),
            exec_root=args.exec_root,
            max_epochs=args.max_epochs,
            patience=args.patience,
        )
        _emit({"status": "prepared", "scripts": [str(path) for path in scripts]}, args.output)
        return 0
    if args.score_cmd == "exec":
        from giga_mario_bench.score.execute import execute

        return execute(args.exec_root, use_nextflow=not args.no_nextflow)
    raise SystemExit(f"unknown score command {args.score_cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
