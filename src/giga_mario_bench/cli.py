"""Command-line entry for giga_mario_bench."""

from __future__ import annotations

import argparse
import json
import sys

from giga_mario_bench import __version__
from giga_mario_bench.baseline import run_pipeline


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and run the baseline pipeline."""
    parser = argparse.ArgumentParser(prog="giga_mario_bench", description="Benchmarks of data-split methods for in silico ligand hypotheses, with shared models and scores.")
    parser.add_argument("--version", action="store_true", help="print version and exit")
    parser.add_argument("-o", "--output", default="-", help="output path or - for stdout")
    args = parser.parse_args(argv)
    if args.version:
        print(__version__)
        return 0
    result = run_pipeline()
    text = json.dumps(result, indent=2)
    if args.output in {"", "-"}:
        print(text)
    else:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
