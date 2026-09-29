# giga_mario_bench

[![version](https://img.shields.io/badge/dynamic/yaml?url=https%3A%2F%2Fraw.githubusercontent.com%2Fdsmutin%2FGiga_Mario_bench%2Fmain%2FVERSION&query=%24&label=version&color=blue)](VERSION)
[![required tests](https://img.shields.io/github/actions/workflow/status/dsmutin/Giga_Mario_bench/required-tests.yml?branch=main&label=required%20tests)](https://github.com/dsmutin/Giga_Mario_bench/actions/workflows/required-tests.yml)
[![full tests](https://img.shields.io/github/actions/workflow/status/dsmutin/Giga_Mario_bench/full-tests.yml?branch=main&label=full%20tests)](https://github.com/dsmutin/Giga_Mario_bench/actions/workflows/full-tests.yml)
[![warning](https://img.shields.io/badge/warning-in%20development-yellow)](https://shields.io/badges/static-badge)

Benchmarks of data-split methods for in silico ligand hypotheses, with shared models and scores.

**Warning: in development.** Interfaces may change. See `VERSION` (single source of truth).

## Install

Conda is the only supported install. Clone [Giga_Mario](https://github.com/Sirius-Back/Giga_Mario) beside this repository (split methods) and set `GIGA_MARIO_ROOT` if it is not the sibling directory.

```bash
conda env create -f environment.yml
conda activate giga_mario_bench
export GIGA_MARIO_ROOT=../Giga_Mario
```

`environment.yml` sets `PYTHONPATH=src`. Do not publish a pip-first install path.

## Usage

```bash
giga_mario_bench --version
giga_mario_bench
giga_mario_bench build --bench reverse_complement --outdir out/bench
giga_mario_bench traintestsplit --bench-dir out/bench --method random
giga_mario_bench score prepare --all \
  --data-out out/data --model-out out/models --score-out out/scores
giga_mario_bench score exec
python examples/toy/run.py
```

The toy panel is 10 DNA sequences plus 10 mutants. Two pairs are zero-shot. Giga_Mario assigns the other 16 at random. Many-to-many RNN and encoder-decoder train until validation loss plateaus. Scores are macro F1, R2, and ROC AUC.

## Tests

```bash
pytest -m mandatory    # every commit
pytest                 # mandatory + optional (release / manual CI)
```

## License

MIT. See [CONTRIBUTING.md](CONTRIBUTING.md).
