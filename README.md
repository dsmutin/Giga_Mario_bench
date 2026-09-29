# giga_mario_bench

[![version](https://img.shields.io/badge/dynamic/yaml?url=https%3A%2F%2Fraw.githubusercontent.com%2Fdsmutin%2FGiga_Mario_bench%2Fmain%2FVERSION&query=%24&label=version&color=blue)](VERSION)
[![required tests](https://img.shields.io/github/actions/workflow/status/dsmutin/Giga_Mario_bench/required-tests.yml?branch=main&label=required%20tests)](https://github.com/dsmutin/Giga_Mario_bench/actions/workflows/required-tests.yml)
[![full tests](https://img.shields.io/github/actions/workflow/status/dsmutin/Giga_Mario_bench/full-tests.yml?branch=main&label=full%20tests)](https://github.com/dsmutin/Giga_Mario_bench/actions/workflows/full-tests.yml)
[![warning](https://img.shields.io/badge/warning-in%20development-yellow)](https://shields.io/badges/static-badge)

Benchmarks of data-split methods for in silico ligand hypotheses, with shared models and scores.

**Warning: in development.** Interfaces may change. See `VERSION` (single source of truth).

## Install

Conda is the only supported install:

```bash
conda env create -f environment.yml
conda activate giga_mario_bench
```

`environment.yml` sets `PYTHONPATH=src`. Do not publish a pip-first install path.

## Usage

```bash
giga_mario_bench --version
giga_mario_bench
python examples/toy/run.py
```

## Tests

```bash
pytest -m mandatory    # every commit
pytest                 # mandatory + optional (release / manual CI)
```

## License

MIT. See [CONTRIBUTING.md](CONTRIBUTING.md).
