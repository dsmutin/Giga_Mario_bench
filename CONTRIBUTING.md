# Contributing to giga_mario_bench

Read this guide before developing a new feature or editing an existing one.

All development follows this guide. Project rules require it (`follow-contributing`).

English is required for every public function, class, module, CLI flag, data file, and user-facing document. Other languages or missing documentation are not allowed.

## What this repository is

This repository builds benchmarks and downloads or initializes models used to test in silico hypotheses about ligands. The benchmark is the **data-split method**, not the model. Split methods live in [Giga_Mario](https://github.com/Sirius-Back/Giga_Mario). Clone that repository and its wiki next to this one (or set `GIGA_MARIO_ROOT`) and keep the two compatible.

```text
data_leakage/
  Giga_Mario_bench/     # this tool
  Giga_Mario/           # split methods and genomic model adapters
  Giga_Mario.wiki/      # Giga_Mario GitHub wiki
```

The checkouts are one ecosystem. Change whichever repository already owns the behavior. Reuse a debugged function instead of copying it. Train/test/val assignment calls `src.pipeline.split_predict.run_split_predict` in Giga_Mario (`ID|train_test|fold`, with `zsv` held out). Genomic window adapters that Giga_Mario already implements (`parse_data` for Caduceus and LegNet) should be extended there. This package keeps adapters that Giga_Mario does not emit.

## Layout

Code lives under `src/giga_mario_bench/`:

| Path | Role |
|------|------|
| `bench/data/` | Generators such as `randomCDS` |
| `bench/<bench_name>/` | `adaptor/`, `scoring` via `score/`, and `contract.md` |
| `score/` | Metrics on a true-versus-predicted similarity matrix (`F1`, `R2`, `ROC AUC`) |
| `vizualisation/` | Altair and cnsplots figures |
| `models/general.py` | Initialize a model (`model get`) |
| `models/data_prepare/<model>.py` | Raw rows to the rows that model accepts |
| `models/<bench>/<model>.py` | Train and test that model on a Giga_Mario split |
| `bin/` | `python -m giga_mario_bench.bin` (same CLI) and `traintestsplit.py` |
| `cli.py` | Subcommands below. Build, prepare, train, and score call the libraries next to the data they own |
| `exec/<bench>/<params>/<split>/<model>_<params>.py` | Start-to-end scripts written by `score prepare` |

`bin` commands:

| Command | Behavior |
|---------|----------|
| `build` | If that bench is already in `outdir`, say so and stop. Else, if `outdir/datadir` or `--datadir` already has records, say so and use them. Else generate `outdir/input` and `outdir/output`. `--all` writes `outdir/<bench_name>/`. |
| `model get` | Initialize or download. `--all` covers every registered model. Skip a model that is already initialized. |
| `model prepare` | Raw benchmark rows to model-ready rows. |
| `model train` / `model test` | Train until validation loss plateaus, then test. |
| `traintestsplit --method` | One Giga_Mario method. `--methods a,b` writes one subdirectory per method. |
| `score prepare` | Writes the exec scripts above. `--all` crosses registered benches, the main parameter set, grid splits, and models. Requires `--data-out`, `--model-out`, and `--score-out`, or a JSON file with those keys. |
| `score exec` | Runs those scripts. Nextflow is the default orchestrator and each script skips artifacts that already exist. `--no-nextflow` runs them in-process. |

Hydra: `python -m giga_mario_bench.score.hydra_app data_out=... model_out=... score_out=...` reads `configs/score.yaml`.

## Architecture

```
CLI (giga_mario_bench.cli)
  → bin commands
      → bench build / Giga_Mario split-predict / model train / score_general
          → exec scripts and JSON scores

tests/          mandatory, optional, and contracts
examples/toy/   reverse-complement panel (10 + 10 sequences, fixed length 16)
cite/           BibTeX for integrated tools
wiki/           GitHub wiki (separate git repo)
agents/         portable rules and skills (any IDE)
```

The pipeline default panel is 10_000 sequences of length 100, SNP rate 0.1, with 2_000 zero-shot sequences and the rest split evenly across train, test, and val (`DEFAULT_SPEC`). The toy example stays at 10 + 10 sequences of length 16.

The no-subcommand CLI still returns the baseline JSON keys `status`, `ok`, and `input_path`.

## Testing architecture

| Kind | Marker | Command | When |
|------|--------|---------|------|
| Required | `mandatory` | `pytest -m mandatory` | every commit; GitHub Action `required-tests` |
| Contracts | `contracts` | `pytest -m contracts` | with mandatory; raw→model-ready, and model-ready→random split→model→score |
| Optional | `optional` | `pytest` (all) | release or workflow_dispatch; Action `full-tests` |
| Examples | — | `python examples/toy/run.py` | full CI; after features that touch the CLI or the toy |
| Vignettes | — | any `vignettes/` or extra `examples/*` | full CI when those files exist |

Do not mark a contract test `optional`. When a new `bench/<name>/<model>` pair is added, add both contract tests in the same change. Optional tests are slow, extra, or nice-to-have.

After **any new feature**, run the **mandatory** suite (and the toy if the CLI or the toy bench changed) before you stop.

## Wiki

`wiki/` is the GitHub wiki, not a second copy of the README. Any change that will be committed must update those pages. Commit the pages here and push the wiki repository. "Fix the wiki" means that checkout.

## Feature checklist (`todo.md`)

Track work in `todo.md` (checkboxes). One line per feature or fix. Check it off only when mandatory tests pass. This is a **feature list**, not a `/do` analysis graph.

## Versioning

Edit **only** `VERSION`. Everything else reads it.

Starting value: `0.0.1`. Current feature level is `0.1.0`.

| Change | Bump |
|--------|------|
| New feature | **minor** (`0.0.1` → `0.1.0`) |
| Fix or update of an existing feature | **patch** (`0.1.0` → `0.1.1`) |
| Release | **major** (`0.1.1` → `1.0.0`) |

## Install (conda only)

```bash
conda env create -f environment.yml
conda activate giga_mario_bench
# sibling checkout of the split library
export GIGA_MARIO_ROOT=../Giga_Mario
```

## GitHub

Do not `git push` the tool repository unless the human explicitly asks. The wiki checkout may be pushed when its pages change. CI runs on GitHub after the human pushes the tool.

## Citations

Add a `.bib` entry in `cite/` only for tools this package actually integrates. Do not invent papers.
