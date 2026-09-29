# Contributing to giga_mario_bench

All development follows this guide. Project rules require it (`follow-contributing`).

English is required for every public function, class, module, CLI flag, data file, and user-facing document. Other languages or missing documentation are not allowed.

## Architecture

```
CLI (giga_mario_bench.cli)
  → baseline pipeline (giga_mario_bench.baseline.run_pipeline)
      → JSON result {status, ok, input_path}

tests/          mandatory vs optional pytest
examples/toy/   end-to-end run of the current (baseline) tool
cite/           BibTeX for integrated third-party tools
wiki/           contracts, test data, integrative testing
agents/         portable rules and skills (any IDE)
```

Replace baseline bodies with real implementations. Keep the documented return keys until you change the contract in `wiki/Contracts.md` and the tests together.

## Testing architecture

| Kind | Marker | Command | When |
|------|--------|---------|------|
| Required | `mandatory` | `pytest -m mandatory` | every commit; GitHub Action `required-tests` |
| Optional | `optional` | `pytest` (all) | release or workflow_dispatch; Action `full-tests` |
| Examples | — | `python examples/toy/run.py` | full CI; after features that touch the CLI |
| Vignettes | — | any `vignettes/` or extra `examples/*` | full CI when those files exist |

Do not mark a contract test `optional`. Optional tests are slow, extra, or nice-to-have.

After **any new feature**, run the **mandatory** suite (and toy if the CLI changed) before you stop.

## Feature checklist (`todo.md`)

Track work in `todo.md` (checkboxes). One line per feature or fix. Check it off only when mandatory tests pass. This is a **feature list**, not a `/do` analysis graph.

## Versioning

Edit **only** `VERSION`. Everything else reads it.

Starting value: `0.0.1`.

| Change | Bump |
|--------|------|
| New feature | **minor** (`0.0.1` → `0.1.0`) |
| Fix or update of an existing feature | **patch** (`0.1.0` → `0.1.1`) |
| Release | **major** (`0.1.1` → `1.0.0`) |

## Install (conda only)

```bash
conda env create -f environment.yml
conda activate giga_mario_bench
```

## GitHub

Never `git push` unless the human explicitly asks. CI runs on GitHub after they push.

## Citations

Add a `.bib` entry in `cite/` only for tools this package actually integrates. Do not invent papers.
