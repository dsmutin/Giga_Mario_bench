# Testing

## Test data

| Path | Role |
|------|------|
| `examples/toy/data/` | toy outputs (created by `run.py`) |
| `tests/` | contract tests (no large fixtures in the scaffold) |

Add real fixtures under `tests/data/` or `examples/toy/data/` and document them here. Do not invent datasets.

## Integrative testing

1. Mandatory unit/contract tests: `pytest -m mandatory`
2. CLI write path: `tests/test_integration.py`
3. Toy: `python examples/toy/run.py` (same baseline JSON contract)
4. Full suite (optional + toy + any vignettes): GitHub Action `full-tests.yml` on **release** or **workflow_dispatch**

Required CI on every push/PR: `required-tests.yml` (mandatory only), conda from `environment.yml`.
