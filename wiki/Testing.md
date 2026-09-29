# Testing

## Test data

| Path | Role |
|------|------|
| `examples/toy/data/` | summary JSON from `run.py` (gitignored result file) |
| `examples/toy/out/` | bench, checkpoints, scores, figures (gitignored) |
| `tests/test_contracts.py` | both contract types for `reverse_complement` and both toy models |
| `tests/test_bench_and_scores.py` | idempotent build, rate parsing, perfect-prediction scores, BPTT gradient check, Altair HTML |

No external sequence fixtures are shipped. The generator is deterministic given `seed`.

Giga_Mario must be importable (`GIGA_MARIO_ROOT` or a sibling `Giga_Mario` directory). CI checks out `Sirius-Back/Giga_Mario` and sets `GIGA_MARIO_ROOT`.

## Integrative testing

1. Mandatory tests, including contracts: `pytest -m mandatory`
2. Contract subset: `pytest -m contracts`
3. CLI baseline write path: `tests/test_integration.py`
4. Toy: `python examples/toy/run.py`
5. Full suite: GitHub Action `full-tests.yml` on **release** or **workflow_dispatch**

Required CI on every push or pull request: `required-tests.yml` (mandatory only), conda from `environment.yml`.

A new `<bench>/<model>` pair adds a raw→model-ready test and a model-ready→random split→train→score test in the same change. Both stay on the `mandatory` and `contracts` markers.
