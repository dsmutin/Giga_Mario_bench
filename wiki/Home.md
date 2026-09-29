# giga_mario_bench wiki

Benchmarks of data-split methods for in silico ligand hypotheses, with shared models and scores.

**Status:** in development. Version: see repository file `VERSION` (starts at 0.0.1).

## Starting interconnections

```
user → giga_mario_bench CLI → run_pipeline() → JSON {status, ok, input_path}
                ↑
         tests (mandatory / optional)
                ↑
         examples/toy/run.py
```

Until real logic exists, `run_pipeline` is a baseline stub.

## Pages

- [Contracts](Contracts)
- [Testing](Testing)
