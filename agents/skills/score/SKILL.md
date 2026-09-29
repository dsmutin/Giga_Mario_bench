---
name: score
description: Build and run the giga_mario_bench score grid (bench × params × split × model), including the full grid, Hydra, and Nextflow.
---

# Score grid

Read `CONTRIBUTING.md` and `wiki/Contracts.md` first.

## Prepare

`score prepare` writes one start-to-end script per cell:

`src/giga_mario_bench/exec/<bench>/<params>/<split>/<model>_<model_params>.py`

Each script must be runnable by itself. It calls `giga_mario_bench.pipeline.run_cell`, which skips a bench, split, checkpoint, prediction file, or score file that already exists.

Required inputs, as flags or as a JSON object: `data_out`, `model_out`, `score_out`.

`--all` crosses every registered benchmark, the default parameter set, grid splits (`random` until a bench supplies MARKED/FNA for the other Giga_Mario methods), and every registered model.

## Hydra

```bash
python -m giga_mario_bench.score.hydra_app \
  data_out=out/data model_out=out/models score_out=out/scores
```

Config: `configs/score.yaml`. Set `run=true` to prepare and then execute.

## Execute

`score exec` runs Nextflow `workflows/score.nf` on the generated scripts. The workflow does not reimplement training; the scripts do, and they perform the skip checks. `--no-nextflow` runs the same scripts under the current Python.

After a grid change, update `wiki/Contracts.md` and run `pytest -m mandatory`.
