# Layout

Package root: `src/giga_mario_bench/`.

```text
bench/data/random_cds.py          randomCDS generator
bench/reverse_complement/         adaptor, contract.md
score/general.py                  F1, R2, ROC AUC
score/prepare.py                  writes exec scripts
score/execute.py                  Nextflow, or Python if asked
vizualisation/metrics.py          Altair + cnsplots
models/general.py                 model get
models/data_prepare/              raw → model-ready
models/reverse_complement/        train and test
models/seqmodels.py               NumPy RNN and encoder-decoder
bin/                              build, traintestsplit, and the CLI package entry
pipeline.py                       one start-to-end cell
exec/                             generated scripts (not the library)
```

Commands are also `giga_mario_bench` and `python -m giga_mario_bench.bin`.

Nextflow workflow: `workflows/score.nf`.

Hydra config: `configs/score.yaml`.

Portable agent rules: `agents/rules/` (mirrored in `.cursor/rules/`).

Skills: `agents/skills/model-add` (with `model-adapt` and `model-pytest`) and `agents/skills/score`.
