# giga_mario_bench wiki

Benchmarks of data-split methods for in silico ligand hypotheses. The thing under test is the **split**, not the model. Models are instruments. Split assignment is [Giga_Mario](https://github.com/Sirius-Back/Giga_Mario) `run_split_predict`.

**Version:** repository file `VERSION` (currently the feature level after `0.0.1`).

When a change is about to be committed, update this wiki and push this wiki repository. "Fix the wiki" means these pages.

## Starting interconnections

```text
randomCDS
  → bench/reverse_complement adaptor (reverse complement answers)
  → models/data_prepare (ACGT integers)
  → Giga_Mario split-predict (train/val/test + zsv)
  → many-to-many RNN or encoder-decoder, trained to a validation plateau
  → score_general (F1, R2, ROC AUC) and vizualisation
```

`score prepare` writes a start-to-end Python script per grid cell under `src/giga_mario_bench/exec/`. `score exec` runs those scripts with Nextflow. Each script skips artifacts that are already on disk.

With no subcommand, the CLI still returns baseline JSON `{status, ok, input_path}`.

## Pages

- [Contracts](Contracts)
- [Testing](Testing)
- [Layout](Layout)
