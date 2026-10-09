# Contracts

Change a row only together with tests and with this page.

## Implementation table

| Contract | Implementation |
|----------|----------------|
| Single version source (`VERSION`) | `giga_mario_bench.__version__` reads `VERSION` |
| CLI `--version` and baseline JSON `{status, ok, input_path}` | `giga_mario_bench.cli.main` with no subcommand; `baseline.run_pipeline` |
| Conda-only install | `environment.yml`, `PYTHONPATH=src` |
| Mandatory pytest on every commit | `pytest -m mandatory` (`required-tests.yml`) |
| Contract tests | marker `contracts` (also `mandatory`): raw→model-ready, and model-ready→random split→model→score |
| Optional pytest on release / manual | `pytest -m optional`, full suite in `full-tests.yml` |
| Toy example | `examples/toy/run.py` |
| Build is idempotent | `bench.build.build_one`: existing `input/`+`output/` prints `benchmark already generated`; pre-generated `records.jsonl` prints `using pre-generated data` |
| `--all` benches | `outdir/<bench_name>/input` and `output` |
| Reverse-complement answers | `bench/reverse_complement/adaptor.py` |
| Model-ready rows | `models/data_prepare`: `x` and `y` integer lists, `A=0,C=1,G=2,T=3` |
| Split | Giga_Mario `src.pipeline.split_predict.run_split_predict`. Output `split.csv` columns `ID\|train_test\|fold`. Fold `zsv` is held out. Several `--methods` use one subdirectory each |
| Pipeline panel default | `DEFAULT_SPEC`: 10_000 sequences, length 100, SNP rate 0.1. 1_000 pairs (2_000 sequences) are `zsv`. The other 8_000 are train/test/val at 1:1:1, stratified by original vs mutant. Label is the reverse complement of that sequence |
| Toy panel counts | 10 originals + 10 mutants, fixed length 16; 2 pairs (4 sequences) `zsv`; 16 sequences assigned; train, val, and test all non-empty |
| Train until plateau | `models.seqmodels.fit_plateau`. Stop reason `plateau` or `max_epochs`. Best validation weights restored |
| Scores | `score.general.score_general`: macro F1 from the 4×4 count matrix, R2 of true one-hot vs probabilities, macro one-vs-rest ROC AUC |
| Figures | `vizualisation.metrics.plot_scores` (Altair HTML, cnsplots PNG when installed) |
| Exec scripts | `score prepare` → `src/giga_mario_bench/exec/<bench>/<params>/<split>/<model>_<params>.py`, runnable alone |
| Score grid `--all` | registered benches × default parameters × `random` × registered models. Other Giga_Mario methods need MARKED/FNA and are not in the default grid |
| Hydra | `python -m giga_mario_bench.score.hydra_app` with `configs/score.yaml` |
| Model init skip | `model already initialized` when `initialized.json` exists |
| English public APIs | module and function docstrings |
| Tool repo push | only when the human asks. Wiki checkout may be pushed |

## Giga_Mario boundary

| Need | Where it lives |
|------|----------------|
| `split.csv` assignment | Giga_Mario `run_split_predict` |
| Caduceus / LegNet window parsing | Giga_Mario `parse_data` — add new models of that kind there |
| Toy ACGT integer adapter | this repo `models/data_prepare`, because Giga_Mario does not emit it |
| Toy RNN and encoder-decoder | this repo, trained on the Giga_Mario split |

## reverse_complement input / output

See `src/giga_mario_bench/bench/reverse_complement/contract.md`.

Input records: `id`, `pair_id`, `role`, `sequence`, `rate`.

Answers: `id`, `sequence` = reverse complement.

Score rows: `id`, `y_true`, `y_pred`, `proba` with shape `(length, 4)`.
