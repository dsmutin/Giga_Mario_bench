# reverse_complement

The benchmark is the **split method**. Models are instruments. The question
answered here is whether a model trained on one partition can emit the reverse
complement of held-out DNA.

## Input

`input/records.jsonl` from `randomCDS` (`bench.data.random_cds.generate_pairs`).

| Field | Meaning |
|-------|---------|
| `id` | `pXX_orig` or `pXX_mut` |
| `pair_id` | shared by an original and its mutant |
| `role` | `original` or `mutant` |
| `sequence` | ACGT string |
| `rate` | mutation rate used for mutants; null for originals |

Default toy panel: 10 originals, 10 mutants. Two pairs (4 sequences) are marked
zero-shot later. The other 8 pairs (16 sequences) go through Giga_Mario
`run_split_predict` (`type=random` unless another method is requested).

## Output

`output/answers.jsonl`: `id`, `sequence` = reverse complement of the input.

## Model-ready

`models.data_prepare` encodes each base as an integer in `A=0,C=1,G=2,T=3`.
`x` is the input sequence. `y` is the reverse complement. This encoding is the
bench adapter for the toy RNNs. Windowed genomic models that Giga_Mario already
adapts (`parse_data` for Caduceus / LegNet) should be added there and called,
not reimplemented here.

## Score-ready

Each prediction row has `id`, `y_true`, `y_pred`, and `proba` (per-position
4-class probabilities). `score.general.score_general` returns `f1`, `r2`,
`rocauc`, and a 4×4 true-versus-predicted count matrix.
