# Toy reverse complement

`run.py` builds the default panel and trains both baseline models until validation loss plateaus.

- 10 random ACGT sequences and 10 mutants at rate `0.1` (a range `low:high` or a comma-separated list draws one rate per mutant)
- 2 pairs (4 sequences) are zero-shot
- the other 16 sequences are assigned by Giga_Mario `type=random`
- models: many-to-many bidirectional RNN and an encoder-decoder
- scores: macro F1, R2 of probabilities against the true one-hot bases, macro ROC AUC

```bash
python examples/toy/run.py
```

Outputs land in `examples/toy/out/` (gitignored). A short JSON summary is written to `examples/toy/data/toy_result.json`.
