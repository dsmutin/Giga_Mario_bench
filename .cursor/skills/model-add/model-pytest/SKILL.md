---
name: model-pytest
description: Add the two mandatory contract tests for a new bench/model pair and run them on the toy panel.
---

# Contract tests for a new model

Use this from [model-add](../SKILL.md). Read `CONTRIBUTING.md` and `wiki/Testing.md` first.

Every new `<bench_name>/<model_name>` pair gets both tests, marked `mandatory` and `contracts`:

1. **raw → model-ready.** Build or load the bench records, run the adapter, and check ids, lengths, and the alphabet.
2. **model-ready → score-ready.** Giga_Mario random `traintestsplit` (keep the bench's zero-shot ids), train with a small `max_epochs` so the suite stays short, test, and require finite `f1`, `r2`, and `rocauc` from `score_general`.

Also run the toy entry (`examples/toy/run.py`) when the new model is part of the default toy grid. Training in the toy may use a higher epoch cap; it must stop on `plateau` or `max_epochs`.

Do not mark these tests `optional`. After they pass, update `wiki/Contracts.md` and `wiki/Testing.md`.
