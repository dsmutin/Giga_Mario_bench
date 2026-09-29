---
name: model-add
description: Add a model to giga_mario_bench, reuse Giga_Mario when it already adapts that model, and register contract tests plus a toy run.
---

# Add a model

Read `CONTRIBUTING.md` and `wiki/Contracts.md` first. Read the Giga_Mario wiki page for `parse_data` / train before writing an adapter.

## Decide where the model lives

1. If Giga_Mario already downloads, parses, or trains this model (Caduceus, LegNet, and later windowed genomic models), add the missing piece **there** and call it from this repo. Do not copy that code.
2. Otherwise add it here.

## Checklist

- [ ] `models/general.py`: registry entry (`model get` / `--all`)
- [ ] `models/data_prepare/<model_name>.py`: raw rows → the rows that model accepts. Reuse `encode.py` when the encoding is shared.
- [ ] `models/<bench_name>/<model_name>.py`: train until validation plateaus, then test. Reuse `models/reverse_complement/common.py` or the Giga_Mario train wrapper.
- [ ] Subskill [model-adapt](model-adapt/SKILL.md) for the raw→ready contract
- [ ] Subskill [model-pytest](model-pytest/SKILL.md) for the two contract tests and a toy run
- [ ] `wiki/Contracts.md` row for the new pair
- [ ] Mandatory pytest passes
- [ ] Version bump in `VERSION` only (feature → minor)

Do not push the tool repository. Push the wiki checkout after its pages change.
