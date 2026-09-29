---
name: model-adapt
description: Adapt raw benchmark rows into the tensor or record format one model accepts, reusing Giga_Mario parse_data when it already does that.
---

# Adapt a model

Use this from [model-add](../SKILL.md). Read `CONTRIBUTING.md` first.

## Input / output

- Input: `input/records.jsonl` and `output/answers.jsonl` for the bench (`contract.md`).
- Output: JSONL the train loop can batch. For the reverse-complement toy that is `id`, `pair_id`, `role`, `x`, `y` with ACGT integers (`A=0,C=1,G=2,T=3`).

## Steps

1. Search Giga_Mario `src/pipeline/parse_data.py` and the conversion wiki pages. If the model consumes `PARSED` windows, call that function and stop.
2. If the encoding is the same as an existing adapter, import it. Do not duplicate `encode_sequence`.
3. Put the new adapter in `src/giga_mario_bench/models/data_prepare/<model_name>.py` with a `prepare(records, answers)` function.
4. Wire `giga_mario_bench model prepare --model <model_name>`.
5. Hand off to [model-pytest](../model-pytest/SKILL.md) for the raw→model-ready contract.
