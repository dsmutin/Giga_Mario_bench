"""Adapter for the encoder-decoder. Reuses the shared ACGT encoding."""

from giga_mario_bench.models.data_prepare.encode import prepare_records

MODEL_NAME = "encoder_decoder"


def prepare(records, answers):
    """Encode raw rows for the encoder-decoder."""
    return prepare_records(records, answers)
