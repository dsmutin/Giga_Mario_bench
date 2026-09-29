"""Adapter for the many-to-many RNN. Same integers as the encoder-decoder."""

from giga_mario_bench.models.data_prepare.encode import prepare_records

MODEL_NAME = "many_to_many_rnn"


def prepare(records, answers):
    """Encode raw rows for the many-to-many RNN."""
    return prepare_records(records, answers)
