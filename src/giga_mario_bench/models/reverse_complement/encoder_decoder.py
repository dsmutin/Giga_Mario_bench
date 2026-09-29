"""Encoder-decoder entry for the reverse-complement benchmark."""

from giga_mario_bench.models.reverse_complement.common import test_model, train_model

MODEL_NAME = "encoder_decoder"


def train(*args, **kwargs):
    """Train the encoder-decoder."""
    return train_model(MODEL_NAME, *args, **kwargs)


def test(*args, **kwargs):
    """Test the encoder-decoder."""
    return test_model(MODEL_NAME, *args, **kwargs)
