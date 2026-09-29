"""Many-to-many RNN entry for the reverse-complement benchmark."""

from giga_mario_bench.models.reverse_complement.common import test_model, train_model

MODEL_NAME = "many_to_many_rnn"


def train(*args, **kwargs):
    """Train the many-to-many RNN."""
    return train_model(MODEL_NAME, *args, **kwargs)


def test(*args, **kwargs):
    """Test the many-to-many RNN."""
    return test_model(MODEL_NAME, *args, **kwargs)
