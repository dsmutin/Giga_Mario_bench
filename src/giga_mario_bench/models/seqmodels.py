"""NumPy many-to-many RNN and encoder-decoder trained until validation plateaus.

Both models map an integer DNA sequence to a same-length integer sequence.
Gradients are analytical (BPTT). Adam updates the parameters.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np

N_BASE = 4


def softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax."""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _xavier(rng: np.random.Generator, rows: int, cols: int) -> np.ndarray:
    limit = math.sqrt(6.0 / (rows + cols))
    return rng.uniform(-limit, limit, size=(rows, cols))


@dataclass
class TrainResult:
    """Plateau training summary.

    ``train_f1`` and ``val_f1`` are per-character macro F1 of the emitted
    sequence (argmax over the output distribution, not teacher-forced tokens).
    """

    train_loss: list[float]
    val_loss: list[float]
    stopped_reason: str
    epochs_ran: int
    train_f1: list[float] = field(default_factory=list)
    val_f1: list[float] = field(default_factory=list)
    best_epoch: int = 0


class _Adam:
    def __init__(self, params: dict[str, np.ndarray], lr: float) -> None:
        self.lr = lr
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        # Global norm clip keeps short toy runs from diverging.
        total = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
        scale = 1.0 if total < 5.0 else 5.0 / total
        for key, grad in grads.items():
            g = grad * scale
            self.m[key] = b1 * self.m[key] + (1.0 - b1) * g
            self.v[key] = b2 * self.v[key] + (1.0 - b2) * (g * g)
            mhat = self.m[key] / (1.0 - b1**self.t)
            vhat = self.v[key] / (1.0 - b2**self.t)
            params[key] -= self.lr * mhat / (np.sqrt(vhat) + eps)


def _ce(probs: np.ndarray, y_idx: np.ndarray) -> float:
    picked = probs[np.arange(len(y_idx)), y_idx]
    return float(-np.mean(np.log(np.clip(picked, 1e-12, 1.0))))


class ManyToManyRNN:
    """Bidirectional Elman RNN. Each position emits a base distribution."""

    kind = "many_to_many_rnn"

    def __init__(self, hidden: int = 32, seed: int = 0, lr: float = 0.05) -> None:
        rng = np.random.default_rng(seed)
        h = hidden
        self.hidden = h
        self.p = {
            "W_xh_f": _xavier(rng, h, N_BASE) * 0.5,
            "W_hh_f": _xavier(rng, h, h) * 0.5,
            "b_h_f": np.zeros(h),
            "W_xh_b": _xavier(rng, h, N_BASE) * 0.5,
            "W_hh_b": _xavier(rng, h, h) * 0.5,
            "b_h_b": np.zeros(h),
            "W_y": _xavier(rng, N_BASE, 2 * h) * 0.5,
            "b_y": np.zeros(N_BASE),
        }
        self.opt = _Adam(self.p, lr)

    def forward(self, x_idx: np.ndarray) -> dict[str, np.ndarray]:
        length = len(x_idx)
        hidden = self.hidden
        one = np.eye(N_BASE)[x_idx]
        h_f = np.zeros((length, hidden))
        h_b = np.zeros((length, hidden))
        prev = np.zeros(hidden)
        for t in range(length):
            z = self.p["W_xh_f"] @ one[t] + self.p["W_hh_f"] @ prev + self.p["b_h_f"]
            prev = np.tanh(z)
            h_f[t] = prev
        prev = np.zeros(hidden)
        for t in range(length - 1, -1, -1):
            z = self.p["W_xh_b"] @ one[t] + self.p["W_hh_b"] @ prev + self.p["b_h_b"]
            prev = np.tanh(z)
            h_b[t] = prev
        cat = np.concatenate([h_f, h_b], axis=1)
        logits = cat @ self.p["W_y"].T + self.p["b_y"]
        return {"x": one, "h_f": h_f, "h_b": h_b, "probs": softmax(logits)}

    def loss_grad(
        self, x_idx: np.ndarray, y_idx: np.ndarray
    ) -> tuple[float, dict[str, np.ndarray], np.ndarray]:
        cache = self.forward(x_idx)
        length = len(x_idx)
        hidden = self.hidden
        probs = cache["probs"]
        loss = _ce(probs, y_idx)
        dlogit = probs.copy()
        dlogit[np.arange(length), y_idx] -= 1.0
        dlogit /= length
        grads = {k: np.zeros_like(v) for k, v in self.p.items()}
        cat = np.concatenate([cache["h_f"], cache["h_b"]], axis=1)
        grads["W_y"] = dlogit.T @ cat
        grads["b_y"] = dlogit.sum(axis=0)
        dcat = dlogit @ self.p["W_y"]
        dh_f = dcat[:, :hidden]
        dh_b = dcat[:, hidden:]
        dh_next = np.zeros(hidden)
        for t in range(length - 1, -1, -1):
            dh = dh_f[t] + dh_next
            dtanh = dh * (1.0 - cache["h_f"][t] ** 2)
            h_prev = cache["h_f"][t - 1] if t else np.zeros(hidden)
            grads["W_xh_f"] += np.outer(dtanh, cache["x"][t])
            grads["W_hh_f"] += np.outer(dtanh, h_prev)
            grads["b_h_f"] += dtanh
            dh_next = self.p["W_hh_f"].T @ dtanh
        dh_next = np.zeros(hidden)
        for t in range(length):
            dh = dh_b[t] + dh_next
            dtanh = dh * (1.0 - cache["h_b"][t] ** 2)
            h_next = cache["h_b"][t + 1] if t + 1 < length else np.zeros(hidden)
            grads["W_xh_b"] += np.outer(dtanh, cache["x"][t])
            grads["W_hh_b"] += np.outer(dtanh, h_next)
            grads["b_h_b"] += dtanh
            dh_next = self.p["W_hh_b"].T @ dtanh
        return loss, grads, probs

    def predict_proba(self, x_idx: np.ndarray) -> np.ndarray:
        return self.forward(x_idx)["probs"]

    def apply_grad(self, grads: dict[str, np.ndarray]) -> None:
        self.opt.step(self.p, grads)


class EncoderDecoder:
    """Encoder RNN plus a teacher-forced decoder RNN of the same length."""

    kind = "encoder_decoder"

    def __init__(self, hidden: int = 32, seed: int = 0, lr: float = 0.05) -> None:
        rng = np.random.default_rng(seed)
        h = hidden
        self.hidden = h
        self.p = {
            "W_xh": _xavier(rng, h, N_BASE) * 0.5,
            "W_hh": _xavier(rng, h, h) * 0.5,
            "b_h": np.zeros(h),
            "W_xh_d": _xavier(rng, h, N_BASE) * 0.5,
            "W_hh_d": _xavier(rng, h, h) * 0.5,
            "b_h_d": np.zeros(h),
            "W_y": _xavier(rng, N_BASE, h) * 0.5,
            "b_y": np.zeros(N_BASE),
        }
        self.opt = _Adam(self.p, lr)

    def _encode(self, one: np.ndarray) -> np.ndarray:
        hidden = self.hidden
        length = len(one)
        h = np.zeros((length, hidden))
        prev = np.zeros(hidden)
        for t in range(length):
            z = self.p["W_xh"] @ one[t] + self.p["W_hh"] @ prev + self.p["b_h"]
            prev = np.tanh(z)
            h[t] = prev
        return h

    def _decode(
        self, context: np.ndarray, y_idx: np.ndarray | None, length: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        hidden = self.hidden
        states = np.zeros((length, hidden))
        prev_y = np.zeros(N_BASE)
        prev_h = context
        logits_in = []
        for t in range(length):
            z = self.p["W_xh_d"] @ prev_y + self.p["W_hh_d"] @ prev_h + self.p["b_h_d"]
            prev_h = np.tanh(z)
            states[t] = prev_h
            logits_in.append(prev_y.copy())
            if y_idx is not None and t + 1 < length:
                prev_y = np.eye(N_BASE)[y_idx[t]]
            elif y_idx is None:
                logit = self.p["W_y"] @ prev_h + self.p["b_y"]
                prev_y = np.eye(N_BASE)[int(np.argmax(logit))]
        inputs = np.stack(logits_in, axis=0)
        logits = states @ self.p["W_y"].T + self.p["b_y"]
        return states, inputs, softmax(logits)

    def forward_teacher(self, x_idx: np.ndarray, y_idx: np.ndarray) -> dict[str, np.ndarray]:
        one = np.eye(N_BASE)[x_idx]
        enc = self._encode(one)
        states, inputs, probs = self._decode(enc[-1], y_idx, len(x_idx))
        return {"x": one, "h_enc": enc, "s": states, "dec_in": inputs, "probs": probs}

    def loss_grad(
        self, x_idx: np.ndarray, y_idx: np.ndarray
    ) -> tuple[float, dict[str, np.ndarray], np.ndarray]:
        cache = self.forward_teacher(x_idx, y_idx)
        length = len(x_idx)
        hidden = self.hidden
        probs = cache["probs"]
        loss = _ce(probs, y_idx)
        dlogit = probs.copy()
        dlogit[np.arange(length), y_idx] -= 1.0
        dlogit /= length
        grads = {k: np.zeros_like(v) for k, v in self.p.items()}
        grads["W_y"] = dlogit.T @ cache["s"]
        grads["b_y"] = dlogit.sum(axis=0)
        dh = dlogit @ self.p["W_y"]
        # Decoder BPTT. s[t] depends on s[t-1] and on the teacher token y[t-1].
        dh_next = np.zeros(hidden)
        dcontext = np.zeros(hidden)
        for t in range(length - 1, -1, -1):
            dh_t = dh[t] + dh_next
            dtanh = dh_t * (1.0 - cache["s"][t] ** 2)
            h_prev = cache["s"][t - 1] if t else cache["h_enc"][-1]
            grads["W_xh_d"] += np.outer(dtanh, cache["dec_in"][t])
            grads["W_hh_d"] += np.outer(dtanh, h_prev)
            grads["b_h_d"] += dtanh
            dh_next = self.p["W_hh_d"].T @ dtanh
            if t == 0:
                dcontext = dh_next
        # Encoder BPTT from the final state only.
        dh_next = dcontext
        for t in range(length - 1, -1, -1):
            dh_t = dh_next
            dtanh = dh_t * (1.0 - cache["h_enc"][t] ** 2)
            h_prev = cache["h_enc"][t - 1] if t else np.zeros(hidden)
            grads["W_xh"] += np.outer(dtanh, cache["x"][t])
            grads["W_hh"] += np.outer(dtanh, h_prev)
            grads["b_h"] += dtanh
            dh_next = self.p["W_hh"].T @ dtanh
        return loss, grads, probs

    def predict_proba(self, x_idx: np.ndarray) -> np.ndarray:
        one = np.eye(N_BASE)[x_idx]
        enc = self._encode(one)
        _states, _inputs, probs = self._decode(enc[-1], None, len(x_idx))
        return probs

    def apply_grad(self, grads: dict[str, np.ndarray]) -> None:
        self.opt.step(self.p, grads)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


def _lstm_pack(rng: np.random.Generator, hidden: int) -> dict[str, np.ndarray]:
    """One LSTM: input, forget, cell, output gates stacked on axis 0.

    The forget-gate bias starts at 1 so the cell is open at initialization.
    """
    bias = np.zeros(4 * hidden)
    bias[hidden : 2 * hidden] = 1.0
    return {
        "W_x": _xavier(rng, 4 * hidden, N_BASE) * 0.5,
        "W_h": _xavier(rng, 4 * hidden, hidden) * 0.5,
        "b": bias,
    }


def _lstm_forward(
    W_x: np.ndarray,
    W_h: np.ndarray,
    b: np.ndarray,
    inputs: np.ndarray,
    h0: np.ndarray,
    c0: np.ndarray,
) -> dict[str, np.ndarray]:
    length, _features = inputs.shape
    hidden = h0.shape[0]
    h = h0
    c = c0
    states = np.zeros((length, hidden))
    cells = np.zeros((length, hidden))
    prev_h = np.zeros((length, hidden))
    prev_c = np.zeros((length, hidden))
    forget = np.zeros((length, hidden))
    ingate = np.zeros((length, hidden))
    cell_in = np.zeros((length, hidden))
    outgate = np.zeros((length, hidden))
    for t in range(length):
        prev_h[t] = h
        prev_c[t] = c
        z = W_x @ inputs[t] + W_h @ h + b
        f = _sigmoid(z[hidden : 2 * hidden])
        i = _sigmoid(z[:hidden])
        g = np.tanh(z[2 * hidden : 3 * hidden])
        o = _sigmoid(z[3 * hidden :])
        c = f * c + i * g
        h = o * np.tanh(c)
        states[t] = h
        cells[t] = c
        forget[t] = f
        ingate[t] = i
        cell_in[t] = g
        outgate[t] = o
    return {
        "h": states,
        "c": cells,
        "h_prev": prev_h,
        "c_prev": prev_c,
        "f": forget,
        "i": ingate,
        "g": cell_in,
        "o": outgate,
    }


def _lstm_bptt(
    cache: dict[str, np.ndarray],
    inputs: np.ndarray,
    W_h: np.ndarray,
    d_h: np.ndarray,
    dh0: np.ndarray,
    dc0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Backprop one LSTM. ``d_h`` is the output gradient at each step."""
    length = inputs.shape[0]
    hidden = d_h.shape[1]
    dW_x = np.zeros((4 * hidden, inputs.shape[1]))
    dW_h = np.zeros((4 * hidden, hidden))
    db = np.zeros(4 * hidden)
    dh_next = dh0
    dc_next = dc0
    for t in range(length - 1, -1, -1):
        f = cache["f"][t]
        i = cache["i"][t]
        g = cache["g"][t]
        o = cache["o"][t]
        c = cache["c"][t]
        tanh_c = np.tanh(c)
        dh = d_h[t] + dh_next
        dc = dc_next + dh * o * (1.0 - tanh_c**2)
        df = dc * cache["c_prev"][t]
        di = dc * g
        dg = dc * i
        do = dh * tanh_c
        dz = np.concatenate(
            [
                di * i * (1.0 - i),
                df * f * (1.0 - f),
                dg * (1.0 - g**2),
                do * o * (1.0 - o),
            ]
        )
        dW_x += np.outer(dz, inputs[t])
        dW_h += np.outer(dz, cache["h_prev"][t])
        db += dz
        dh_next = W_h.T @ dz
        dc_next = dc * f
    return dW_x, dW_h, db, dh_next, dc_next


class ManyToManyLSTM:
    """Bidirectional LSTM. Each position emits a base distribution.

    The cell state is the change relative to the tanh Elman baseline: it can
    carry a base across the reverse-complement lag. The loss stays per-base
    cross-entropy.
    """

    kind = "many_to_many_lstm"

    def __init__(self, hidden: int = 32, seed: int = 0, lr: float = 0.05) -> None:
        rng = np.random.default_rng(seed)
        self.hidden = hidden
        forward = _lstm_pack(rng, hidden)
        backward = _lstm_pack(rng, hidden)
        self.p = {
            "W_x_f": forward["W_x"],
            "W_h_f": forward["W_h"],
            "b_f": forward["b"],
            "W_x_b": backward["W_x"],
            "W_h_b": backward["W_h"],
            "b_b": backward["b"],
            "W_y": _xavier(rng, N_BASE, 2 * hidden) * 0.5,
            "b_y": np.zeros(N_BASE),
        }
        self.opt = _Adam(self.p, lr)

    def _directions(self, one: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        hidden = self.hidden
        zeros_h = np.zeros(hidden)
        zeros_c = np.zeros(hidden)
        forward = _lstm_forward(self.p["W_x_f"], self.p["W_h_f"], self.p["b_f"], one, zeros_h, zeros_c)
        backward = _lstm_forward(
            self.p["W_x_b"], self.p["W_h_b"], self.p["b_b"], one[::-1], zeros_h, zeros_c
        )
        return forward, backward

    def forward(self, x_idx: np.ndarray) -> dict[str, np.ndarray]:
        one = np.eye(N_BASE)[x_idx]
        forward, backward = self._directions(one)
        cat = np.concatenate([forward["h"], backward["h"][::-1]], axis=1)
        return {"x": one, "fwd": forward, "bwd": backward, "probs": softmax(cat @ self.p["W_y"].T + self.p["b_y"])}

    def loss_grad(
        self, x_idx: np.ndarray, y_idx: np.ndarray
    ) -> tuple[float, dict[str, np.ndarray], np.ndarray]:
        cache = self.forward(x_idx)
        length = len(x_idx)
        hidden = self.hidden
        probs = cache["probs"]
        loss = _ce(probs, y_idx)
        dlogit = probs.copy()
        dlogit[np.arange(length), y_idx] -= 1.0
        dlogit /= length
        cat = np.concatenate([cache["fwd"]["h"], cache["bwd"]["h"][::-1]], axis=1)
        grads = {key: np.zeros_like(value) for key, value in self.p.items()}
        grads["W_y"] = dlogit.T @ cat
        grads["b_y"] = dlogit.sum(axis=0)
        dcat = dlogit @ self.p["W_y"]
        zeros = np.zeros(hidden)
        grads["W_x_f"], grads["W_h_f"], grads["b_f"], _dh, _dc = _lstm_bptt(
            cache["fwd"], cache["x"], self.p["W_h_f"], dcat[:, :hidden], zeros, zeros
        )
        grads["W_x_b"], grads["W_h_b"], grads["b_b"], _dh, _dc = _lstm_bptt(
            cache["bwd"], cache["x"][::-1], self.p["W_h_b"], dcat[:, hidden:][::-1], zeros, zeros
        )
        return loss, grads, probs

    def predict_proba(self, x_idx: np.ndarray) -> np.ndarray:
        return self.forward(x_idx)["probs"]

    def apply_grad(self, grads: dict[str, np.ndarray]) -> None:
        self.opt.step(self.p, grads)


class EncoderDecoderLSTM:
    """LSTM encoder-decoder. Training loss is teacher-forced cross-entropy.

    Free-run ``predict_proba`` feeds its own argmax back into the decoder.
    The decoder starts from the encoder's final hidden state and cell.
    """

    kind = "encoder_decoder_lstm"

    def __init__(self, hidden: int = 32, seed: int = 0, lr: float = 0.05) -> None:
        rng = np.random.default_rng(seed)
        self.hidden = hidden
        enc = _lstm_pack(rng, hidden)
        dec = _lstm_pack(rng, hidden)
        self.p = {
            "W_x": enc["W_x"],
            "W_h": enc["W_h"],
            "b": enc["b"],
            "W_x_d": dec["W_x"],
            "W_h_d": dec["W_h"],
            "b_d": dec["b"],
            "W_y": _xavier(rng, N_BASE, hidden) * 0.5,
            "b_y": np.zeros(N_BASE),
        }
        self.opt = _Adam(self.p, lr)

    def _encode(self, one: np.ndarray) -> dict[str, np.ndarray]:
        hidden = self.hidden
        return _lstm_forward(
            self.p["W_x"], self.p["W_h"], self.p["b"], one, np.zeros(hidden), np.zeros(hidden)
        )

    def _decode(
        self, h0: np.ndarray, c0: np.ndarray, y_idx: np.ndarray | None, length: int
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        inputs = np.zeros((length, N_BASE))
        prev = np.zeros(N_BASE)
        hidden = self.hidden
        h = h0.copy()
        c = c0.copy()
        states = np.zeros((length, hidden))
        cells = np.zeros((length, hidden))
        prev_h = np.zeros((length, hidden))
        prev_c = np.zeros((length, hidden))
        forget = np.zeros((length, hidden))
        ingate = np.zeros((length, hidden))
        cell_in = np.zeros((length, hidden))
        outgate = np.zeros((length, hidden))
        for t in range(length):
            inputs[t] = prev
            prev_h[t] = h
            prev_c[t] = c
            z = self.p["W_x_d"] @ prev + self.p["W_h_d"] @ h + self.p["b_d"]
            f = _sigmoid(z[hidden : 2 * hidden])
            i = _sigmoid(z[:hidden])
            g = np.tanh(z[2 * hidden : 3 * hidden])
            o = _sigmoid(z[3 * hidden :])
            c = f * c + i * g
            h = o * np.tanh(c)
            states[t] = h
            cells[t] = c
            forget[t] = f
            ingate[t] = i
            cell_in[t] = g
            outgate[t] = o
            if y_idx is not None and t + 1 < length:
                prev = np.eye(N_BASE)[y_idx[t]]
            elif y_idx is None:
                logit = self.p["W_y"] @ h + self.p["b_y"]
                prev = np.eye(N_BASE)[int(np.argmax(logit))]
        cache = {
            "h": states,
            "c": cells,
            "h_prev": prev_h,
            "c_prev": prev_c,
            "f": forget,
            "i": ingate,
            "g": cell_in,
            "o": outgate,
        }
        return cache, inputs, softmax(states @ self.p["W_y"].T + self.p["b_y"])

    def loss_grad(
        self, x_idx: np.ndarray, y_idx: np.ndarray
    ) -> tuple[float, dict[str, np.ndarray], np.ndarray]:
        one = np.eye(N_BASE)[x_idx]
        enc = self._encode(one)
        dec, inputs, probs = self._decode(enc["h"][-1], enc["c"][-1], y_idx, len(x_idx))
        length = len(x_idx)
        hidden = self.hidden
        loss = _ce(probs, y_idx)
        dlogit = probs.copy()
        dlogit[np.arange(length), y_idx] -= 1.0
        dlogit /= length
        grads = {key: np.zeros_like(value) for key, value in self.p.items()}
        grads["W_y"] = dlogit.T @ dec["h"]
        grads["b_y"] = dlogit.sum(axis=0)
        d_h = dlogit @ self.p["W_y"]
        grads["W_x_d"], grads["W_h_d"], grads["b_d"], dh0, dc0 = _lstm_bptt(
            dec, inputs, self.p["W_h_d"], d_h, np.zeros(hidden), np.zeros(hidden)
        )
        d_enc = np.zeros((length, hidden))
        grads["W_x"], grads["W_h"], grads["b"], _dh, _dc = _lstm_bptt(
            enc, one, self.p["W_h"], d_enc, dh0, dc0
        )
        return loss, grads, probs

    def predict_proba(self, x_idx: np.ndarray) -> np.ndarray:
        one = np.eye(N_BASE)[x_idx]
        enc = self._encode(one)
        _cache, _inputs, probs = self._decode(enc["h"][-1], enc["c"][-1], None, len(x_idx))
        return probs

    def apply_grad(self, grads: dict[str, np.ndarray]) -> None:
        self.opt.step(self.p, grads)


def per_character_f1(
    model: ManyToManyRNN | EncoderDecoder | ManyToManyLSTM | EncoderDecoderLSTM,
    pairs: list[tuple[np.ndarray, np.ndarray]],
) -> float:
    """Macro F1 of argmax outputs, counted once per character."""
    from giga_mario_bench.score.general import character_macro_f1

    if not pairs:
        return float("nan")
    truth_parts: list[np.ndarray] = []
    pred_parts: list[np.ndarray] = []
    for x_idx, y_idx in pairs:
        pred = np.argmax(model.predict_proba(x_idx), axis=1)
        truth_parts.append(np.asarray(y_idx, dtype=int))
        pred_parts.append(pred.astype(int))
    return character_macro_f1(np.concatenate(truth_parts), np.concatenate(pred_parts))


def fit_plateau(
    model: ManyToManyRNN | EncoderDecoder | ManyToManyLSTM | EncoderDecoderLSTM,
    train: list[tuple[np.ndarray, np.ndarray]],
    val: list[tuple[np.ndarray, np.ndarray]],
    *,
    max_epochs: int = 30,
    patience: int = 4,
    min_delta: float = 1e-4,
    seed: int = 0,
    restore_best: bool = True,
    on_epoch: Callable[[int, bool], None] | None = None,
) -> TrainResult:
    """Train until validation loss stops improving, then restore the best weights.

    Per-character F1 is recorded after each epoch on the weights of that epoch.
    ``on_epoch`` receives the 1-based epoch index and whether it is the best
    validation loss so far. Set ``restore_best=False`` to keep the last epoch.
    """
    if not train:
        raise ValueError("train split is empty")
    monitor = val if val else train
    best = math.inf
    best_params = deepcopy(model.p)
    best_epoch = 0
    wait = 0
    train_hist: list[float] = []
    val_hist: list[float] = []
    train_f1: list[float] = []
    val_f1: list[float] = []
    reason = "max_epochs"
    rng = np.random.default_rng(seed)
    epochs_ran = 0
    for _epoch in range(max_epochs):
        epochs_ran += 1
        order = rng.permutation(len(train))
        total = 0.0
        for index in order:
            x_idx, y_idx = train[int(index)]
            loss, grads, _probs = model.loss_grad(x_idx, y_idx)
            model.apply_grad(grads)
            total += loss
        train_hist.append(total / len(train))
        val_loss = 0.0
        for x_idx, y_idx in monitor:
            loss, _grads, _probs = model.loss_grad(x_idx, y_idx)
            val_loss += loss
        val_loss /= len(monitor)
        val_hist.append(val_loss)
        train_f1.append(per_character_f1(model, train))
        val_f1.append(per_character_f1(model, val) if val else float("nan"))
        is_best = val_loss < best - min_delta
        if is_best:
            best = val_loss
            best_params = deepcopy(model.p)
            best_epoch = epochs_ran
            wait = 0
        else:
            wait += 1
        model._epoch_trace = {
            "epoch": epochs_ran,
            "train_loss": train_hist[-1],
            "val_loss": val_hist[-1],
            "train_f1": train_f1[-1],
            "val_f1": val_f1[-1],
            "is_best": is_best,
        }
        if on_epoch is not None:
            on_epoch(epochs_ran, is_best)
        if not is_best and wait >= patience:
            reason = "plateau"
            break
    if restore_best:
        model.p = best_params
    return TrainResult(
        train_hist,
        val_hist,
        reason,
        epochs_ran,
        train_f1,
        val_f1,
        best_epoch,
    )


def finite_difference_ok(
    model: ManyToManyRNN | EncoderDecoder | ManyToManyLSTM | EncoderDecoderLSTM,
    key: str,
    eps: float = 1e-5,
) -> float:
    """Return the max absolute error between analytical and numeric gradients."""
    rng = np.random.default_rng(1)
    x_idx = rng.integers(0, N_BASE, size=4)
    y_idx = rng.integers(0, N_BASE, size=4)
    _loss, grads, _probs = model.loss_grad(x_idx, y_idx)
    original = model.p[key].copy()
    numeric = np.zeros_like(original)
    flat = original.ravel()
    num_flat = numeric.ravel()
    # Check a handful of entries so the test stays small.
    picks = [0, flat.size // 2, flat.size - 1]
    for index in picks:
        saved = flat[index]
        flat[index] = saved + eps
        model.p[key] = flat.reshape(original.shape).copy()
        loss_plus, _, _ = model.loss_grad(x_idx, y_idx)
        flat[index] = saved - eps
        model.p[key] = flat.reshape(original.shape).copy()
        loss_minus, _, _ = model.loss_grad(x_idx, y_idx)
        num_flat[index] = (loss_plus - loss_minus) / (2 * eps)
        flat[index] = saved
        model.p[key] = original.copy()
    analytical = grads[key].ravel()
    return float(np.max(np.abs(analytical[picks] - num_flat[picks])))
