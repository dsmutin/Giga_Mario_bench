"""NumPy many-to-many RNN and encoder-decoder trained until validation plateaus.

Both models map an integer DNA sequence to a same-length integer sequence.
Gradients are analytical (BPTT). Adam updates the parameters.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass

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
    """Plateau training summary."""

    train_loss: list[float]
    val_loss: list[float]
    stopped_reason: str
    epochs_ran: int


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


def fit_plateau(
    model: ManyToManyRNN | EncoderDecoder,
    train: list[tuple[np.ndarray, np.ndarray]],
    val: list[tuple[np.ndarray, np.ndarray]],
    *,
    max_epochs: int = 30,
    patience: int = 4,
    min_delta: float = 1e-4,
    seed: int = 0,
) -> TrainResult:
    """Train until validation loss stops improving, then restore the best weights."""
    if not train:
        raise ValueError("train split is empty")
    monitor = val if val else train
    best = math.inf
    best_params = deepcopy(model.p)
    wait = 0
    train_hist: list[float] = []
    val_hist: list[float] = []
    reason = "max_epochs"
    rng = np.random.default_rng(seed)
    epochs_ran = 0
    for epoch in range(max_epochs):
        epochs_ran = epoch + 1
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
            # Recompute loss without a second parameter update.
            loss, _grads, _probs = model.loss_grad(x_idx, y_idx)
            val_loss += loss
        val_loss /= len(monitor)
        val_hist.append(val_loss)
        if val_loss < best - min_delta:
            best = val_loss
            best_params = deepcopy(model.p)
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                reason = "plateau"
                break
    model.p = best_params
    return TrainResult(train_hist, val_hist, reason, epochs_ran)


def finite_difference_ok(model: ManyToManyRNN | EncoderDecoder, key: str, eps: float = 1e-5) -> float:
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
