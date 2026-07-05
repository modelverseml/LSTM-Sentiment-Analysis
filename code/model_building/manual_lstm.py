"""
An LSTM written from scratch in NumPy -- no deep-learning library.

Same role as manual_rnn.py in the Vanilla-RNN project, but a full LSTM cell
(forget / input / candidate / output gates + a cell state) instead of a plain
Elman cell. Single layer, take the hidden state at the last real word, then a
linear layer into the class scores. Trained with full backprop-through-time
(BPTT) and Adam, with gradient clipping and best-dev checkpointing.

Batch-first, row-vector convention, with the four gates sharing a [a_prev, x]
concatenation (so each gate weight has shape (hidden, hidden + input_dim)):

    concat = [a_prev, x]                         (batch, hidden + input_dim)
    f = sigmoid(concat @ Wf.T + bf)              forget gate
    i = sigmoid(concat @ Wi.T + bi)              input gate
    g = tanh   (concat @ Wc.T + bc)              candidate cell
    o = sigmoid(concat @ Wo.T + bo)              output gate
    c = f * c_prev + i * g                        cell state
    a = o * tanh(c)                               hidden state
"""

import numpy as np

DEFAULT_SEED = 42
_GATES = ("Wf", "bf", "Wi", "bi", "Wc", "bc", "Wo", "bo")


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class ManualLSTM:
    def __init__(self, input_dim=None, hidden_dim=None, num_classes=None,
                 seed=DEFAULT_SEED):
        if input_dim is None:
            return  # empty shell, to be filled by load()
        rng = np.random.default_rng(seed)
        concat = hidden_dim + input_dim
        scale = 1.0 / np.sqrt(concat)
        self.params = {
            "Wf": rng.standard_normal((hidden_dim, concat)) * scale,
            "Wi": rng.standard_normal((hidden_dim, concat)) * scale,
            "Wc": rng.standard_normal((hidden_dim, concat)) * scale,
            "Wo": rng.standard_normal((hidden_dim, concat)) * scale,
            # forget-gate bias starts at 1.0 -- a standard trick that lets
            # gradients flow through the cell state early in training
            "bf": np.ones((1, hidden_dim)),
            "bi": np.zeros((1, hidden_dim)),
            "bc": np.zeros((1, hidden_dim)),
            "bo": np.zeros((1, hidden_dim)),
            "Wy": rng.standard_normal((num_classes, hidden_dim)) / np.sqrt(hidden_dim),
            "by": np.zeros((1, num_classes)),
        }
        self._init_adam()

    # ---- shapes / state ----
    @property
    def hidden_dim(self):
        return self.params["Wf"].shape[0]

    def _init_adam(self):
        self._m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._t = 0

    @staticmethod
    def _last_real_idx(X):
        mask = np.abs(X).sum(axis=-1) > 0          # (batch, seq_len)
        lengths = np.clip(mask.sum(axis=1), 1, None)
        return lengths - 1

    @staticmethod
    def _softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    # ---- forward / backward ----
    def forward(self, X):
        B, T, _ = X.shape
        H = self.hidden_dim
        p = self.params

        a_prev = np.zeros((B, H))
        c_prev = np.zeros((B, H))
        a_seq = np.zeros((B, T, H))
        cache = []
        for t in range(T):
            xt = X[:, t, :]
            concat = np.concatenate([a_prev, xt], axis=1)      # (B, H+D)
            ft = _sigmoid(concat @ p["Wf"].T + p["bf"])
            it = _sigmoid(concat @ p["Wi"].T + p["bi"])
            gt = np.tanh(concat @ p["Wc"].T + p["bc"])
            ot = _sigmoid(concat @ p["Wo"].T + p["bo"])
            c_next = ft * c_prev + it * gt
            a_next = ot * np.tanh(c_next)
            a_seq[:, t, :] = a_next
            cache.append((a_prev, c_prev, ft, it, gt, ot, c_next, xt))
            a_prev, c_prev = a_next, c_next

        last_idx = self._last_real_idx(X)
        a_last = a_seq[np.arange(B), last_idx]
        logits = a_last @ p["Wy"].T + p["by"]
        return logits, (cache, last_idx, a_last, B, T, H)

    def backward(self, logits, y, ctx):
        cache, last_idx, a_last, B, T, H = ctx
        p = self.params

        probs = self._softmax(logits)
        dlogits = probs.copy()
        dlogits[np.arange(B), y] -= 1.0
        dlogits /= B

        grads = {k: np.zeros_like(v) for k, v in p.items()}
        grads["Wy"] = dlogits.T @ a_last
        grads["by"] = dlogits.sum(axis=0, keepdims=True)

        # output gradient enters the hidden state only at each example's last word
        dA = np.zeros((B, T, H))
        dA[np.arange(B), last_idx] = dlogits @ p["Wy"]

        da_next = np.zeros((B, H))
        dc_next = np.zeros((B, H))
        for t in reversed(range(T)):
            a_prev, c_prev, ft, it, gt, ot, c_next, xt = cache[t]
            da = dA[:, t, :] + da_next
            tanh_c = np.tanh(c_next)

            dot = da * tanh_c * ot * (1 - ot)
            dc = da * ot * (1 - tanh_c ** 2) + dc_next
            dgt = dc * it * (1 - gt ** 2)
            dit = dc * gt * it * (1 - it)
            dft = dc * c_prev * ft * (1 - ft)

            concat = np.concatenate([a_prev, xt], axis=1)
            grads["Wf"] += dft.T @ concat
            grads["Wi"] += dit.T @ concat
            grads["Wc"] += dgt.T @ concat
            grads["Wo"] += dot.T @ concat
            grads["bf"] += dft.sum(axis=0, keepdims=True)
            grads["bi"] += dit.sum(axis=0, keepdims=True)
            grads["bc"] += dgt.sum(axis=0, keepdims=True)
            grads["bo"] += dot.sum(axis=0, keepdims=True)

            dconcat = dft @ p["Wf"] + dit @ p["Wi"] + dgt @ p["Wc"] + dot @ p["Wo"]
            da_next = dconcat[:, :H]
            dc_next = dc * ft
        return grads

    def _clip(self, grads, max_norm=5.0):
        total = np.sqrt(sum((g ** 2).sum() for g in grads.values()))
        if total > max_norm:
            for k in grads:
                grads[k] *= max_norm / (total + 1e-6)

    def _adam_step(self, grads, lr, b1=0.9, b2=0.999, eps=1e-8):
        self._t += 1
        for k in self.params:
            self._m[k] = b1 * self._m[k] + (1 - b1) * grads[k]
            self._v[k] = b2 * self._v[k] + (1 - b2) * (grads[k] ** 2)
            mhat = self._m[k] / (1 - b1 ** self._t)
            vhat = self._v[k] / (1 - b2 ** self._t)
            self.params[k] -= lr * mhat / (np.sqrt(vhat) + eps)

    # ---- public API (matches ManualRNN) ----
    def predict_proba(self, X):
        logits, _ = self.forward(X)
        return self._softmax(logits)

    def accuracy(self, X, y):
        return float((self.predict_proba(X).argmax(axis=1) == y).mean())

    def fit(self, X_train, y_train, X_dev, y_dev, epochs, batch_size, lr):
        self._init_adam()
        n = len(X_train)
        best_dev, best_params = -1.0, None
        for epoch in range(epochs):
            order = np.random.permutation(n)
            for s in range(0, n, batch_size):
                idx = order[s:s + batch_size]
                logits, ctx = self.forward(X_train[idx])
                grads = self.backward(logits, y_train[idx], ctx)
                self._clip(grads)
                self._adam_step(grads, lr)
            dev_acc = self.accuracy(X_dev, y_dev)
            if dev_acc > best_dev:
                best_dev = dev_acc
                best_params = {k: v.copy() for k, v in self.params.items()}
            print(f"  [manual] epoch {epoch + 1}/{epochs}  dev_acc={dev_acc:.3f}")
        if best_params is not None:
            self.params = best_params
        print(f"  [manual] best dev_acc={best_dev:.3f}")

    def save(self, path):
        np.savez(path, **self.params)

    @classmethod
    def load(cls, path):
        model = cls()
        npz = np.load(path)
        model.params = {k: npz[k] for k in (*_GATES, "Wy", "by")}
        return model
