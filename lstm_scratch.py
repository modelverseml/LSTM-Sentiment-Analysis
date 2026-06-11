"""From-scratch, multi-layer LSTM in pure NumPy.

This is the reference implementation the README derives step by step. It uses the
**batch-first** layout `(m, T_x, n_x)` (examples are rows), so every gate is computed
as `sigmoid(z @ W.T + b)` with `z = [a_prev, x_t]` — the transpose of the textbook
column-vector form. The forward pass (`layer_forward` / `lstm_forward`) implements the
six LSTM equations; the backward pass (`layer_backward` / `lstm_backward`) implements
the batch-first BPTT gradients from §7 of the README, line for line.

Tensor convention:
    m   : number of sequences in the batch
    T_x : time steps per sequence
    n_x : input feature size
    n_y : output feature size (vocab size)
    n_a : hidden / cell state size of a layer
"""

import numpy as np


class LSTM:

    def __init__(self, X, Y, hidden_layers=(100,),
                 learning_rate=0.01, epochs=15,
                 batch_size=32, task='classification'):

        self.X = X
        self.Y = Y

        self.hidden_layers = hidden_layers
        self.L = len(hidden_layers)

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.task = task

        self.n_x = X.shape[-1]
        self.n_y = Y.shape[-1]

        self.parameters = self.initialize_parameters()

    def initialize_parameters(self):
        # One set of gate weights per stacked layer. Convention (batch-first):
        #   a, x are (m, ...) row vectors; weights are (n_a, n_a + in_size);
        #   a gate is computed as  concat @ W.T + b  with b of shape (n_a,).
        Wf, bf, Wi, bi, Wc, bc, Wo, bo = [], [], [], [], [], [], [], []

        for l in range(self.L):
            n_a = self.hidden_layers[l]
            in_size = self.n_x if l == 0 else self.hidden_layers[l - 1]
            concat = n_a + in_size
            # Xavier/Glorot init: scaling by 1/sqrt(fan_in) keeps the gate
            # pre-activations in a sensible range so gradients neither vanish
            # nor explode -- a tiny fixed scale (e.g. 0.01) leaves the gates
            # almost frozen and the model barely learns under plain SGD.
            scale = 1.0 / np.sqrt(concat)
            Wf.append(np.random.randn(n_a, concat) * scale)
            # Forget-gate bias of 1.0 starts the cell in "remember" mode, the
            # standard trick that greatly eases learning long dependencies.
            bf.append(np.ones(n_a))
            Wi.append(np.random.randn(n_a, concat) * scale)
            bi.append(np.zeros(n_a))
            Wc.append(np.random.randn(n_a, concat) * scale)
            bc.append(np.zeros(n_a))
            Wo.append(np.random.randn(n_a, concat) * scale)
            bo.append(np.zeros(n_a))

        # Output (dense) layer maps the top hidden state to the targets.
        Wy = np.random.randn(self.n_y, self.hidden_layers[-1]) / np.sqrt(self.hidden_layers[-1])
        by = np.zeros(self.n_y)

        return {
            'Wf': Wf, 'bf': bf,
            'Wi': Wi, 'bi': bi,
            'Wc': Wc, 'bc': bc,
            'Wo': Wo, 'bo': bo,
            'Wy': Wy, 'by': by,
        }

    def layer_forward(self, X_seq, a0, c0, l):

        Wf = self.parameters['Wf'][l]
        bf = self.parameters['bf'][l]
        Wi = self.parameters['Wi'][l]
        bi = self.parameters['bi'][l]
        Wc = self.parameters['Wc'][l]
        bc = self.parameters['bc'][l]
        Wo = self.parameters['Wo'][l]
        bo = self.parameters['bo'][l]

        m, T_x, _ = X_seq.shape
        n_a = self.hidden_layers[l]

        a_prev = a0
        c_prev = c0

        a = np.zeros((m, T_x, n_a))
        c = np.zeros((m, T_x, n_a))

        layer_cache = []

        for t in range(T_x):
            xt = X_seq[:, t, :]                              # (m, in_size)
            concat = np.concatenate([a_prev, xt], axis=1)    # (m, n_a + in_size)

            ft = self.sigmoid(concat @ Wf.T + bf)            # forget gate
            it = self.sigmoid(concat @ Wi.T + bi)            # input gate
            cct = np.tanh(concat @ Wc.T + bc)                # candidate cell
            ot = self.sigmoid(concat @ Wo.T + bo)            # output gate

            c_next = ft * c_prev + it * cct
            a_next = ot * np.tanh(c_next)

            a[:, t, :] = a_next
            c[:, t, :] = c_next
            layer_cache.append((a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt))

            a_prev = a_next
            c_prev = c_next

        return a, c, layer_cache

    def lstm_forward(self, X, a0, c0):

        caches_per_layer = []

        inp = X
        for l in range(self.L):
            a, c, layer_cache = self.layer_forward(inp, a0[l], c0[l], l)
            caches_per_layer.append(layer_cache)
            inp = a                                          # feed states up the stack

        a_top = inp                                          # (m, T_x, n_a_top)

        # Many-to-many: a prediction at every timestep (next-token modelling).
        Wy = self.parameters['Wy']
        by = self.parameters['by']
        scores = a_top @ Wy.T + by                           # (m, T_x, n_y)

        if self.task == 'classification':
            y_pred = self.softmax(scores)
        else:
            y_pred = scores

        caches = (caches_per_layer, X, a_top)
        return caches, a_top, y_pred

    def layer_backward(self, da_above, layer_cache, l):

        Wf = self.parameters['Wf'][l]
        Wi = self.parameters['Wi'][l]
        Wc = self.parameters['Wc'][l]
        Wo = self.parameters['Wo'][l]

        m, T_x, n_a = da_above.shape
        in_size = Wf.shape[1] - n_a

        dWf_l = np.zeros_like(Wf)
        dbf_l = np.zeros_like(self.parameters['bf'][l])
        dWi_l = np.zeros_like(Wi)
        dbi_l = np.zeros_like(self.parameters['bi'][l])
        dWc_l = np.zeros_like(Wc)
        dbc_l = np.zeros_like(self.parameters['bc'][l])
        dWo_l = np.zeros_like(Wo)
        dbo_l = np.zeros_like(self.parameters['bo'][l])

        dx = np.zeros((m, T_x, in_size))
        da_next = np.zeros((m, n_a))
        dc_next = np.zeros((m, n_a))

        for t in reversed(range(T_x)):

            (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt) = layer_cache[t]

            da = da_above[:, t, :] + da_next                 # total grad into a_next
            tanh_c = np.tanh(c_next)

            # Backprop through a_next = ot * tanh(c_next) and the cell update.
            dot = da * tanh_c * ot * (1 - ot)
            dc = da * ot * (1 - tanh_c ** 2) + dc_next        # total grad into c_next

            dcct = dc * it * (1 - cct ** 2)
            dit = dc * cct * it * (1 - it)
            dft = dc * c_prev * ft * (1 - ft)

            concat = np.concatenate([a_prev, xt], axis=1)     # (m, n_a + in_size)

            dWf_l += dft.T @ concat
            dWi_l += dit.T @ concat
            dWc_l += dcct.T @ concat
            dWo_l += dot.T @ concat
            dbf_l += dft.sum(axis=0)
            dbi_l += dit.sum(axis=0)
            dbc_l += dcct.sum(axis=0)
            dbo_l += dot.sum(axis=0)

            # Gradient w.r.t. the [a_prev, xt] concatenation, then split it.
            dconcat = dft @ Wf + dit @ Wi + dcct @ Wc + dot @ Wo
            da_prev = dconcat[:, :n_a]
            dx[:, t, :] = dconcat[:, n_a:]
            dc_prev = dc * ft

            da_next = da_prev
            dc_next = dc_prev

        grads = {
            'dWf_l': dWf_l, 'dbf_l': dbf_l,
            'dWi_l': dWi_l, 'dbi_l': dbi_l,
            'dWc_l': dWc_l, 'dbc_l': dbc_l,
            'dWo_l': dWo_l, 'dbo_l': dbo_l,
        }
        return grads, dx

    def lstm_backward(self, dscores, caches):

        (caches_per_layer, X, a_top) = caches

        # Output-layer gradients. dscores is d(loss)/d(scores) at every timestep,
        # e.g. (y_pred - Y) / (m * T_x) for softmax+cross-entropy or linear+MSE.
        Wy = self.parameters['Wy']
        dWy = np.einsum('mty,mta->ya', dscores, a_top)        # (n_y, n_a_top)
        dby = dscores.sum(axis=(0, 1))                        # (n_y,)
        da_top = dscores @ Wy                                 # (m, T_x, n_a_top)

        dWf = [None] * self.L
        dbf = [None] * self.L
        dWi = [None] * self.L
        dbi = [None] * self.L
        dWc = [None] * self.L
        dbc = [None] * self.L
        dWo = [None] * self.L
        dbo = [None] * self.L

        # The top layer receives the output-layer gradient at every timestep.
        da_above = da_top

        for l in reversed(range(self.L)):
            grads, dx = self.layer_backward(da_above, caches_per_layer[l], l)
            dWf[l] = grads['dWf_l']
            dbf[l] = grads['dbf_l']
            dWi[l] = grads['dWi_l']
            dbi[l] = grads['dbi_l']
            dWc[l] = grads['dWc_l']
            dbc[l] = grads['dbc_l']
            dWo[l] = grads['dWo_l']
            dbo[l] = grads['dbo_l']
            da_above = dx                                     # pass down the stack

        return {
            'dWf': dWf, 'dbf': dbf,
            'dWi': dWi, 'dbi': dbi,
            'dWc': dWc, 'dbc': dbc,
            'dWo': dWo, 'dbo': dbo,
            'dWy': dWy, 'dby': dby,
        }

    def update_parameters(self, gradients):
        lr = self.learning_rate
        for l in range(self.L):
            self.parameters['Wf'][l] -= lr * gradients['dWf'][l]
            self.parameters['bf'][l] -= lr * gradients['dbf'][l]
            self.parameters['Wi'][l] -= lr * gradients['dWi'][l]
            self.parameters['bi'][l] -= lr * gradients['dbi'][l]
            self.parameters['Wc'][l] -= lr * gradients['dWc'][l]
            self.parameters['bc'][l] -= lr * gradients['dbc'][l]
            self.parameters['Wo'][l] -= lr * gradients['dWo'][l]
            self.parameters['bo'][l] -= lr * gradients['dbo'][l]
        self.parameters['Wy'] -= lr * gradients['dWy']
        self.parameters['by'] -= lr * gradients['dby']

    def compute_loss(self, y_pred, Y):
        # Averaged over both the batch (m) and the time axis (T_x), matching the
        # per-timestep next-token objective the framework wrappers also use.
        m, T_x = Y.shape[0], Y.shape[1]
        if self.task == 'classification':
            eps = 1e-12
            return -np.sum(Y * np.log(y_pred + eps)) / (m * T_x)
        return np.sum((y_pred - Y) ** 2) / (2 * m * T_x)

    def _clip_gradients(self, gradients, max_norm=5.0):
        # Global-norm gradient clipping keeps BPTT through long sequences stable.
        sq = 0.0
        for k, v in gradients.items():
            if isinstance(v, list):
                sq += sum(float(np.sum(g ** 2)) for g in v)
            else:
                sq += float(np.sum(v ** 2))
        norm = np.sqrt(sq)
        if norm > max_norm:
            scale = max_norm / (norm + 1e-12)
            for k, v in gradients.items():
                if isinstance(v, list):
                    gradients[k] = [g * scale for g in v]
                else:
                    gradients[k] = v * scale
        return gradients

    def train(self):
        m = self.X.shape[0]
        batch_size = self.batch_size or m

        for epoch in range(self.epochs):
            perm = np.random.permutation(m)
            epoch_loss, n_batches = 0.0, 0

            for start in range(0, m, batch_size):
                idx = perm[start:start + batch_size]
                Xb, Yb = self.X[idx], self.Y[idx]
                mb, T_x = Xb.shape[0], Xb.shape[1]

                a0 = [np.zeros((mb, n_a)) for n_a in self.hidden_layers]
                c0 = [np.zeros((mb, n_a)) for n_a in self.hidden_layers]

                caches, a_top, y_pred = self.lstm_forward(Xb, a0, c0)

                epoch_loss += self.compute_loss(y_pred, Yb)
                n_batches += 1

                # Gradient of the (mean) loss w.r.t. the scores at every timestep:
                # (y_pred - Y) / (m * T_x) for softmax+CE and for linear+MSE.
                dscores = (y_pred - Yb) / (mb * T_x)

                gradients = self.lstm_backward(dscores, caches)
                gradients = self._clip_gradients(gradients)
                self.update_parameters(gradients)

            print(f"epoch {epoch + 1}/{self.epochs} - loss: {epoch_loss / max(n_batches, 1):.4f}")

        return self

    def predict(self, X):
        # Returns per-timestep probabilities, shape (m, T_x, n_y), as the shared
        # utils.evaluate / utils.generate helpers expect.
        m = X.shape[0]
        a0 = [np.zeros((m, n_a)) for n_a in self.hidden_layers]
        c0 = [np.zeros((m, n_a)) for n_a in self.hidden_layers]
        _, _, y_pred = self.lstm_forward(X, a0, c0)
        return y_pred

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        # Stable softmax over the last axis (works for 2-D and 3-D inputs).
        x = x - np.max(x, axis=-1, keepdims=True)
        e = np.exp(x)
        return e / np.sum(e, axis=-1, keepdims=True)
