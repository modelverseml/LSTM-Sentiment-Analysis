"""PyTorch version of the stacked LSTM.

`PyTorchLSTM` mirrors the from-scratch `lstm_scratch.LSTM`: same constructor inputs and the
same `train()` / `predict()` interface, so `utils.py` and `compare.py` drive it unchanged.
It is standalone (a stack of `nn.LSTM` layers trained with Adam) and does not share weights
with the manual model.
"""

import torch
import torch.nn as nn
import numpy as np


class _Net(nn.Module):
    """Stacked LSTM + dense head. batch_first=True keeps the (m, T_x, n_x) layout."""

    def __init__(self, n_x, hidden_layers, n_y):
        super().__init__()
        self.lstms = nn.ModuleList()
        # Chain the layers: each LSTM's hidden size becomes the next one's input size.
        in_size = n_x
        for units in hidden_layers:
            self.lstms.append(nn.LSTM(in_size, units, batch_first=True))
            in_size = units
        # Output projection applied at every timestep (many-to-many).
        self.fc = nn.Linear(in_size, n_y)

    def forward(self, x):
        # Feed the hidden-state sequence up the stack; we only need the outputs, not
        # the final (h, c) tuple.
        for lstm in self.lstms:
            x, _ = lstm(x)
        return self.fc(x)                      # raw logits (m, T_x, n_y)


class PyTorchLSTM:

    def __init__(self, X, Y, hidden_layers=(100, 50), learning_rate=0.01,
                 epochs=15, batch_size=15, task='classification'):
        # X/Y are batch-first: (m, T_x, n_x) and (m, T_x, n_y).
        self.X = X
        self.Y = Y
        self.hidden_layers = list(hidden_layers)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        if task not in ("classification", "regression"):
            raise ValueError("task must be 'classification' or 'regression'")
        self.task = task
        self.n_x = X.shape[2]
        self.n_y = Y.shape[2]
        self.model = None

    def train(self):
        Xk = torch.tensor(self.X, dtype=torch.float32)
        self.model = _Net(self.n_x, self.hidden_layers, self.n_y)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        if self.task == 'classification':
            # CrossEntropyLoss wants integer class indices, so collapse the one-hot
            # targets to the index of the hot column.
            criterion = nn.CrossEntropyLoss()
            targets = torch.tensor(np.argmax(self.Y, axis=2), dtype=torch.long)
        else:
            # MSELoss compares against the real-valued targets directly (float).
            criterion = nn.MSELoss()
            targets = torch.tensor(self.Y, dtype=torch.float32)

        m = Xk.shape[0]
        batch_size = self.batch_size or m

        self.model.train()
        for epoch in range(self.epochs):
            perm = torch.randperm(m)                       # shuffle each epoch
            epoch_loss, n_batches = 0.0, 0
            for start in range(0, m, batch_size):
                idx = perm[start:start + batch_size]
                Xb, tb = Xk[idx], targets[idx]

                optimizer.zero_grad()
                logits = self.model(Xb)

                if self.task == "classification":
                    # Flatten (batch, T_x) into one long axis so cross-entropy scores
                    # every timestep independently.
                    b, T_x, n_y = logits.shape
                    loss = criterion(logits.reshape(b * T_x, n_y),
                                     tb.reshape(b * T_x))
                else:
                    loss = criterion(logits, tb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            print(f"[PyTorch] epoch {epoch + 1}/{self.epochs}, "
                  f"loss {epoch_loss / n_batches:.4f}")
        return self

    def predict(self, X):
        """Run the trained network forward. Returns (m, T_x, n_y)."""
        self.model.eval()
        with torch.no_grad():
            Xk = torch.tensor(X, dtype=torch.float32)
            logits = self.model(Xk)                        # (m, T_x, n_y)
            # Match the manual model's predict: probabilities for classification.
            out = (torch.softmax(logits, dim=2) if self.task == "classification"
                   else logits)
        return out.numpy()
