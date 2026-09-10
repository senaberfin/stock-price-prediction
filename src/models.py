"""
models.py

LSTM and GRU regression models for next-step stock price prediction.
Both models take a sequence of `lookback` past prices (shape: batch, seq_len, 1)
and output a single predicted value (the next day's price).
"""
import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_dim: int = 1, hidden_dim: int = 32,
                 num_layers: int = 2, output_dim: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out.squeeze(-1)


class GRUModel(nn.Module):
    def __init__(self, input_dim: int = 1, hidden_dim: int = 32,
                 num_layers: int = 2, output_dim: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)

        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out.squeeze(-1)
