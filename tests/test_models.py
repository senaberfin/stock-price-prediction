"""
Unit tests for src/models.py

Run with:
    pytest tests/
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models import GRUModel, LSTMModel  # noqa: E402


def _make_batch(batch_size=4, seq_len=20, input_dim=1):
    return torch.randn(batch_size, seq_len, input_dim)


def test_lstm_model_output_shape():
    model = LSTMModel(input_dim=1, hidden_dim=16, num_layers=2, output_dim=1)
    x = _make_batch()
    out = model(x)
    assert out.shape == (4,)


def test_gru_model_output_shape():
    model = GRUModel(input_dim=1, hidden_dim=16, num_layers=2, output_dim=1)
    x = _make_batch()
    out = model(x)
    assert out.shape == (4,)


def test_lstm_model_output_is_finite():
    model = LSTMModel(input_dim=1, hidden_dim=8, num_layers=1)
    x = _make_batch(batch_size=2, seq_len=10)
    out = model(x)
    assert torch.isfinite(out).all()


def test_gru_model_output_is_finite():
    model = GRUModel(input_dim=1, hidden_dim=8, num_layers=1)
    x = _make_batch(batch_size=2, seq_len=10)
    out = model(x)
    assert torch.isfinite(out).all()


def test_models_are_trainable_single_step():
    torch.manual_seed(0)
    for ModelClass in (LSTMModel, GRUModel):
        model = ModelClass(input_dim=1, hidden_dim=8, num_layers=1)
        x = _make_batch(batch_size=8, seq_len=10)
        y = torch.randn(8)

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

        loss_before = criterion(model(x), y).item()
        for _ in range(5):
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        loss_after = criterion(model(x), y).item()

        assert loss_after < loss_before, f"{ModelClass.__name__} did not learn on toy batch"


def test_models_accept_variable_batch_size():
    model = LSTMModel()
    for batch_size in (1, 4, 16):
        x = _make_batch(batch_size=batch_size)
        out = model(x)
        assert out.shape == (batch_size,)
