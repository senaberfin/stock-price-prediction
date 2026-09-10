"""
Unit tests for src/train.py and src/evaluate.py

Run with:
    pytest tests/
"""
import os
import sys
import math

import numpy as np
import pytest
import torch
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluate import evaluate_model, inverse_transform  # noqa: E402
from models import GRUModel  # noqa: E402
from train import train_model  # noqa: E402


def test_train_model_returns_losses_and_time():
    torch.manual_seed(0)
    model = GRUModel(input_dim=1, hidden_dim=8, num_layers=1)
    X = torch.randn(20, 10, 1)
    y = torch.randn(20)

    losses, elapsed = train_model(model, X, y, num_epochs=5, lr=0.01, print_every=100)

    assert len(losses) == 5
    assert all(isinstance(loss_value, float) for loss_value in losses)
    assert elapsed >= 0


def test_train_model_loss_decreases_over_more_epochs():
    torch.manual_seed(0)
    model = GRUModel(input_dim=1, hidden_dim=8, num_layers=1)
    X = torch.randn(20, 10, 1)
    y = torch.randn(20)

    losses, _ = train_model(model, X, y, num_epochs=50, lr=0.01, print_every=1000)
    assert np.mean(losses[-10:]) < np.mean(losses[:10])


def test_inverse_transform_recovers_original_scale():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0]).reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(values)
    scaled = scaler.transform(values).flatten()

    recovered = inverse_transform(scaler, scaled)
    np.testing.assert_allclose(recovered, values.flatten(), atol=1e-6)


def test_evaluate_model_returns_expected_keys_and_metrics():
    torch.manual_seed(0)
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(np.linspace(0, 100, 50).reshape(-1, 1))

    model = GRUModel(input_dim=1, hidden_dim=8, num_layers=1)
    X_test = torch.randn(6, 10, 1)
    y_test = torch.randn(6)

    result = evaluate_model(model, X_test, y_test, scaler)

    for key in ("y_true", "y_pred", "mse", "rmse"):
        assert key in result
    assert len(result["y_true"]) == 6
    assert len(result["y_pred"]) == 6
    assert result["mse"] >= 0
    assert result["rmse"] >= 0
    assert result["rmse"] == pytest.approx(math.sqrt(result["mse"]))
