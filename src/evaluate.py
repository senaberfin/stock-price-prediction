"""
evaluate.py

Evaluation helpers: inverse-transform scaled predictions back to real
prices, compute MSE/RMSE, and plot actual vs. predicted prices.
"""
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_squared_error


def inverse_transform(scaler, values: np.ndarray) -> np.ndarray:
    """Map values from the [-1, 1] scaled space back to real price units."""
    values = np.asarray(values).reshape(-1, 1)
    return scaler.inverse_transform(values).flatten()


def evaluate_model(
    model: torch.nn.Module, X_test: torch.Tensor, y_test: torch.Tensor, scaler
) -> Dict[str, np.ndarray]:
    """Run inference on the test set and return real-scale predictions/targets + metrics."""
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test).numpy()

    y_true_scaled = y_test.numpy()

    y_pred = inverse_transform(scaler, y_pred_scaled)
    y_true = inverse_transform(scaler, y_true_scaled)

    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))

    return {"y_true": y_true, "y_pred": y_pred, "mse": mse, "rmse": rmse}


def plot_predictions(
    dates, y_true, y_pred_lstm, y_pred_gru,
    title="LSTM vs GRU: Actual vs Predicted", save_path=None,
):
    plt.figure(figsize=(12, 6))
    plt.plot(dates, y_true, label="Actual", color="black", linewidth=1.5)
    plt.plot(dates, y_pred_lstm, label="LSTM Predicted", linestyle="--")
    plt.plot(dates, y_pred_gru, label="GRU Predicted", linestyle="--")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_loss_curves(lstm_losses, gru_losses, save_path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(lstm_losses, label="LSTM training loss")
    plt.plot(gru_losses, label="GRU training loss")
    plt.title("Training Loss (MSE) over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (scaled space)")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
