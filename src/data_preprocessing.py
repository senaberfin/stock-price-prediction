"""
data_preprocessing.py

Utilities for turning a raw stock-price CSV into PyTorch-ready
sliding-window sequences for LSTM/GRU training.
"""
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler


def load_close_prices(
    csv_path: str, date_col: str = "Date", price_col: str = "Close"
) -> pd.DataFrame:
    """Load a CSV and return a DataFrame indexed by date with one 'Close' column."""
    df = pd.read_csv(csv_path, parse_dates=[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    return df[[date_col, price_col]].rename(columns={date_col: "Date", price_col: "Close"})


def scale_series(values: np.ndarray, feature_range=(-1, 1)) -> Tuple[np.ndarray, MinMaxScaler]:
    """Scale a 1D price series into `feature_range` using MinMaxScaler."""
    scaler = MinMaxScaler(feature_range=feature_range)
    scaled = scaler.fit_transform(values.reshape(-1, 1))
    return scaled, scaler


def create_sliding_windows(series: np.ndarray, lookback: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) sequences from a scaled 1D series.

    X[i] = series[i : i+lookback]      (lookback days of prices)
    y[i] = series[i+lookback]          (the next day's price)
    """
    X, y = [], []
    for i in range(len(series) - lookback):
        X.append(series[i: i + lookback])
        y.append(series[i + lookback])
    return np.array(X), np.array(y)


def train_test_split_sequences(X: np.ndarray, y: np.ndarray, train_frac: float = 0.8):
    """Chronological (non-shuffled) train/test split, since order matters for time series."""
    split_idx = int(len(X) * train_frac)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    return X_train, X_test, y_train, y_test


def to_tensors(X_train, X_test, y_train, y_test):
    """Convert numpy arrays to PyTorch tensors shaped (batch, seq_len, 1)."""
    X_train_t = torch.from_numpy(X_train).float().unsqueeze(-1)
    X_test_t = torch.from_numpy(X_test).float().unsqueeze(-1)
    y_train_t = torch.from_numpy(y_train).float()
    y_test_t = torch.from_numpy(y_test).float()
    return X_train_t, X_test_t, y_train_t, y_test_t


def prepare_dataset(csv_path: str, lookback: int = 20, train_frac: float = 0.8):
    """
    End-to-end pipeline: CSV -> scaled series -> sliding windows -> train/test tensors.

    Returns a dict with tensors, the fitted scaler, and the raw dates/prices
    (useful later for plotting predictions against real dates).
    """
    df = load_close_prices(csv_path)
    prices = df["Close"].values.astype(float)

    scaled, scaler = scale_series(prices)
    X, y = create_sliding_windows(scaled.flatten(), lookback=lookback)
    X_train, X_test, y_train, y_test = train_test_split_sequences(X, y, train_frac=train_frac)
    X_train_t, X_test_t, y_train_t, y_test_t = to_tensors(X_train, X_test, y_train, y_test)

    split_idx = int(len(X) * train_frac)
    test_dates = df["Date"].values[lookback:][split_idx:]

    return {
        "X_train": X_train_t, "X_test": X_test_t,
        "y_train": y_train_t, "y_test": y_test_t,
        "scaler": scaler,
        "dates": df["Date"].values,
        "raw_prices": prices,
        "test_dates": test_dates,
        "lookback": lookback,
    }
