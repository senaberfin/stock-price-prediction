"""
Unit tests for src/data_preprocessing.py

Run with:
    pytest tests/
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_preprocessing import (  # noqa: E402
    create_sliding_windows,
    load_close_prices,
    prepare_dataset,
    scale_series,
    to_tensors,
    train_test_split_sequences,
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "stock_data.csv")


def test_load_close_prices_returns_date_and_close_columns():
    df = load_close_prices(DATA_PATH)
    assert list(df.columns) == ["Date", "Close"]
    assert len(df) > 0
    assert pd.api.types.is_datetime64_any_dtype(df["Date"])


def test_load_close_prices_is_sorted_chronologically():
    df = load_close_prices(DATA_PATH)
    assert df["Date"].is_monotonic_increasing


def test_scale_series_respects_feature_range():
    values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    scaled, scaler = scale_series(values, feature_range=(-1, 1))
    assert scaled.min() == pytest.approx(-1.0)
    assert scaled.max() == pytest.approx(1.0)
    recovered = scaler.inverse_transform(scaled).flatten()
    np.testing.assert_allclose(recovered, values)


def test_create_sliding_windows_shapes():
    series = np.arange(0, 100, dtype=float)
    lookback = 20
    X, y = create_sliding_windows(series, lookback=lookback)

    expected_n_samples = len(series) - lookback
    assert X.shape == (expected_n_samples, lookback)
    assert y.shape == (expected_n_samples,)


def test_create_sliding_windows_values_are_correct():
    series = np.arange(0, 30, dtype=float)
    lookback = 5
    X, y = create_sliding_windows(series, lookback=lookback)

    np.testing.assert_allclose(X[0], series[0:5])
    assert y[0] == series[5]

    np.testing.assert_allclose(X[-1], series[-lookback - 1:-1])
    assert y[-1] == series[-1]


def test_train_test_split_sequences_is_chronological_and_covers_all_data():
    X = np.arange(100).reshape(50, 2)
    y = np.arange(50)
    X_train, X_test, y_train, y_test = train_test_split_sequences(X, y, train_frac=0.8)

    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)
    assert y_train[-1] < y_test[0]


def test_to_tensors_shapes_and_dtype():
    X_train = np.random.rand(10, 20)
    X_test = np.random.rand(3, 20)
    y_train = np.random.rand(10)
    y_test = np.random.rand(3)

    X_train_t, X_test_t, y_train_t, y_test_t = to_tensors(X_train, X_test, y_train, y_test)

    assert X_train_t.shape == (10, 20, 1)
    assert X_test_t.shape == (3, 20, 1)
    assert y_train_t.shape == (10,)
    assert y_test_t.shape == (3,)
    assert X_train_t.dtype.is_floating_point


def test_prepare_dataset_end_to_end():
    result = prepare_dataset(DATA_PATH, lookback=20, train_frac=0.8)

    for key in ("X_train", "X_test", "y_train", "y_test", "scaler", "test_dates", "lookback"):
        assert key in result

    assert result["X_train"].shape[1] == 20
    assert result["X_train"].shape[2] == 1
    assert len(result["test_dates"]) == result["X_test"].shape[0]
    assert result["lookback"] == 20
