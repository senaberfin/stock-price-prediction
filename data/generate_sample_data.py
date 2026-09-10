"""
generate_sample_data.py

Generates a synthetic daily stock-price CSV (Date, Open, High, Low, Close, Volume)
using Geometric Brownian Motion so the notebook can run end-to-end out of the box.

For real results, replace `data/stock_data.csv` with an actual dataset, e.g.:
  - Kaggle "DJIA 30 Stock Time Series" dataset
  - Any CSV from Yahoo Finance / your broker's export with a "Close" column

Usage:
    python generate_sample_data.py --ticker AMZN --years 12 --out stock_data.csv
"""
import argparse
import numpy as np
import pandas as pd


def generate_gbm_prices(n_days: int, start_price: float = 50.0,
                         mu: float = 0.0005, sigma: float = 0.018,
                         seed: int = 42) -> np.ndarray:
    """Simulate a daily close-price series with Geometric Brownian Motion."""
    rng = np.random.default_rng(seed)
    daily_returns = rng.normal(loc=mu, scale=sigma, size=n_days)
    price_relatives = np.exp(daily_returns)
    prices = start_price * np.cumprod(price_relatives)
    return prices


def build_ohlcv(close: np.ndarray, seed: int = 42):
    rng = np.random.default_rng(seed + 1)
    noise = rng.normal(0, 0.004, size=len(close))
    open_ = close * (1 + noise)
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, size=len(close))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, size=len(close))))
    volume = rng.integers(2_000_000, 8_000_000, size=len(close))
    return open_, high, low, volume


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AMZN")
    parser.add_argument("--years", type=int, default=12)
    parser.add_argument("--out", default="stock_data.csv")
    args = parser.parse_args()

    n_days = args.years * 252
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)

    close = generate_gbm_prices(n_days)
    open_, high, low, volume = build_ohlcv(close)

    df = pd.DataFrame({
        "Date": dates,
        "Open": open_.round(2),
        "High": high.round(2),
        "Low": low.round(2),
        "Close": close.round(2),
        "Volume": volume,
    })

    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows of synthetic '{args.ticker}'-like data to {args.out}")


if __name__ == "__main__":
    main()
