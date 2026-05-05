from __future__ import annotations

import pandas as pd


def build_features(
    prices: pd.DataFrame,
    return_lags: int = 5,
    volatility_window: int = 20,
    momentum_windows: list[int] | None = None,
) -> pd.DataFrame:
    """Create lagged returns, volatility, and momentum features."""
    if momentum_windows is None:
        momentum_windows = [5, 20, 60]

    df = prices.copy()
    df["return"] = df["close"].pct_change()

    for lag in range(1, return_lags + 1):
        df[f"return_lag_{lag}"] = df["return"].shift(lag)

    df[f"vol_{volatility_window}"] = df["return"].rolling(volatility_window).std()

    for window in momentum_windows:
        df[f"momentum_{window}"] = df["close"].pct_change(window)

    df["target_return"] = df["return"].shift(-1)

    feature_cols = [
        c for c in df.columns
        if c.startswith("return_lag_") or c.startswith("vol_") or c.startswith("momentum_")
    ]

    dataset = df[feature_cols + ["target_return"]].dropna()
    return dataset
