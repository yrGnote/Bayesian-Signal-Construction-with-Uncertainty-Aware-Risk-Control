import numpy as np
import pandas as pd

from src.features import build_features
from src.bayesian_model import rolling_bayesian_predictions
from src.signals import build_probability_signal


def test_smoke_pipeline_on_synthetic_prices():
    dates = pd.date_range("2020-01-01", periods=220, freq="B")
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0002, 0.01, size=len(dates))
    close = 100 * np.cumprod(1 + returns)
    prices = pd.DataFrame({"close": close}, index=dates)

    dataset = build_features(prices, return_lags=3, volatility_window=10, momentum_windows=[5, 10])
    preds = rolling_bayesian_predictions(dataset, train_window=60, ridge_alpha=10.0)
    sigs = build_probability_signal(preds)

    assert not preds.empty
    assert "prob_positive" in preds.columns
    assert "raw_position" in sigs.columns
