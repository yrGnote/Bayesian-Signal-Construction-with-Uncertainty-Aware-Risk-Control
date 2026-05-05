from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_performance(
    returns: pd.Series,
    positions: pd.Series,
    annualization: int = 252,
) -> dict[str, float]:
    """Compute standard backtest metrics."""
    returns = returns.dropna()

    if returns.empty:
        raise ValueError("No returns available for performance summary.")

    ann_return = float(returns.mean() * annualization)
    ann_vol = float(returns.std(ddof=1) * np.sqrt(annualization))
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    max_drawdown = float(drawdown.min())

    turnover = float(positions.diff().abs().mean())

    return {
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "average_turnover": turnover,
    }
