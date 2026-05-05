from __future__ import annotations

import pandas as pd


def run_backtest(
    positions: pd.DataFrame,
    realized_returns: pd.DataFrame,
    transaction_cost_bps: float = 2.0,
) -> pd.DataFrame:
    """
    Backtest one-period-ahead returns.

    Position at date t is applied to target_return at date t.
    """
    df = positions.join(realized_returns, how="inner").copy()

    df["position_lag"] = df["position"].shift(1).fillna(0.0)
    df["turnover"] = df["position"].diff().abs().fillna(df["position"].abs())

    cost = transaction_cost_bps / 10000.0
    df["transaction_cost"] = cost * df["turnover"]

    df["strategy_return"] = df["position_lag"] * df["target_return"] - df["transaction_cost"]
    df["benchmark_return"] = df["target_return"]

    df["equity_curve"] = (1.0 + df["strategy_return"]).cumprod()
    df["benchmark_curve"] = (1.0 + df["benchmark_return"]).cumprod()

    running_max = df["equity_curve"].cummax()
    df["drawdown"] = df["equity_curve"] / running_max - 1.0

    return df
