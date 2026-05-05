from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def plot_equity_curve(backtest: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(backtest.index, backtest["equity_curve"], label="Strategy")
    plt.plot(backtest.index, backtest["benchmark_curve"], label="Benchmark")
    plt.title("Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_drawdown(backtest: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(backtest.index, backtest["drawdown"], label="Drawdown")
    plt.title("Strategy Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
