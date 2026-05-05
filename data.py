from __future__ import annotations

import pandas as pd
import yfinance as yf


def load_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance."""
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)

    if data.empty:
        raise ValueError(f"No data downloaded for ticker={ticker}")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    if "Close" not in data.columns:
        raise ValueError("Expected a 'Close' column in downloaded data.")

    prices = data[["Close"]].rename(columns={"Close": "close"}).dropna()
    prices.index.name = "date"
    return prices
