# Bayesian Signal Construction with Uncertainty-Aware Risk Control

This project implements a compact quant research pipeline:

1. Build return and volatility features from market prices.
2. Estimate predictive return distributions with Bayesian linear regression.
3. Construct probabilistic trading signals using posterior uncertainty.
4. Use Monte Carlo simulation to estimate tail risk and risk-aware position size.
5. Backtest the strategy and evaluate Sharpe ratio, drawdown, turnover, and cumulative return.

The goal is not to maximize historical performance, but to demonstrate a clean research workflow:

**model → uncertainty → signal → risk control → backtest → evaluation**

## Project Structure

```text
bayesian_signal_risk_control/
├── README.md
├── requirements.txt
├── config.yaml
├── run_pipeline.py
├── src/
│   ├── data.py
│   ├── features.py
│   ├── bayesian_model.py
│   ├── signals.py
│   ├── risk.py
│   ├── backtest.py
│   ├── metrics.py
│   └── plots.py
├── notebooks/
│   └── research_notebook_template.ipynb
├── data/
├── results/
└── tests/
    └── test_smoke.py
```

## Methodology

### Bayesian predictive model

For each date, the model estimates a predictive distribution of next-period returns using Bayesian linear regression with a Gaussian prior.

### Probabilistic signal

The trading signal is based on the posterior probability of a positive next-period return:

```text
signal_t = P(r_{t+1} > 0 | data_t)
```

This avoids using only point forecasts and explicitly incorporates uncertainty.

### Monte Carlo risk control

Predictive distributions are sampled to estimate downside risk, including VaR and expected shortfall. Position size is reduced when tail risk is high.

### Backtest

The strategy is evaluated with:

- Annualized return
- Annualized volatility
- Sharpe ratio
- Maximum drawdown
- Turnover
- Cumulative return

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the full pipeline:

```bash
python run_pipeline.py
```

By default, the project downloads daily adjusted close prices for SPY using `yfinance`.
You can change the ticker and dates in `config.yaml`.

## Example Resume Bullet

```text
Built a Bayesian time-series signal research pipeline using posterior predictive distributions, Monte Carlo tail-risk estimation, and risk-aware position sizing; evaluated strategy performance via Sharpe ratio, drawdown, and turnover.
```

## Notes

This project is intended for research and educational purposes only. It is not financial advice.
