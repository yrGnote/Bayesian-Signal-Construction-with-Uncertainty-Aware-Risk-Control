from __future__ import annotations

import numpy as np
import pandas as pd


def monte_carlo_tail_risk(
    predictions: pd.DataFrame,
    n_simulations: int = 5000,
    var_level: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """Simulate predictive returns and estimate VaR / expected shortfall."""
    rng = np.random.default_rng(random_state)

    rows = []
    for date, row in predictions.iterrows():
        sims = rng.normal(
            loc=float(row["pred_mean"]),
            scale=float(row["pred_std"]),
            size=n_simulations,
        )

        var = float(np.quantile(sims, var_level))
        expected_shortfall = float(sims[sims <= var].mean()) if np.any(sims <= var) else var

        rows.append({
            "date": date,
            "var": var,
            "expected_shortfall": expected_shortfall,
            "tail_risk": abs(expected_shortfall),
        })

    return pd.DataFrame(rows).set_index("date")


def apply_risk_scaling(
    signals: pd.DataFrame,
    tail_risk: pd.DataFrame,
    target_daily_var: float = 0.015,
    min_scale: float = 0.1,
    max_scale: float = 1.0,
) -> pd.DataFrame:
    """Scale positions down when estimated tail risk is high."""
    out = signals.join(tail_risk, how="inner")

    raw_scale = target_daily_var / out["tail_risk"].replace(0.0, np.nan)
    out["risk_scale"] = raw_scale.clip(lower=min_scale, upper=max_scale).fillna(min_scale)
    out["position"] = out["raw_position"] * out["risk_scale"]

    return out[["raw_position", "confidence", "var", "expected_shortfall", "tail_risk", "risk_scale", "position"]]
