from __future__ import annotations

import numpy as np
import pandas as pd


def build_probability_signal(
    predictions: pd.DataFrame,
    long_threshold: float = 0.55,
    short_threshold: float = 0.45,
    max_position: float = 1.0,
) -> pd.DataFrame:
    """
    Convert posterior probability into raw trading position.

    Long when P(r > 0) is sufficiently high.
    Short when P(r > 0) is sufficiently low.
    Flat otherwise.
    """
    out = predictions[["prob_positive"]].copy()
    out["raw_position"] = 0.0

    out.loc[out["prob_positive"] >= long_threshold, "raw_position"] = max_position
    out.loc[out["prob_positive"] <= short_threshold, "raw_position"] = -max_position

    out["confidence"] = np.abs(out["prob_positive"] - 0.5) * 2.0
    out["raw_position"] = out["raw_position"] * out["confidence"]

    return out
