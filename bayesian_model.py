from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def _fit_bayesian_ridge_closed_form(
    x_train: np.ndarray,
    y_train: np.ndarray,
    ridge_alpha: float,
    noise_variance_floor: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Bayesian linear regression with a zero-mean Gaussian prior.

    The implementation uses a closed-form posterior approximation similar to
    Bayesian ridge regression. It returns the posterior mean and covariance
    of the regression coefficients, plus an estimated observation variance.
    """
    n_features = x_train.shape[1]

    y_centered = y_train - y_train.mean()
    residual_var = float(np.var(y_centered))
    sigma2 = max(residual_var, noise_variance_floor)

    prior_precision = ridge_alpha * np.eye(n_features)
    likelihood_precision = (x_train.T @ x_train) / sigma2

    posterior_cov = np.linalg.pinv(prior_precision + likelihood_precision)
    posterior_mean = posterior_cov @ (x_train.T @ y_train) / sigma2

    return posterior_mean, posterior_cov, sigma2


def rolling_bayesian_predictions(
    dataset: pd.DataFrame,
    train_window: int = 504,
    ridge_alpha: float = 10.0,
    noise_variance_floor: float = 1e-8,
) -> pd.DataFrame:
    """
    Estimate rolling posterior predictive distributions.

    Output columns:
        pred_mean: posterior predictive mean
        pred_std: posterior predictive standard deviation
        prob_positive: posterior probability of positive next-period return
        target_return: realized next-period return
    """
    feature_cols = [c for c in dataset.columns if c != "target_return"]

    rows = []
    dates = []

    x = dataset[feature_cols].values.astype(float)
    y = dataset["target_return"].values.astype(float)

    for i in range(train_window, len(dataset)):
        x_train = x[i - train_window:i]
        y_train = y[i - train_window:i]
        x_test = x[i]

        x_mean = x_train.mean(axis=0)
        x_std = x_train.std(axis=0)
        x_std[x_std == 0] = 1.0

        x_train_std = (x_train - x_mean) / x_std
        x_test_std = (x_test - x_mean) / x_std

        beta_mean, beta_cov, sigma2 = _fit_bayesian_ridge_closed_form(
            x_train_std,
            y_train,
            ridge_alpha=ridge_alpha,
            noise_variance_floor=noise_variance_floor,
        )

        pred_mean = float(x_test_std @ beta_mean)
        pred_var = float(sigma2 + x_test_std @ beta_cov @ x_test_std.T)
        pred_std = float(np.sqrt(max(pred_var, noise_variance_floor)))
        prob_positive = float(1.0 - norm.cdf(0.0, loc=pred_mean, scale=pred_std))

        rows.append({
            "pred_mean": pred_mean,
            "pred_std": pred_std,
            "prob_positive": prob_positive,
            "target_return": y[i],
        })
        dates.append(dataset.index[i])

    return pd.DataFrame(rows, index=pd.Index(dates, name="date"))
