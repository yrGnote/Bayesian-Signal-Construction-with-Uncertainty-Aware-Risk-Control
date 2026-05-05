from pathlib import Path
import yaml

from src.data import load_price_data
from src.features import build_features
from src.bayesian_model import rolling_bayesian_predictions
from src.signals import build_probability_signal
from src.risk import monte_carlo_tail_risk, apply_risk_scaling
from src.backtest import run_backtest
from src.metrics import summarize_performance
from src.plots import plot_equity_curve, plot_drawdown


def main() -> None:
    root = Path(__file__).resolve().parent
    config_path = root / "config.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    results_dir = root / "results"
    results_dir.mkdir(exist_ok=True)

    prices = load_price_data(
        ticker=config["ticker"],
        start_date=config["start_date"],
        end_date=config["end_date"],
    )

    dataset = build_features(
        prices,
        return_lags=config["features"]["return_lags"],
        volatility_window=config["features"]["volatility_window"],
        momentum_windows=config["features"]["momentum_windows"],
    )

    predictions = rolling_bayesian_predictions(
        dataset,
        train_window=config["model"]["train_window"],
        ridge_alpha=config["model"]["ridge_alpha"],
        noise_variance_floor=float(config["model"]["noise_variance_floor"]),
    )

    signals = build_probability_signal(
        predictions,
        long_threshold=config["signal"]["long_threshold"],
        short_threshold=config["signal"]["short_threshold"],
        max_position=config["signal"]["max_position"],
    )

    tail_risk = monte_carlo_tail_risk(
        predictions,
        n_simulations=config["risk"]["n_simulations"],
        var_level=config["risk"]["var_level"],
        random_state=42,
    )

    positions = apply_risk_scaling(
        signals,
        tail_risk,
        target_daily_var=config["risk"]["target_daily_var"],
        min_scale=config["risk"]["min_position_scale"],
        max_scale=config["risk"]["max_position_scale"],
    )

    backtest = run_backtest(
        positions,
        predictions[["target_return"]],
        transaction_cost_bps=config["backtest"]["transaction_cost_bps"],
    )

    summary = summarize_performance(
        backtest["strategy_return"],
        positions["position"],
        annualization=config["backtest"]["annualization"],
    )

    print("\nPerformance Summary")
    print("-------------------")
    for k, v in summary.items():
        print(f"{k}: {v:.4f}")

    predictions.to_csv(results_dir / "predictions.csv")
    positions.to_csv(results_dir / "positions.csv")
    backtest.to_csv(results_dir / "backtest.csv")

    plot_equity_curve(backtest, results_dir / "equity_curve.png")
    plot_drawdown(backtest, results_dir / "drawdown.png")

    with open(results_dir / "summary.txt", "w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v:.6f}\n")


if __name__ == "__main__":
    main()
