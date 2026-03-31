
"""
05_evaluation_and_charts.py
Evaluate model outputs and portfolio outcomes, then export tables and charts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    CHART_DIR,
    MODEL_METRICS_FILE,
    WARNING_METRICS_FILE,
    PORTFOLIO_RESULTS_FILE,
    PORTFOLIO_KPIS_FILE,
    FEATURE_IMPORTANCE_FILE,
    TRADING_DAYS_PER_YEAR,
)


def ann_return(ret: pd.Series) -> float:
    return float(ret.mean() * TRADING_DAYS_PER_YEAR)


def ann_vol(ret: pd.Series) -> float:
    return float(ret.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR))


def sharpe_ratio(ret: pd.Series) -> float:
    vol = ann_vol(ret)
    return np.nan if vol == 0 else ann_return(ret) / vol


def max_drawdown(ret: pd.Series) -> float:
    wealth = np.exp(ret.cumsum())
    running_max = wealth.cummax()
    drawdown = (wealth - running_max) / running_max
    return float(drawdown.min())


def compute_portfolio_kpis(df: pd.DataFrame) -> pd.DataFrame:
    strategies = [
        "static",
        "baseline",
        "ets",
        "arima",
        "garch",
        "rf",
    ]
    rows = []
    for name in strategies:
        ret = df[f"{name}_net_return"]
        turnover = df[f"{name}_turnover"] if f"{name}_turnover" in df.columns else pd.Series(np.zeros(len(df)))
        rows.append({
            "strategy": name,
            "annual_return": ann_return(ret),
            "annual_volatility": ann_vol(ret),
            "sharpe": sharpe_ratio(ret),
            "max_drawdown": max_drawdown(ret),
            "avg_daily_turnover": float(turnover.mean()),
            "total_turnover": float(turnover.sum()),
        })
    return pd.DataFrame(rows)


def plot_actual_vs_pred(df: pd.DataFrame, asset_prefix: str, chart_dir: Path) -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(df["date"], df[f"{asset_prefix}_actual_vol"], label="Actual")
    plt.plot(df["date"], df[f"{asset_prefix}_baseline_pred"], label="Baseline")
    plt.plot(df["date"], df[f"{asset_prefix}_ets_pred"], label="ETS")
    plt.plot(df["date"], df[f"{asset_prefix}_arima_pred"], label="ARIMA")
    plt.plot(df["date"], df[f"{asset_prefix}_garch_pred"], label="GARCH")
    plt.plot(df["date"], df[f"{asset_prefix}_rf_pred"], label="RF")
    plt.title(f"{asset_prefix.upper()} Actual vs Predicted 5-Day Volatility")
    plt.xlabel("Date")
    plt.ylabel("Annualised Volatility")
    plt.legend()
    plt.tight_layout()
    plt.savefig(chart_dir / f"{asset_prefix}_actual_vs_predicted_vol.png", dpi=180)
    plt.close()


def plot_cumulative_returns(df: pd.DataFrame, chart_dir: Path) -> None:
    plt.figure(figsize=(12, 6))
    for name in ["static", "baseline", "ets", "arima", "garch", "rf"]:
        wealth = np.exp(df[f"{name}_net_return"].cumsum())
        plt.plot(df["date"], wealth, label=name.upper())
    plt.title("Static vs Dynamic Portfolio Backtest")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Wealth")
    plt.legend()
    plt.tight_layout()
    plt.savefig(chart_dir / "cumulative_returns.png", dpi=180)
    plt.close()


def plot_drawdowns(df: pd.DataFrame, chart_dir: Path) -> None:
    plt.figure(figsize=(12, 6))
    for name in ["static", "rf", "garch", "baseline"]:
        wealth = np.exp(df[f"{name}_net_return"].cumsum())
        running_max = wealth.cummax()
        drawdown = (wealth - running_max) / running_max
        plt.plot(df["date"], drawdown, label=name.upper())
    plt.title("Drawdown Comparison")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(chart_dir / "drawdown_comparison.png", dpi=180)
    plt.close()


def plot_weights(df: pd.DataFrame, chart_dir: Path, model_name: str = "rf") -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(df["date"], df[f"{model_name}_spy_weight"], label="SPY Weight")
    plt.plot(df["date"], df[f"{model_name}_tlt_weight"], label="TLT Weight")
    plt.title(f"{model_name.upper()} Dynamic Allocation Weights")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Weight")
    plt.legend()
    plt.tight_layout()
    plt.savefig(chart_dir / f"{model_name}_allocation_weights.png", dpi=180)
    plt.close()


def plot_feature_importance(chart_dir: Path) -> None:
    imp = pd.read_csv(FEATURE_IMPORTANCE_FILE)
    for asset in imp["asset"].unique():
        sub = imp[imp["asset"] == asset].sort_values("importance", ascending=False).head(10)
        plt.figure(figsize=(10, 5))
        plt.barh(sub["feature"][::-1], sub["importance"][::-1])
        plt.title(f"{asset} Random Forest Feature Importance (Top 10)")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(chart_dir / f"{asset.lower()}_rf_feature_importance.png", dpi=180)
        plt.close()


def main() -> None:
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    portfolio_df = pd.read_csv(PORTFOLIO_RESULTS_FILE, parse_dates=["date"])
    kpis = compute_portfolio_kpis(portfolio_df)
    kpis.to_csv(PORTFOLIO_KPIS_FILE, index=False)

    # Charts
    plot_actual_vs_pred(portfolio_df, "spy", CHART_DIR)
    plot_actual_vs_pred(portfolio_df, "tlt", CHART_DIR)
    plot_cumulative_returns(portfolio_df, CHART_DIR)
    plot_drawdowns(portfolio_df, CHART_DIR)
    plot_weights(portfolio_df, CHART_DIR, model_name="rf")
    plot_feature_importance(CHART_DIR)

    print("=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Saved KPI table: {PORTFOLIO_KPIS_FILE}")
    print("\nPortfolio KPIs:")
    print(kpis.round(4).to_string(index=False))

    print("\nModel metrics:")
    print(pd.read_csv(MODEL_METRICS_FILE).round(4).to_string(index=False))

    print("\nWarning metrics:")
    print(pd.read_csv(WARNING_METRICS_FILE).round(4).to_string(index=False))


if __name__ == "__main__":
    main()