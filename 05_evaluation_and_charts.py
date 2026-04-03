"""
05_evaluation_and_charts.py

Updated evaluation + charting script for the current BC2407 project pipeline.

What this script does:
1. Reads model_metrics.csv, warning_metrics.csv, predictions.csv,
   portfolio_results.csv, and portfolio_kpis.csv.
2. Regenerates portfolio KPI tables if needed.
3. Exports presentation-ready charts for:
   - model forecasting performance
   - warning-system performance
   - actual vs predicted volatility
   - cumulative portfolio value
   - drawdown comparison
   - dynamic allocation weights
   - turnover / transaction cost behaviour
   - portfolio KPI comparison
4. Optionally exports feature importance charts if the file exists.

The script is designed to work with the current project outputs:
- predictions.csv from 03_modeling.py
- portfolio_results.csv and portfolio_kpis.csv from 04_decision_engine.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    CHART_DIR,
    MODEL_METRICS_FILE,
    WARNING_METRICS_FILE,
    PREDICTIONS_FILE,
    PORTFOLIO_RESULTS_FILE,
    PORTFOLIO_KPIS_FILE,
    FEATURE_IMPORTANCE_FILE,
    TRADING_DAYS_PER_YEAR,
)


# ================================================================
# GENERAL HELPERS
# ================================================================

def ensure_chart_dir() -> None:
    CHART_DIR.mkdir(parents=True, exist_ok=True)


def annualised_return(daily_returns: pd.Series) -> float:
    daily_returns = daily_returns.dropna()
    if len(daily_returns) == 0:
        return np.nan

    cumulative = (1.0 + daily_returns).prod()
    years = len(daily_returns) / TRADING_DAYS_PER_YEAR
    if years <= 0:
        return np.nan
    return float(cumulative ** (1.0 / years) - 1.0)


def annualised_volatility(daily_returns: pd.Series) -> float:
    daily_returns = daily_returns.dropna()
    if len(daily_returns) < 2:
        return np.nan
    return float(daily_returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def sharpe_ratio(daily_returns: pd.Series, rf_rate: float = 0.0) -> float:
    ann_ret = annualised_return(daily_returns)
    ann_vol = annualised_volatility(daily_returns)
    if pd.isna(ann_ret) or pd.isna(ann_vol) or ann_vol <= 1e-12:
        return np.nan
    return float((ann_ret - rf_rate) / ann_vol)


def max_drawdown(value_series: pd.Series) -> float:
    value_series = value_series.dropna()
    if len(value_series) == 0:
        return np.nan
    running_peak = value_series.cummax()
    drawdown = (value_series / running_peak) - 1.0
    return float(drawdown.min())


def savefig(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(CHART_DIR / filename, dpi=220, bbox_inches="tight")
    plt.close()


def clean_strategy_name(name: str) -> str:
    replacements = {
        "Dynamic Decision Engine": "Dynamic Strategy",
        "Dynamic Strategy": "Dynamic Strategy",
        "Static 60/40": "Static 60/40",
        "Naive Vol Target": "Naive Vol Target",
        "100% SPY": "100% SPY",
    }
    return replacements.get(name, name)


# ================================================================
# KPI RECOMPUTATION (SAFE FALLBACK)
# ================================================================

def compute_portfolio_kpis_from_results(portfolio_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy, sub in portfolio_results.groupby("strategy"):
        sub = sub.sort_values("date").copy()
        strategy_name = clean_strategy_name(strategy)

        row = {
            "strategy": strategy_name,
            "n_days": int(sub["net_return"].notna().sum()),
            "annualised_return": annualised_return(sub["net_return"]),
            "annualised_volatility": annualised_volatility(sub["net_return"]),
            "target_vol_gap": abs(annualised_volatility(sub["net_return"]) - 0.10)
            if sub["net_return"].notna().sum() > 1
            else np.nan,
            "sharpe_ratio": sharpe_ratio(sub["net_return"]),
            "max_drawdown": max_drawdown(sub["portfolio_value"]),
            "ending_value": float(sub["portfolio_value"].iloc[-1]) if len(sub) > 0 else np.nan,
        }

        if "turnover" in sub.columns:
            row["avg_daily_turnover"] = float(sub["turnover"].mean())
            row["total_turnover"] = float(sub["turnover"].sum())
        else:
            row["avg_daily_turnover"] = np.nan
            row["total_turnover"] = np.nan

        if "transaction_cost" in sub.columns:
            row["avg_daily_cost"] = float(sub["transaction_cost"].mean())
            row["total_cost"] = float(sub["transaction_cost"].sum())
        else:
            row["avg_daily_cost"] = np.nan
            row["total_cost"] = np.nan

        for col, out_name in [("w_spy", "avg_spy_weight"), ("w_tlt", "avg_tlt_weight"), ("w_cash", "avg_cash_weight")]:
            row[out_name] = float(sub[col].mean()) if col in sub.columns else np.nan

        rows.append(row)

    order = ["Dynamic Strategy", "Static 60/40", "Naive Vol Target", "100% SPY"]
    out = pd.DataFrame(rows)
    if not out.empty:
        out["strategy"] = pd.Categorical(out["strategy"], categories=order, ordered=True)
        out = out.sort_values("strategy").reset_index(drop=True)
        out["strategy"] = out["strategy"].astype(str)
    return out


# ================================================================
# MODEL PERFORMANCE CHARTS
# ================================================================

def plot_model_metric_bars(model_metrics: pd.DataFrame, metric: str, filename: str, title: str) -> None:
    assets = list(model_metrics["asset"].unique())
    models = list(model_metrics["model"].unique())
    x = np.arange(len(models))
    width = 0.35 if len(assets) == 2 else 0.8 / max(len(assets), 1)

    plt.figure(figsize=(12, 6))
    for i, asset in enumerate(assets):
        sub = model_metrics[model_metrics["asset"] == asset].set_index("model")
        y = [sub.loc[m, metric] if m in sub.index else np.nan for m in models]
        plt.bar(x + (i - (len(assets) - 1) / 2) * width, y, width=width, label=asset)

    plt.xticks(x, models, rotation=30)
    plt.title(title)
    plt.xlabel("Model")
    plt.ylabel(metric.upper())
    plt.legend()
    savefig(filename)


def plot_warning_metric_bars(warning_metrics: pd.DataFrame, metric: str, filename: str, title: str) -> None:
    assets = list(warning_metrics["asset"].unique())
    models = list(warning_metrics["model"].unique())
    x = np.arange(len(models))
    width = 0.35 if len(assets) == 2 else 0.8 / max(len(assets), 1)

    plt.figure(figsize=(12, 6))
    for i, asset in enumerate(assets):
        sub = warning_metrics[warning_metrics["asset"] == asset].set_index("model")
        y = [sub.loc[m, metric] if m in sub.index else np.nan for m in models]
        plt.bar(x + (i - (len(assets) - 1) / 2) * width, y, width=width, label=asset)

    plt.xticks(x, models, rotation=30)
    plt.title(title)
    plt.xlabel("Model")
    plt.ylabel(metric.upper())
    plt.legend()
    savefig(filename)


def plot_actual_vs_pred(predictions: pd.DataFrame, asset_prefix: str, filename: str) -> None:
    asset_name = asset_prefix.upper()
    actual_col = f"{asset_prefix}_actual_vol"

    candidate_cols = [
        f"{asset_prefix}_baseline_pred",
        f"{asset_prefix}_garch_pred",
        f"{asset_prefix}_rf_pred",
        f"{asset_prefix}_calibrated_rf_pred",
    ]
    pred_cols = [c for c in candidate_cols if c in predictions.columns]

    plt.figure(figsize=(13, 6))
    plt.plot(predictions["date"], predictions[actual_col], label="Actual Vol")
    for col in pred_cols:
        label = col.replace(f"{asset_prefix}_", "").replace("_pred", "").replace("_", " ").title()
        plt.plot(predictions["date"], predictions[col], label=label)

    plt.title(f"{asset_name} Actual vs Predicted 5-Day Volatility")
    plt.xlabel("Date")
    plt.ylabel("Annualised Volatility")
    plt.legend()
    savefig(filename)


def plot_best_model_focus(predictions: pd.DataFrame, asset_prefix: str, filename: str) -> None:
    asset_name = asset_prefix.upper()
    actual_col = f"{asset_prefix}_actual_vol"
    best_col = f"{asset_prefix}_calibrated_rf_pred"
    baseline_col = f"{asset_prefix}_baseline_pred"

    plt.figure(figsize=(13, 6))
    plt.plot(predictions["date"], predictions[actual_col], label="Actual Vol")
    if baseline_col in predictions.columns:
        plt.plot(predictions["date"], predictions[baseline_col], label="Baseline")
    if best_col in predictions.columns:
        plt.plot(predictions["date"], predictions[best_col], label="Calibrated RF")
    plt.title(f"{asset_name} Focused Forecast Comparison: Actual vs Baseline vs Calibrated RF")
    plt.xlabel("Date")
    plt.ylabel("Annualised Volatility")
    plt.legend()
    savefig(filename)


# ================================================================
# PORTFOLIO CHARTS
# ================================================================

def plot_cumulative_portfolio_value(portfolio_results: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 6))
    for strategy, sub in portfolio_results.groupby("strategy"):
        sub = sub.sort_values("date")
        plt.plot(sub["date"], sub["portfolio_value"], label=clean_strategy_name(strategy))

    plt.title("Cumulative Portfolio Value Comparison")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    savefig("portfolio_cumulative_value.png")


def plot_drawdown_comparison(portfolio_results: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 6))
    for strategy, sub in portfolio_results.groupby("strategy"):
        sub = sub.sort_values("date")
        plt.plot(sub["date"], sub["drawdown"], label=clean_strategy_name(strategy))

    plt.title("Portfolio Drawdown Comparison")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    savefig("portfolio_drawdown_comparison.png")


def plot_dynamic_weights(portfolio_results: pd.DataFrame) -> None:
    dynamic = portfolio_results[portfolio_results["strategy"] == "Dynamic Strategy"].sort_values("date")
    if dynamic.empty:
        return

    plt.figure(figsize=(12, 6))
    plt.plot(dynamic["date"], dynamic["w_spy"], label="SPY")
    plt.plot(dynamic["date"], dynamic["w_tlt"], label="TLT")
    plt.plot(dynamic["date"], dynamic["w_cash"], label="Cash")
    plt.title("Dynamic Strategy Asset Allocation Over Time")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Weight")
    plt.legend()
    savefig("dynamic_strategy_weights.png")


def plot_dynamic_turnover_and_cost(portfolio_results: pd.DataFrame) -> None:
    dynamic = portfolio_results[portfolio_results["strategy"] == "Dynamic Strategy"].sort_values("date")
    if dynamic.empty:
        return

    plt.figure(figsize=(12, 5))
    plt.plot(dynamic["date"], dynamic["turnover"], label="Turnover")
    plt.title("Dynamic Strategy Daily Turnover")
    plt.xlabel("Date")
    plt.ylabel("Turnover")
    plt.legend()
    savefig("dynamic_strategy_turnover.png")

    plt.figure(figsize=(12, 5))
    plt.plot(dynamic["date"], dynamic["transaction_cost"], label="Transaction Cost")
    plt.title("Dynamic Strategy Daily Transaction Cost")
    plt.xlabel("Date")
    plt.ylabel("Cost")
    plt.legend()
    savefig("dynamic_strategy_transaction_cost.png")


def plot_strategy_kpi_bars(kpis: pd.DataFrame) -> None:
    kpis = kpis.copy()
    kpis["strategy"] = kpis["strategy"].apply(clean_strategy_name)

    metrics = [
        ("annualised_return", "Annualised Return"),
        ("annualised_volatility", "Annualised Volatility"),
        ("sharpe_ratio", "Sharpe Ratio"),
        ("max_drawdown", "Max Drawdown"),
        ("target_vol_gap", "Target Volatility Gap"),
    ]

    for metric, title in metrics:
        if metric not in kpis.columns:
            continue
        plt.figure(figsize=(10, 5))
        plt.bar(kpis["strategy"], kpis[metric])
        plt.title(title)
        plt.xlabel("Strategy")
        plt.ylabel(title)
        plt.xticks(rotation=20)
        savefig(f"kpi_{metric}.png")


def plot_predicted_vol_and_risk_signals(portfolio_results: pd.DataFrame) -> None:
    dynamic = portfolio_results[portfolio_results["strategy"] == "Dynamic Strategy"].sort_values("date")
    if dynamic.empty:
        return

    plt.figure(figsize=(12, 6))
    plt.plot(dynamic["date"], dynamic["spy_calibrated_rf_pred"], label="Predicted SPY Vol")
    plt.plot(dynamic["date"], dynamic["tlt_calibrated_rf_pred"], label="Predicted TLT Vol")
    plt.axhline(0.10, linestyle="--", label="Target Vol (10%)")
    plt.axhline(0.20, linestyle=":", label="Stress Threshold")
    plt.title("Predicted Volatility Signals Used by Decision Engine")
    plt.xlabel("Date")
    plt.ylabel("Annualised Volatility")
    plt.legend()
    savefig("decision_engine_predicted_vol_signals.png")


def plot_feature_importance_if_available() -> None:
    if not FEATURE_IMPORTANCE_FILE.exists():
        return

    imp = pd.read_csv(FEATURE_IMPORTANCE_FILE)
    if imp.empty:
        return

    value_col = "importance"
    if value_col not in imp.columns:
        alt_cols = [c for c in imp.columns if c not in {"asset", "feature"}]
        if not alt_cols:
            return
        value_col = alt_cols[0]

    for asset in imp["asset"].unique():
        sub = imp[imp["asset"] == asset].sort_values(value_col, ascending=False).head(10)
        plt.figure(figsize=(10, 6))
        plt.barh(sub["feature"][::-1], sub[value_col][::-1])
        plt.title(f"{asset} Feature Importance (Top 10)")
        plt.xlabel(value_col.replace("_", " ").title())
        savefig(f"{asset.lower()}_feature_importance_top10.png")


# ================================================================
# TABLE EXPORTS
# ================================================================

def export_summary_tables(model_metrics: pd.DataFrame, warning_metrics: pd.DataFrame, kpis: pd.DataFrame) -> None:
    model_summary = model_metrics.pivot(index="model", columns="asset", values=["rmse", "mae", "correlation"])
    model_summary.to_csv(CHART_DIR / "table_model_metrics_summary.csv")

    warning_summary = warning_metrics.pivot(index="model", columns="asset", values=["precision", "recall", "f1"])
    warning_summary.to_csv(CHART_DIR / "table_warning_metrics_summary.csv")

    kpis.to_csv(CHART_DIR / "table_portfolio_kpis_summary.csv", index=False)


# ================================================================
# MAIN
# ================================================================

def main() -> None:
    ensure_chart_dir()

    model_metrics = pd.read_csv(MODEL_METRICS_FILE)
    warning_metrics = pd.read_csv(WARNING_METRICS_FILE)
    predictions = pd.read_csv(PREDICTIONS_FILE, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    portfolio_results = pd.read_csv(PORTFOLIO_RESULTS_FILE, parse_dates=["date"]).sort_values(["strategy", "date"]).reset_index(drop=True)

    portfolio_results["strategy"] = portfolio_results["strategy"].apply(clean_strategy_name)

    # Recompute KPIs from portfolio results to ensure consistency with the latest outputs.
    kpis = compute_portfolio_kpis_from_results(portfolio_results)
    kpis.to_csv(PORTFOLIO_KPIS_FILE, index=False)

    # Model charts
    plot_model_metric_bars(model_metrics, "rmse", "model_rmse_comparison.png", "Volatility Forecast RMSE by Model")
    plot_model_metric_bars(model_metrics, "correlation", "model_correlation_comparison.png", "Volatility Forecast Correlation by Model")

    # Warning charts
    plot_warning_metric_bars(warning_metrics, "precision", "warning_precision_comparison.png", "Warning Precision by Model")
    plot_warning_metric_bars(warning_metrics, "recall", "warning_recall_comparison.png", "Warning Recall by Model")
    plot_warning_metric_bars(warning_metrics, "f1", "warning_f1_comparison.png", "Warning F1 by Model")

    # Forecast vs actual charts
    plot_actual_vs_pred(predictions, "spy", "spy_actual_vs_predicted_vol.png")
    plot_actual_vs_pred(predictions, "tlt", "tlt_actual_vs_predicted_vol.png")
    plot_best_model_focus(predictions, "spy", "spy_focus_actual_vs_calibrated_rf.png")
    plot_best_model_focus(predictions, "tlt", "tlt_focus_actual_vs_calibrated_rf.png")

    # Portfolio charts
    plot_cumulative_portfolio_value(portfolio_results)
    plot_drawdown_comparison(portfolio_results)
    plot_dynamic_weights(portfolio_results)
    plot_dynamic_turnover_and_cost(portfolio_results)
    plot_strategy_kpi_bars(kpis)
    plot_predicted_vol_and_risk_signals(portfolio_results)

    # Optional feature importance charts
    plot_feature_importance_if_available()

    # Tables
    export_summary_tables(model_metrics, warning_metrics, kpis)

    print("=" * 70)
    print("UPDATED EVALUATION + CHART EXPORT COMPLETE")
    print("=" * 70)
    print(f"Charts saved to: {CHART_DIR}")
    print(f"Portfolio KPI table saved to: {PORTFOLIO_KPIS_FILE}")
    print("\nPortfolio KPIs:")
    print(kpis.round(4).to_string(index=False))
    print("\nModel metrics:")
    print(model_metrics.round(4).to_string(index=False))
    print("\nWarning metrics:")
    print(warning_metrics.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
