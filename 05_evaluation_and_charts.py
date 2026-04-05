from __future__ import annotations

# ================================================================
# 05_evaluation_and_charts.py (ANNOTATED VERSION)
# ================================================================
# This script turns result CSVs into presentation-ready charts and summary
# tables.
#
#
# Main workflow:
# 1. Load model_metrics.csv, warning_metrics.csv, predictions.csv,
#    portfolio_results.csv, and portfolio_kpis.csv
# 2. Standardise strategy names so charts do not break
# 3. Recompute KPI table if needed
# 4. Export charts for forecasting, warning-system quality, portfolio
#    performance, drawdowns, weights, costs, and feature importance
# 5. Export clean summary tables for report / slide appendix

from pathlib import Path

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

TARGET_VOL = 0.10
STRATEGY_ORDER = ["Dynamic Strategy", "Static 60/40", "Naive Vol Target", "100% SPY"]


# ================================================================
# BASIC HELPERS
# ================================================================
def ensure_chart_dir() -> None:
    """Create output folder for charts if it does not exist."""
    CHART_DIR.mkdir(parents=True, exist_ok=True)



def clean_strategy_name(name: str) -> str:
    """Standardise strategy labels so plotting stays consistent."""
    mapping = {
        "Dynamic Decision Engine": "Dynamic Strategy",
        "Dynamic Strategy": "Dynamic Strategy",
        "Static 60/40": "Static 60/40",
        "Naive Vol Target": "Naive Vol Target",
        "100% SPY": "100% SPY",
    }
    return mapping.get(name, name)



def annualised_return(daily_returns: pd.Series) -> float:
    """Annualised compounded return from daily returns."""
    daily_returns = daily_returns.dropna()
    if len(daily_returns) == 0:
        return np.nan
    cumulative = (1.0 + daily_returns).prod()
    years = len(daily_returns) / TRADING_DAYS_PER_YEAR
    return np.nan if years <= 0 else float(cumulative ** (1.0 / years) - 1.0)



def annualised_volatility(daily_returns: pd.Series) -> float:
    """Annualised standard deviation of daily returns."""
    daily_returns = daily_returns.dropna()
    if len(daily_returns) < 2:
        return np.nan
    return float(daily_returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))



def sharpe_ratio(daily_returns: pd.Series, rf_rate: float = 0.0) -> float:
    """Annualised Sharpe ratio."""
    ann_ret = annualised_return(daily_returns)
    ann_vol = annualised_volatility(daily_returns)
    if pd.isna(ann_ret) or pd.isna(ann_vol) or ann_vol <= 1e-12:
        return np.nan
    return float((ann_ret - rf_rate) / ann_vol)



def max_drawdown(value_series: pd.Series) -> float:
    """Worst peak-to-trough loss in portfolio value."""
    value_series = value_series.dropna()
    if len(value_series) == 0:
        return np.nan
    running_peak = value_series.cummax()
    return float(((value_series / running_peak) - 1.0).min())



def savefig(filename: str) -> None:
    """Save current matplotlib figure cleanly."""
    plt.tight_layout()
    plt.savefig(CHART_DIR / filename, dpi=220, bbox_inches="tight")
    plt.close()



def load_portfolio_results() -> pd.DataFrame:
    """
    Load portfolio_results.csv and clean the strategy column immediately.

    Why this matters:
    the portfolio results file is in long format, with multiple strategies stacked
    together. Strategy names must be standardised before filtering or plotting,
    otherwise charts can become misleading.
    """
    df = pd.read_csv(PORTFOLIO_RESULTS_FILE, parse_dates=["date"])
    if "strategy" not in df.columns:
        raise ValueError("portfolio_results.csv must contain a strategy column.")
    df["strategy"] = df["strategy"].apply(clean_strategy_name)
    df["strategy"] = pd.Categorical(df["strategy"], categories=STRATEGY_ORDER, ordered=True)
    df = df.sort_values(["strategy", "date"]).reset_index(drop=True)
    df["strategy"] = df["strategy"].astype(str)
    return df



def compute_portfolio_kpis_from_results(portfolio_results: pd.DataFrame) -> pd.DataFrame:
    """
    Rebuild KPI table directly from portfolio_results.csv.

    This acts as a safe fallback in case the saved KPI file is missing or needs to
    be refreshed after rerunning the decision engine.
    """
    rows = []
    for strategy, sub in portfolio_results.groupby("strategy"):
        sub = sub.sort_values("date")
        rows.append({
            "strategy": strategy,
            "n_days": int(sub["net_return"].notna().sum()),
            "annualised_return": annualised_return(sub["net_return"]),
            "annualised_volatility": annualised_volatility(sub["net_return"]),
            "target_vol_gap": abs(annualised_volatility(sub["net_return"]) - TARGET_VOL),
            "sharpe_ratio": sharpe_ratio(sub["net_return"]),
            "max_drawdown": max_drawdown(sub["portfolio_value"]),
            "ending_value": float(sub["portfolio_value"].iloc[-1]) if len(sub) > 0 else np.nan,
            "avg_daily_turnover": float(sub["turnover"].mean()) if "turnover" in sub.columns else np.nan,
            "total_turnover": float(sub["turnover"].sum()) if "turnover" in sub.columns else np.nan,
            "avg_daily_cost": float(sub["transaction_cost"].mean()) if "transaction_cost" in sub.columns else np.nan,
            "total_cost": float(sub["transaction_cost"].sum()) if "transaction_cost" in sub.columns else np.nan,
            "avg_spy_weight": float(sub["w_spy"].mean()) if "w_spy" in sub.columns else np.nan,
            "avg_tlt_weight": float(sub["w_tlt"].mean()) if "w_tlt" in sub.columns else np.nan,
            "avg_cash_weight": float(sub["w_cash"].mean()) if "w_cash" in sub.columns else np.nan,
        })
    out = pd.DataFrame(rows)
    out["strategy"] = pd.Categorical(out["strategy"], categories=STRATEGY_ORDER, ordered=True)
    out = out.sort_values("strategy").reset_index(drop=True)
    out["strategy"] = out["strategy"].astype(str)
    return out


# ================================================================
# MODEL CHARTS
# ================================================================
def plot_model_metric_bars(model_metrics: pd.DataFrame, metric: str, filename: str, title: str) -> None:
    """
    Compare forecasting models across SPY and TLT for one metric.

    Example use cases:
    - RMSE comparison
    - Correlation comparison
    """
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
    """
    Compare warning-system performance across models.

    This helps explain which models are better at flagging high-volatility regimes.
    """
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
    """
    Plot actual volatility against several model predictions.

    Purpose:
    show whether the models are tracking realised volatility well over time.
    """
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
    plt.title(f"{asset_prefix.upper()} Actual vs Predicted 5-Day Volatility")
    plt.xlabel("Date")
    plt.ylabel("Annualised Volatility")
    plt.legend()
    savefig(filename)



def plot_best_model_focus(predictions: pd.DataFrame, asset_prefix: str, filename: str) -> None:
    """
    Focused comparison: actual vs baseline vs calibrated RF.

    Purpose:
    simpler chart for presentation when too many forecast lines become cluttered.
    """
    actual_col = f"{asset_prefix}_actual_vol"
    baseline_col = f"{asset_prefix}_baseline_pred"
    best_col = f"{asset_prefix}_calibrated_rf_pred"
    plt.figure(figsize=(13, 6))
    plt.plot(predictions["date"], predictions[actual_col], label="Actual Vol")
    if baseline_col in predictions.columns:
        plt.plot(predictions["date"], predictions[baseline_col], label="Baseline")
    if best_col in predictions.columns:
        plt.plot(predictions["date"], predictions[best_col], label="Calibrated RF")
    plt.title(f"{asset_prefix.upper()} Focused Forecast Comparison")
    plt.xlabel("Date")
    plt.ylabel("Annualised Volatility")
    plt.legend()
    savefig(filename)


# ================================================================
# PORTFOLIO CHARTS
# ================================================================
def plot_cumulative_portfolio_value(portfolio_results: pd.DataFrame) -> None:
    """
    Plot cumulative portfolio value for all strategies.

    Purpose:
    easiest visual for showing overall portfolio growth paths.
    """
    plt.figure(figsize=(12, 6))
    for strategy, sub in portfolio_results.groupby("strategy"):
        plt.plot(sub["date"], sub["portfolio_value"], label=strategy)
    plt.title("Cumulative Portfolio Value Comparison")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    savefig("portfolio_cumulative_value.png")



def plot_drawdown_comparison(portfolio_results: pd.DataFrame) -> None:
    """
    Plot drawdown paths for all strategies.

    Also marks the minimum point of each strategy so viewers can clearly see
    where the maximum drawdown occurred.
    """
    plt.figure(figsize=(12, 6))
    for strategy, sub in portfolio_results.groupby("strategy"):
        plt.plot(sub["date"], sub["drawdown"], label=strategy)
        min_idx = sub["drawdown"].idxmin()
        plt.scatter(sub.loc[min_idx, "date"], sub.loc[min_idx, "drawdown"], s=20)
    plt.title("Portfolio Drawdown Comparison")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    savefig("portfolio_drawdown_comparison.png")



def plot_dynamic_weights(portfolio_results: pd.DataFrame) -> None:
    """
    Plot SPY/TLT/Cash weights for the Dynamic Strategy only.

    Purpose:
    helps audience understand how the decision engine actually behaves over time.
    """
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
    """
    Plot turnover and transaction cost for Dynamic Strategy.

    Purpose:
    shows implementation realism: the strategy may work on paper, but how much
    trading does it require?
    """
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
    """
    Plot one bar chart per KPI across strategies.

    Purpose:
    gives clean side-by-side management visuals for:
    - return
    - volatility
    - Sharpe
    - max drawdown
    - target-vol gap
    """
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
    """
    Plot the predicted volatility signals actually used by the decision engine.

    Purpose:
    makes the link between forecasting and portfolio action explicit.
    """
    dynamic = portfolio_results[portfolio_results["strategy"] == "Dynamic Strategy"].sort_values("date")
    if dynamic.empty or "spy_calibrated_rf_pred" not in dynamic.columns:
        return
    plt.figure(figsize=(12, 6))
    plt.plot(dynamic["date"], dynamic["spy_calibrated_rf_pred"], label="Predicted SPY Vol")
    if "tlt_calibrated_rf_pred" in dynamic.columns:
        plt.plot(dynamic["date"], dynamic["tlt_calibrated_rf_pred"], label="Predicted TLT Vol")
    plt.axhline(TARGET_VOL, linestyle="--", label="Target Vol (10%)")
    plt.axhline(0.20, linestyle=":", label="Stress Threshold")
    plt.title("Predicted Volatility Signals Used by Decision Engine")
    plt.xlabel("Date")
    plt.ylabel("Annualised Volatility")
    plt.legend()
    savefig("decision_engine_predicted_vol_signals.png")



def plot_feature_importance_if_available() -> None:
    """
    Plot top-10 feature importance per asset if the file exists.

    Purpose:
    helps explain which predictors the Random Forest relied on most.
    """
    if not FEATURE_IMPORTANCE_FILE.exists():
        return
    imp = pd.read_csv(FEATURE_IMPORTANCE_FILE)
    if imp.empty:
        return
    value_col = "importance" if "importance" in imp.columns else next((c for c in imp.columns if c not in {"asset", "feature"}), None)
    if value_col is None:
        return
    for asset in imp["asset"].unique():
        sub = imp[imp["asset"] == asset].sort_values(value_col, ascending=False).head(10)
        plt.figure(figsize=(10, 6))
        plt.barh(sub["feature"][::-1], sub[value_col][::-1])
        plt.title(f"{asset} Feature Importance (Top 10)")
        plt.xlabel(value_col.replace("_", " ").title())
        savefig(f"{asset.lower()}_feature_importance_top10.png")



def export_summary_tables(model_metrics: pd.DataFrame, warning_metrics: pd.DataFrame, kpis: pd.DataFrame) -> None:
    """
    Export pivoted tables for appendix / report use.

    Purpose:
    not all insights need a chart; some are clearer in summary-table form.
    """
    model_summary = model_metrics.pivot(index="model", columns="asset", values=["rmse", "mae", "correlation"])
    model_summary.to_csv(CHART_DIR / "table_model_metrics_summary.csv")

    warning_summary = warning_metrics.pivot(index="model", columns="asset", values=["precision", "recall", "f1"])
    warning_summary.to_csv(CHART_DIR / "table_warning_metrics_summary.csv")

    kpis.to_csv(CHART_DIR / "table_portfolio_kpis_summary.csv", index=False)



def main() -> None:
    """
    Run the full chart-export pipeline.

    Steps:
    1. make sure chart output folder exists
    2. load all result files
    3. clean strategy names and KPI table
    4. export model charts
    5. export portfolio charts
    6. export summary tables
    """
    ensure_chart_dir()

    # Load all required inputs from previous pipeline stages.
    model_metrics = pd.read_csv(MODEL_METRICS_FILE)
    warning_metrics = pd.read_csv(WARNING_METRICS_FILE)
    predictions = pd.read_csv(PREDICTIONS_FILE, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    portfolio_results = load_portfolio_results()

    # Load existing KPI file if available; otherwise rebuild from results.
    if PORTFOLIO_KPIS_FILE.exists():
        kpis = pd.read_csv(PORTFOLIO_KPIS_FILE)
        if "strategy" in kpis.columns:
            kpis["strategy"] = kpis["strategy"].apply(clean_strategy_name)
            kpis["strategy"] = pd.Categorical(kpis["strategy"], categories=STRATEGY_ORDER, ordered=True)
            kpis = kpis.sort_values("strategy").reset_index(drop=True)
            kpis["strategy"] = kpis["strategy"].astype(str)
        else:
            kpis = compute_portfolio_kpis_from_results(portfolio_results)
    else:
        kpis = compute_portfolio_kpis_from_results(portfolio_results)

    # Save cleaned KPI file back out, so future use is consistent.
    kpis.to_csv(PORTFOLIO_KPIS_FILE, index=False)

    # =====================
    # Model-level charts
    # =====================
    plot_model_metric_bars(model_metrics, "rmse", "model_rmse_comparison.png", "Model RMSE Comparison")
    plot_model_metric_bars(model_metrics, "correlation", "model_correlation_comparison.png", "Model Correlation Comparison")
    plot_warning_metric_bars(warning_metrics, "precision", "warning_precision_comparison.png", "Warning Precision Comparison")
    plot_warning_metric_bars(warning_metrics, "recall", "warning_recall_comparison.png", "Warning Recall Comparison")
    plot_warning_metric_bars(warning_metrics, "f1", "warning_f1_comparison.png", "Warning F1 Comparison")

    # Forecast path charts.
    plot_actual_vs_pred(predictions, "spy", "spy_actual_vs_predicted_vol.png")
    plot_actual_vs_pred(predictions, "tlt", "tlt_actual_vs_predicted_vol.png")
    plot_best_model_focus(predictions, "spy", "spy_best_model_focus.png")
    plot_best_model_focus(predictions, "tlt", "tlt_best_model_focus.png")

    # =====================
    # Strategy / portfolio charts
    # =====================
    plot_cumulative_portfolio_value(portfolio_results)
    plot_drawdown_comparison(portfolio_results)
    plot_dynamic_weights(portfolio_results)
    plot_dynamic_turnover_and_cost(portfolio_results)
    plot_strategy_kpi_bars(kpis)
    plot_predicted_vol_and_risk_signals(portfolio_results)

    # Optional technical appendix chart.
    plot_feature_importance_if_available()

    # Export clean summary tables.
    export_summary_tables(model_metrics, warning_metrics, kpis)

    print("=" * 60)
    print("EVALUATION AND CHART EXPORT COMPLETE")
    print("=" * 60)
    print(f"Charts saved to: {CHART_DIR}")
    print("\nPortfolio KPIs:")
    print(kpis.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
