"""
04_decision_engine.py

Decision engine for:
- SPY (growth / risky asset)
- TLT (defensive / hedge asset)
- Cash (capital preservation)

This script:
1. loads predictions.csv from the modelling stage
2. splits the prediction period into:
   - validation segment (for parameter tuning)
   - holdout segment (for final honest evaluation)
3. runs a structured grid search over a SMALL set of engine parameters
4. scores parameter sets using business-aligned objectives:
   - realised volatility close to target
   - lower max drawdown
   - higher Sharpe ratio
   - reasonable annualised return
   - lower turnover / cost
5. selects the best parameter set on validation
6. evaluates that parameter set on the later unseen holdout segment
7. compares the dynamic strategy against benchmarks
8. saves:
   - validation search results
   - daily portfolio results on holdout
   - KPI summary on holdout
   - best parameters

This is a portfolio construction / decision-engine script,
NOT a predictive modelling script.
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import (
    PREDICTIONS_FILE,
    OUTPUT_DIR,
    PORTFOLIO_RESULTS_FILE,
    PORTFOLIO_KPIS_FILE,
    STATIC_SPY_WEIGHT,
    STATIC_TLT_WEIGHT,
    TRANSACTION_COST_BPS,
    TRADING_DAYS_PER_YEAR,
)

# ================================================================
# FIXED ENGINE SETTINGS (STRUCTURE, NOT TUNED)
# ================================================================

# Business objective: keep realised portfolio risk near ~10% annualised
TARGET_VOL = 0.10

# Asset weight bounds
MAX_SPY_WEIGHT = 0.90
MIN_SPY_WEIGHT = 0.00

MAX_TLT_WEIGHT = 0.90
MIN_TLT_WEIGHT = 0.00

MIN_CASH_WEIGHT = 0.00

# IMPORTANT:
# We now FIX these instead of tuning them, because over a short sample
# they may not trigger often enough to tune reliably.
STRESS_VOL_THRESHOLD = 0.20
STRESS_MAX_SPY_WEIGHT = 0.50

DRAWDOWN_TRIGGER = 0.10
DRAWDOWN_SPY_MULTIPLIER = 0.85

# Force at least some of the defensive bucket into TLT before hiding fully in cash
# This avoids the engine becoming too cash-heavy.
MIN_TLT_SHARE_OF_DEFENSIVE = 0.20

# Whether to include 100% SPY benchmark
INCLUDE_SPY_ONLY_BENCHMARK = True

# Convert transaction cost from bps to decimal
TRANSACTION_COST_RATE = TRANSACTION_COST_BPS / 10000.0

# Validation / holdout split INSIDE predictions.csv
# First 60% = validation for tuning
# Last 40%  = holdout for final honest evaluation
ENGINE_VALIDATION_RATIO = 0.60

# Output files for tuning
SEARCH_RESULTS_FILE = OUTPUT_DIR / "decision_engine_search_results.csv"
BEST_ENGINE_PARAMS_FILE = OUTPUT_DIR / "decision_engine_best_params.json"


# ================================================================
# PARAMETER GRID (TUNE ONLY A FEW BEHAVIOURAL PARAMETERS)
# ================================================================
# These are engine-behaviour settings, not model hyperparameters.
# Keep the grid small and interpretable.

PARAM_GRID = {
    "SPY_UNCERTAINTY_PENALTY": [0.10, 0.20, 0.30],
    "TLT_UNCERTAINTY_PENALTY": [0.05, 0.15, 0.25],
    "REBALANCE_THRESHOLD": [0.02, 0.05, 0.08],
    "SMOOTHING_LAMBDA": [0.20, 0.30, 0.50],
}


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def clip(value: float, low: float, high: float) -> float:
    """Clip a scalar into [low, high]."""
    return float(np.clip(value, low, high))


def normalise_weights(w_spy: float, w_tlt: float, w_cash: float) -> tuple[float, float, float]:
    """
    Ensure weights are non-negative and sum to 1.
    This keeps the engine numerically safe after caps / floors / smoothing.
    """
    weights = np.array(
        [max(w_spy, 0.0), max(w_tlt, 0.0), max(w_cash, 0.0)],
        dtype=float
    )
    total = weights.sum()

    if total <= 0:
        return 0.0, 0.0, 1.0

    weights = weights / total
    return float(weights[0]), float(weights[1]), float(weights[2])


def annualised_return(daily_returns: pd.Series) -> float:
    """Annualised compounded return."""
    if len(daily_returns) == 0:
        return np.nan

    cumulative = (1.0 + daily_returns).prod()
    years = len(daily_returns) / TRADING_DAYS_PER_YEAR

    if years <= 0:
        return np.nan

    return float(cumulative ** (1.0 / years) - 1.0)


def annualised_volatility(daily_returns: pd.Series) -> float:
    """Annualised standard deviation of daily returns."""
    if len(daily_returns) < 2:
        return np.nan
    return float(daily_returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def sharpe_ratio(daily_returns: pd.Series, rf_rate: float = 0.0) -> float:
    """Annualised Sharpe ratio with default risk-free rate = 0."""
    ann_ret = annualised_return(daily_returns)
    ann_vol = annualised_volatility(daily_returns)

    if pd.isna(ann_ret) or pd.isna(ann_vol) or ann_vol <= 1e-12:
        return np.nan

    return float((ann_ret - rf_rate) / ann_vol)


def max_drawdown(value_series: pd.Series) -> float:
    """Maximum peak-to-trough drawdown of a cumulative value series."""
    if len(value_series) == 0:
        return np.nan

    running_peak = value_series.cummax()
    dd = (value_series / running_peak) - 1.0
    return float(dd.min())


def compute_running_uncertainty_scale(history: list[float], current_value: float) -> float:
    """
    Convert raw RF uncertainty into a rough 0-to-1 style scale using only
    information available up to the current day.

    Why:
    - RF uncertainty = std deviation across tree predictions
    - larger value means trees disagree more
    - the engine should become more conservative when uncertainty is high
    """
    hist = history + [current_value]

    if len(hist) < 5:
        # Early days: not enough history yet
        return 0.5

    p90 = np.percentile(hist, 90)
    if p90 <= 1e-8:
        return 0.0

    scaled = current_value / p90
    return clip(scaled, 0.0, 1.0)


def split_validation_holdout(df: pd.DataFrame, ratio: float = ENGINE_VALIDATION_RATIO) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronologically split predictions into:
    - validation period for engine tuning
    - later holdout period for honest final evaluation
    """
    split_idx = int(len(df) * ratio)

    if split_idx <= 0 or split_idx >= len(df):
        raise ValueError("Invalid validation / holdout split. Check ENGINE_VALIDATION_RATIO.")

    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def make_param_combinations(grid: dict) -> list[dict]:
    """Generate all parameter combinations from a small grid."""
    keys = list(grid.keys())
    values = list(grid.values())

    combos = []
    for vals in product(*values):
        combos.append(dict(zip(keys, vals)))

    return combos


# ================================================================
# KPI + SCORING FUNCTIONS
# ================================================================

def compute_strategy_kpis(
    df: pd.DataFrame,
    strategy_name: str,
    return_col: str,
    value_col: str,
    turnover_col: str | None = None,
    cost_col: str | None = None,
    spy_weight_col: str | None = None,
    tlt_weight_col: str | None = None,
    cash_weight_col: str | None = None,
) -> dict:
    """
    Compute KPI summary for one strategy.
    """
    returns = df[return_col].dropna()
    values = df[value_col].dropna()

    out = {
        "strategy": strategy_name,
        "n_days": int(len(returns)),
        "annualised_return": annualised_return(returns),
        "annualised_volatility": annualised_volatility(returns),
        "target_vol_gap": abs(annualised_volatility(returns) - TARGET_VOL) if len(returns) > 1 else np.nan,
        "sharpe_ratio": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(values),
        "ending_value": float(values.iloc[-1]) if len(values) > 0 else np.nan,
    }

    if turnover_col is not None:
        out["avg_daily_turnover"] = float(df[turnover_col].mean())
        out["total_turnover"] = float(df[turnover_col].sum())

    if cost_col is not None:
        out["avg_daily_cost"] = float(df[cost_col].mean())
        out["total_cost"] = float(df[cost_col].sum())

    if spy_weight_col is not None:
        out["avg_spy_weight"] = float(df[spy_weight_col].mean())

    if tlt_weight_col is not None:
        out["avg_tlt_weight"] = float(df[tlt_weight_col].mean())

    if cash_weight_col is not None:
        out["avg_cash_weight"] = float(df[cash_weight_col].mean())

    return out


def strategy_score(kpi_row: pd.Series) -> float:
    """
    Composite score for parameter selection.

    Higher is better.

    Revised from the earlier version:
    - still rewards low drawdown and target-vol adherence
    - but now rewards Sharpe and return more strongly
    - this reduces the chance that an overly defensive, cash-heavy
      strategy gets selected just because it is "safe"
    """
    vol_gap = abs(kpi_row["annualised_volatility"] - TARGET_VOL)
    mdd = abs(kpi_row["max_drawdown"])
    sharpe = kpi_row["sharpe_ratio"]
    ann_ret = kpi_row["annualised_return"]
    turnover = kpi_row.get("avg_daily_turnover", 0.0)

    # Safe handling of NaN
    if pd.isna(sharpe):
        sharpe = -999.0
    if pd.isna(ann_ret):
        ann_ret = -999.0
    if pd.isna(turnover):
        turnover = 999.0

    score = (
        - 3.0 * vol_gap
        - 2.0 * mdd
        + 3.0 * sharpe
        + 2.0 * ann_ret
        - 1.0 * turnover
    )
    return float(score)


# ================================================================
# CORE DECISION LOGIC
# ================================================================

def compute_target_weights(
    spy_pred_vol: float,
    tlt_pred_vol: float,
    spy_unc_scaled: float,
    tlt_unc_scaled: float,
    current_drawdown: float,
    params: dict,
) -> tuple[float, float, float]:
    """
    Convert today's predicted vol + uncertainty into target weights.

    Layer 1: decide how much SPY exposure is affordable
    Layer 2: allocate the defensive bucket between TLT and cash
    Layer 3: apply uncertainty, stress, and drawdown overlays
    """
    spy_unc_penalty = params["SPY_UNCERTAINTY_PENALTY"]
    tlt_unc_penalty = params["TLT_UNCERTAINTY_PENALTY"]

    # ------------------------------------------------------------
    # 1) Base SPY allocation from volatility targeting
    # ------------------------------------------------------------
    # If predicted SPY vol > target, reduce SPY weight.
    # If predicted SPY vol < target, allow higher SPY exposure.
    spy_base = TARGET_VOL / max(spy_pred_vol, 1e-8)
    spy_base = clip(spy_base, MIN_SPY_WEIGHT, MAX_SPY_WEIGHT)

    # ------------------------------------------------------------
    # 2) Reduce SPY exposure when RF uncertainty is high
    # ------------------------------------------------------------
    spy_after_unc = spy_base * (1.0 - spy_unc_penalty * spy_unc_scaled)
    spy_after_unc = clip(spy_after_unc, MIN_SPY_WEIGHT, MAX_SPY_WEIGHT)

    # ------------------------------------------------------------
    # 3) Stress overlay
    # ------------------------------------------------------------
    if spy_pred_vol > STRESS_VOL_THRESHOLD:
        spy_after_unc = min(spy_after_unc, STRESS_MAX_SPY_WEIGHT)

    # ------------------------------------------------------------
    # 4) Drawdown overlay
    # ------------------------------------------------------------
    if abs(current_drawdown) > DRAWDOWN_TRIGGER:
        spy_after_unc *= DRAWDOWN_SPY_MULTIPLIER

    spy_target = clip(spy_after_unc, MIN_SPY_WEIGHT, MAX_SPY_WEIGHT)

    # ------------------------------------------------------------
    # 5) Defensive bucket = everything not in SPY
    # ------------------------------------------------------------
    defensive_bucket = 1.0 - spy_target

    # ------------------------------------------------------------
    # 6) Allocate defensive bucket between TLT and cash
    # ------------------------------------------------------------
    # TLT is preferred when:
    # - its predicted risk is low relative to SPY
    # - its uncertainty is not too high
    #
    # This baseline intentionally tries to use TLT before hiding in cash.
    rel_tlt_risk = tlt_pred_vol / max(spy_pred_vol, 1e-8)

    base_tlt_share = 1.2 - rel_tlt_risk
    base_tlt_share = clip(base_tlt_share, 0.30, 0.90)

    tlt_share_after_unc = base_tlt_share * (1.0 - tlt_unc_penalty * tlt_unc_scaled)
    tlt_share_after_unc = clip(tlt_share_after_unc, 0.0, 1.0)

    # Mild floor: at least some of the defensive bucket should go to TLT
    # unless the bucket itself is near zero.
    tlt_share_after_unc = max(tlt_share_after_unc, MIN_TLT_SHARE_OF_DEFENSIVE)

    w_tlt_target = defensive_bucket * tlt_share_after_unc
    w_tlt_target = clip(w_tlt_target, MIN_TLT_WEIGHT, MAX_TLT_WEIGHT)

    w_cash_target = 1.0 - spy_target - w_tlt_target
    w_cash_target = max(w_cash_target, MIN_CASH_WEIGHT)

    return normalise_weights(spy_target, w_tlt_target, w_cash_target)


def apply_rebalancing_rules(
    prev_weights: tuple[float, float, float],
    target_weights: tuple[float, float, float],
    params: dict,
) -> tuple[float, float, float, float]:
    """
    Apply:
    - rebalance threshold
    - smoothing / partial adjustment

    Turnover convention:
    0.5 * sum(abs(weight change))
    """
    rebalance_threshold = params["REBALANCE_THRESHOLD"]
    smoothing_lambda = params["SMOOTHING_LAMBDA"]

    prev_spy, prev_tlt, prev_cash = prev_weights
    tgt_spy, tgt_tlt, tgt_cash = target_weights

    max_change = max(
        abs(tgt_spy - prev_spy),
        abs(tgt_tlt - prev_tlt),
        abs(tgt_cash - prev_cash),
    )

    # If desired change is too small, do nothing
    if max_change < rebalance_threshold:
        new_weights = (prev_spy, prev_tlt, prev_cash)
    else:
        new_spy = prev_spy + smoothing_lambda * (tgt_spy - prev_spy)
        new_tlt = prev_tlt + smoothing_lambda * (tgt_tlt - prev_tlt)
        new_cash = prev_cash + smoothing_lambda * (tgt_cash - prev_cash)
        new_weights = normalise_weights(new_spy, new_tlt, new_cash)

    turnover = 0.5 * (
        abs(new_weights[0] - prev_spy) +
        abs(new_weights[1] - prev_tlt) +
        abs(new_weights[2] - prev_cash)
    )

    return new_weights[0], new_weights[1], new_weights[2], float(turnover)


# ================================================================
# BACKTEST
# ================================================================

def run_dynamic_strategy(pred_df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Run the dynamic decision engine on a given period.

    Timing assumption:
    - signals observed at date t
    - weights decided at date t
    - next-day returns earned using *_next_day_return
    """
    df = pred_df.copy().sort_values("date").reset_index(drop=True)

    required_cols = [
        "date",
        "spy_calibrated_rf_pred",
        "tlt_calibrated_rf_pred",
        "spy_rf_uncertainty",
        "tlt_rf_uncertainty",
        "spy_next_day_return",
        "tlt_next_day_return",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"predictions.csv is missing required columns: {missing}")

    # Start from the static benchmark weights
    w_spy = STATIC_SPY_WEIGHT
    w_tlt = STATIC_TLT_WEIGHT
    w_cash = 0.0

    portfolio_value = 1.0
    running_peak = 1.0

    spy_unc_history: list[float] = []
    tlt_unc_history: list[float] = []

    records = []

    for _, row in df.iterrows():
        spy_pred = float(row["spy_calibrated_rf_pred"])
        tlt_pred = float(row["tlt_calibrated_rf_pred"])
        spy_unc = float(row["spy_rf_uncertainty"])
        tlt_unc = float(row["tlt_rf_uncertainty"])

        spy_unc_scaled = compute_running_uncertainty_scale(spy_unc_history, spy_unc)
        tlt_unc_scaled = compute_running_uncertainty_scale(tlt_unc_history, tlt_unc)

        current_drawdown = (portfolio_value / running_peak) - 1.0

        tgt_spy, tgt_tlt, tgt_cash = compute_target_weights(
            spy_pred_vol=spy_pred,
            tlt_pred_vol=tlt_pred,
            spy_unc_scaled=spy_unc_scaled,
            tlt_unc_scaled=tlt_unc_scaled,
            current_drawdown=current_drawdown,
            params=params,
        )

        new_spy, new_tlt, new_cash, turnover = apply_rebalancing_rules(
            prev_weights=(w_spy, w_tlt, w_cash),
            target_weights=(tgt_spy, tgt_tlt, tgt_cash),
            params=params,
        )

        trading_cost = turnover * TRANSACTION_COST_RATE

        next_spy_ret = float(row["spy_next_day_return"])
        next_tlt_ret = float(row["tlt_next_day_return"])

        portfolio_return_gross = (
            new_spy * next_spy_ret +
            new_tlt * next_tlt_ret +
            new_cash * 0.0
        )
        portfolio_return_net = portfolio_return_gross - trading_cost

        portfolio_value *= (1.0 + portfolio_return_net)
        running_peak = max(running_peak, portfolio_value)
        new_drawdown = (portfolio_value / running_peak) - 1.0

        records.append({
            "date": row["date"],

            # Signals
            "spy_pred_vol": spy_pred,
            "tlt_pred_vol": tlt_pred,
            "spy_uncertainty_raw": spy_unc,
            "tlt_uncertainty_raw": tlt_unc,
            "spy_uncertainty_scaled": spy_unc_scaled,
            "tlt_uncertainty_scaled": tlt_unc_scaled,

            # Targets
            "target_spy_weight": tgt_spy,
            "target_tlt_weight": tgt_tlt,
            "target_cash_weight": tgt_cash,

            # Actual weights after rebalance logic
            "dyn_spy_weight": new_spy,
            "dyn_tlt_weight": new_tlt,
            "dyn_cash_weight": new_cash,

            # Trading + performance
            "dyn_turnover": turnover,
            "dyn_trading_cost": trading_cost,
            "spy_next_day_return": next_spy_ret,
            "tlt_next_day_return": next_tlt_ret,
            "dyn_return_gross": portfolio_return_gross,
            "dyn_return_net": portfolio_return_net,
            "dyn_portfolio_value": portfolio_value,
            "dyn_drawdown": new_drawdown,
        })

        # Roll forward
        w_spy, w_tlt, w_cash = new_spy, new_tlt, new_cash
        spy_unc_history.append(spy_unc)
        tlt_unc_history.append(tlt_unc)

    return pd.DataFrame(records)


def add_benchmarks(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add benchmark strategies to the daily results dataframe:
    1. Static 60/40 SPY/TLT
    2. Optional 100% SPY
    """
    df = results_df.copy()

    # Static 60/40 benchmark
    static_value = 1.0
    static_values = []
    static_returns = []

    for _, row in df.iterrows():
        r = (
            STATIC_SPY_WEIGHT * row["spy_next_day_return"] +
            STATIC_TLT_WEIGHT * row["tlt_next_day_return"]
        )
        static_value *= (1.0 + r)
        static_returns.append(r)
        static_values.append(static_value)

    df["static_return_net"] = static_returns
    df["static_portfolio_value"] = static_values
    static_peak = df["static_portfolio_value"].cummax()
    df["static_drawdown"] = (df["static_portfolio_value"] / static_peak) - 1.0

    # 100% SPY benchmark
    if INCLUDE_SPY_ONLY_BENCHMARK:
        spy_only_value = 1.0
        spy_only_values = []
        spy_only_returns = []

        for _, row in df.iterrows():
            r = row["spy_next_day_return"]
            spy_only_value *= (1.0 + r)
            spy_only_returns.append(r)
            spy_only_values.append(spy_only_value)

        df["spy_only_return_net"] = spy_only_returns
        df["spy_only_portfolio_value"] = spy_only_values
        spy_only_peak = df["spy_only_portfolio_value"].cummax()
        df["spy_only_drawdown"] = (df["spy_only_portfolio_value"] / spy_only_peak) - 1.0

    return df


# ================================================================
# PARAMETER SEARCH
# ================================================================

def run_parameter_search(validation_df: pd.DataFrame, param_grid: dict) -> pd.DataFrame:
    """
    Run a structured grid search on the validation segment only.
    """
    combinations = make_param_combinations(param_grid)
    results = []

    print("=" * 60)
    print("RUNNING DECISION ENGINE PARAMETER SEARCH")
    print("=" * 60)
    print(f"Validation rows: {len(validation_df)}")
    print(f"Parameter combinations: {len(combinations)}")

    for i, params in enumerate(combinations, start=1):
        daily = run_dynamic_strategy(validation_df, params=params)

        kpis = compute_strategy_kpis(
            df=daily,
            strategy_name="Dynamic Decision Engine (Validation)",
            return_col="dyn_return_net",
            value_col="dyn_portfolio_value",
            turnover_col="dyn_turnover",
            cost_col="dyn_trading_cost",
            spy_weight_col="dyn_spy_weight",
            tlt_weight_col="dyn_tlt_weight",
            cash_weight_col="dyn_cash_weight",
        )

        score = strategy_score(pd.Series(kpis))

        row = {
            **params,
            **kpis,
            "score": score,
        }
        results.append(row)

        if i % 20 == 0 or i == len(combinations):
            print(f"  Completed {i}/{len(combinations)} combinations")

    search_df = pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)
    return search_df


# ================================================================
# MAIN
# ================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # 1) Load predictions from modelling stage
    # ------------------------------------------------------------
    pred_df = pd.read_csv(PREDICTIONS_FILE, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    # ------------------------------------------------------------
    # 2) Split into validation and holdout periods
    # ------------------------------------------------------------
    validation_df, holdout_df = split_validation_holdout(pred_df, ratio=ENGINE_VALIDATION_RATIO)

    print("=" * 60)
    print("DECISION ENGINE DATA SPLIT")
    print("=" * 60)
    print(f"Total rows:       {len(pred_df)}")
    print(f"Validation rows:  {len(validation_df)}")
    print(f"Holdout rows:     {len(holdout_df)}")
    print(f"Validation dates: {validation_df['date'].min().date()} -> {validation_df['date'].max().date()}")
    print(f"Holdout dates:    {holdout_df['date'].min().date()} -> {holdout_df['date'].max().date()}")

    # ------------------------------------------------------------
    # 3) Parameter search on validation
    # ------------------------------------------------------------
    search_df = run_parameter_search(validation_df, PARAM_GRID)
    search_df.to_csv(SEARCH_RESULTS_FILE, index=False)

    best_params = {
        key: search_df.loc[0, key]
        for key in PARAM_GRID.keys()
    }

    with open(BEST_ENGINE_PARAMS_FILE, "w") as f:
        json.dump(best_params, f, indent=2)

    print("\nTop 10 parameter sets on validation:")
    display_cols = list(PARAM_GRID.keys()) + [
        "score",
        "annualised_return",
        "annualised_volatility",
        "target_vol_gap",
        "sharpe_ratio",
        "max_drawdown",
        "avg_daily_turnover",
        "avg_spy_weight",
        "avg_tlt_weight",
        "avg_cash_weight",
    ]
    print(search_df[display_cols].head(10).round(4).to_string(index=False))

    print("\nBest parameters selected from validation:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    # ------------------------------------------------------------
    # 4) Final honest evaluation on later unseen holdout
    # ------------------------------------------------------------
    holdout_daily = run_dynamic_strategy(holdout_df, params=best_params)
    holdout_daily = add_benchmarks(holdout_daily)

    # ------------------------------------------------------------
    # 5) KPI comparison on holdout
    # ------------------------------------------------------------
    kpis = []

    # Dynamic strategy
    kpis.append(
        compute_strategy_kpis(
            df=holdout_daily,
            strategy_name="Dynamic Decision Engine",
            return_col="dyn_return_net",
            value_col="dyn_portfolio_value",
            turnover_col="dyn_turnover",
            cost_col="dyn_trading_cost",
            spy_weight_col="dyn_spy_weight",
            tlt_weight_col="dyn_tlt_weight",
            cash_weight_col="dyn_cash_weight",
        )
    )

    # Static 60/40 benchmark
    kpis.append(
        compute_strategy_kpis(
            df=holdout_daily,
            strategy_name="Static 60/40",
            return_col="static_return_net",
            value_col="static_portfolio_value",
        )
    )

    # 100% SPY benchmark
    if INCLUDE_SPY_ONLY_BENCHMARK:
        kpis.append(
            compute_strategy_kpis(
                df=holdout_daily,
                strategy_name="100% SPY",
                return_col="spy_only_return_net",
                value_col="spy_only_portfolio_value",
            )
        )

    kpi_df = pd.DataFrame(kpis)

    # ------------------------------------------------------------
    # 6) Save outputs
    # ------------------------------------------------------------
    holdout_daily.to_csv(PORTFOLIO_RESULTS_FILE, index=False)
    kpi_df.to_csv(PORTFOLIO_KPIS_FILE, index=False)

    # ------------------------------------------------------------
    # 7) Print final summary
    # ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("DECISION ENGINE COMPLETE")
    print("=" * 60)
    print(f"Saved search results:  {SEARCH_RESULTS_FILE}")
    print(f"Saved best params:     {BEST_ENGINE_PARAMS_FILE}")
    print(f"Saved daily results:   {PORTFOLIO_RESULTS_FILE}")
    print(f"Saved KPI summary:     {PORTFOLIO_KPIS_FILE}")

    print("\nFinal KPI summary on holdout period:")
    final_display_cols = [
        "strategy",
        "annualised_return",
        "annualised_volatility",
        "target_vol_gap",
        "sharpe_ratio",
        "max_drawdown",
        "ending_value",
    ]
    print(kpi_df[final_display_cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()