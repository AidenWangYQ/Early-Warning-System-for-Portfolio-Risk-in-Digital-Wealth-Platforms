"""
04_decision_engine.py

Upgraded decision engine for:
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

Key upgrades vs earlier version:
- momentum / trend overlay
- relative attractiveness between SPY and TLT
- gradual drawdown scaling
- re-entry logic after stress eases
- reduced cash-hoarding behaviour
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
# FIXED ENGINE SETTINGS
# ================================================================

# Business objective: keep realised portfolio risk around ~10% annualised
TARGET_VOL = 0.10

# Asset weight bounds
MAX_SPY_WEIGHT = 0.90
MIN_SPY_WEIGHT = 0.00

MAX_TLT_WEIGHT = 0.90
MIN_TLT_WEIGHT = 0.00

MIN_CASH_WEIGHT = 0.00
MAX_CASH_WEIGHT = 0.60

# Stress controls
STRESS_VOL_THRESHOLD = 0.20
STRESS_MAX_SPY_WEIGHT = 0.55

# Drawdown controls
DRAWDOWN_TRIGGER = 0.10

# Defensive allocation preferences
MIN_TLT_SHARE_OF_DEFENSIVE = 0.25

# Re-entry / recovery control
RECOVERY_VOL_BUFFER = 0.015

# Optional benchmark
INCLUDE_SPY_ONLY_BENCHMARK = True

# Convert transaction cost from bps to decimal
TRANSACTION_COST_RATE = TRANSACTION_COST_BPS / 10000.0

# Validation / holdout split INSIDE predictions.csv
ENGINE_VALIDATION_RATIO = 0.60

# Output files
SEARCH_RESULTS_FILE = OUTPUT_DIR / "decision_engine_search_results.csv"
BEST_ENGINE_PARAMS_FILE = OUTPUT_DIR / "decision_engine_best_params.json"

SPY_UNCERTAINTY_PENALTY = 0.1
TLT_UNCERTAINTY_PENALTY = 0.2
REBALANCE_THRESHOLD = 0.05
SMOOTHING_LAMBDA = 0.2
MOMENTUM_STRENGTH = 0.2
RELATIVE_STRENGTH = 0.3
DRAWDOWN_SENSITIVITY = 0.5
REENTRY_STRENGTH = 0.1
CASH_PENALTY = 0.02

# ================================================================
# PARAMETER GRID
# ================================================================
# Keep the grid small and interpretable.

PARAM_GRID = {
    "MOMENTUM_STRENGTH": [0.20, 0.30],
    "CASH_PENALTY": [0.02, 0.05],
}

# ================================================================
# HELPER FUNCTIONS
# ================================================================

def clip(value: float, low: float, high: float) -> float:
    """Clip a scalar into [low, high]."""
    return float(np.clip(value, low, high))


def safe_div(a: float, b: float, fallback: float = 0.0) -> float:
    """Safe scalar division."""
    if abs(b) <= 1e-12:
        return fallback
    return float(a / b)


def normalise_weights(w_spy: float, w_tlt: float, w_cash: float) -> tuple[float, float, float]:
    """
    Ensure weights are non-negative and sum to 1.
    """
    weights = np.array(
        [max(w_spy, 0.0), max(w_tlt, 0.0), max(w_cash, 0.0)],
        dtype=float,
    )
    total = weights.sum()

    if total <= 0:
        return 0.0, 0.0, 1.0

    weights = weights / total

    # Re-cap cash if needed and renormalise remaining assets
    if weights[2] > MAX_CASH_WEIGHT:
        excess = weights[2] - MAX_CASH_WEIGHT
        weights[2] = MAX_CASH_WEIGHT

        risky_total = weights[0] + weights[1]
        if risky_total > 0:
            weights[0] += excess * weights[0] / risky_total
            weights[1] += excess * weights[1] / risky_total
        else:
            weights[1] += excess

        weights = weights / weights.sum()

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
    """
    hist = history + [current_value]

    if len(hist) < 5:
        return 0.5

    p90 = np.percentile(hist, 90)
    if p90 <= 1e-8:
        return 0.0

    scaled = current_value / p90
    return clip(scaled, 0.0, 1.25)


def split_validation_holdout(
    df: pd.DataFrame,
    ratio: float = ENGINE_VALIDATION_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronologically split predictions into:
    - validation period for engine tuning
    - later holdout period for final evaluation
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


def squash_signal(x: float, scale: float = 1.0) -> float:
    """
    Smoothly squash signal into roughly [-1, 1].
    Useful for momentum / attractiveness modifiers.
    """
    return float(np.tanh(scale * x))


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

    This version rewards:
    - better Sharpe
    - reasonable return
    - lower drawdown
    - closeness to target volatility
    - lower turnover
    """
    vol_gap = abs(kpi_row["annualised_volatility"] - TARGET_VOL)
    mdd = abs(kpi_row["max_drawdown"])
    sharpe = kpi_row["sharpe_ratio"]
    ann_ret = kpi_row["annualised_return"]
    turnover = kpi_row.get("avg_daily_turnover", 0.0)

    if pd.isna(sharpe):
        sharpe = -999.0
    if pd.isna(ann_ret):
        ann_ret = -999.0
    if pd.isna(turnover):
        turnover = 999.0

    score = (
        -2.5 * vol_gap
        -1.8 * mdd
        +3.5 * sharpe
        +2.5 * ann_ret
        -0.8 * turnover
    )
    return float(score)


# ================================================================
# CORE DECISION LOGIC
# ================================================================

def compute_momentum_signal(
    spy_ma5_gap: float,
    spy_ma20_gap: float,
    spy_ret_lag1: float,
    spy_ret_lag5: float,
) -> float:
    """
    Build a simple direction / trend score for SPY using:
    - short and medium MA gaps
    - very recent return
    - slightly longer lagged return

    Positive = favourable trend
    Negative = weak trend
    """
    raw = (
        0.35 * spy_ma5_gap
        + 0.35 * spy_ma20_gap
        + 8.0 * spy_ret_lag1
        + 4.0 * spy_ret_lag5
    )
    return squash_signal(raw, scale=3.0)


def compute_relative_attractiveness(
    spy_pred_vol: float,
    tlt_pred_vol: float,
    spy_unc_scaled: float,
    tlt_unc_scaled: float,
) -> float:
    """
    Relative attractiveness of SPY vs TLT using inverse risk adjusted by uncertainty.

    Positive => SPY relatively more attractive
    Negative => TLT relatively more attractive
    """
    spy_score = 1.0 / max(spy_pred_vol * (1.0 + spy_unc_scaled), 1e-8)
    tlt_score = 1.0 / max(tlt_pred_vol * (1.0 + tlt_unc_scaled), 1e-8)

    raw = safe_div(spy_score - tlt_score, spy_score + tlt_score, fallback=0.0)
    return squash_signal(raw, scale=2.0)


def compute_target_weights(
    row: pd.Series,
    spy_unc_scaled: float,
    tlt_unc_scaled: float,
    current_drawdown: float,
    params: dict,
) -> tuple[float, float, float]:
    """
    Convert today's predictions + state into target weights.

    Layers:
    1) volatility targeting
    2) uncertainty adjustment
    3) relative attractiveness SPY vs TLT
    4) momentum / trend overlay
    5) stress and drawdown controls
    6) re-entry logic
    7) defensive bucket split between TLT and cash
    """
    spy_pred_vol = float(row["spy_calibrated_rf_pred"])
    tlt_pred_vol = float(row["tlt_calibrated_rf_pred"])

    spy_unc_penalty = params["SPY_UNCERTAINTY_PENALTY"]
    tlt_unc_penalty = params["TLT_UNCERTAINTY_PENALTY"]
    momentum_strength = params["MOMENTUM_STRENGTH"]
    relative_strength = params["RELATIVE_STRENGTH"]
    drawdown_sensitivity = params["DRAWDOWN_SENSITIVITY"]
    reentry_strength = params["REENTRY_STRENGTH"]
    cash_penalty = params["CASH_PENALTY"]

    # ------------------------------------------------------------
    # 1) Base SPY weight from volatility targeting
    # ------------------------------------------------------------
    spy_base = TARGET_VOL / max(spy_pred_vol, 1e-8)
    spy_base = clip(spy_base, MIN_SPY_WEIGHT, MAX_SPY_WEIGHT)

    # ------------------------------------------------------------
    # 2) Reduce SPY based on uncertainty
    # ------------------------------------------------------------
    spy_after_unc = spy_base * (1.0 - spy_unc_penalty * spy_unc_scaled)
    spy_after_unc = clip(spy_after_unc, MIN_SPY_WEIGHT, MAX_SPY_WEIGHT)

    # ------------------------------------------------------------
    # 3) Relative attractiveness between SPY and TLT
    # ------------------------------------------------------------
    rel_attr = compute_relative_attractiveness(
        spy_pred_vol=spy_pred_vol,
        tlt_pred_vol=tlt_pred_vol,
        spy_unc_scaled=spy_unc_scaled,
        tlt_unc_scaled=tlt_unc_scaled,
    )

    # Push SPY slightly up when relatively attractive, down otherwise
    spy_after_relative = spy_after_unc * (1.0 + relative_strength * rel_attr)
    spy_after_relative = clip(spy_after_relative, MIN_SPY_WEIGHT, MAX_SPY_WEIGHT)

    # ------------------------------------------------------------
    # 4) Momentum / trend overlay
    # ------------------------------------------------------------
    momentum_signal = compute_momentum_signal(
        spy_ma5_gap=float(row.get("spy_price_ma5_gap", 0.0)),
        spy_ma20_gap=float(row.get("spy_price_ma20_gap", 0.0)),
        spy_ret_lag1=float(row.get("spy_ret_lag1", 0.0)),
        spy_ret_lag5=float(row.get("spy_ret_lag5", 0.0)),
    )

    spy_after_momentum = spy_after_relative * (1.0 + momentum_strength * momentum_signal)
    spy_after_momentum = clip(spy_after_momentum, MIN_SPY_WEIGHT, MAX_SPY_WEIGHT)

    # ------------------------------------------------------------
    # 5) Stress overlay
    # ------------------------------------------------------------
    if spy_pred_vol > STRESS_VOL_THRESHOLD:
        spy_after_momentum = min(spy_after_momentum, STRESS_MAX_SPY_WEIGHT)

    # ------------------------------------------------------------
    # 6) Gradual drawdown overlay
    # ------------------------------------------------------------
    # current_drawdown is negative or zero
    if abs(current_drawdown) > DRAWDOWN_TRIGGER:
        dd_excess = abs(current_drawdown) - DRAWDOWN_TRIGGER
        dd_scale = 1.0 - drawdown_sensitivity * dd_excess
        dd_scale = clip(dd_scale, 0.65, 1.0)
        spy_after_momentum *= dd_scale

    # ------------------------------------------------------------
    # 7) Re-entry logic
    # ------------------------------------------------------------
    # If predicted vol is near/below target and momentum is positive,
    # allow some additional SPY exposure to participate in recovery.
    if (spy_pred_vol < TARGET_VOL + RECOVERY_VOL_BUFFER) and (momentum_signal > 0):
        reentry_boost = 1.0 + reentry_strength * momentum_signal
        spy_after_momentum *= reentry_boost

    spy_target = clip(spy_after_momentum, MIN_SPY_WEIGHT, MAX_SPY_WEIGHT)

    # ------------------------------------------------------------
    # 8) Defensive bucket between TLT and cash
    # ------------------------------------------------------------
    defensive_bucket = 1.0 - spy_target

    # TLT share higher when:
    # - TLT predicted vol is not too high vs SPY
    # - TLT uncertainty is manageable
    rel_tlt_risk = safe_div(tlt_pred_vol, spy_pred_vol, fallback=1.0)

    base_tlt_share = 1.10 - 0.55 * rel_tlt_risk
    base_tlt_share = clip(base_tlt_share, 0.25, 0.90)

    # Apply TLT uncertainty penalty
    tlt_share_after_unc = base_tlt_share * (1.0 - tlt_unc_penalty * tlt_unc_scaled)
    tlt_share_after_unc = clip(tlt_share_after_unc, 0.0, 1.0)

    # Mild floor so defensive allocation does not go too aggressively into cash
    tlt_share_after_unc = max(tlt_share_after_unc, MIN_TLT_SHARE_OF_DEFENSIVE)

    # Cash penalty = prefer staying invested rather than hiding in cash too much
    tlt_share_after_unc = clip(tlt_share_after_unc + cash_penalty, 0.0, 1.0)

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

    if max_change < rebalance_threshold:
        new_weights = (prev_spy, prev_tlt, prev_cash)
    else:
        new_spy = prev_spy + smoothing_lambda * (tgt_spy - prev_spy)
        new_tlt = prev_tlt + smoothing_lambda * (tgt_tlt - prev_tlt)
        new_cash = prev_cash + smoothing_lambda * (tgt_cash - prev_cash)
        new_weights = normalise_weights(new_spy, new_tlt, new_cash)

    turnover = 0.5 * (
        abs(new_weights[0] - prev_spy)
        + abs(new_weights[1] - prev_tlt)
        + abs(new_weights[2] - prev_cash)
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

    # Optional columns used for momentum overlay
    optional_cols = [
        "spy_price_ma5_gap",
        "spy_price_ma20_gap",
        "spy_ret_lag1",
        "spy_ret_lag5",
    ]
    for col in optional_cols:
        if col not in df.columns:
            df[col] = 0.0

    # Start from static benchmark weights
    w_spy = STATIC_SPY_WEIGHT
    w_tlt = STATIC_TLT_WEIGHT
    w_cash = max(0.0, 1.0 - w_spy - w_tlt)

    portfolio_value = 1.0
    running_peak = 1.0

    spy_unc_history: list[float] = []
    tlt_unc_history: list[float] = []

    records = []

    for _, row in df.iterrows():
        spy_unc = float(row["spy_rf_uncertainty"])
        tlt_unc = float(row["tlt_rf_uncertainty"])

        spy_unc_scaled = compute_running_uncertainty_scale(spy_unc_history, spy_unc)
        tlt_unc_scaled = compute_running_uncertainty_scale(tlt_unc_history, tlt_unc)

        current_drawdown = (portfolio_value / running_peak) - 1.0

        tgt_spy, tgt_tlt, tgt_cash = compute_target_weights(
            row=row,
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

        transaction_cost = turnover * TRANSACTION_COST_RATE

        gross_return = (
            new_spy * float(row["spy_next_day_return"])
            + new_tlt * float(row["tlt_next_day_return"])
            + new_cash * 0.0
        )

        net_return = gross_return - transaction_cost

        portfolio_value *= (1.0 + net_return)
        running_peak = max(running_peak, portfolio_value)
        next_drawdown = (portfolio_value / running_peak) - 1.0

        records.append(
            {
                "date": row["date"],
                "spy_calibrated_rf_pred": float(row["spy_calibrated_rf_pred"]),
                "tlt_calibrated_rf_pred": float(row["tlt_calibrated_rf_pred"]),
                "spy_rf_uncertainty": spy_unc,
                "tlt_rf_uncertainty": tlt_unc,
                "spy_unc_scaled": spy_unc_scaled,
                "tlt_unc_scaled": tlt_unc_scaled,
                "spy_price_ma5_gap": float(row["spy_price_ma5_gap"]),
                "spy_price_ma20_gap": float(row["spy_price_ma20_gap"]),
                "spy_ret_lag1": float(row["spy_ret_lag1"]),
                "spy_ret_lag5": float(row["spy_ret_lag5"]),
                "w_spy": new_spy,
                "w_tlt": new_tlt,
                "w_cash": new_cash,
                "turnover": turnover,
                "transaction_cost": transaction_cost,
                "gross_return": gross_return,
                "net_return": net_return,
                "portfolio_value": portfolio_value,
                "drawdown": next_drawdown,
            }
        )

        w_spy, w_tlt, w_cash = new_spy, new_tlt, new_cash
        spy_unc_history.append(spy_unc)
        tlt_unc_history.append(tlt_unc)

    return pd.DataFrame(records)


# ================================================================
# BENCHMARKS
# ================================================================

def run_static_6040_benchmark(pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Static 60/40 benchmark using next-day SPY and TLT returns.
    """
    df = pred_df.copy().sort_values("date").reset_index(drop=True)

    w_spy = STATIC_SPY_WEIGHT
    w_tlt = STATIC_TLT_WEIGHT
    w_cash = max(0.0, 1.0 - w_spy - w_tlt)

    portfolio_value = 1.0
    running_peak = 1.0
    records = []

    for _, row in df.iterrows():
        gross_return = (
            w_spy * float(row["spy_next_day_return"])
            + w_tlt * float(row["tlt_next_day_return"])
            + w_cash * 0.0
        )
        net_return = gross_return
        portfolio_value *= (1.0 + net_return)
        running_peak = max(running_peak, portfolio_value)
        dd = (portfolio_value / running_peak) - 1.0

        records.append(
            {
                "date": row["date"],
                "w_spy": w_spy,
                "w_tlt": w_tlt,
                "w_cash": w_cash,
                "turnover": 0.0,
                "transaction_cost": 0.0,
                "gross_return": gross_return,
                "net_return": net_return,
                "portfolio_value": portfolio_value,
                "drawdown": dd,
            }
        )

    return pd.DataFrame(records)


def run_static_spy_benchmark(pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    100% SPY benchmark.
    """
    df = pred_df.copy().sort_values("date").reset_index(drop=True)

    portfolio_value = 1.0
    running_peak = 1.0
    records = []

    for _, row in df.iterrows():
        gross_return = float(row["spy_next_day_return"])
        net_return = gross_return
        portfolio_value *= (1.0 + net_return)
        running_peak = max(running_peak, portfolio_value)
        dd = (portfolio_value / running_peak) - 1.0

        records.append(
            {
                "date": row["date"],
                "w_spy": 1.0,
                "w_tlt": 0.0,
                "w_cash": 0.0,
                "turnover": 0.0,
                "transaction_cost": 0.0,
                "gross_return": gross_return,
                "net_return": net_return,
                "portfolio_value": portfolio_value,
                "drawdown": dd,
            }
        )

    return pd.DataFrame(records)


def run_naive_vol_target_benchmark(pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple benchmark:
    - weight SPY by target_vol / predicted_vol
    - put the remainder into TLT
    - no uncertainty, no momentum, no drawdown overlay
    - partial cap for realism

    This is useful because it shows whether the full decision engine
    adds value beyond a much simpler dynamic rule.
    """
    df = pred_df.copy().sort_values("date").reset_index(drop=True)

    portfolio_value = 1.0
    running_peak = 1.0

    prev_spy, prev_tlt, prev_cash = STATIC_SPY_WEIGHT, STATIC_TLT_WEIGHT, max(0.0, 1.0 - STATIC_SPY_WEIGHT - STATIC_TLT_WEIGHT)
    records = []

    for _, row in df.iterrows():
        spy_pred = float(row["spy_calibrated_rf_pred"])

        w_spy = clip(TARGET_VOL / max(spy_pred, 1e-8), 0.0, 0.90)
        w_tlt = 1.0 - w_spy
        w_cash = 0.0
        w_spy, w_tlt, w_cash = normalise_weights(w_spy, w_tlt, w_cash)

        turnover = 0.5 * (
            abs(w_spy - prev_spy)
            + abs(w_tlt - prev_tlt)
            + abs(w_cash - prev_cash)
        )
        transaction_cost = turnover * TRANSACTION_COST_RATE

        gross_return = (
            w_spy * float(row["spy_next_day_return"])
            + w_tlt * float(row["tlt_next_day_return"])
        )
        net_return = gross_return - transaction_cost

        portfolio_value *= (1.0 + net_return)
        running_peak = max(running_peak, portfolio_value)
        dd = (portfolio_value / running_peak) - 1.0

        records.append(
            {
                "date": row["date"],
                "w_spy": w_spy,
                "w_tlt": w_tlt,
                "w_cash": w_cash,
                "turnover": turnover,
                "transaction_cost": transaction_cost,
                "gross_return": gross_return,
                "net_return": net_return,
                "portfolio_value": portfolio_value,
                "drawdown": dd,
            }
        )

        prev_spy, prev_tlt, prev_cash = w_spy, w_tlt, w_cash

    return pd.DataFrame(records)


# ================================================================
# ENGINE TUNING
# ================================================================

def tune_engine(validation_df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """
    Grid-search only the selected behavioural parameters on validation period,
    while keeping all other engine settings fixed at their default values.
    """
    combos = make_param_combinations(PARAM_GRID)
    results = []

    # Full default parameter set
    DEFAULT_PARAMS = {
        "SPY_UNCERTAINTY_PENALTY": SPY_UNCERTAINTY_PENALTY,
        "TLT_UNCERTAINTY_PENALTY": TLT_UNCERTAINTY_PENALTY,
        "REBALANCE_THRESHOLD": REBALANCE_THRESHOLD,
        "SMOOTHING_LAMBDA": SMOOTHING_LAMBDA,
        "MOMENTUM_STRENGTH": MOMENTUM_STRENGTH,
        "RELATIVE_STRENGTH": RELATIVE_STRENGTH,
        "DRAWDOWN_SENSITIVITY": DRAWDOWN_SENSITIVITY,
        "REENTRY_STRENGTH": REENTRY_STRENGTH,
        "CASH_PENALTY": CASH_PENALTY,
    }

    print(f"\nTuning decision engine over {len(combos)} parameter combinations...")

    for i, tuned_params in enumerate(combos, start=1):
        if i % 50 == 0 or i == 1 or i == len(combos):
            print(f"  Progress: {i}/{len(combos)}")

        # Merge tuned params into full default param set
        params = DEFAULT_PARAMS.copy()
        params.update(tuned_params)

        try:
            strat_df = run_dynamic_strategy(validation_df, params)

            kpis = compute_strategy_kpis(
                strat_df,
                strategy_name="dynamic_validation",
                return_col="net_return",
                value_col="portfolio_value",
                turnover_col="turnover",
                cost_col="transaction_cost",
                spy_weight_col="w_spy",
                tlt_weight_col="w_tlt",
                cash_weight_col="w_cash",
            )

            score = strategy_score(pd.Series(kpis))

            # Save both tuned values and KPI outputs
            result_row = dict(params)
            result_row.update(kpis)
            result_row["score"] = score
            results.append(result_row)

        except Exception as e:
            result_row = dict(params)
            result_row["strategy"] = "dynamic_validation"
            result_row["score"] = -999999.0
            result_row["error"] = str(e)
            results.append(result_row)

    results_df = pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)

    if results_df.empty:
        raise ValueError("No tuning results generated.")

    best_row = results_df.iloc[0]

    # Reconstruct full best parameter set
    best_params = DEFAULT_PARAMS.copy()
    for k in PARAM_GRID.keys():
        best_params[k] = best_row[k]

    return best_params, results_df


# ================================================================
# MAIN
# ================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("UPGRADED DECISION ENGINE")
    print("=" * 70)

    pred_df = pd.read_csv(PREDICTIONS_FILE, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    print(f"Loaded predictions: {len(pred_df)} rows from {PREDICTIONS_FILE}")

    validation_df, holdout_df = split_validation_holdout(pred_df)

    print(f"Validation rows: {len(validation_df)}")
    print(f"Holdout rows:    {len(holdout_df)}")

    # ------------------------------------------------------------
    # Tune on validation
    # ------------------------------------------------------------
    best_params, search_results_df = tune_engine(validation_df)

    print("\nBest engine parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    search_results_df.to_csv(SEARCH_RESULTS_FILE, index=False)
    with open(BEST_ENGINE_PARAMS_FILE, "w") as f:
        json.dump(best_params, f, indent=2)

    # ------------------------------------------------------------
    # Evaluate on holdout
    # ------------------------------------------------------------
    print("\nRunning final holdout evaluation...")

    dynamic_df = run_dynamic_strategy(holdout_df, best_params)
    static_6040_df = run_static_6040_benchmark(holdout_df)
    naive_vol_df = run_naive_vol_target_benchmark(holdout_df)

    kpi_rows = []

    kpi_rows.append(
        compute_strategy_kpis(
            dynamic_df,
            strategy_name="Dynamic Strategy",
            return_col="net_return",
            value_col="portfolio_value",
            turnover_col="turnover",
            cost_col="transaction_cost",
            spy_weight_col="w_spy",
            tlt_weight_col="w_tlt",
            cash_weight_col="w_cash",
        )
    )

    kpi_rows.append(
        compute_strategy_kpis(
            static_6040_df,
            strategy_name="Static 60/40",
            return_col="net_return",
            value_col="portfolio_value",
            turnover_col="turnover",
            cost_col="transaction_cost",
            spy_weight_col="w_spy",
            tlt_weight_col="w_tlt",
            cash_weight_col="w_cash",
        )
    )

    kpi_rows.append(
        compute_strategy_kpis(
            naive_vol_df,
            strategy_name="Naive Vol Target",
            return_col="net_return",
            value_col="portfolio_value",
            turnover_col="turnover",
            cost_col="transaction_cost",
            spy_weight_col="w_spy",
            tlt_weight_col="w_tlt",
            cash_weight_col="w_cash",
        )
    )

    if INCLUDE_SPY_ONLY_BENCHMARK:
        spy_df = run_static_spy_benchmark(holdout_df)
        kpi_rows.append(
            compute_strategy_kpis(
                spy_df,
                strategy_name="100% SPY",
                return_col="net_return",
                value_col="portfolio_value",
                turnover_col="turnover",
                cost_col="transaction_cost",
                spy_weight_col="w_spy",
                tlt_weight_col="w_tlt",
                cash_weight_col="w_cash",
            )
        )
    else:
        spy_df = None

    kpis_df = pd.DataFrame(kpi_rows)

    # ------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------
    output_frames = []

    dyn_out = dynamic_df.copy()
    dyn_out["strategy"] = "Dynamic Strategy"
    output_frames.append(dyn_out)

    s6040_out = static_6040_df.copy()
    s6040_out["strategy"] = "Static 60/40"
    output_frames.append(s6040_out)

    naive_out = naive_vol_df.copy()
    naive_out["strategy"] = "Naive Vol Target"
    output_frames.append(naive_out)

    if spy_df is not None:
        spy_out = spy_df.copy()
        spy_out["strategy"] = "100% SPY"
        output_frames.append(spy_out)

    portfolio_results_df = pd.concat(output_frames, ignore_index=True)

    portfolio_results_df.to_csv(PORTFOLIO_RESULTS_FILE, index=False)
    kpis_df.to_csv(PORTFOLIO_KPIS_FILE, index=False)

    # ------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------
    print("\nHoldout KPI summary:")
    print(kpis_df.round(4).to_string(index=False))

    print(f"\nSaved:")
    print(f"  Search results:    {SEARCH_RESULTS_FILE}")
    print(f"  Best params:       {BEST_ENGINE_PARAMS_FILE}")
    print(f"  Portfolio results: {PORTFOLIO_RESULTS_FILE}")
    print(f"  Portfolio KPIs:    {PORTFOLIO_KPIS_FILE}")

    print("\nDone.")


if __name__ == "__main__":
    main()