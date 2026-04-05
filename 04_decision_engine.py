from __future__ import annotations

# ================================================================
# 04_decision_engine.py (ANNOTATED VERSION)
# ================================================================
# Purpose of this file:
# This script takes the model outputs from predictions.csv and turns them
# into portfolio decisions.
#
# In other words:
# - 03_modeling.py answers: "How risky do we think SPY and TLT will be soon?"
# - 04_decision_engine.py answers: "Given that forecast, how should we allocate
#   between SPY, TLT, and Cash?"
#
# It also benchmarks the final strategy against simpler alternatives:
# - Static 60/40
# - Naive volatility targeting
# - 100% SPY
#
# Main workflow:
# 1. Read predictions.csv from the modelling stage
# 2. Split that period into validation and holdout
# 3. Tune a small number of interpretable decision-engine parameters on validation
# 4. Run the final strategy on holdout only
# 5. Save portfolio results and KPI summaries

import json
from itertools import product
from pathlib import Path
from typing import Dict

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
# These are the business rules / structural choices of the engine.
# They are NOT all tuned. Most are fixed deliberately so the system stays
# interpretable and does not become overly optimized to historical data.

# Target annualised portfolio volatility.
# The whole engine is built around trying to keep realised risk near this level.
TARGET_VOL = 0.10

# Hard allocation bounds so the strategy stays realistic.
MAX_SPY_WEIGHT = 0.90
MIN_SPY_WEIGHT = 0.00
MAX_TLT_WEIGHT = 0.90
MIN_TLT_WEIGHT = 0.00
MIN_CASH_WEIGHT = 0.00
MAX_CASH_WEIGHT = 0.60

# Stress control:
# if predicted SPY volatility is very high, cap SPY exposure more aggressively.
STRESS_VOL_THRESHOLD = 0.20
STRESS_MAX_SPY_WEIGHT = 0.55

# Drawdown control:
# once portfolio drawdown becomes meaningful, reduce aggression.
DRAWDOWN_TRIGGER = 0.10

# Even in defensive mode, try not to dump everything into cash too easily.
MIN_TLT_SHARE_OF_DEFENSIVE = 0.25

# If conditions improve, allow some re-entry into SPY.
RECOVERY_VOL_BUFFER = 0.015

# Include a 100% SPY benchmark for comparison.
INCLUDE_SPY_ONLY_BENCHMARK = True

# Convert bps to decimal rate.
TRANSACTION_COST_RATE = TRANSACTION_COST_BPS / 10000.0

# Inside predictions.csv, first 60% is used for decision-engine tuning,
# last 40% is reserved for final honest evaluation.
ENGINE_VALIDATION_RATIO = 0.60

# Output locations.
SEARCH_RESULTS_FILE = OUTPUT_DIR / "decision_engine_search_results.csv"
BEST_ENGINE_PARAMS_FILE = OUTPUT_DIR / "decision_engine_best_params.json"
STRATEGY_BREAKOUT_DIR = OUTPUT_DIR / "strategy_breakouts"


# Default engine parameters.
# These values define the behaviour of the proposed strategy unless a parameter
# is specifically included in the small tuning grid below.
SPY_UNCERTAINTY_PENALTY = 0.10
TLT_UNCERTAINTY_PENALTY = 0.20
REBALANCE_THRESHOLD = 0.05
SMOOTHING_LAMBDA = 0.20
MOMENTUM_STRENGTH = 0.20
RELATIVE_STRENGTH = 0.30
DRAWDOWN_SENSITIVITY = 0.50
REENTRY_STRENGTH = 0.10
CASH_PENALTY = 0.02

# Full default parameter dictionary.
# Important: even if we tune only 2 parameters, the engine still needs all
# parameters to exist. This avoids missing-key errors.
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

# Small tuning grid.
# We tune only the most interpretable behavioural levers:
# - MOMENTUM_STRENGTH: how much the strategy responds to recent trend
# - CASH_PENALTY: how strongly the engine prefers TLT over idle cash
PARAM_GRID = {
    "MOMENTUM_STRENGTH": [0.20, 0.30],
    "CASH_PENALTY": [0.02, 0.05],
}

# Desired final display order in outputs.
STRATEGY_ORDER = ["Dynamic Strategy", "Static 60/40", "Naive Vol Target", "100% SPY"]


# ================================================================
# HELPER FUNCTIONS
# ================================================================
def clip(value: float, low: float, high: float) -> float:
    """Keep a number inside a valid range."""
    return float(np.clip(value, low, high))



def safe_div(a: float, b: float, fallback: float = 0.0) -> float:
    """
    Safely divide a by b.
    If denominator is too close to zero, return fallback instead.
    """
    return fallback if abs(b) <= 1e-12 else float(a / b)



def normalise_weights(w_spy: float, w_tlt: float, w_cash: float) -> tuple[float, float, float]:
    """
    Make sure weights are:
    1. non-negative
    2. sum to 1
    3. do not exceed the max cash cap

    Why this matters:
    during overlays and adjustments, weights can drift slightly away from
    valid portfolio proportions. This function brings them back into a clean,
    realistic form.
    """
    weights = np.array([max(w_spy, 0.0), max(w_tlt, 0.0), max(w_cash, 0.0)], dtype=float)
    total = weights.sum()
    if total <= 0:
        return 0.0, 0.0, 1.0

    weights = weights / total

    # If cash is too high, redistribute the excess to SPY/TLT proportionally.
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
    """
    Convert a daily return series into annualised compounded return.
    This is the portfolio growth metric used in the KPI summary.
    """
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
    """Annualised Sharpe ratio using annualised return / annualised volatility."""
    ann_ret = annualised_return(daily_returns)
    ann_vol = annualised_volatility(daily_returns)
    if pd.isna(ann_ret) or pd.isna(ann_vol) or ann_vol <= 1e-12:
        return np.nan
    return float((ann_ret - rf_rate) / ann_vol)



def max_drawdown(value_series: pd.Series) -> float:
    """
    Worst peak-to-trough loss in the cumulative portfolio value path.
    Very important for business interpretation because it captures downside pain.
    """
    value_series = value_series.dropna()
    if len(value_series) == 0:
        return np.nan
    running_peak = value_series.cummax()
    return float(((value_series / running_peak) - 1.0).min())



def compute_running_uncertainty_scale(history: list[float], current_value: float) -> float:
    """
    Convert raw RF uncertainty into a rough 0-to-1 scale using only past data.

    Why:
    RF uncertainty here is tree disagreement.
    Higher disagreement means the model is less sure, so the engine should become
    more conservative.
    """
    hist = history + [current_value]
    if len(hist) < 5:
        return 0.5
    p90 = np.percentile(hist, 90)
    if p90 <= 1e-8:
        return 0.0
    return clip(current_value / p90, 0.0, 1.25)



def split_validation_holdout(df: pd.DataFrame, ratio: float = ENGINE_VALIDATION_RATIO) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split predictions chronologically into:
    - validation: for tuning decision-engine parameters
    - holdout: final unseen evaluation period
    """
    split_idx = int(len(df) * ratio)
    if split_idx <= 0 or split_idx >= len(df):
        raise ValueError("Invalid validation / holdout split. Check ENGINE_VALIDATION_RATIO.")
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()



def make_param_combinations(grid: dict) -> list[dict]:
    """Create all parameter combinations from the tuning grid."""
    keys = list(grid.keys())
    vals = list(grid.values())
    return [dict(zip(keys, combo)) for combo in product(*vals)]



def squash_signal(x: float, scale: float = 1.0) -> float:
    """
    Smoothly compress a raw signal into roughly [-1, 1].
    This stops momentum / attractiveness inputs from exploding.
    """
    return float(np.tanh(scale * x))



def clean_strategy_name(name: str) -> str:
    """Standardise strategy labels so outputs and charts stay consistent."""
    mapping = {
        "Dynamic Decision Engine": "Dynamic Strategy",
        "Dynamic Strategy": "Dynamic Strategy",
        "Static 60/40": "Static 60/40",
        "Naive Vol Target": "Naive Vol Target",
        "100% SPY": "100% SPY",
    }
    return mapping.get(name, name)


# ================================================================
# KPI + SCORING
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
    Compute the summary metrics used for evaluation.

    Why this exists:
    The raw backtest gives daily rows, but management and report sections need
    concise strategy-level KPIs.
    """
    returns = df[return_col].dropna()
    values = df[value_col].dropna()

    out = {
        "strategy": clean_strategy_name(strategy_name),
        "n_days": int(len(returns)),
        "annualised_return": annualised_return(returns),
        "annualised_volatility": annualised_volatility(returns),
        "target_vol_gap": abs(annualised_volatility(returns) - TARGET_VOL) if len(returns) > 1 else np.nan,
        "sharpe_ratio": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(values),
        "ending_value": float(values.iloc[-1]) if len(values) > 0 else np.nan,
    }

    # Implementation / business realism metrics.
    out["avg_daily_turnover"] = float(df[turnover_col].mean()) if turnover_col and turnover_col in df.columns else np.nan
    out["total_turnover"] = float(df[turnover_col].sum()) if turnover_col and turnover_col in df.columns else np.nan
    out["avg_daily_cost"] = float(df[cost_col].mean()) if cost_col and cost_col in df.columns else np.nan
    out["total_cost"] = float(df[cost_col].sum()) if cost_col and cost_col in df.columns else np.nan
    out["avg_spy_weight"] = float(df[spy_weight_col].mean()) if spy_weight_col and spy_weight_col in df.columns else np.nan
    out["avg_tlt_weight"] = float(df[tlt_weight_col].mean()) if tlt_weight_col and tlt_weight_col in df.columns else np.nan
    out["avg_cash_weight"] = float(df[cash_weight_col].mean()) if cash_weight_col and cash_weight_col in df.columns else np.nan
    return out



def strategy_score(kpi_row: pd.Series) -> float:
    """
    Scoring rule used during validation tuning.

    Interpretation:
    - rewards better Sharpe and reasonable return
    - rewards closeness to target volatility
    - rewards lower drawdown
    - penalises excessive turnover
    """
    vol_gap = abs(kpi_row["annualised_volatility"] - TARGET_VOL)
    mdd = abs(kpi_row["max_drawdown"])
    sharpe = -999.0 if pd.isna(kpi_row["sharpe_ratio"]) else kpi_row["sharpe_ratio"]
    ann_ret = -999.0 if pd.isna(kpi_row["annualised_return"]) else kpi_row["annualised_return"]
    turnover = 999.0 if pd.isna(kpi_row.get("avg_daily_turnover", np.nan)) else kpi_row.get("avg_daily_turnover", 0.0)
    return float(-2.5 * vol_gap - 1.8 * mdd + 3.5 * sharpe + 2.5 * ann_ret - 0.8 * turnover)


# ================================================================
# CORE DECISION LOGIC
# ================================================================
def compute_momentum_signal(spy_ma5_gap: float, spy_ma20_gap: float, spy_ret_lag1: float, spy_ret_lag5: float) -> float:
    """
    Combine short-term trend indicators into one momentum score for SPY.

    Purpose:
    a portfolio should not only react to risk, but also recognize when trend is
    favourable and allow more participation.
    """
    raw = 0.35 * spy_ma5_gap + 0.35 * spy_ma20_gap + 8.0 * spy_ret_lag1 + 4.0 * spy_ret_lag5
    return squash_signal(raw, scale=3.0)



def compute_relative_attractiveness(spy_pred_vol: float, tlt_pred_vol: float, spy_unc_scaled: float, tlt_unc_scaled: float) -> float:
    """
    Compare SPY vs TLT on a risk-adjusted basis.

    Positive result: SPY looks more attractive.
    Negative result: TLT looks more attractive.
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
    This is the heart of the strategy.

    It converts today's forecasts and portfolio state into desired target weights.

    The logic is layered intentionally:
    1. Start with volatility targeting
    2. Reduce aggression if model uncertainty is high
    3. Compare SPY vs TLT attractiveness
    4. Adjust for momentum
    5. Apply stress and drawdown safeguards
    6. Allow re-entry when conditions improve
    7. Split defensive capital between TLT and cash
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

    # 1) Base SPY allocation from volatility targeting.
    # If predicted SPY vol is above target, reduce SPY weight.
    spy_base = clip(TARGET_VOL / max(spy_pred_vol, 1e-8), MIN_SPY_WEIGHT, MAX_SPY_WEIGHT)

    # 2) Uncertainty overlay.
    # If RF trees disagree more, trust the forecast less and reduce SPY more.
    spy_after_unc = clip(spy_base * (1.0 - spy_unc_penalty * spy_unc_scaled), MIN_SPY_WEIGHT, MAX_SPY_WEIGHT)

    # 3) Relative attractiveness overlay.
    rel_attr = compute_relative_attractiveness(spy_pred_vol, tlt_pred_vol, spy_unc_scaled, tlt_unc_scaled)
    spy_after_relative = clip(spy_after_unc * (1.0 + relative_strength * rel_attr), MIN_SPY_WEIGHT, MAX_SPY_WEIGHT)

    # 4) Momentum overlay.
    momentum_signal = compute_momentum_signal(
        spy_ma5_gap=float(row.get("spy_price_ma5_gap", 0.0)),
        spy_ma20_gap=float(row.get("spy_price_ma20_gap", 0.0)),
        spy_ret_lag1=float(row.get("spy_ret_lag1", 0.0)),
        spy_ret_lag5=float(row.get("spy_ret_lag5", 0.0)),
    )
    spy_after_momentum = clip(spy_after_relative * (1.0 + momentum_strength * momentum_signal), MIN_SPY_WEIGHT, MAX_SPY_WEIGHT)

    # 5) Stress overlay.
    # If market risk is very high, hard-cap SPY exposure.
    if spy_pred_vol > STRESS_VOL_THRESHOLD:
        spy_after_momentum = min(spy_after_momentum, STRESS_MAX_SPY_WEIGHT)

    # 6) Drawdown overlay.
    # If the portfolio is already in drawdown, scale back risk further.
    if abs(current_drawdown) > DRAWDOWN_TRIGGER:
        dd_excess = abs(current_drawdown) - DRAWDOWN_TRIGGER
        dd_scale = clip(1.0 - drawdown_sensitivity * dd_excess, 0.65, 1.0)
        spy_after_momentum *= dd_scale

    # 7) Re-entry overlay.
    # If volatility improves and trend is positive, let SPY come back up gradually.
    if (spy_pred_vol < TARGET_VOL + RECOVERY_VOL_BUFFER) and (momentum_signal > 0):
        spy_after_momentum *= 1.0 + reentry_strength * momentum_signal

    spy_target = clip(spy_after_momentum, MIN_SPY_WEIGHT, MAX_SPY_WEIGHT)

    # Defensive bucket = everything not allocated to SPY.
    defensive_bucket = 1.0 - spy_target

    # Allocate defensive bucket between TLT and cash.
    rel_tlt_risk = safe_div(tlt_pred_vol, spy_pred_vol, fallback=1.0)
    base_tlt_share = clip(1.10 - 0.55 * rel_tlt_risk, 0.25, 0.90)
    tlt_share_after_unc = clip(base_tlt_share * (1.0 - tlt_unc_penalty * tlt_unc_scaled), 0.0, 1.0)
    tlt_share_after_unc = max(tlt_share_after_unc, MIN_TLT_SHARE_OF_DEFENSIVE)

    # Cash penalty makes the engine less willing to hide in cash too much.
    tlt_share_after_unc = clip(tlt_share_after_unc + cash_penalty, 0.0, 1.0)

    w_tlt_target = clip(defensive_bucket * tlt_share_after_unc, MIN_TLT_WEIGHT, MAX_TLT_WEIGHT)
    w_cash_target = max(1.0 - spy_target - w_tlt_target, MIN_CASH_WEIGHT)
    return normalise_weights(spy_target, w_tlt_target, w_cash_target)



def apply_rebalancing_rules(prev_weights: tuple[float, float, float], target_weights: tuple[float, float, float], params: dict) -> tuple[float, float, float, float]:
    """
    Convert target weights into executed weights.

    Why this step exists:
    in real life, portfolios do not jump instantly to target every day because:
    - small trades create unnecessary cost
    - smoother rebalancing is more realistic
    """
    rebalance_threshold = params["REBALANCE_THRESHOLD"]
    smoothing_lambda = params["SMOOTHING_LAMBDA"]

    prev_spy, prev_tlt, prev_cash = prev_weights
    tgt_spy, tgt_tlt, tgt_cash = target_weights
    max_change = max(abs(tgt_spy - prev_spy), abs(tgt_tlt - prev_tlt), abs(tgt_cash - prev_cash))

    # If change is too small, do nothing.
    if max_change < rebalance_threshold:
        new_weights = (prev_spy, prev_tlt, prev_cash)
    else:
        # Otherwise move only partially toward the target.
        new_spy = prev_spy + smoothing_lambda * (tgt_spy - prev_spy)
        new_tlt = prev_tlt + smoothing_lambda * (tgt_tlt - prev_tlt)
        new_cash = prev_cash + smoothing_lambda * (tgt_cash - prev_cash)
        new_weights = normalise_weights(new_spy, new_tlt, new_cash)

    # Standard turnover convention.
    turnover = 0.5 * (
        abs(new_weights[0] - prev_spy) + abs(new_weights[1] - prev_tlt) + abs(new_weights[2] - prev_cash)
    )
    return new_weights[0], new_weights[1], new_weights[2], float(turnover)


# ================================================================
# STRATEGY RUNNERS
# ================================================================
def run_dynamic_strategy(pred_df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Run the proposed dynamic strategy on a set of prediction rows.

    Timing assumption:
    - use information on date t
    - decide weights on date t
    - earn next-day return from those weights
    """
    df = pred_df.copy().sort_values("date").reset_index(drop=True)

    required_cols = [
        "date", "spy_calibrated_rf_pred", "tlt_calibrated_rf_pred",
        "spy_rf_uncertainty", "tlt_rf_uncertainty", "spy_next_day_return", "tlt_next_day_return",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"predictions.csv is missing required columns: {missing}")

    # Some momentum features may be missing depending on file version.
    # If so, default to zero so the engine still runs safely.
    for col in ["spy_price_ma5_gap", "spy_price_ma20_gap", "spy_ret_lag1", "spy_ret_lag5"]:
        if col not in df.columns:
            df[col] = 0.0

    # Start from the static 60/40 mix.
    w_spy = STATIC_SPY_WEIGHT
    w_tlt = STATIC_TLT_WEIGHT
    w_cash = max(0.0, 1.0 - w_spy - w_tlt)
    portfolio_value = 1.0
    running_peak = 1.0
    spy_unc_history: list[float] = []
    tlt_unc_history: list[float] = []
    records = []

    for _, row in df.iterrows():
        # Convert today's raw uncertainty into scaled uncertainty.
        spy_unc = float(row["spy_rf_uncertainty"])
        tlt_unc = float(row["tlt_rf_uncertainty"])
        spy_unc_scaled = compute_running_uncertainty_scale(spy_unc_history, spy_unc)
        tlt_unc_scaled = compute_running_uncertainty_scale(tlt_unc_history, tlt_unc)
        current_drawdown = (portfolio_value / running_peak) - 1.0

        # Step 1: decide desired target weights.
        tgt_spy, tgt_tlt, tgt_cash = compute_target_weights(row, spy_unc_scaled, tlt_unc_scaled, current_drawdown, params)

        # Step 2: convert target weights to executed weights with turnover logic.
        new_spy, new_tlt, new_cash, turnover = apply_rebalancing_rules((w_spy, w_tlt, w_cash), (tgt_spy, tgt_tlt, tgt_cash), params)

        # Step 3: apply transaction costs.
        transaction_cost = turnover * TRANSACTION_COST_RATE

        # Step 4: compute gross and net portfolio return.
        gross_return = new_spy * float(row["spy_next_day_return"]) + new_tlt * float(row["tlt_next_day_return"])
        net_return = gross_return - transaction_cost

        # Step 5: update portfolio value and drawdown.
        portfolio_value *= (1.0 + net_return)
        running_peak = max(running_peak, portfolio_value)
        drawdown = (portfolio_value / running_peak) - 1.0

        # Save all useful diagnostics.
        records.append({
            "date": row["date"],
            "strategy": "Dynamic Strategy",
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
            "drawdown": drawdown,
        })

        # Carry weights and uncertainty history forward.
        w_spy, w_tlt, w_cash = new_spy, new_tlt, new_cash
        spy_unc_history.append(spy_unc)
        tlt_unc_history.append(tlt_unc)

    return pd.DataFrame(records)



def run_static_6040_benchmark(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Run a plain fixed 60/40 benchmark with no dynamic logic."""
    df = pred_df.copy().sort_values("date").reset_index(drop=True)
    w_spy, w_tlt, w_cash = STATIC_SPY_WEIGHT, STATIC_TLT_WEIGHT, max(0.0, 1.0 - STATIC_SPY_WEIGHT - STATIC_TLT_WEIGHT)
    portfolio_value = 1.0
    running_peak = 1.0
    records = []

    for _, row in df.iterrows():
        gross_return = w_spy * float(row["spy_next_day_return"]) + w_tlt * float(row["tlt_next_day_return"])
        net_return = gross_return
        portfolio_value *= (1.0 + net_return)
        running_peak = max(running_peak, portfolio_value)
        drawdown = (portfolio_value / running_peak) - 1.0
        records.append({
            "date": row["date"],
            "strategy": "Static 60/40",
            "w_spy": w_spy,
            "w_tlt": w_tlt,
            "w_cash": w_cash,
            "turnover": 0.0,
            "transaction_cost": 0.0,
            "gross_return": gross_return,
            "net_return": net_return,
            "portfolio_value": portfolio_value,
            "drawdown": drawdown,
        })
    return pd.DataFrame(records)



def run_static_spy_benchmark(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Run a 100% SPY benchmark."""
    df = pred_df.copy().sort_values("date").reset_index(drop=True)
    portfolio_value = 1.0
    running_peak = 1.0
    records = []

    for _, row in df.iterrows():
        gross_return = float(row["spy_next_day_return"])
        net_return = gross_return
        portfolio_value *= (1.0 + net_return)
        running_peak = max(running_peak, portfolio_value)
        drawdown = (portfolio_value / running_peak) - 1.0
        records.append({
            "date": row["date"],
            "strategy": "100% SPY",
            "w_spy": 1.0,
            "w_tlt": 0.0,
            "w_cash": 0.0,
            "turnover": 0.0,
            "transaction_cost": 0.0,
            "gross_return": gross_return,
            "net_return": net_return,
            "portfolio_value": portfolio_value,
            "drawdown": drawdown,
        })
    return pd.DataFrame(records)



def run_naive_vol_target_benchmark(pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run a simple benchmark that uses the same SPY volatility forecast,
    but without uncertainty, momentum, drawdown, or relative-attractiveness logic.

    Purpose:
    shows whether the full decision engine adds value beyond a much simpler
    rule-based system.
    """
    df = pred_df.copy().sort_values("date").reset_index(drop=True)
    portfolio_value = 1.0
    running_peak = 1.0
    prev_spy, prev_tlt, prev_cash = STATIC_SPY_WEIGHT, STATIC_TLT_WEIGHT, max(0.0, 1.0 - STATIC_SPY_WEIGHT - STATIC_TLT_WEIGHT)
    records = []

    for _, row in df.iterrows():
        spy_pred = float(row["spy_calibrated_rf_pred"])

        # Simple rule: risk-target SPY directly, put the remainder into TLT.
        w_spy = clip(TARGET_VOL / max(spy_pred, 1e-8), 0.0, 0.90)
        w_tlt = 1.0 - w_spy
        w_cash = 0.0
        w_spy, w_tlt, w_cash = normalise_weights(w_spy, w_tlt, w_cash)

        turnover = 0.5 * (abs(w_spy - prev_spy) + abs(w_tlt - prev_tlt) + abs(w_cash - prev_cash))
        transaction_cost = turnover * TRANSACTION_COST_RATE
        gross_return = w_spy * float(row["spy_next_day_return"]) + w_tlt * float(row["tlt_next_day_return"])
        net_return = gross_return - transaction_cost
        portfolio_value *= (1.0 + net_return)
        running_peak = max(running_peak, portfolio_value)
        drawdown = (portfolio_value / running_peak) - 1.0

        records.append({
            "date": row["date"],
            "strategy": "Naive Vol Target",
            "spy_calibrated_rf_pred": spy_pred,
            "w_spy": w_spy,
            "w_tlt": w_tlt,
            "w_cash": w_cash,
            "turnover": turnover,
            "transaction_cost": transaction_cost,
            "gross_return": gross_return,
            "net_return": net_return,
            "portfolio_value": portfolio_value,
            "drawdown": drawdown,
        })

        prev_spy, prev_tlt, prev_cash = w_spy, w_tlt, w_cash

    return pd.DataFrame(records)


# ================================================================
# TUNING
# ================================================================
def tune_engine(validation_df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """
    Tune only the selected parameters on the validation period.

    Important idea:
    the engine is not tuned on the final holdout period.
    This keeps the final backtest more honest.
    """
    combos = make_param_combinations(PARAM_GRID)
    results = []
    print(f"\nTuning decision engine over {len(combos)} parameter combinations...")

    for i, tuned_params in enumerate(combos, start=1):
        print(f"  Progress: {i}/{len(combos)}")
        params = DEFAULT_PARAMS.copy()
        params.update(tuned_params)
        try:
            strat_df = run_dynamic_strategy(validation_df, params)
            kpis = compute_strategy_kpis(
                strat_df,
                strategy_name="Dynamic Strategy",
                return_col="net_return",
                value_col="portfolio_value",
                turnover_col="turnover",
                cost_col="transaction_cost",
                spy_weight_col="w_spy",
                tlt_weight_col="w_tlt",
                cash_weight_col="w_cash",
            )
            result_row = dict(params)
            result_row.update(kpis)
            result_row["score"] = strategy_score(pd.Series(kpis))
            results.append(result_row)
        except Exception as e:
            result_row = dict(params)
            result_row["strategy"] = "Dynamic Strategy"
            result_row["score"] = -999999.0
            result_row["error"] = str(e)
            results.append(result_row)

    results_df = pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)
    if results_df.empty:
        raise ValueError("No tuning results generated.")

    best_row = results_df.iloc[0]
    best_params = DEFAULT_PARAMS.copy()
    for k in PARAM_GRID.keys():
        best_params[k] = best_row[k]
    return best_params, results_df


# ================================================================
# MAIN
# ================================================================
def save_strategy_breakouts(portfolio_results_df: pd.DataFrame) -> None:
    """Save one CSV per strategy so inspection is easier than using only one stacked master file."""
    STRATEGY_BREAKOUT_DIR.mkdir(parents=True, exist_ok=True)
    for strategy, sub in portfolio_results_df.groupby("strategy"):
        slug = strategy.lower().replace("%", "pct").replace("/", "_").replace(" ", "_")
        sub.sort_values("date").to_csv(STRATEGY_BREAKOUT_DIR / f"portfolio_results_{slug}.csv", index=False)



def main() -> None:
    """
    Run the full decision-engine pipeline:
    1. load model predictions
    2. split into validation/holdout
    3. tune decision-engine parameters on validation
    4. run final strategies on holdout
    5. save results and KPIs
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 70)
    print("DECISION ENGINE")
    print("=" * 70)

    # Load forecasting outputs from modelling stage.
    pred_df = pd.read_csv(PREDICTIONS_FILE, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    print(f"Loaded predictions: {len(pred_df)} rows from {PREDICTIONS_FILE}")

    # Split into validation and holdout for the decision-engine stage.
    validation_df, holdout_df = split_validation_holdout(pred_df)
    print(f"Validation rows: {len(validation_df)}")
    print(f"Holdout rows:    {len(holdout_df)}")

    # Tune on validation only.
    best_params, search_results_df = tune_engine(validation_df)
    search_results_df.to_csv(SEARCH_RESULTS_FILE, index=False)
    with open(BEST_ENGINE_PARAMS_FILE, "w") as f:
        json.dump(best_params, f, indent=2)

    print("\nBest engine parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    # Final honest evaluation on holdout.
    dynamic_df = run_dynamic_strategy(holdout_df, best_params)
    static_6040_df = run_static_6040_benchmark(holdout_df)
    naive_vol_df = run_naive_vol_target_benchmark(holdout_df)
    output_frames = [dynamic_df, static_6040_df, naive_vol_df]
    if INCLUDE_SPY_ONLY_BENCHMARK:
        spy_df = run_static_spy_benchmark(holdout_df)
        output_frames.append(spy_df)

    # One combined long-format table for charts and grouped analysis.
    portfolio_results_df = pd.concat(output_frames, ignore_index=True)
    portfolio_results_df["strategy"] = portfolio_results_df["strategy"].apply(clean_strategy_name)
    portfolio_results_df["strategy"] = pd.Categorical(portfolio_results_df["strategy"], categories=STRATEGY_ORDER, ordered=True)
    portfolio_results_df = portfolio_results_df.sort_values(["strategy", "date"]).reset_index(drop=True)
    portfolio_results_df["strategy"] = portfolio_results_df["strategy"].astype(str)

    # Compute one KPI row per strategy.
    kpi_rows = []
    for strategy, sub in portfolio_results_df.groupby("strategy"):
        kpi_rows.append(
            compute_strategy_kpis(
                sub,
                strategy_name=strategy,
                return_col="net_return",
                value_col="portfolio_value",
                turnover_col="turnover",
                cost_col="transaction_cost",
                spy_weight_col="w_spy",
                tlt_weight_col="w_tlt",
                cash_weight_col="w_cash",
            )
        )
    kpis_df = pd.DataFrame(kpi_rows)
    kpis_df["strategy"] = pd.Categorical(kpis_df["strategy"], categories=STRATEGY_ORDER, ordered=True)
    kpis_df = kpis_df.sort_values("strategy").reset_index(drop=True)
    kpis_df["strategy"] = kpis_df["strategy"].astype(str)

    # Save outputs.
    portfolio_results_df.to_csv(PORTFOLIO_RESULTS_FILE, index=False)
    kpis_df.to_csv(PORTFOLIO_KPIS_FILE, index=False)
    save_strategy_breakouts(portfolio_results_df)

    print("\nSaved:")
    print(f"  Search results:    {SEARCH_RESULTS_FILE}")
    print(f"  Best params:       {BEST_ENGINE_PARAMS_FILE}")
    print(f"  Portfolio results: {PORTFOLIO_RESULTS_FILE}")
    print(f"  Portfolio KPIs:    {PORTFOLIO_KPIS_FILE}")
    print(f"  Breakout CSVs:     {STRATEGY_BREAKOUT_DIR}")

    print("\nHoldout KPI summary:")
    print(kpis_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
