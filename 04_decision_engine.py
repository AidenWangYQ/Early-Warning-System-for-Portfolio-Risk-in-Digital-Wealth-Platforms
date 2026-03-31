"""
04_decision_engine.py

Upgraded from the teammate R logic:
- keep the simple threshold-based warning rule
- but make it a true SPY/TLT allocation strategy instead of scaling SPY alone

This file is intentionally a baseline portfolio layer so it is easy to extend later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    PREDICTIONS_FILE,
    PORTFOLIO_RESULTS_FILE,
    STATIC_SPY_WEIGHT,
    STATIC_TLT_WEIGHT,
    RISK_OFF_SPY_WEIGHT,
    RISK_OFF_TLT_WEIGHT,
    RISK_ON_SPY_WEIGHT,
    RISK_ON_TLT_WEIGHT,
    BOTH_HIGH_SPY_WEIGHT,
    BOTH_HIGH_TLT_WEIGHT,
    WARNING_THRESHOLD,
    TRANSACTION_COST_BPS,
)


MODEL_NAMES = ["baseline", "ets", "arima", "garch", "rf"]


def load_predictions() -> pd.DataFrame:
    return pd.read_csv(PREDICTIONS_FILE, parse_dates=["date"]).sort_values("date").reset_index(drop=True)


def choose_weights(spy_pred_vol: float, tlt_pred_vol: float) -> tuple[float, float]:
    spy_high = spy_pred_vol > WARNING_THRESHOLD
    tlt_high = tlt_pred_vol > WARNING_THRESHOLD

    if spy_high and not tlt_high:
        return RISK_OFF_SPY_WEIGHT, RISK_OFF_TLT_WEIGHT
    if (not spy_high) and tlt_high:
        return RISK_ON_SPY_WEIGHT, RISK_ON_TLT_WEIGHT
    if spy_high and tlt_high:
        return BOTH_HIGH_SPY_WEIGHT, BOTH_HIGH_TLT_WEIGHT
    return STATIC_SPY_WEIGHT, STATIC_TLT_WEIGHT


def compute_strategy_returns(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    out = df.copy()

    spy_pred_col = f"spy_{model_name}_pred"
    tlt_pred_col = f"tlt_{model_name}_pred"

    spy_weights = []
    tlt_weights = []
    turnovers = []
    costs = []

    prev_spy_w = STATIC_SPY_WEIGHT
    prev_tlt_w = STATIC_TLT_WEIGHT

    for _, row in out.iterrows():
        spy_w, tlt_w = choose_weights(row[spy_pred_col], row[tlt_pred_col])
        turnover = abs(spy_w - prev_spy_w) + abs(tlt_w - prev_tlt_w)
        cost = turnover * (TRANSACTION_COST_BPS / 10000.0)

        spy_weights.append(spy_w)
        tlt_weights.append(tlt_w)
        turnovers.append(turnover)
        costs.append(cost)

        prev_spy_w, prev_tlt_w = spy_w, tlt_w

    out[f"{model_name}_spy_weight"] = spy_weights
    out[f"{model_name}_tlt_weight"] = tlt_weights
    out[f"{model_name}_turnover"] = turnovers
    out[f"{model_name}_transaction_cost"] = costs

    # Use next-day returns so today's forecast drives tomorrow's portfolio
    gross_ret = (
        out[f"{model_name}_spy_weight"] * out["spy_next_day_return"] +
        out[f"{model_name}_tlt_weight"] * out["tlt_next_day_return"]
    )
    out[f"{model_name}_gross_return"] = gross_ret
    out[f"{model_name}_net_return"] = gross_ret - out[f"{model_name}_transaction_cost"]

    return out


def main() -> None:
    df = load_predictions()

    # Static benchmark 60/40
    df["static_spy_weight"] = STATIC_SPY_WEIGHT
    df["static_tlt_weight"] = STATIC_TLT_WEIGHT
    df["static_turnover"] = 0.0
    df["static_transaction_cost"] = 0.0
    df["static_gross_return"] = (
        STATIC_SPY_WEIGHT * df["spy_next_day_return"] +
        STATIC_TLT_WEIGHT * df["tlt_next_day_return"]
    )
    df["static_net_return"] = df["static_gross_return"]

    for model_name in MODEL_NAMES:
        df = compute_strategy_returns(df, model_name)

    # Drop last row if next-day return missing
    df = df.dropna(subset=["spy_next_day_return", "tlt_next_day_return"]).reset_index(drop=True)
    df.to_csv(PORTFOLIO_RESULTS_FILE, index=False)

    print("=" * 60)
    print("DECISION ENGINE COMPLETE")
    print("=" * 60)
    print(f"Saved: {PORTFOLIO_RESULTS_FILE}")


if __name__ == "__main__":
    main()
