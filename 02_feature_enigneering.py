"""
02_feature_engineering.py
Build asset-specific modelling datasets using notebook-style features:
- lagged returns
- rolling vols
- skewness
- MA gaps
- range / volume change
- drawdowns
- VIX features
- SPY-TLT rolling correlation
Targets:
- next 5-day realised volatility (annualised)
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd

from config import (
    MASTER_DATASET_FILE,
    SPY_MODEL_DATASET_FILE,
    TLT_MODEL_DATASET_FILE,
    OUTPUT_DIR,
)


def load_master_dataset() -> pd.DataFrame:
    df = pd.read_csv(MASTER_DATASET_FILE, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def compute_cross_asset_features(master_df: pd.DataFrame) -> pd.Series:
    spy_ret = np.log(master_df["spy_adj_close"] / master_df["spy_adj_close"].shift(1))
    tlt_ret = np.log(master_df["tlt_adj_close"] / master_df["tlt_adj_close"].shift(1))
    return spy_ret.rolling(20).corr(tlt_ret)


def build_features_and_target(
    asset_df: pd.DataFrame,
    vix_series: pd.Series,
    spy_tlt_corr: pd.Series,
    asset_name: str,
) -> Tuple[pd.DataFrame, List[str]]:
    data = asset_df.copy().sort_values("date").reset_index(drop=True)

    # Daily log returns
    data["return"] = np.log(data["adj_close"] / data["adj_close"].shift(1))

    # Forward target: next 5-day realised volatility (annualised)
    future_vol = data["return"].rolling(window=5).std().shift(-4)
    data["target_vol_5d"] = future_vol * np.sqrt(252)

    # Backtest next-day return
    data["next_day_return"] = data["return"].shift(-1)

    # Lagged returns
    for lag in range(1, 6):
        data[f"ret_lag{lag}"] = data["return"].shift(lag)

    # Rolling volatility and skewness
    data["vol_5d"] = data["return"].rolling(5).std() * np.sqrt(252)
    data["vol_10d"] = data["return"].rolling(10).std() * np.sqrt(252)
    data["vol_20d"] = data["return"].rolling(20).std() * np.sqrt(252)
    data["skew_20d"] = data["return"].rolling(20).skew()

    # Trend / moving-average gaps
    data["ma_5"] = data["adj_close"].rolling(5).mean()
    data["ma_20"] = data["adj_close"].rolling(20).mean()
    data["price_ma5_gap"] = (data["adj_close"] - data["ma_5"]) / data["ma_5"]
    data["price_ma20_gap"] = (data["adj_close"] - data["ma_20"]) / data["ma_20"]

    # Intraday range
    data["range_pct"] = (data["high"] - data["low"]) / data["adj_close"]

    # Volume change
    data["vol_change"] = np.log(data["volume"] / data["volume"].shift(1))

    # Drawdowns
    data["max_5d"] = data["adj_close"].rolling(5).max()
    data["drawdown_5d"] = (data["adj_close"] - data["max_5d"]) / data["max_5d"]
    data["max_20d"] = data["adj_close"].rolling(20).max()
    data["drawdown_20d"] = (data["adj_close"] - data["max_20d"]) / data["max_20d"]

    # VIX features
    data["vix_level"] = vix_series.values
    data["vix_change_1d"] = data["vix_level"].pct_change(1)
    data["vix_change_5d"] = data["vix_level"].pct_change(5)
    data["vix_ma5"] = data["vix_level"].rolling(5).mean()
    data["vix_above_20"] = (data["vix_level"] > 20).astype(int)
    data["vix_above_25"] = (data["vix_level"] > 25).astype(int)

    # Cross-asset feature
    data["spy_tlt_corr_20d"] = spy_tlt_corr.values

    feature_cols = [
        "ret_lag1", "ret_lag2", "ret_lag3", "ret_lag4", "ret_lag5",
        "vol_5d", "vol_10d", "vol_20d", "skew_20d",
        "price_ma5_gap", "price_ma20_gap",
        "range_pct", "vol_change",
        "drawdown_5d", "drawdown_20d",
        "vix_level", "vix_change_1d", "vix_change_5d",
        "vix_ma5", "vix_above_20", "vix_above_25",
        "spy_tlt_corr_20d",
    ]

    keep_cols = ["date", "return", "next_day_return", "target_vol_5d"] + feature_cols
    model_df = data[keep_cols].dropna().reset_index(drop=True)

    print(f"{asset_name}: {len(model_df)} usable rows")
    print(f"  date range: {model_df['date'].min().date()} -> {model_df['date'].max().date()}")
    print(f"  target mean={model_df['target_vol_5d'].mean():.4f}, std={model_df['target_vol_5d'].std():.4f}")

    return model_df, feature_cols


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    master = load_master_dataset()
    spy_tlt_corr = compute_cross_asset_features(master)

    spy_df = master[["date", "spy_open", "spy_high", "spy_low", "spy_close", "spy_volume", "spy_adj_close"]].rename(
        columns={
            "spy_open": "open",
            "spy_high": "high",
            "spy_low": "low",
            "spy_close": "close",
            "spy_volume": "volume",
            "spy_adj_close": "adj_close",
        }
    )
    tlt_df = master[["date", "tlt_open", "tlt_high", "tlt_low", "tlt_adj_close", "tlt_volume"]].rename(
        columns={
            "tlt_open": "open",
            "tlt_high": "high",
            "tlt_low": "low",
            "tlt_adj_close": "adj_close",
            "tlt_volume": "volume",
        }
    )
    # TLT may not have a separate 'close'. Use adj_close for range denominator.
    tlt_df["close"] = tlt_df["adj_close"]

    spy_model, feature_cols = build_features_and_target(spy_df, master["vix_close"], spy_tlt_corr, "SPY")
    tlt_model, _ = build_features_and_target(tlt_df, master["vix_close"], spy_tlt_corr, "TLT")

    spy_model.to_csv(SPY_MODEL_DATASET_FILE, index=False)
    tlt_model.to_csv(TLT_MODEL_DATASET_FILE, index=False)

    print("=" * 60)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 60)
    print(f"Saved: {SPY_MODEL_DATASET_FILE}")
    print(f"Saved: {TLT_MODEL_DATASET_FILE}")
    print(f"Features ({len(feature_cols)}): {feature_cols}")


if __name__ == "__main__":
    main()
