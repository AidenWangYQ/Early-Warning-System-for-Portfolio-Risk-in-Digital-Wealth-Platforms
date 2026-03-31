"""
02_feature_engineering.py

Build asset-specific modelling datasets from the master dataset.

This script engineers predictive features for each asset (SPY and TLT),
including:
- lagged returns
- rolling volatility
- skewness
- moving-average gaps
- intraday range and volume change
- drawdowns
- VIX-based market stress features
- SPY-TLT rolling correlation

It also creates the modelling target:
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
    """
    Load the master dataset and sort it chronologically.

    Returns:
        pd.DataFrame: Master dataset containing aligned SPY, TLT, and VIX data.
    """
    df = pd.read_csv(MASTER_DATASET_FILE, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def compute_cross_asset_features(master_df: pd.DataFrame) -> pd.Series:
    """
    Compute the 20-day rolling correlation between SPY and TLT log returns.

    This feature captures the evolving relationship between equities (SPY)
    and bonds (TLT). It may help identify changing market regimes, such as
    risk-on / risk-off behaviour or periods of market stress.

    Parameters:
        master_df (pd.DataFrame): DataFrame containing:
            - 'spy_adj_close'
            - 'tlt_adj_close'

    Returns:
        pd.Series: 20-day rolling correlation between SPY and TLT returns.
                   The first observations will be NaN due to return and
                   rolling-window calculations.
    """
    # Compute daily log returns for SPY and TLT
    spy_ret = (master_df["spy_adj_close"] / master_df["spy_adj_close"].shift(1)).apply(np.log)
    tlt_ret = (master_df["tlt_adj_close"] / master_df["tlt_adj_close"].shift(1)).apply(np.log)

    # Measure how strongly SPY and TLT have moved together over the past 20 days
    return spy_ret.rolling(20).corr(tlt_ret)


def build_features_and_target(
    asset_df: pd.DataFrame,
    vix_series: pd.Series,
    spy_tlt_corr: pd.Series,
    asset_name: str,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create the full modelling dataset for a single asset.

    This function:
    1. computes the asset's daily return series
    2. constructs the prediction target (next 5-day realised volatility)
    3. engineers technical, volatility, drawdown, VIX, and cross-asset features
    4. removes rows with missing values caused by rolling windows / lagging

    Parameters:
        asset_df (pd.DataFrame): Asset-level OHLCV data with columns such as
            'date', 'high', 'low', 'volume', and 'adj_close'
        vix_series (pd.Series): VIX closing levels aligned by date
        spy_tlt_corr (pd.Series): 20-day rolling SPY-TLT return correlation
        asset_name (str): Asset label used for logging output

    Returns:
        Tuple[pd.DataFrame, List[str]]:
            - model_df: final cleaned modelling dataset
            - feature_cols: list of engineered predictor column names
    """
    # Work on a copy to avoid mutating the original input
    data = asset_df.copy().sort_values("date").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Return series and targets
    # ------------------------------------------------------------------

    # Daily log return for the asset
    data["return"] = np.log(data["adj_close"] / data["adj_close"].shift(1))

    # Prediction target:
    # realised volatility over the NEXT 5 trading days, annualised
    #
    # Steps:
    # - rolling(5).std() computes 5-day realised volatility using returns
    # - shift(-4) aligns that future volatility with today's row, so today's
    #   features predict the next 5-day volatility period
    data["target_vol_5d"] = data["return"].rolling(window=5).std().shift(-4) * np.sqrt(252)

    # Optional backtesting field:
    # tomorrow's return, useful later if you want to test directional strategies
    data["next_day_return"] = data["return"].shift(-1)

    # ------------------------------------------------------------------
    # Lagged return features
    # ------------------------------------------------------------------

    # Include the last 5 daily returns as short-term momentum / reversal signals
    for lag in range(1, 6):
        data[f"ret_lag{lag}"] = data["return"].shift(lag)

    # ------------------------------------------------------------------
    # Rolling distribution / volatility features
    # ------------------------------------------------------------------

    # Recent annualised volatility over different lookback windows
    data["vol_5d"] = data["return"].rolling(5).std() * np.sqrt(252)
    data["vol_10d"] = data["return"].rolling(10).std() * np.sqrt(252)
    data["vol_20d"] = data["return"].rolling(20).std() * np.sqrt(252)

    # 20-day rolling skewness of returns
    # This captures asymmetry in the recent return distribution
    data["skew_20d"] = data["return"].rolling(20).skew()

    # ------------------------------------------------------------------
    # Trend / moving-average features
    # ------------------------------------------------------------------

    # Short-term and medium-term moving averages of adjusted close price
    data["ma_5"] = data["adj_close"].rolling(5).mean()
    data["ma_20"] = data["adj_close"].rolling(20).mean()

    # Relative gap between price and moving averages
    # Positive values suggest price is above trend; negative values suggest below
    data["price_ma5_gap"] = (data["adj_close"] - data["ma_5"]) / data["ma_5"]
    data["price_ma20_gap"] = (data["adj_close"] - data["ma_20"]) / data["ma_20"]

    # ------------------------------------------------------------------
    # Intraday activity features
    # ------------------------------------------------------------------

    # Intraday trading range scaled by adjusted close
    # This is a simple proxy for daily price dispersion / turbulence
    data["range_pct"] = (data["high"] - data["low"]) / data["adj_close"]

    # Log change in trading volume from the previous day
    # Helps capture unusual surges or drops in market activity
    data["vol_change"] = np.log(data["volume"] / data["volume"].shift(1))

    # ------------------------------------------------------------------
    # Drawdown features
    # ------------------------------------------------------------------

    # Rolling recent peaks over short and medium horizons
    data["max_5d"] = data["adj_close"].rolling(5).max()
    data["max_20d"] = data["adj_close"].rolling(20).max()

    # Percentage drawdown from recent peak
    # More negative values indicate the asset has fallen further below its
    # recent high
    data["drawdown_5d"] = (data["adj_close"] - data["max_5d"]) / data["max_5d"]
    data["drawdown_20d"] = (data["adj_close"] - data["max_20d"]) / data["max_20d"]

    # ------------------------------------------------------------------
    # VIX-based market stress features
    # ------------------------------------------------------------------

    # Current VIX level aligned to each date
    data["vix_level"] = vix_series.values

    # Short-horizon changes in VIX to capture shifts in market fear
    data["vix_change_1d"] = data["vix_level"].pct_change(1)
    data["vix_change_5d"] = data["vix_level"].pct_change(5)

    # Smoothed VIX level over the past 5 days
    data["vix_ma5"] = data["vix_level"].rolling(5).mean()

    # Simple stress regime flags using common VIX thresholds
    data["vix_above_20"] = (data["vix_level"] > 20).astype(int)
    data["vix_above_25"] = (data["vix_level"] > 25).astype(int)

    # ------------------------------------------------------------------
    # Cross-asset market regime feature
    # ------------------------------------------------------------------

    # Add the rolling stock-bond correlation feature computed from the master set
    data["spy_tlt_corr_20d"] = spy_tlt_corr.values

    # Final list of predictors used for modelling
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

    # Keep only the fields needed for modelling and evaluation
    keep_cols = ["date", "return", "next_day_return", "target_vol_5d"] + feature_cols

    # Drop rows with missing values caused by:
    # - shifted returns
    # - rolling-window statistics
    # - future target alignment
    model_df = data[keep_cols].dropna().reset_index(drop=True)

    # Print a quick summary for verification / sanity checking
    print(f"{asset_name}: {len(model_df)} usable rows")
    print(f"  date range: {model_df['date'].min().date()} -> {model_df['date'].max().date()}")
    print(f"  target mean={model_df['target_vol_5d'].mean():.4f}, std={model_df['target_vol_5d'].std():.4f}")

    return model_df, feature_cols


def main() -> None:
    """
    Main pipeline for feature engineering.

    Workflow:
    1. load the master aligned dataset
    2. compute cross-asset features
    3. split the master data into SPY and TLT asset-level tables
    4. build modelling datasets for each asset
    5. save the final feature-engineered datasets to disk
    """
    # Ensure output folder exists before saving files
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load combined dataset and compute shared cross-asset signal
    master = load_master_dataset()
    spy_tlt_corr = compute_cross_asset_features(master)

    # Build a SPY-specific dataframe with standardised column names
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

    # Build a TLT-specific dataframe with standardised column names
    tlt_df = master[["date", "tlt_open", "tlt_high", "tlt_low", "tlt_adj_close", "tlt_volume"]].rename(
        columns={
            "tlt_open": "open",
            "tlt_high": "high",
            "tlt_low": "low",
            "tlt_adj_close": "adj_close",
            "tlt_volume": "volume",
        }
    )

    # TLT may not have a separate raw close field in the master dataset.
    # Using adjusted close ensures downstream calculations still work consistently.
    tlt_df["close"] = tlt_df["adj_close"]

    # Build final modelling datasets for each asset
    spy_model, feature_cols = build_features_and_target(
        spy_df, master["vix_close"], spy_tlt_corr, "SPY"
    )
    tlt_model, _ = build_features_and_target(
        tlt_df, master["vix_close"], spy_tlt_corr, "TLT"
    )

    # Save engineered datasets for later modelling
    spy_model.to_csv(SPY_MODEL_DATASET_FILE, index=False)
    tlt_model.to_csv(TLT_MODEL_DATASET_FILE, index=False)

    # Final summary output
    print("=" * 60)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 60)
    print(f"Saved: {SPY_MODEL_DATASET_FILE}")
    print(f"Saved: {TLT_MODEL_DATASET_FILE}")
    print(f"Features ({len(feature_cols)}): {feature_cols}")


if __name__ == "__main__":
    main()