"""
01_data_prep.py
Load SPY, TLT, and VIX raw files, align on common dates, and save one clean master dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from config import (
    SPY_FILE,
    TLT_FILE,
    VIX_FILE,
    OUTPUT_DIR,
    MASTER_DATASET_FILE,
)


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def load_spy(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _standardise_columns(df)
    df["date"] = pd.to_datetime(df["date"])
    # teammate R script uses adjusted; notebook renames adjusted -> adj_close
    if "adjusted" in df.columns:
        df = df.rename(columns={"adjusted": "adj_close"})
    required = ["date", "open", "high", "low", "close", "volume", "adj_close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"SPY file missing columns: {missing}")
    return df[required].sort_values("date").reset_index(drop=True)


def load_tlt(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _standardise_columns(df)
    df["date"] = pd.to_datetime(df["date"])

    # Try to infer adjusted close column
    if "adj_close" not in df.columns:
        if "adjusted" in df.columns:
            df = df.rename(columns={"adjusted": "adj_close"})
        elif "close" in df.columns:
            # notebook treated yfinance close as adjusted close
            df["adj_close"] = df["close"]

    required = ["date", "open", "high", "low", "volume", "adj_close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"TLT file missing columns: {missing}")
    return df[required].sort_values("date").reset_index(drop=True)


def load_vix(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _standardise_columns(df)
    df["date"] = pd.to_datetime(df["date"])
    if "vix_close" not in df.columns:
        if "close" in df.columns:
            df = df.rename(columns={"close": "vix_close"})
        elif "adj_close" in df.columns:
            df = df.rename(columns={"adj_close": "vix_close"})
    required = ["date", "vix_close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"VIX file missing columns: {missing}")
    return df[required].sort_values("date").reset_index(drop=True)


def merge_assets(spy: pd.DataFrame, tlt: pd.DataFrame, vix: pd.DataFrame) -> pd.DataFrame:
    spy = spy.add_prefix("spy_").rename(columns={"spy_date": "date"})
    tlt = tlt.add_prefix("tlt_").rename(columns={"tlt_date": "date"})
    merged = spy.merge(tlt, on="date", how="inner").merge(vix, on="date", how="inner")
    merged = merged.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    return merged


def run_data_checks(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Merged dataset is empty after alignment.")
    if df["date"].isna().any():
        raise ValueError("Date column contains missing values.")
    if not df["date"].is_monotonic_increasing:
        raise ValueError("Dates are not sorted in ascending order.")
    if df.duplicated("date").any():
        raise ValueError("Duplicate dates detected.")
    if df.isna().sum().sum() > 0:
        # keep strict here so issues are caught early
        na_cols = df.columns[df.isna().any()].tolist()
        raise ValueError(f"Merged dataset still has missing values in: {na_cols}")


def main() -> None:
    ensure_output_dir()
    spy = load_spy(SPY_FILE)
    tlt = load_tlt(TLT_FILE)
    vix = load_vix(VIX_FILE)
    merged = merge_assets(spy, tlt, vix)
    run_data_checks(merged)
    merged.to_csv(MASTER_DATASET_FILE, index=False)

    print("=" * 60)
    print("DATA PREP COMPLETE")
    print("=" * 60)
    print(f"Rows: {len(merged)}")
    print(f"Date range: {merged['date'].min().date()} -> {merged['date'].max().date()}")
    print(f"Saved: {MASTER_DATASET_FILE}")


if __name__ == "__main__":
    main()
