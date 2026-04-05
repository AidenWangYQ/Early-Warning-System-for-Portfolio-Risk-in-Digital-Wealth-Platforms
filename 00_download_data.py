import yfinance as yf
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

def download_and_save(ticker: str, out_path: Path) -> None:
    # auto_adjust=False keeps both Close and Adj Close
    df = yf.download(
        ticker,
        start="2000-01-01",
        auto_adjust=False,
        progress=False,
    )

    # If yfinance returns MultiIndex columns, flatten them
    if isinstance(df.columns, pd.MultiIndex):
        # keep only the first level: Open, High, Low, Close, Adj Close, Volume
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df.to_csv(out_path, index=False)
    print(f"Saved {ticker} -> {out_path}")

download_and_save("SPY", DATA_DIR / "spy_full.csv")
download_and_save("TLT", DATA_DIR / "tlt_full.csv")
download_and_save("^VIX", DATA_DIR / "vix_full.csv")