import time
import pandas as pd
import yfinance as yf


def _download_yfinance(ticker: str, start: str, end: str | None):
    return yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        threads=False,
    )


def _download_stooq_direct(ticker: str):
    # Stooq direct CSV URL
    stooq_ticker = "spy.us" if ticker.upper() == "SPY" else f"{ticker.lower()}.us"
    url = f"https://stooq.com/q/d/l/?s={stooq_ticker}&i=d"
    df = pd.read_csv(url)
    return df


def download_spy_data(
    ticker: str = "SPY",
    start: str = "1994-01-01",
    end: str | None = None,
    output_file: str = "Data/spy_market_data.csv",
) -> pd.DataFrame:
    df = pd.DataFrame()

    # Try Yahoo first
    for attempt in range(3):
        try:
            print(f"Trying Yahoo Finance (attempt {attempt + 1}/3)...")
            df = _download_yfinance(ticker, start, end)
            if not df.empty:
                print("Downloaded data from Yahoo Finance.")
                break
        except Exception as e:
            print(f"Yahoo attempt {attempt + 1} failed: {e}")
        time.sleep(5)

    # Fallback to direct Stooq CSV
    if df.empty:
        try:
            print("Falling back to Stooq direct CSV...")
            df = _download_stooq_direct(ticker)
            if df.empty:
                raise ValueError("Stooq returned an empty dataframe.")
            print("Downloaded data from Stooq.")
        except Exception as e:
            raise ValueError(
                f"No data downloaded from Yahoo Finance or Stooq. Last fallback error: {e}"
            )

    # If data came from Yahoo, reset index
    if "Date" not in df.columns:
        df = df.reset_index()

    # Flatten columns if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            col[0] if isinstance(col, tuple) and col[1] == ""
            else "_".join([str(c) for c in col if c != ""])
            for col in df.columns
        ]

    # Standardize names
    rename_map = {}
    for col in df.columns:
        col_str = str(col)
        if col_str.startswith("Date"):
            rename_map[col] = "Date"
        elif col_str.startswith("Open"):
            rename_map[col] = "Open"
        elif col_str.startswith("High"):
            rename_map[col] = "High"
        elif col_str.startswith("Low"):
            rename_map[col] = "Low"
        elif col_str.startswith("Close") and "Adj" not in col_str:
            rename_map[col] = "Close"
        elif col_str.startswith("Adj Close"):
            rename_map[col] = "Adj Close"
        elif col_str.startswith("Volume"):
            rename_map[col] = "Volume"

    df = df.rename(columns=rename_map)

    # Convert date + sort
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Filter date range manually for Stooq fallback
    df = df[df["Date"] >= pd.to_datetime(start)]
    if end is not None:
        df = df[df["Date"] <= pd.to_datetime(end)]

    # Create Adj Close if missing
    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]

    # Feature engineering
    df["daily_return"] = df["Close"].pct_change()
    df["lag_return_1"] = df["daily_return"].shift(1)
    df["lag_return_2"] = df["daily_return"].shift(2)
    df["moving_avg_5"] = df["Close"].rolling(window=5).mean()
    df["moving_avg_20"] = df["Close"].rolling(window=20).mean()
    df["volatility_5"] = df["daily_return"].rolling(window=5).std()
    df["target_direction"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    # Clean
    df = df.dropna().reset_index(drop=True)

    # Save
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} rows to {output_file}")

    return df


if __name__ == "__main__":
    spy_df = download_spy_data()
    print(spy_df.head())