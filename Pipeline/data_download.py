import time
import logging
import os
import pandas as pd
import yfinance as yf


# Logging setup

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "data_download.log")

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger()


# Ensure Data folder exists

DATA_DIR = "Data"
os.makedirs(DATA_DIR, exist_ok=True)



# Download functions

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
    stooq_ticker = "spy.us" if ticker.upper() == "SPY" else f"{ticker.lower()}.us"
    url = f"https://stooq.com/q/d/l/?s={stooq_ticker}&i=d"
    logger.info(f"Downloading from Stooq: {url}")
    return pd.read_csv(url)



# Main function

def download_spy_data(
    ticker: str = "SPY",
    start: str = "2015-01-01",
    end: str | None = None,
    output_file: str = os.path.join(DATA_DIR, "spy_market_data.parquet"),
) -> pd.DataFrame:

    df = pd.DataFrame()

    
    # Try Yahoo Finance
    
    for attempt in range(3):
        try:
            logger.info(f"Trying Yahoo Finance (attempt {attempt + 1}/3)")
            df = _download_yfinance(ticker, start, end)

            if not df.empty:
                logger.info("Downloaded data from Yahoo Finance")
                break

        except Exception as e:
            logger.error(f"Yahoo attempt {attempt + 1} failed: {e}")

        time.sleep(5)

    
    # Fallback to Stooq
    
    if df.empty:
        try:
            logger.warning("Yahoo failed, falling back to Stooq")
            df = _download_stooq_direct(ticker)

            if df.empty:
                raise ValueError("Stooq returned empty dataframe")

            logger.info("Downloaded data from Stooq")

        except Exception as e:
            logger.critical(f"All data sources failed: {e}")
            raise ValueError(f"No data downloaded from Yahoo or Stooq. Error: {e}")

    
    # Data cleaning
    
    try:
        if "Date" not in df.columns:
            df = df.reset_index()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                col[0] if isinstance(col, tuple) and col[1] == ""
                else "_".join([str(c) for c in col if c != ""])
                for col in df.columns
            ]

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

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        df = df[df["Date"] >= pd.to_datetime(start)]
        if end is not None:
            df = df[df["Date"] <= pd.to_datetime(end)]

        if "Adj Close" not in df.columns and "Close" in df.columns:
            df["Adj Close"] = df["Close"]

        logger.info("Data cleaning complete")

    except Exception as e:
        logger.error(f"Data cleaning failed: {e}")
        raise

    
    # Feature engineering
    
    try:
        df["daily_return"] = df["Close"].pct_change()
        df["lag_return_1"] = df["daily_return"].shift(1)
        df["lag_return_2"] = df["daily_return"].shift(2)
        df["moving_avg_5"] = df["Close"].rolling(5).mean()
        df["moving_avg_20"] = df["Close"].rolling(20).mean()
        df["volatility_5"] = df["daily_return"].rolling(5).std()
        df["target_direction"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

        df = df.dropna().reset_index(drop=True)

        logger.info(f"Feature engineering complete. Rows: {len(df)}")

    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        raise

    
    # Save as Parquet in Data folder
    
    try:
        df.to_parquet(output_file, index=False)
        logger.info(f"Saved {len(df)} rows to {output_file}")

    except Exception as e:
        logger.error(f"Failed to save parquet file: {e}")
        raise

    return df



# Run script

if __name__ == "__main__":
    try:
        df = download_spy_data()
        print(df.head())
        logger.info("Script completed successfully")

    except Exception as e:
        logger.critical(f"Script failed: {e}")
        print("Script failed. Check logs.")