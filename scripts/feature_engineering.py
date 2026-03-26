import os
import logging
import pandas as pd
import numpy as np


# Logging setup

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "feature_engineering.log")

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger()


def engineer_features(
    input_file: str = "Data/spy_market_data.parquet",
    output_file: str = "Data/spy_features.parquet",
) -> pd.DataFrame:
    """
    Load SPY market data, create predictive features, and save dataset.
    """

    try:
        logger.info(f"Loading data from {input_file}")

        # Support both parquet and csv
        if input_file.endswith(".parquet"):
            df = pd.read_parquet(input_file)
        else:
            df = pd.read_csv(input_file)

    except Exception as e:
        logger.critical(f"Failed to load input file: {e}")
        raise

    
    # Validate required columns
    
    try:
        required_cols = ["Date", "Close", "Volume"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        logger.info("Input validation passed")

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise

    
    # Date handling
    
    try:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        logger.info("Date processing complete")

    except Exception as e:
        logger.error(f"Date processing failed: {e}")
        raise

    
    # Feature engineering
    
    try:
        if "daily_return" not in df.columns:
            df["daily_return"] = df["Close"].pct_change()

        # Lagged returns
        for lag in [1, 2, 3, 5]:
            df[f"lag_return_{lag}"] = df["daily_return"].shift(lag)

        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f"moving_avg_{window}"] = df["Close"].rolling(window=window).mean()

        # Ratios
        for window in [5, 10, 20, 50]:
            df[f"close_to_ma_{window}"] = df["Close"] / df[f"moving_avg_{window}"]

        # Volatility
        for window in [5, 10, 20]:
            df[f"volatility_{window}"] = df["daily_return"].rolling(window=window).std()

        # Momentum
        df["momentum_5"] = df["Close"] / df["Close"].shift(5) - 1
        df["momentum_10"] = df["Close"] / df["Close"].shift(10) - 1
        df["momentum_20"] = df["Close"] / df["Close"].shift(20) - 1

        # Volume
        df["volume_change"] = df["Volume"].pct_change()
        df["volume_ma_5"] = df["Volume"].rolling(window=5).mean()
        df["volume_ma_20"] = df["Volume"].rolling(window=20).mean()
        df["volume_ratio_5"] = df["Volume"] / df["volume_ma_5"]
        df["volume_ratio_20"] = df["Volume"] / df["volume_ma_20"]

        # Intraday features
        if {"Open", "High", "Low", "Close"}.issubset(df.columns):
            df["intraday_return"] = (df["Close"] - df["Open"]) / df["Open"]
            df["high_low_range"] = (df["High"] - df["Low"]) / df["Close"]
            df["open_close_range"] = (df["Close"] - df["Open"]) / df["Close"]

        # Target
        if "target_direction" not in df.columns:
            df["target_direction"] = (
                df["Close"].shift(-1) > df["Close"]
            ).astype(int)

        logger.info("Feature engineering complete")

    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        raise

    
    # Cleanup
    
    try:
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna().reset_index(drop=True)

        logger.info(f"Data cleaned. Final row count: {len(df)}")

    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        raise

    
    # Save output
    
    try:
        os.makedirs("Data", exist_ok=True)

        if output_file.endswith(".parquet"):
            df.to_parquet(output_file, index=False)
        else:
            df.to_csv(output_file, index=False)

        logger.info(f"Saved dataset to {output_file}")

    except Exception as e:
        logger.critical(f"Failed to save output: {e}")
        raise

    return df



# Run script

if __name__ == "__main__":
    try:
        features_df = engineer_features()
        print(features_df.head())
        print("\nColumns:")
        print(features_df.columns.tolist())

        logger.info("Feature engineering script completed successfully")

    except Exception as e:
        logger.critical(f"Script failed: {e}")
        print("Script failed. Check logs.")