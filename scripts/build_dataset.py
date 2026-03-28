import os
import logging
import pandas as pd

# Logging setup
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "build_dataset.log")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger()

# Paths
DATA_DIR = "Data"
os.makedirs(DATA_DIR, exist_ok=True)


def build_relational_dataset(
    input_file: str = os.path.join(DATA_DIR, "spy_features.parquet"),
) -> dict[str, pd.DataFrame]:
    """
    Split the engineered SPY dataset into a relational dataset with 4 tables.

    Tables created:
    prices.parquet
        Raw market data by Date and Ticker.
    returns.parquet
        Return-based features by Date and Ticker.
    technical_indicators.parquet
        Technical indicator features by Date and Ticker.
    targets.parquet
        Prediction target by Date and Ticker.

    Parameters:
    input_file : str, default="Data/spy_features.parquet"
        Path to the engineered feature dataset.

    Returns:
    dict[str, pandas.DataFrame]
        Dictionary containing the four tables.
    """

    try:
        logger.info(f"Loading engineered dataset from {input_file}")

        if input_file.endswith(".parquet"):
            df = pd.read_parquet(input_file)
        else:
            df = pd.read_csv(input_file)

    except Exception as e:
        logger.critical(f"Failed to load input dataset: {e}")
        raise

    try:
        # Required keys
        required_cols = ["Date"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required key column: {col}")

        # Add ticker if not already present
        if "Ticker" not in df.columns:
            df["Ticker"] = "SPY"
            logger.info("Ticker column was missing. Added default value 'SPY'.")

        # Standardize types
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

        logger.info(f"Input dataset loaded successfully with {len(df)} rows.")

    except Exception as e:
        logger.error(f"Input validation/cleanup failed: {e}")
        raise

    try:
        # Table 1: prices
        prices_cols = [
            "Date", "Ticker", "Open", "High", "Low", "Close", "Volume"
        ]
        prices_cols = [c for c in prices_cols if c in df.columns]
        prices = df[prices_cols].copy()

        # Table 2: returns
        returns_cols = [
            "Date", "Ticker",
            "daily_return", "lag_return_1", "lag_return_2", "lag_return_3", "lag_return_5"
        ]
        returns_cols = [c for c in returns_cols if c in df.columns]
        returns = df[returns_cols].copy()

        # Table 3: technical indicators
        technical_cols = [
            "Date", "Ticker",
            "moving_avg_5", "moving_avg_10", "moving_avg_20", "moving_avg_50",
            "close_to_ma_5", "close_to_ma_10", "close_to_ma_20", "close_to_ma_50",
            "volatility_5", "volatility_10", "volatility_20",
            "momentum_5", "momentum_10", "momentum_20",
            "volume_change", "volume_ma_5", "volume_ma_20",
            "volume_ratio_5", "volume_ratio_20",
            "intraday_return", "high_low_range", "open_close_range"
        ]
        technical_cols = [c for c in technical_cols if c in df.columns]
        technical_indicators = df[technical_cols].copy()

        # Table 4: targets
        target_cols = ["Date", "Ticker", "target_direction"]
        target_cols = [c for c in target_cols if c in df.columns]
        targets = df[target_cols].copy()

        logger.info("Successfully split dataset into 4 relational tables.")

    except Exception as e:
        logger.error(f"Failed to split dataset into relational tables: {e}")
        raise
    
    # Data cleaning and validation    
    try:
        outputs = {
            "prices": prices,
            "returns": returns,
            "technical_indicators": technical_indicators,
            "targets": targets,
        }

        for name, table in outputs.items():
            output_path = os.path.join(DATA_DIR, f"{name}.parquet")
            table.to_parquet(output_path, index=False)
            logger.info(f"Saved {name} table with {len(table)} rows to {output_path}")

        return outputs
    
    # Data saving
    except Exception as e:
        logger.critical(f"Failed to save relational tables: {e}")
        raise


if __name__ == "__main__":
    try:
        tables = build_relational_dataset()

        for name, table in tables.items():
            print(f"\n{name.upper()} TABLE")
            print(table.head())
            print(f"Rows: {len(table)} | Columns: {len(table.columns)}")

        logger.info("build_dataset.py completed successfully.")

    except Exception as e:
        logger.critical(f"Script failed: {e}")
        print("Script failed. Check logs/build_dataset.log")