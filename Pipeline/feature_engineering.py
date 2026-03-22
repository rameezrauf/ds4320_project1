import pandas as pd
import numpy as np


def engineer_features(
    input_file: str = "/Data/spy_market_data.csv",
    output_file: str = "/Data/spy_features.csv",
) -> pd.DataFrame:
    """
    Load downloaded SPY market data, create additional predictive features,
    and save a model-ready dataset.

    Parameters
    input_file : str, default="/Data/spy_market_data.csv"
        CSV created by data_download.py.
    output_file : str, default="/Data/spy_features.csv"
        Destination CSV for engineered dataset.

    Returns
    pandas.DataFrame
        DataFrame with engineered features.
    """

    df = pd.read_csv(input_file)

    if "Date" not in df.columns:
        raise ValueError("Expected a 'Date' column in the input file.")
    if "Close" not in df.columns:
        raise ValueError("Expected a 'Close' column in the input file.")
    if "Volume" not in df.columns:
        raise ValueError("Expected a 'Volume' column in the input file.")

    # Date handling
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # If daily_return does not exist, make it
    if "daily_return" not in df.columns:
        df["daily_return"] = df["Close"].pct_change()

    # Lagged return features
    for lag in [1, 2, 3, 5]:
        df[f"lag_return_{lag}"] = df["daily_return"].shift(lag)

    # Moving averages
    for window in [5, 10, 20, 50]:
        df[f"moving_avg_{window}"] = df["Close"].rolling(window=window).mean()

    # Ratios of close to moving average
    for window in [5, 10, 20, 50]:
        df[f"close_to_ma_{window}"] = df["Close"] / df[f"moving_avg_{window}"]

    # Rolling volatility
    for window in [5, 10, 20]:
        df[f"volatility_{window}"] = df["daily_return"].rolling(window=window).std()

    # Momentum features
    df["momentum_5"] = df["Close"] / df["Close"].shift(5) - 1
    df["momentum_10"] = df["Close"] / df["Close"].shift(10) - 1
    df["momentum_20"] = df["Close"] / df["Close"].shift(20) - 1

    # Volume features
    df["volume_change"] = df["Volume"].pct_change()
    df["volume_ma_5"] = df["Volume"].rolling(window=5).mean()
    df["volume_ma_20"] = df["Volume"].rolling(window=20).mean()
    df["volume_ratio_5"] = df["Volume"] / df["volume_ma_5"]
    df["volume_ratio_20"] = df["Volume"] / df["volume_ma_20"]

    # Intraday / range features
    if {"Open", "High", "Low", "Close"}.issubset(df.columns):
        df["intraday_return"] = (df["Close"] - df["Open"]) / df["Open"]
        df["high_low_range"] = (df["High"] - df["Low"]) / df["Close"]
        df["open_close_range"] = (df["Close"] - df["Open"]) / df["Close"]

    # Target variable
    # Keep existing target if already present, otherwise create it
    if "target_direction" not in df.columns:
        df["target_direction"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    # Clean up
    # Replace inf values if any
    df = df.replace([np.inf, -np.inf], np.nan)

    # Drop rows created by rolling windows / shifts
    df = df.dropna().reset_index(drop=True)

    # Save
    df.to_csv(output_file, index=False)
    print(f"Saved engineered dataset with {len(df)} rows to {output_file}")

    return df


if __name__ == "__main__":
    features_df = engineer_features()
    print(features_df.head())
    print("\nColumns:")
    print(features_df.columns.tolist())