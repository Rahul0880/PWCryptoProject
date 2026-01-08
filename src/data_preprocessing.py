"""Data ingestion, cleaning, feature engineering, and feature store persistence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_PROCESSED, DATA_RAW

TARGET_COL = "future_volatility_7d"
FUTURE_WINDOW = 7
ROLLING_WINDOWS = (3, 7, 14, 30)
NUMERIC_COLS = ["open", "high", "low", "close", "volume", "market_cap"]


def load_raw_data(path: str | bytes | "os.PathLike[str]") -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns=str.strip)
    if df.columns[0].lower().startswith("unnamed") or df.columns[0] == "":
        df = df.drop(columns=df.columns[0])
    df = df.rename(columns={"marketCap": "market_cap"})
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["crypto_name"] = df["crypto_name"].astype(str).str.strip()
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.drop_duplicates(subset=["crypto_name", "date"])
    df = df.sort_values(["crypto_name", "date"]).reset_index(drop=True)
    df.loc[df["volume"] <= 0, "volume"] = np.nan
    df.loc[df["market_cap"] <= 0, "market_cap"] = np.nan

    df[NUMERIC_COLS] = (
        df.groupby("crypto_name")[NUMERIC_COLS]
        .transform(lambda g: g.interpolate(method="linear", limit_direction="both"))
    )
    df[NUMERIC_COLS] = df.groupby("crypto_name")[NUMERIC_COLS].transform(lambda g: g.ffill().bfill())
    df = df.dropna(subset=["date", "crypto_name"])
    return df


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _apply_asset_features(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("date").reset_index(drop=True)
    group["log_close"] = np.log(group["close"].clip(lower=1e-9))
    group["log_return"] = group["log_close"].diff()
    group["pct_change_close"] = group["close"].pct_change()

    prev_close = group["close"].shift(1)
    true_range = np.maximum(group["high"] - group["low"], np.maximum((group["high"] - prev_close).abs(), (group["low"] - prev_close).abs()))
    group["true_range"] = true_range

    for window in ROLLING_WINDOWS:
        group[f"close_ma_{window}"] = group["close"].rolling(window, min_periods=1).mean()
        group[f"close_std_{window}"] = group["close"].rolling(window, min_periods=1).std()
        group[f"return_std_{window}"] = group["log_return"].rolling(window, min_periods=1).std()
        group[f"volume_ma_{window}"] = group["volume"].rolling(window, min_periods=1).mean()
    group["ema_12"] = group["close"].ewm(span=12, adjust=False).mean()
    group["ema_26"] = group["close"].ewm(span=26, adjust=False).mean()
    group["macd"] = group["ema_12"] - group["ema_26"]
    group["rsi_14"] = _compute_rsi(group["close"], period=14)
    group["atr_14"] = group["true_range"].rolling(14, min_periods=1).mean()
    rolling_mean_20 = group["close"].rolling(20, min_periods=1).mean()
    rolling_std_20 = group["close"].rolling(20, min_periods=1).std()
    group["bollinger_bandwidth_20"] = (4 * rolling_std_20) / (rolling_mean_20 + 1e-9)
    group["momentum_3"] = group["close"].pct_change(periods=3)
    group["momentum_7"] = group["close"].pct_change(periods=7)
    group["liquidity_ratio"] = group["volume"] / (group["high"] - group["low"] + 1e-6)
    group["volume_to_market_cap"] = group["volume"] / (group["market_cap"] + 1e-9)
    group["volume_zscore_7"] = (group["volume"] - group["volume_ma_7"]) / (group["volume"].rolling(7, min_periods=1).std() + 1e-9)
    group["price_range"] = group["high"] - group["low"]
    group["normalized_range"] = group["price_range"] / (group["close"] + 1e-9)
    group["body_size"] = group["close"] - group["open"]
    group["volatility_ratio"] = group["return_std_7"] / (group["return_std_30"] + 1e-9)
    group["liquidity_trend_5"] = group["volume_to_market_cap"].rolling(5, min_periods=1).mean()
    future_vol = group["log_return"].rolling(FUTURE_WINDOW, min_periods=FUTURE_WINDOW).std().shift(-FUTURE_WINDOW)
    group[TARGET_COL] = future_vol
    return group


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    enriched_groups = [_apply_asset_features(group.copy()) for _, group in df.groupby("crypto_name", sort=False)]
    df = pd.concat(enriched_groups, ignore_index=True)
    df["date_ordinal"] = df["date"].map(pd.Timestamp.toordinal)
    df["day_of_week"] = df["date"].dt.dayofweek
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["days_since_listing"] = df.groupby("crypto_name").cumcount()
    df["log_market_cap"] = np.log(df["market_cap"] + 1e-9)
    df.drop(columns=["log_close"], inplace=True, errors="ignore")
    return df


def finalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=[TARGET_COL])
    df = df.sort_values(["date", "crypto_name"]).reset_index(drop=True)
    return df


def run_pipeline(raw_path=DATA_RAW, output_path=DATA_PROCESSED) -> pd.DataFrame:
    raw_path = Path(raw_path)
    output_path = Path(output_path)
    raw_df = load_raw_data(raw_path)
    cleaned_df = clean_data(raw_df)
    featured_df = engineer_features(cleaned_df)
    final_df = finalize_dataset(featured_df)
    final_df.to_parquet(output_path, index=False)
    return final_df


def main() -> None:
    final_df = run_pipeline()
    print(f"Final dataset shape: {final_df.shape}")
    print(f"Date range: {final_df['date'].min().date()} -> {final_df['date'].max().date()}")


if __name__ == "__main__":
    main()
