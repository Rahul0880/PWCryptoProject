"""Exploratory data analysis utilities and Markdown report generator."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import DATA_RAW, EDA_REPORT_PATH, FIGURES_DIR

matplotlib.use("Agg")
sns.set_theme(style="darkgrid")


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.columns[0].lower().startswith("unnamed") or df.columns[0] == "":
        df = df.drop(columns=df.columns[0])
    df = df.rename(columns={"marketCap": "market_cap"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["crypto_name"] = df["crypto_name"].astype(str)
    numeric_cols = ["open", "high", "low", "close", "volume", "market_cap"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values(["crypto_name", "date"]).reset_index(drop=True)
    return df


def _figure_path(name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR / name


def plot_price_trends(df: pd.DataFrame) -> Path:
    latest_caps = df.groupby("crypto_name")["market_cap"].last().sort_values(ascending=False)
    top_assets = latest_caps.head(5).index.tolist()
    subset = df[df["crypto_name"].isin(top_assets)]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=subset, x="date", y="close", hue="crypto_name", ax=ax)
    ax.set_title("Closing Price Trends (Top Market Cap Assets)")
    ax.set_ylabel("Close Price (USD)")
    fig.tight_layout()
    path = _figure_path("price_trends.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_volatility_distribution(df: pd.DataFrame) -> Path:
    df = df.copy()
    df["log_return"] = np.log(df["close"].clip(lower=1e-9)).diff()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df["log_return"].dropna(), bins=100, kde=True, ax=ax, color="#ff7f0e")
    ax.set_title("Distribution of Daily Log Returns")
    ax.set_xlabel("Log Return")
    fig.tight_layout()
    path = _figure_path("volatility_distribution.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_volume_vs_market_cap(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=df.sample(min(5000, len(df)), random_state=42),
        x="market_cap",
        y="volume",
        hue="crypto_name",
        ax=ax,
        alpha=0.6,
        legend=False,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Volume vs Market Cap")
    fig.tight_layout()
    path = _figure_path("volume_vs_market_cap.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_correlation_heatmap(df: pd.DataFrame) -> Path:
    base_cols = ["open", "high", "low", "close", "volume", "market_cap"]
    corr = df[base_cols].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Base Feature Correlations")
    fig.tight_layout()
    path = _figure_path("base_feature_correlation.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def build_eda_report(df: pd.DataFrame) -> str:
    row_count = len(df)
    asset_count = df["crypto_name"].nunique()
    start_date = df["date"].min()
    end_date = df["date"].max()
    missing_pct = (df.isna().mean() * 100).round(2).sort_values(ascending=False).head(5)
    returns = df.groupby("crypto_name")["close"].pct_change()
    rolling_vol = returns.rolling(7).std()
    mean_vol = rolling_vol.groupby(df["crypto_name"]).mean().sort_values(ascending=False)
    lines = [
        "# Exploratory Data Analysis",
        f"- Observations: {row_count:,}",
        f"- Assets covered: {asset_count}",
        f"- Date range: {start_date.date()} → {end_date.date()}",
        "",
        "## Missing Data (Top 5 Columns)",
    ]
    for col, pct in missing_pct.items():
        lines.append(f"- {col}: {pct}% missing")
    lines.append("")
    lines.append("## Highest Average 7-day Volatility")
    for name, vol in mean_vol.head(5).items():
        lines.append(f"- {name}: {vol:.4f} mean σ")
    lines.extend(
        [
            "",
            "## Key Visuals",
            "1. Closing price trends (`reports/figures/price_trends.png`)",
            "2. Log return distribution (`reports/figures/volatility_distribution.png`)",
            "3. Volume vs market cap (`reports/figures/volume_vs_market_cap.png`)",
            "4. Base feature correlations (`reports/figures/base_feature_correlation.png`)",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    df = load_dataset(Path(DATA_RAW))
    figure_paths = [
        plot_price_trends(df),
        plot_volatility_distribution(df),
        plot_volume_vs_market_cap(df),
        plot_correlation_heatmap(df),
    ]
    report_text = build_eda_report(df)
    EDA_REPORT_PATH.write_text(report_text, encoding="utf-8")
    print("Saved EDA report to", EDA_REPORT_PATH)
    for path in figure_paths:
        print("Saved figure:", path)


if __name__ == "__main__":
    main()
