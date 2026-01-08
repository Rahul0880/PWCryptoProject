"""Streamlit dashboard for monitoring predicted cryptocurrency volatility."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.config import DATA_PROCESSED, PIPELINE_ARTIFACT  # type: ignore  # pylint: disable=wrong-import-position

CALM_LEVEL = 0.02
ALERT_LEVEL = 0.05

st.set_page_config(page_title="Crypto Volatility Radar", page_icon="📈", layout="wide")


@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(PIPELINE_ARTIFACT)


@st.cache_data(show_spinner=False)
def load_dataset():
    return pd.read_parquet(DATA_PROCESSED)


def _feature_names(model) -> np.ndarray:
    if hasattr(model, "feature_names_in_"):
        return model.feature_names_in_
    if hasattr(model, "regressor_") and hasattr(model.regressor_, "feature_names_in_"):
        return model.regressor_.feature_names_in_
    raise AttributeError("Model does not expose feature_names_in_ for inference")


def predict(data: pd.DataFrame, model) -> np.ndarray:
    feature_cols = _feature_names(model)
    return model.predict(data[feature_cols])


def _format_currency(value: float) -> str:
    if not np.isfinite(value) or value <= 0:
        return "–"
    for unit in ["", "K", "M", "B", "T"]:
        if abs(value) < 1000.0:
            return f"${value:,.1f}{unit}"
        value /= 1000.0
    return f"${value:,.1f}P"


def _risk_label(value: float, calm: float = CALM_LEVEL, alert: float = ALERT_LEVEL) -> tuple[str, str]:
    if value < calm:
        return "Calm seas", "🟢"
    if value < alert:
        return "Keep watch", "🟡"
    return "High alert", "🔴"


def render_overview(latest_scores: pd.DataFrame, threshold: float) -> None:
    st.subheader("Market Snapshot")
    share_high = float((latest_scores["predicted_volatility"] >= threshold).mean())
    calm_label, calm_icon = _risk_label(latest_scores["predicted_volatility"].median())
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Assets tracked", latest_scores["crypto_name"].nunique())
    col2.metric("Avg predicted σ", f"{latest_scores['predicted_volatility'].mean():.4f}")
    col3.metric("Top σ", f"{latest_scores['predicted_volatility'].max():.4f}")
    col4.metric("Typical mood", f"{calm_icon} {calm_label}")
    st.progress(share_high, text=f"{share_high*100:,.1f}% of assets exceed your alert level")


def _apply_focus_filter(latest_scores: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "Large caps":
        return latest_scores.sort_values("market_cap", ascending=False).head(12)
    if mode == "Smaller caps":
        median_cap = latest_scores["market_cap"].median()
        return latest_scores[latest_scores["market_cap"] <= median_cap]
    return latest_scores


def render_watchlist(latest_scores: pd.DataFrame, threshold: float, focus_mode: str) -> None:
    st.subheader("Volatility Watchlist")
    focus_df = _apply_focus_filter(latest_scores, focus_mode).copy()
    focus_df["Risk level"] = focus_df["predicted_volatility"].apply(lambda v: "{} {}".format(*_risk_label(v)))
    focus_df["Predicted σ"] = focus_df["predicted_volatility"].map(lambda v: f"{v:.3f}")
    focus_df["Market cap"] = focus_df["market_cap"].apply(_format_currency)
    focus_df = focus_df.sort_values("predicted_volatility", ascending=False)
    top_watchlist = focus_df.head(10)
    st.caption("Sorted by predicted risk so newcomers can focus on what matters most.")
    st.dataframe(
        top_watchlist[["crypto_name", "Risk level", "Predicted σ", "Market cap"]],
        width="stretch",
        hide_index=True,
    )

    elevated = focus_df[focus_df["predicted_volatility"] >= threshold]
    if not elevated.empty:
        st.info(
            f"{len(elevated)} asset(s) are above your alert level. Consider widening spreads or reducing exposure if you hold them."
        )


def render_asset_detail(df: pd.DataFrame, model) -> None:
    st.subheader("Asset Drill-down")
    asset = st.selectbox("Pick an asset to inspect", sorted(df["crypto_name"].unique()))
    horizon = st.slider("History window (days)", 60, 365, 180)
    asset_df = df[df["crypto_name"] == asset].sort_values("date").tail(horizon)
    if asset_df.empty:
        st.info("Not enough data for the selected asset.")
        return
    asset_df = asset_df.copy()
    asset_df["predicted_volatility"] = predict(asset_df, model)

    tab_trend, tab_table = st.tabs(["Trend", "Latest numbers"])
    with tab_trend:
        chart_df = asset_df[["date", "predicted_volatility", "future_volatility_7d"]].melt(
            id_vars="date", var_name="series", value_name="volatility"
        )
        st.line_chart(chart_df, x="date", y="volatility", color="series")
        st.caption("Blue = model estimate, orange = realized future volatility (when available).")
    with tab_table:
        latest_rows = asset_df[["date", "close", "predicted_volatility", "future_volatility_7d"]].tail(20)
        st.dataframe(latest_rows, hide_index=True, width="stretch")


def render_glossary() -> None:
    with st.expander("What am I looking at? (Friendly glossary)"):
        st.markdown(
            "- **Volatility (σ)** – how much price wiggles day to day. Higher values mean bigger swings."\
            "\n- **Predicted σ** – the model's forward-looking guess for the next week."\
            "\n- **High-alert threshold** – anything above this is highlighted so you can act early."\
            "\n- **Watchlist table** – sorted so newcomers instantly see which coins are calm vs. stormy."
        )

    st.caption("Need a reminder later? Collapse the glossary and it will stay out of the way.")


def main() -> None:
    model = load_model()
    data = load_dataset()
    latest = data.sort_values("date").groupby("crypto_name").tail(1)
    latest = latest.copy()
    latest["predicted_volatility"] = predict(latest, model)

    st.sidebar.header("👋 Start here")
    st.sidebar.markdown(
        "1. Pick an alert level that matches your risk appetite.\n"
        "2. Glance at the watchlist – red badges mean rough waters.\n"
        "3. Drill into a coin to see how predictions line up with history."
    )
    st.sidebar.info("The dashboard uses daily prices, volumes, and market caps to guess next week's turbulence.")

    threshold = st.sidebar.slider(
        "Choose your high-alert threshold",
        min_value=0.0,
        max_value=float(latest["predicted_volatility"].max()),
        value=float(ALERT_LEVEL),
        step=0.005,
    )
    focus_mode = st.sidebar.selectbox(
        "Focus on", ["All assets", "Large caps", "Smaller caps"], help="Pick a simple lens if you're new here."
    )

    st.title("Crypto Volatility Radar")
    st.caption("Plain-language risk insights for busy humans – no quant background required.")

    render_overview(latest, threshold)
    render_watchlist(latest, threshold, focus_mode)
    render_asset_detail(data, model)
    render_glossary()


if __name__ == "__main__":
    main()
