# Pipeline Architecture

## Stage 1 – Ingestion
1. Load raw CSV from `data/raw/cryptocurrency_prices.csv`.
2. Normalize headers and parse timestamps.
3. Persist metadata: observation count, assets, and temporal coverage (logged via `src.data_preprocessing`).

## Stage 2 – Cleaning
1. Enforce numeric dtypes on OHLCV + market cap.
2. Replace non-positive volume/market cap with `NaN` and interpolate per asset.
3. Apply forward/backward fill to remaining gaps; drop duplicates.

## Stage 3 – Feature Engineering
1. Compute log returns and rolling statistics (3/7/14/30 day means + stds).
2. Technical indicators: RSI(14), ATR(14), Bollinger bandwidth(20), MACD, momentum windows.
3. Liquidity metrics: volume/market cap, liquidity ratio, liquidity trend.
4. Temporal features: day-of-week, week-of-year, month, years since listing, date ordinal.
5. Label generation: future 7-day realized volatility computed per asset with negative shift to avoid leakage.
6. Persist enriched parquet store (`data/processed/volatility_features.parquet`).

## Stage 4 – Modeling
1. Split chronologically (80/20) for hold-out evaluation.
2. Build sklearn pipeline: numeric scaler + categorical one-hot feeding HistGradientBoostingRegressor.
3. Hyperparameter tuning via `RandomizedSearchCV` with `TimeSeriesSplit`.
4. Save best pipeline to `models/volatility_pipeline.pkl`.

## Stage 5 – Evaluation & Explainability
1. Compute metrics on hold-out: RMSE, MAE, $R^2$ recorded in `reports/metrics.json`.
2. Run permutation importance on validation data → `reports/feature_importances.csv`.
3. Visual inspection via `reports/figures/*.png`.

## Stage 6 – Serving
1. Streamlit app loads model + processed store.
2. Scores latest observations and lists assets breaching user-set volatility thresholds.
3. Provides per-asset historical trend overlay (predicted vs realized future volatility).

## Automation Hooks
- All stages can be chained inside a CI job or `make` target: preprocess → EDA → train → deploy.
- Because every stage writes deterministic artifacts, incremental reruns only touch downstream files when upstream data change.
