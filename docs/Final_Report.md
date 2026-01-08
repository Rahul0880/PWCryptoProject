# Final Report

## Problem Statement
Cryptocurrency markets exhibit pronounced volatility, exposing market participants to sharp drawdowns. The goal of this project is to predict the next 7-day realized volatility for 50+ assets using daily OHLCV and market capitalization data so desks can rebalance positions proactively.

## Data Summary
- Observations: 72,560 rows (2013-05-05 → 2022-09-12).
- Assets: 50+ cryptocurrencies; Bitcoin and Litecoin dominate early history.
- Raw columns: `open`, `high`, `low`, `close`, `volume`, `marketCap`, timestamps, and asset labels.
- Target definition: $\sigma_{t+1:t+7} = std(\log p_{t+1} - \log p_t)$ per asset, derived without lookahead.

## Methodology
1. **Cleaning** – removed duplicate dates per asset, coerced numeric types, interpolated/fill-forward missing values, masked zero liquidity.
2. **Feature Engineering** –
   - Rolling stats on prices/returns (3/7/14/30 day means & stds).
   - Momentum (3- and 7-day), RSI(14), ATR(14), Bollinger bandwidth(20), MACD.
   - Liquidity indicators: volume/market cap, liquidity ratios, liquidity trend.
   - Temporal signals: day-of-week, ISO week, month, years since listing, date ordinal.
3. **Modeling** – HistGradientBoostingRegressor inside a sklearn pipeline (scaler + one-hot). Hyperparameters tuned through `RandomizedSearchCV` with `TimeSeriesSplit` (5 folds, 20 trials).
4. **Evaluation** – Chronological 80/20 split; permutation importance computed on the hold-out to interpret drivers.

## Results
- Hold-out metrics (`reports/metrics.json`):
  - $RMSE = 0.0768$
  - $MAE = 0.0268$
  - $R^2 = 0.0650$
- Top drivers (`reports/feature_importances.csv`): 30-day return volatility, normalized intraday range, ISO week-of-year, short-term return volatility, and Bollinger bandwidth.
- Insight: Seasonality (week-of-year) and liquidity-normalized price swings materially influence predicted volatility, validating the inclusion of both technical and calendar features.

## Visual Insights
- `reports/figures/price_trends.png` shows prolonged bull/bear regimes; volatility spikes track Bitcoin drawdowns.
- `reports/figures/volatility_distribution.png` reveals heavy-tailed log returns, justifying robust estimators.
- `reports/figures/base_feature_correlation.png` confirms strong OHLC collinearity, motivating tree-based models.

## Limitations & Next Steps
1. **Exogenous Signals** – Incorporate macro or on-chain metrics (funding rates, flows) to boost $R^2$.
2. **Model Classes** – Experiment with temporal architectures (Temporal Fusion Transformer, seq2seq) or quantile regressors for probabilistic forecasts.
3. **Multi-Horizon Targets** – Extend labels beyond 7-day volatility for tactical vs strategic planning.
4. **Deployment** – Containerize the Streamlit app or expose a REST endpoint for automation.
5. **Monitoring** – Add dataset drift detection and scheduled retraining triggers.

## Deliverables
- Feature store: `data/processed/volatility_features.parquet`
- Model artifact: `models/volatility_pipeline.pkl`
- Metrics & explanations: `reports/metrics.json`, `reports/feature_importances.csv`
- UI prototype: `app/streamlit_app.py`
- Documentation: HLD, LLD, Pipeline (`docs/`), EDA write-up (`reports/EDA.md`).
