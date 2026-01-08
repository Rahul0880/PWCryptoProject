# PWCryptoProject

## Cryptocurrency Volatility Forecasting

Predict future price turbulence for 50+ cryptocurrencies using engineered technical indicators and tree-based regression. The target variable is the 7-day forward realized volatility computed from [...]

## Dataset
- Source: `data/raw/cryptocurrency_prices.csv` (daily OHLCV + market cap per asset).
- Time range: 2013-05 onward (Bitcoin, Litecoin, and other large caps).
- Target: $\sigma_{t+1:t+7}$, the rolling standard deviation of log returns over the next 7 trading days.

> Note: Large raw datasets, processed feature stores, and model binaries are intentionally ignored by `.gitignore`.
> After cloning, place your CSV at `data/raw/cryptocurrency_prices.csv`, then run the pipeline to regenerate artifacts.

## Project Structure
```
├── app/                  # Streamlit prototype for monitoring predicted vol levels
├── data/
│   ├── processed/        # Feature store (parquet)
│   └── raw/              # Immutable source data
├── docs/                 # HLD, LLD, pipeline, and final report
├── models/               # Serialized sklearn pipeline + metadata
├── reports/
│   ├── figures/          # PNG visualizations created during EDA
│   └── *.md/json         # EDA write-up and metric snapshots
├── src/
│   ├── config.py         # Centralized file/dir paths
│   ├── data_preprocessing.py
│   ├── eda.py
│   └── model_training.py
├── README.md
└── requirements.txt
```

## Quickstart
1. **Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
2. **Data prep + feature store**
   ```bash
   python -m src.data_preprocessing
   ```
3. **Exploratory analysis artifacts**
   ```bash
   python -m src.eda
   ```
4. **Model training + evaluation**
   ```bash
   python -m src.model_training
   ```
   Metrics and artifacts land in `reports/metrics.json`, `reports/feature_importances.csv`, and `models/volatility_pipeline.pkl`.
5. **Interactive monitoring**
   ```bash
   streamlit run app/streamlit_app.py
   ```

## Streamlit Community Cloud
- App entry point: `app/streamlit_app.py`
- Dependency file: `requirements.txt`
- Live demo: [Crypto Volatility Radar (hosted)](https://pwcryptoproject-ws5wckxejtxzvmjfdvwmhw.streamlit.app/#crypto-volatility-radar)

## Pipeline Summary
1. **Ingestion** – load raw OHLCV + market cap, enforce schema, and repair gaps via asset-wise time interpolation.
2. **Feature Engineering** – compute log returns, rolling stats (7/14/30 day means and volatilities), ATR, Bollinger bandwidth, RSI, liquidity ratios, and date-derived seasonality fields.
3. **Labeling** – future 7-day realized volatility per asset for supervised learning without lookahead bias.
4. **Modeling** – `HistGradientBoostingRegressor` inside a `ColumnTransformer` pipeline with `TimeSeriesSplit` + `RandomizedSearchCV` for hyperparameter tuning.
5. **Evaluation** – hold-out chronological split, logging RMSE/MAE/$R^2$, and exporting feature importances.
6. **Serving** – Streamlit dashboard reloads the processed store, scores the latest rows, and surfaces high-volatility candidates.

## Deliverables
- **Machine Learning Model** – serialized sklearn pipeline + metrics snapshot.
- **Data Processing & Feature Engineering** – reproducible scripts that emit a parquet feature store and document the engineered indicators.
- **EDA Report** – Markdown summary with descriptive stats and figures saved in `reports/figures/`.
- **Documentation** – HLD, LLD, pipeline architecture, and final findings in `docs/`.

- **Deployment Stub** – Streamlit UI for local experimentation.
