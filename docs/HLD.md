# High-Level Design (HLD)

## Objective
Forecast cryptocurrency volatility one week ahead using daily OHLCV + market cap data so that risk desks and traders can anticipate turbulence windows.

## Architecture Overview
1. **Data Layer**
   - Raw Store: `data/raw/cryptocurrency_prices.csv` remains immutable.
   - Feature Store: `data/processed/volatility_features.parquet` contains cleaned, enriched observations.
2. **Processing Layer**
   - `src/data_preprocessing.py` orchestrates ingestion, cleansing, feature engineering, and label creation.
   - `src/eda.py` generates descriptive statistics and figures for stakeholder communication.
3. **Modeling Layer**
   - `src/model_training.py` builds a sklearn `Pipeline` with a `ColumnTransformer` and `HistGradientBoostingRegressor`.
   - Hyperparameter tuning relies on `RandomizedSearchCV` + `TimeSeriesSplit` for chronological robustness.
4. **Serving Layer**
   - `app/streamlit_app.py` loads the serialized model (`models/volatility_pipeline.pkl`) and the feature store to render monitoring views.
5. **Reporting Layer**
   - `reports/EDA.md`, `reports/metrics.json`, and `reports/feature_importances.csv` provide governance artifacts.

## Data Flow
1. **Ingest** raw CSV → enforce schema + sorting by `crypto_name` and `date`.
2. **Clean** missing/zero values via per-asset interpolation followed by ffill/bfill.
3. **Engineer** technical indicators, liquidity metrics, and temporal features.
4. **Label** target volatility: $\sigma_{t+1:t+7} = std(\log p_{t+1} - \log p_t)$.
5. **Persist** feature store (parquet) for reproducible modeling + serving.
6. **Model** training/tuning on chronological splits; evaluation metrics logged to JSON.
7. **Serve** predictions by scoring the latest rows inside Streamlit for human review.

## Technology Choices (Rationale)
- **Python + pandas** for rapid ETL on tabular daily data.
- **Parquet** feature store for compressed analytics-friendly storage.
- **HistGradientBoostingRegressor** balances speed, non-linearity handling, and low maintenance relative to deep models.
- **Permutation importance** for model explainability without relying on estimator-specific APIs.
- **Streamlit** for quick local deployment with minimal boilerplate.

## Non-Functional Requirements
- **Reproducibility**: deterministic preprocessing, pinned dependencies (`requirements.txt`).
- **Observability**: key outputs stored in `reports/` for audit.
- **Extensibility**: modular scripts allow swapping the estimator or adding new indicators without altering the full stack.
- **Performance**: pipeline completes within minutes on commodity hardware (≤ 100k rows, gradient boosting with ~100 fits).
