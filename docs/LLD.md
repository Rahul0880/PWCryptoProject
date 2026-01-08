# Low-Level Design (LLD)

## Modules & Responsibilities

### `src/config.py`
- Centralizes `Path` objects for raw data, feature store, reports, and models to avoid hard-coded strings.

### `src/data_preprocessing.py`
1. **`load_raw_data`** – reads CSV, normalizes headers, drops unnamed index column.
2. **`clean_data`** – enforces dtypes, sorts, replaces invalid values, performs per-asset interpolation and bidirectional filling using `GroupBy.transform`.
3. **`_compute_rsi`, `_apply_asset_features`** – helper routines to compute RSI, ATR, momentum, rolling stats, Bollinger bandwidth, liquidity ratios, and the future-volatility label.
4. **`engineer_features`** – concatenates per-asset feature frames, adds calendar-derived columns, log market cap, and lifecycle counters.
5. **`finalize_dataset`** – drops lookahead NaNs, sorts chronologically, writes parquet.
6. **`run_pipeline` / `main`** – orchestration entry point used by CLI (`python -m src.data_preprocessing`).

### `src/eda.py`
- `load_dataset` enforces schema.
- Plot functions (`plot_price_trends`, `plot_volatility_distribution`, etc.) save PNGs under `reports/figures/`.
- `build_eda_report` composes markdown stats; `main` wires everything to `reports/EDA.md`.

### `src/model_training.py`
1. **`load_features`** – pulls parquet feature store.
2. **`build_pipeline`** – defines `ColumnTransformer` (numeric scaler + one-hot for `crypto_name`) and the boosting estimator.
3. **`derive_train_test`** – chronological hold-out split (80/20).
4. **`tune_model`** – `RandomizedSearchCV` over learning rate, depth, leaves, and `l2_regularization` with `TimeSeriesSplit` to preserve order.
5. **`evaluate`** – computes RMSE, MAE, and $R^2$.
6. **`save_feature_importance`** – permutation importance on the validation set using original features.
7. **`main`** – orchestrates loading, training, evaluation, serialization (`models/volatility_pipeline.pkl`), and metric logging (`reports/metrics.json`).

### `app/streamlit_app.py`
- Injects repo root into `sys.path`, caches model/data loaders, computes on-demand predictions, renders overview KPIs + detailed charts filtered by user selections.

## Data Contracts
- **Feature Store Schema** (subset):
  - `date` (datetime64[ns])
  - `crypto_name` (string)
  - Engineered fields (`return_std_7`, `bollinger_bandwidth_20`, ...)
  - Target `future_volatility_7d` (float)
- All modeling scripts expect the parquet file to include both engineered predictors and target; Streamlit requires identical schema for scoring.

## Error Handling & Logging
- Scripts rely on pandas/sklearn exceptions for hard failures; CLI surfaces stack traces.
- Key checkpoints print dataset shape, date range, and metrics to STDOUT for quick validation.

## Execution Commands
1. `/home/asus/rahul/.venv/bin/python -m src.data_preprocessing`
2. `/home/asus/rahul/.venv/bin/python -m src.eda`
3. `/home/asus/rahul/.venv/bin/python -m src.model_training`
4. `streamlit run app/streamlit_app.py`

## Extensibility Hooks
- Add new indicators inside `_apply_asset_features`.
- Swap/stack models by editing `build_pipeline` and `tune_model`.
- Introduce additional deployment targets by reusing `models/volatility_pipeline.pkl`.
