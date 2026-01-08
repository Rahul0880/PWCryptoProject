"""Model training, hyperparameter tuning, evaluation, and artifact persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import DATA_PROCESSED, FEATURE_IMPORTANCE_PATH, METRICS_PATH, PIPELINE_ARTIFACT

TARGET_COL = "future_volatility_7d"
CATEGORICAL_COLS = ["crypto_name"]
EXCLUDE_COLS = {TARGET_COL, "date", "timestamp"}
CV_SPLITS = 5
SCORING = "neg_root_mean_squared_error"
RANDOM_STATE = 42

ModelBuilder = Callable[[], TransformedTargetRegressor]


def load_features(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def derive_train_test(df: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    return train_df, test_df


def build_preprocessor(numeric_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
        ]
    )


def make_pipeline(numeric_cols: list[str], estimator) -> Pipeline:
    preprocessor = build_preprocessor(numeric_cols)
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


def wrap_with_log_target(pipeline: Pipeline) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )


def build_model_configs(numeric_cols: list[str]) -> dict[str, dict[str, Any]]:
    def hist_builder() -> TransformedTargetRegressor:
        estimator = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_depth=6,
            max_iter=800,
            random_state=RANDOM_STATE,
        )
        pipeline = make_pipeline(numeric_cols, estimator)
        return wrap_with_log_target(pipeline)

    def gbrt_builder() -> TransformedTargetRegressor:
        estimator = GradientBoostingRegressor(
            learning_rate=0.05,
            n_estimators=600,
            max_depth=4,
            subsample=0.8,
            random_state=RANDOM_STATE,
        )
        pipeline = make_pipeline(numeric_cols, estimator)
        return wrap_with_log_target(pipeline)

    return {
        "hist_gradient_boosting": {
            "builder": hist_builder,
            "params": {
                "regressor__model__learning_rate": loguniform(0.005, 0.2),
                "regressor__model__max_depth": [3, 4, 5, 6, None],
                "regressor__model__max_leaf_nodes": randint(31, 255),
                "regressor__model__min_samples_leaf": randint(20, 200),
                "regressor__model__l2_regularization": loguniform(1e-4, 1e1),
                "regressor__model__max_bins": randint(128, 255),
            },
            "n_iter": 25,
        },
        "gradient_boosting": {
            "builder": gbrt_builder,
            "params": {
                "regressor__model__learning_rate": loguniform(0.01, 0.2),
                "regressor__model__n_estimators": randint(200, 1200),
                "regressor__model__max_depth": randint(2, 6),
                "regressor__model__subsample": loguniform(0.6, 1.0),
                "regressor__model__min_samples_leaf": randint(5, 100),
            },
            "n_iter": 35,
        },
    }


def _serialize_params(params: dict[str, Any]) -> dict[str, Any]:
    serialized: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, (np.floating, np.float32, np.float64)):
            serialized[key] = float(value)
        elif isinstance(value, (np.integer, np.int32, np.int64)):
            serialized[key] = int(value)
        else:
            serialized[key] = value
    return serialized


def tune_models(
    model_configs: dict[str, dict[str, Any]],
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[RandomizedSearchCV, str, list[dict[str, Any]]]:
    cv = TimeSeriesSplit(n_splits=CV_SPLITS)
    best_search: RandomizedSearchCV | None = None
    best_name = ""
    leaderboard: list[dict[str, Any]] = []

    for name, config in model_configs.items():
        search = RandomizedSearchCV(
            estimator=config["builder"](),
            param_distributions=config["params"],
            n_iter=config.get("n_iter", 25),
            cv=cv,
            scoring=SCORING,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=1,
        )
        search.fit(X, y)
        leaderboard.append(
            {
                "model": name,
                "cv_neg_rmse": float(search.best_score_),
                "cv_rmse": float(-search.best_score_),
                "best_params": _serialize_params(search.best_params_),
            }
        )
        if best_search is None or search.best_score_ > best_search.best_score_:
            best_search = search
            best_name = name

    if best_search is None:
        raise RuntimeError("No model configurations were evaluated.")

    return best_search, best_name, leaderboard


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def save_feature_importance(model, X_valid: pd.DataFrame, y_valid: pd.Series, path: Path) -> None:
    feature_names = X_valid.columns
    importance = permutation_importance(
        model,
        X_valid,
        y_valid,
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance.importances_mean,
            "std": importance.importances_std,
        }
    )
    importance_df = importance_df.sort_values("importance", ascending=False)
    importance_df.to_csv(path, index=False)


def annotate_feature_names(model, feature_names: list[str]) -> None:
    if hasattr(model, "feature_names_in_"):
        return
    regressor = getattr(model, "regressor_", None)
    if regressor is not None and hasattr(regressor, "feature_names_in_"):
        model.feature_names_in_ = regressor.feature_names_in_
    else:
        model.feature_names_in_ = np.array(feature_names)


def main() -> None:
    data_path = Path(DATA_PROCESSED)
    df = load_features(data_path)
    numeric_cols = [
        col
        for col in df.columns
        if col not in EXCLUDE_COLS and col not in CATEGORICAL_COLS and pd.api.types.is_numeric_dtype(df[col])
    ]
    train_df, test_df = derive_train_test(df)
    X_train = train_df[numeric_cols + CATEGORICAL_COLS]
    y_train = train_df[TARGET_COL]
    X_test = test_df[numeric_cols + CATEGORICAL_COLS]
    y_test = test_df[TARGET_COL]

    model_configs = build_model_configs(numeric_cols)
    search, best_name, leaderboard = tune_models(model_configs, X_train, y_train)
    best_model = search.best_estimator_
    annotate_feature_names(best_model, list(X_train.columns))

    metrics = evaluate(best_model, X_test, y_test)
    metrics.update(
        {
            "best_model": best_name,
            "cv_neg_rmse": float(search.best_score_),
            "cv_rmse": float(-search.best_score_),
            "best_params": _serialize_params(search.best_params_),
            "leaderboard": leaderboard,
        }
    )

    PIPELINE_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, PIPELINE_ARTIFACT)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_feature_importance(best_model, X_test, y_test, FEATURE_IMPORTANCE_PATH)

    print("Saved pipeline to", PIPELINE_ARTIFACT)
    print("Best model:", best_name)
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
