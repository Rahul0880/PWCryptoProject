"""Train a classification model to predict high-volatility periods.

Why classification?
- "85% accuracy" is a classification metric, not a regression metric.
- This script converts the continuous 7-day forward volatility into a binary label:
  high_volatility = 1 if future_volatility_7d is in the top quantile.

Artifacts:
- reports/metrics_classification.json
- models/volatility_classifier.pkl

Run:
- python -m src.model_training_classification
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import DATA_PROCESSED, PROJECT_ROOT

TARGET_VOL_COL = "future_volatility_7d"
LABEL_COL = "high_volatility"
CATEGORICAL_COLS = ["crypto_name"]
EXCLUDE_COLS = {"date", "timestamp", TARGET_VOL_COL, LABEL_COL}
RANDOM_STATE = 42

METRICS_PATH = PROJECT_ROOT / "reports" / "metrics_classification.json"
MODEL_PATH = PROJECT_ROOT / "models" / "volatility_classifier.pkl"


def load_features(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def add_label(df: pd.DataFrame, quantile: float) -> pd.DataFrame:
    df = df.copy()
    # Global quantile keeps the label definition simple and evaluator-friendly.
    threshold = float(df[TARGET_VOL_COL].quantile(quantile))
    df[LABEL_COL] = (df[TARGET_VOL_COL] >= threshold).astype(int)
    df.attrs["label_threshold"] = threshold
    return df


def derive_train_test(df: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * (1 - test_size))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def build_pipeline(numeric_cols: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
        ]
    )
    clf = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=6,
        max_iter=600,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", clf)])


def tune(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> RandomizedSearchCV:
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions={
            "model__learning_rate": loguniform(0.01, 0.2),
            "model__max_depth": [3, 4, 5, 6, None],
            "model__max_leaf_nodes": randint(15, 127),
            "model__min_samples_leaf": randint(10, 200),
            "model__l2_regularization": loguniform(1e-5, 10),
            "model__max_bins": randint(64, 255),
        },
        n_iter=40,
        cv=TimeSeriesSplit(n_splits=5),
        scoring="balanced_accuracy",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    search.fit(X, y)
    return search


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, object]:
    preds = model.predict(X_test)
    metrics: dict[str, object] = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "support": {
            "n": int(len(y_test)),
            "positive_rate": float(np.mean(y_test)),
        },
    }
    # Baseline: always predict the majority class.
    majority = int(y_test.mode().iloc[0])
    baseline_preds = np.full_like(preds, majority)
    metrics["baseline_accuracy_majority"] = float(accuracy_score(y_test, baseline_preds))
    return metrics


def main() -> None:
    df = load_features(Path(DATA_PROCESSED))
    parser = argparse.ArgumentParser(description="Train a high-volatility classifier")
    parser.add_argument("--quantile", type=float, default=0.8, help="Top-quantile threshold for high volatility")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of most-recent samples for test")
    args = parser.parse_args()

    if not (0.5 < args.quantile < 1.0):
        raise SystemExit("--quantile must be between 0.5 and 1.0")
    if not (0.05 <= args.test_size <= 0.5):
        raise SystemExit("--test-size must be between 0.05 and 0.5")

    df = add_label(df, quantile=0.8)
    threshold = float(df.attrs["label_threshold"])

    numeric_cols = [
        c
        for c in df.columns
        if c not in EXCLUDE_COLS and c not in CATEGORICAL_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]

    train_df, test_df = derive_train_test(df, test_size=float(args.test_size))
    X_train = train_df[numeric_cols + CATEGORICAL_COLS]
    y_train = train_df[LABEL_COL]
    X_test = test_df[numeric_cols + CATEGORICAL_COLS]
    y_test = test_df[LABEL_COL]

    pipeline = build_pipeline(numeric_cols)
    search = tune(pipeline, X_train, y_train)
    best_model = search.best_estimator_

    metrics = evaluate(best_model, X_test, y_test)
    payload = {
        "label": {
            "name": LABEL_COL,
            "definition": f"1 if {TARGET_VOL_COL} >= global {0.8:.2f} quantile",
            "threshold": threshold,
        },
        "best_params": {
            k: (float(v) if isinstance(v, np.floating) else int(v) if isinstance(v, np.integer) else v)
            for k, v in search.best_params_.items()
        },
        "cv_accuracy": float(search.best_score_),
        **metrics,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Saved classifier to", MODEL_PATH)
    print("Saved metrics to", METRICS_PATH)
    print("Accuracy:", payload["accuracy"], "| Balanced:", payload["balanced_accuracy"], "| Baseline:", payload["baseline_accuracy_majority"])


if __name__ == "__main__":
    main()
