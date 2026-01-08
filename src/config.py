from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "cryptocurrency_prices.csv"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed" / "volatility_features.parquet"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
EDA_REPORT_PATH = PROJECT_ROOT / "reports" / "EDA.md"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"
FEATURE_IMPORTANCE_PATH = PROJECT_ROOT / "reports" / "feature_importances.csv"
PIPELINE_ARTIFACT = PROJECT_ROOT / "models" / "volatility_pipeline.pkl"
