"""
Central project configuration, filesystem paths, and global parameters.
"""

from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"

# Database & Schema
DEFAULT_DB_PATH = PROCESSED_DATA_DIR / "consumer.duckdb"
SCHEMA_SQL_PATH = PROJECT_ROOT / "src" / "features" / "schema.sql"

# Feature store artifacts
TRAIN_FEATURES_PATH = FEATURES_DIR / "train_features.parquet"
TEST_FEATURES_PATH = FEATURES_DIR / "test_features.parquet"
SELECTED_FEATURES_PATH = FEATURES_DIR / "selected_features.json"
RFM_FEATURES_PATH = FEATURES_DIR / "rfm_features.parquet"
BEHAVIORAL_FEATURES_PATH = FEATURES_DIR / "behavioral_features.parquet"

# Financial & conversion parameters
USD_TO_INR_RATE = 83.0

# Sessionization parameters
DEFAULT_SESSION_INACTIVITY_MINUTES = 30.0
DEFAULT_CROSS_APP_WINDOW_MINUTES = 60.0

# Model artifacts
MODELS_DIR = PROJECT_ROOT / "src" / "models"
ARTIFACTS_DIR = MODELS_DIR / "artifacts"
PROPENSITY_MODEL_PATH = ARTIFACTS_DIR / "propensity_lgbm.joblib"
PROPENSITY_CALIBRATED_PATH = ARTIFACTS_DIR / "propensity_lgbm_calibrated.joblib"
SWITCHING_MODEL_PATH = ARTIFACTS_DIR / "switching_lgbm.joblib"
SEGMENTATION_MODEL_PATH = ARTIFACTS_DIR / "segmentation_kmeans.joblib"
CLV_BGNBD_PATH = ARTIFACTS_DIR / "clv_bgnbd.joblib"
CLV_GAMMAGAMMA_PATH = ARTIFACTS_DIR / "clv_gammagamma.joblib"
USER_CLV_PATH = FEATURES_DIR / "user_clv.parquet"
USER_SEGMENTS_PATH = FEATURES_DIR / "user_segments.parquet"

# Modeling parameters
PRIMARY_TARGET = "will_purchase_next_7d"
SWITCHING_TARGET = "will_switch_app_next_7d"
COUNT_TARGET = "num_purchases_next_7d"
TARGET_COLUMNS = (PRIMARY_TARGET, COUNT_TARGET, SWITCHING_TARGET)
RANDOM_SEED = 42
