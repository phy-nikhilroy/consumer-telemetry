"""
Central project configuration, filesystem paths, and global parameters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

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

# Taxonomy classification. A product whose best cosine similarity to every taxonomy
# description falls below this floor is labelled Unknown instead of being forced into
# the nearest category. Calibrated so every synthetic catalog product still classifies
# (lowest catalog similarity 0.190) while out-of-taxonomy real listings, such as phone
# cases that previously landed in Footwear, abstain. See week_3_real_data_validation.md.
TAXONOMY_MIN_SIMILARITY = 0.16
# Similarity at which a category assignment counts as fully confident when scoring
# extraction confidence. Between the floor and this value confidence rises linearly.
TAXONOMY_CONFIDENT_SIMILARITY = 0.45

# Modeling parameters
PRIMARY_TARGET = "will_purchase_next_7d"
SWITCHING_TARGET = "will_switch_app_next_7d"
COUNT_TARGET = "num_purchases_next_7d"
TARGET_COLUMNS = (PRIMARY_TARGET, COUNT_TARGET, SWITCHING_TARGET)

# Recency features measure days since a user last did something. A null means the user
# has never done it, which is the opposite of "did it today", so these columns must not
# be zero filled. They take a sentinel larger than any observable recency instead; tree
# models only use its ordering, and a fixed constant keeps train, test, and serving
# consistent. Zero filling cost 61 percent of propensity PR-AUC on real clickstream data.
RECENCY_FEATURES = (
    "days_since_last_event",
    "days_since_last_purchase",
    "days_since_last_cart_add",
)
RECENCY_NULL_SENTINEL_DAYS = 9999.0
RANDOM_SEED = 42


def rebase_path(default: Path, directory: Optional[Union[str, Path]] = None) -> Path:
    """
    Returns `default` unchanged, or the same file name inside `directory` when given.

    Every orchestrator writes its outputs through this, so pointing a run at a second
    dataset only requires passing that run's own features and artifacts directories.
    Without it, a real-data run once silently overwrote the committed synthetic
    `selected_features.json`.
    """
    if directory is None:
        return default
    return Path(directory) / default.name
