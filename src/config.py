"""
Central project configuration, filesystem paths, and global parameters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Real Amazon-Reviews-2023 meta shards (not committed), keyed by file stem. The brand rules
# were tuned on meta_shard0, so it is only ever train or tune data. The other three were
# downloaded untouched and supply the held-out rows. meta_fashion_shard0 is the first 180,000
# lines of the Clothing JSONL, because the dataset has no Clothing parquet folder.
AMAZON_META_DIR = DATA_DIR / "external" / "amazon_meta"
AMAZON_SHARDS: Dict[str, Path] = {
    "meta_shard0": AMAZON_META_DIR / "meta_shard0.parquet",
    "meta_cell_shard1": AMAZON_META_DIR / "meta_cell_shard1.parquet",
    "meta_electronics_shard0": AMAZON_META_DIR / "meta_electronics_shard0.parquet",
    "meta_fashion_shard0": AMAZON_META_DIR / "meta_fashion_shard0.parquet",
}
# The shard the brand rules were tuned on; never held-out data.
AMAZON_TUNED_SHARD = "meta_shard0"
# Where each shard came from, under https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/.
# The plan named the parquet folder raw_meta_Clothing_Shoes_and_Jewelry, which the dataset
# does not publish; the fashion shard is the first 180,000 lines of the Clothing JSONL,
# converted by src/ingestion/amazon_jsonl_shard.py (README, reproduce section).
AMAZON_SHARD_SOURCES: Dict[str, str] = {
    "meta_shard0": "raw_meta_Cell_Phones_and_Accessories/full-00000-of-00007.parquet",
    "meta_cell_shard1": "raw_meta_Cell_Phones_and_Accessories/full-00001-of-00007.parquet",
    "meta_electronics_shard0": "raw_meta_Electronics/full-00000-of-00010.parquet",
    "meta_fashion_shard0": "raw/meta_categories/meta_Clothing_Shoes_and_Jewelry.jsonl (first 180,000 lines)",
}

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

# Temporal split. Every train target window (ref, ref + window] must end at or before the
# test reference date, so no train label ever observes an event the test labels observe.
PREDICTION_WINDOW_DAYS = 7
# Five weekly train dates on the 60-day simulation (v1.4); the last train window ends on the
# test reference date.
TRAIN_REFERENCE_DATES = ("2026-10-12 00:00:00", "2026-10-19 00:00:00", "2026-10-26 00:00:00",
                         "2026-11-02 00:00:00", "2026-11-09 00:00:00")
TEST_REFERENCE_DATE = "2026-11-16 00:00:00"

# CLV validation. The holdout fits BG/NBD and Gamma-Gamma on purchases up to the
# calibration end and scores the following holdout window. Shrinkage pulls a user's
# observed mean order value toward the global mean with the weight of K pseudo-orders.
CLV_ORDER_VALUE_SHRINKAGE_K = 3.0
CLV_CALIBRATION_END = "2026-11-02 00:00:00"
CLV_HOLDOUT_DAYS = 21
# The spend regressor trains at CLV_REGRESSOR_TRAIN_REF against the next CLV_HOLDOUT_DAYS and is
# tested at CLV_CALIBRATION_END against the holdout window, so its windows never overlap.
CLV_REGRESSOR_TRAIN_REF = "2026-10-12 00:00:00"

# Model artifacts
MODELS_DIR = PROJECT_ROOT / "src" / "models"
ARTIFACTS_DIR = MODELS_DIR / "artifacts"
PROPENSITY_MODEL_PATH = ARTIFACTS_DIR / "propensity_lgbm.joblib"
PROPENSITY_CALIBRATED_PATH = ARTIFACTS_DIR / "propensity_lgbm_calibrated.joblib"
SWITCHING_MODEL_PATH = ARTIFACTS_DIR / "switching_lgbm.joblib"
SWITCHING_CONDITIONAL_MODEL_PATH = ARTIFACTS_DIR / "switching_conditional_lgbm.joblib"
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
SWITCHING_CONDITIONAL_TARGET = "switched_given_purchase"
CONDITIONAL_BOOTSTRAP_SAMPLES = 500
# Resamples behind the model-versus-baseline confidence intervals in every metrics file.
BASELINE_BOOTSTRAP_SAMPLES = 2000
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

# Price comparison features. A purchase is "informed" when the same user saw the same
# product priced on a different app within this many hours before buying.
CROSS_APP_PRICE_LOOKBACK_HOURS = 24
# Features whose null means "not measurable" (no informed purchase, no comparison). Zero
# and 0.5 are real values for them, so model inputs take an out-of-range sentinel instead.
MISSING_AS_SENTINEL_FEATURES = {"chose_cheaper_app_rate": -1.0, "avg_abs_price_gap_pct": -1.0}
RANDOM_SEED = 42

# Serving. The API writes live events into its own copy of the warehouse so that demo
# traffic can never leak into the batch warehouse the models are trained from. On first
# start the serving copy is cloned from DEFAULT_DB_PATH, or simulated if that is absent.
SERVING_DB_PATH = PROCESSED_DATA_DIR / "serving.duckdb"
SERVING_SIMULATION_USERS = 400
SERVING_SIMULATION_DAYS = 30
# Purchase probability bands reported as risk_segment. The test-set base rate is about
# 40 percent; at or above 0.6 the top deciles bought at roughly twice that rate, and
# below 0.3 almost nobody bought (see propensity_evaluation_summary.json).
PROPENSITY_HIGH_THRESHOLD = 0.6
PROPENSITY_LOW_THRESHOLD = 0.3
# Users with fewer events than this are scored, but flagged as low history: real
# clickstream data showed the feature store says little about sparse visitors.
LOW_HISTORY_EVENT_THRESHOLD = 20
CLV_HORIZON_DAYS = 30
# Warehouse actions are view, cart, and buy. Extracted intents outside that set are
# stored as views so they still count as engagement without inventing a new funnel step.
WAREHOUSE_ACTIONS = ("view", "cart", "buy")


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


# Catalog price tiers in INR (lower bound inclusive, upper bound exclusive). Shared by the
# simulator and the family catalog; kept here so importing the catalog stays cheap.
PRICE_TIERS: Dict[str, Tuple[float, float]] = {
    "budget": (0.0, 10_000.0),
    "mid": (10_000.0, 50_000.0),
    "premium": (50_000.0, float("inf")),
}


# Wording-tolerant product matching (src/ingestion/entity_matcher.py). Chosen as the value in
# {0.50, 0.55, ..., 0.95} that maximises F1 on tune-family pairs; see
# reports/v1_2_entity_resolution.json.
ENTITY_MATCH_THRESHOLD = 0.50


# Supervised taxonomy (src/ingestion/taxonomy_train.py). A label is `<categories[0]> > <first
# later node that is not an audience>`, so fashion is labelled by product type (Shoes,
# Clothing) rather than by Women or Men. Rows whose product node is a merchandising or store
# node are dropped. A label needs TAXONOMY_MIN_LABEL_ROWS train rows to be kept.
TAXONOMY_AUDIENCE_NODES = frozenset({"Women", "Men", "Girls", "Boys", "Baby", "Novelty & More"})
TAXONOMY_NON_PRODUCT_NODES = frozenset({
    "Wireless Trade-In Buy Box Widget", "Electronics Warranties", "Westlake", "Boot Shop",
})
TAXONOMY_ARTIFACT_PATH = PROJECT_ROOT / "src" / "ingestion" / "artifacts" / "taxonomy_lr.joblib"
TAXONOMY_ENCODER_NAME = "all-MiniLM-L6-v2"
TAXONOMY_MIN_LABEL_ROWS = 300
TAXONOMY_MAX_ROWS_PER_LABEL = 3_000
TAXONOMY_MAX_TRAIN_ROWS = 60_000
# Rows of the tune split sampled to choose the abstention threshold tau.
TAXONOMY_TUNE_ROWS = 20_000
# tau is the smallest grid value whose confident tune rows reach this accuracy. The plan set
# 0.90; on the real shards that forces 87 percent abstention against gate G1.2-03's 50
# percent ceiling, so the user chose 0.85 (2026-11-18, recorded in v2_execution_log.md).
TAXONOMY_TARGET_ACCURACY = 0.85
TAXONOMY_TAU_GRID = tuple(round(0.05 * i, 2) for i in range(1, 20))
TAXONOMY_TAU_FALLBACK = 0.95


# Learned extraction confidence (src/ingestion/confidence_model.py).
CONFIDENCE_ARTIFACT_PATH = PROJECT_ROOT / "src" / "ingestion" / "artifacts" / "confidence_lr.joblib"
CONFIDENCE_TRAIN_ROWS = 8_000
CONFIDENCE_EVAL_ROWS = 3_000
CONFIDENCE_PRICE_TOLERANCE = 0.01
CONFIDENCE_ECE_BINS = 10


# Warehouse schema version 2 (src/features/migrations.py). SCHEMA_VERSION is written to
# schema_info by migrate(); EXTRACTOR_VERSION stamps every raw_events row so a later
# re-extraction can tell which extractor produced the stored fields.
SCHEMA_VERSION = "2"
EXTRACTOR_VERSION = "2.0.0"


# Raw event loader (src/features/event_loader.py). Extraction runs in tasks of
# EVENT_LOADER_TASK_SIZE screens (one process_batch call each); writes are flushed every
# EVENT_LOADER_CHUNK_SIZE events. SIMULATION_OCR_NOISE is the character noise rate the
# simulator applies to its rendered screens.
EVENT_LOADER_TASK_SIZE = 256
EVENT_LOADER_CHUNK_SIZE = 5_000
# Extracted screens the loader remembers within one process (see event_loader._SCREEN_MEMO).
EVENT_LOADER_MEMO_SIZE = 20_000
SIMULATION_OCR_NOISE = 0.01
SIMULATION_TRUTH_PATH = PROCESSED_DATA_DIR / "simulation_truth.parquet"
SIMULATION_MAX_CONTEXT_LISTINGS = 3
# Extraction processes a simulation load uses. The tables are identical for any value.
SIMULATION_LOAD_WORKERS = 4


# Behaviour engine (src/simulation/behavior.py, v1.4). The default population and window,
# the apps every shopper can use, and the price coefficient of the purchase-app logit
# U(app) = loyalty_u * [app == primary_app] - PRICE_BETA * sensitivity_u * log(price_inr(app)).
SIMULATION_USERS = 1000
SIMULATION_DAYS = 60
SIMULATION_START = "2026-09-28 08:00:00"
SIMULATION_APPS = ("amazon_in", "flipkart", "walmart", "amazon_us")
PRICE_BETA = 4.0
# Weekly probability that a shopper switches archetype; churn hazards apply from this day.
ARCHETYPE_DRIFT_WEEKLY = 0.03
SIMULATION_CHURN_START_DAY = 7
# A buy needs a view of the same SKU by the same user within this many days.
SIMULATION_VIEW_MEMORY_DAYS = 7
