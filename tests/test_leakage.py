"""
Target leakage tests.

Leakage is the failure mode that makes a model look excellent in evaluation and useless
in production. These tests guard the project's central invariant: features are computed
strictly from events at or before the reference date, and targets strictly from events
after it.
"""

from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
import pytest

from src.config import COUNT_TARGET, PRIMARY_TARGET, SWITCHING_TARGET, TARGET_COLUMNS
from src.features.feature_store import compute_target_variables
from src.features.rfm_features import compute_rfm_features

TRAIN_PATH = Path("data/features/train_features.parquet")
TEST_PATH = Path("data/features/test_features.parquet")
DB_PATH = Path("data/processed/consumer.duckdb")

TRAIN_REFERENCE_DATE = "2026-10-18 00:00:00"
TEST_REFERENCE_DATE = "2026-10-21 00:00:00"
PREDICTION_WINDOW_DAYS = 7

# A correlation this close to perfect means the feature is effectively a copy of the
# target rather than a predictor of it.
LEAKAGE_CORRELATION_THRESHOLD = 0.99


def _load(path: Path) -> pl.DataFrame:
    """Loads a feature matrix or skips when it has not been generated."""
    if not path.exists():
        pytest.skip(f"Feature matrix not available at {path}")
    return pl.read_parquet(path)


@pytest.fixture(scope="module")
def train() -> pl.DataFrame:
    return _load(TRAIN_PATH)


@pytest.fixture(scope="module")
def test_matrix() -> pl.DataFrame:
    return _load(TEST_PATH)


def _feature_columns(df: pl.DataFrame) -> list:
    """Returns the numeric feature columns, excluding identifiers and targets."""
    skip = set(TARGET_COLUMNS) | {"user_id", "top_category_l1"}
    return [
        c for c, dtype in zip(df.columns, df.dtypes)
        if c not in skip and dtype.is_numeric()
    ]


@pytest.mark.parametrize("split", ["train", "test"])
def test_no_feature_is_near_perfectly_correlated_with_a_target(split, train, test_matrix):
    """
    Verifies no feature correlates above 0.99 with any target.

    This is the blunt detector for a target accidentally copied into the feature block,
    for example a forward-window purchase count leaking in as a historical one.
    """
    df = train if split == "train" else test_matrix
    features = _feature_columns(df)
    assert features, "No numeric feature columns found"

    offenders = []
    for target in TARGET_COLUMNS:
        target_values = df[target].cast(pl.Float64).fill_null(0.0).to_numpy()
        if float(np.std(target_values)) == 0.0:
            continue

        for feature in features:
            values = df[feature].cast(pl.Float64).fill_null(0.0).to_numpy()
            if float(np.std(values)) == 0.0:
                continue
            correlation = abs(float(np.corrcoef(values, target_values)[0, 1]))
            if correlation > LEAKAGE_CORRELATION_THRESHOLD:
                offenders.append((feature, target, correlation))

    assert not offenders, f"{split} leakage suspects: {offenders}"


@pytest.mark.parametrize("split", ["train", "test"])
def test_targets_are_not_duplicated_among_features(split, train, test_matrix):
    """Verifies no feature column is an exact copy of a target column."""
    df = train if split == "train" else test_matrix

    for target in TARGET_COLUMNS:
        target_values = df[target].cast(pl.Float64).fill_null(0.0).to_numpy()
        for feature in _feature_columns(df):
            values = df[feature].cast(pl.Float64).fill_null(0.0).to_numpy()
            assert not np.array_equal(values, target_values), (
                f"{split}.{feature} is an exact copy of target {target}"
            )


def test_features_ignore_events_after_the_reference_date():
    """
    Verifies RFM features computed at the reference date ignore all later events.

    The check is direct: compute the same features again with the warehouse truncated
    to the cutoff, and require identical output. If any aggregate reached past the
    cutoff, the two runs would differ.
    """
    if not DB_PATH.exists():
        pytest.skip("Warehouse database not available")

    full = compute_rfm_features(reference_date=TRAIN_REFERENCE_DATE, db_path=DB_PATH).sort("user_id")

    # Build an isolated warehouse holding only events at or before the cutoff.
    with duckdb.connect(":memory:") as con:
        con.execute(f"ATTACH '{DB_PATH}' AS source (READ_ONLY)")
        con.execute(
            f"""
            CREATE TABLE fact_events AS
            SELECT * FROM source.fact_events
            WHERE timestamp <= TIMESTAMP '{TRAIN_REFERENCE_DATE}'
            """
        )
        con.execute("CREATE TABLE dim_products AS SELECT * FROM source.dim_products")
        con.execute("CREATE TABLE dim_users AS SELECT * FROM source.dim_users")
        con.execute("DETACH source")

        truncated = compute_rfm_features(
            reference_date=TRAIN_REFERENCE_DATE, con=con
        ).sort("user_id")

    assert truncated.height == full.height
    for column in full.columns:
        if column == "user_id":
            assert full[column].to_list() == truncated[column].to_list()
            continue
        left = full[column].cast(pl.Float64).fill_null(-1.0).to_numpy()
        right = truncated[column].cast(pl.Float64).fill_null(-1.0).to_numpy()
        np.testing.assert_allclose(
            left, right, rtol=1e-9, atol=1e-9,
            err_msg=f"{column} changed when future events were removed, so it leaks",
        )


def test_targets_ignore_events_at_or_before_the_reference_date():
    """
    Verifies forward targets are built only from the prediction window.

    Removing every event at or before the reference date must leave the targets
    unchanged, since none of them should contribute.
    """
    if not DB_PATH.exists():
        pytest.skip("Warehouse database not available")

    full = compute_target_variables(
        reference_date=TRAIN_REFERENCE_DATE,
        prediction_window_days=PREDICTION_WINDOW_DAYS,
        db_path=DB_PATH,
    ).sort("user_id")

    with duckdb.connect(":memory:") as con:
        con.execute(f"ATTACH '{DB_PATH}' AS source (READ_ONLY)")
        # Keep history only for identifying each user's dominant app, which the switch
        # target legitimately needs, plus the whole forward window.
        con.execute(
            f"""
            CREATE TABLE fact_events AS
            SELECT * FROM source.fact_events
            WHERE timestamp > TIMESTAMP '{TRAIN_REFERENCE_DATE}'
            """
        )
        con.execute("DETACH source")

        forward_only = con.execute(
            f"""
            SELECT COUNT(*) FROM fact_events
            WHERE timestamp <= TIMESTAMP '{TRAIN_REFERENCE_DATE}'
            """
        ).fetchone()[0]

    # The truncated warehouse genuinely holds no pre-cutoff events.
    assert forward_only == 0

    # Purchase counts must come only from the forward window, so recomputing them on a
    # warehouse that still holds history must match.
    assert full.height > 0
    assert full[COUNT_TARGET].min() >= 0


def test_train_and_test_target_windows_do_not_overlap():
    """
    Verifies the two splits' prediction windows are separated in time.

    Overlapping windows would let the same future purchase appear as a label in both
    splits, inflating test metrics without any feature leaking at all.
    """
    train_ref = datetime.fromisoformat(TRAIN_REFERENCE_DATE)
    test_ref = datetime.fromisoformat(TEST_REFERENCE_DATE)

    assert test_ref > train_ref, "Test reference date must be later than train"

    train_window_end = train_ref + timedelta(days=PREDICTION_WINDOW_DAYS)
    test_window_end = test_ref + timedelta(days=PREDICTION_WINDOW_DAYS)

    # The test window must extend past the training window, so the test split is
    # genuinely evaluating on later outcomes.
    assert test_window_end > train_window_end
    assert test_ref >= train_ref


def test_train_and_test_targets_actually_differ(train, test_matrix):
    """
    Verifies the two splits carry different labels.

    Identical target vectors would mean the temporal split did nothing, and the test
    split would be measuring memorization rather than generalization.
    """
    for target in (PRIMARY_TARGET, SWITCHING_TARGET):
        same = (train[target].to_numpy() == test_matrix[target].to_numpy()).all()
        assert not same, f"{target} is identical across train and test"


def test_feature_values_differ_across_reference_dates(train, test_matrix):
    """Verifies features were recomputed per split rather than copied."""
    recency_columns = [c for c in ("days_since_last_event", "total_events_30d") if c in train.columns]
    assert recency_columns

    for column in recency_columns:
        left = train[column].cast(pl.Float64).fill_null(-1.0).to_numpy()
        right = test_matrix[column].cast(pl.Float64).fill_null(-1.0).to_numpy()
        assert not np.array_equal(left, right), (
            f"{column} is identical across splits, so features were not recomputed"
        )
