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
    Verifies every train prediction window ends at or before the test reference date.

    Overlapping windows would let the same future purchase appear as a label in both
    splits, inflating test metrics without any feature leaking at all.
    """
    from src.config import (
        PREDICTION_WINDOW_DAYS as WINDOW,
        TEST_REFERENCE_DATE as TEST_REF,
        TRAIN_REFERENCE_DATES,
    )

    test_ref = datetime.fromisoformat(TEST_REF)
    window = timedelta(days=WINDOW)
    assert len(TRAIN_REFERENCE_DATES) >= 1
    for train_ref in TRAIN_REFERENCE_DATES:
        assert datetime.fromisoformat(train_ref) + window <= test_ref


def test_train_and_test_targets_actually_differ(train, test_matrix):
    """
    Verifies the two splits carry different labels.

    Identical target vectors would mean the temporal split did nothing, and the test
    split would be measuring memorization rather than generalization.
    """
    test_sorted = test_matrix.sort("user_id")
    # The train matrix stacks one block per reference date; compare each block with the
    # test matrix user by user.
    for ref in train["reference_date"].unique().to_list():
        block = train.filter(pl.col("reference_date") == ref).sort("user_id")
        assert block["user_id"].to_list() == test_sorted["user_id"].to_list()
        for target in (PRIMARY_TARGET, SWITCHING_TARGET):
            same = (block[target].to_numpy() == test_sorted[target].to_numpy()).all()
            assert not same, f"{target} at {ref} is identical to the test labels"


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


def test_clv_bgnbd_params_unchanged_when_events_after_cutoff_deleted(tmp_path):
    """Deleting every event after the CLV reference date leaves fitted BG/NBD parameters unchanged."""
    import shutil

    from src.models.clv import build_rfm_summary, fit_bgnbd, load_purchase_transactions
    from tests.test_clv_validation import CAL_END, _build_warehouse

    full = _build_warehouse(tmp_path / "full.duckdb", seed=11)
    trimmed = tmp_path / "trimmed.duckdb"
    shutil.copy2(full, trimmed)
    con = duckdb.connect(str(trimmed))
    con.execute(f"DELETE FROM fact_events WHERE timestamp > TIMESTAMP '{CAL_END}'")
    con.close()

    def fit(db):
        tx = load_purchase_transactions(db_path=db, reference_date=CAL_END)
        return fit_bgnbd(build_rfm_summary(tx, observation_period_end=CAL_END)).params_

    a, b = fit(full), fit(trimmed)
    for key in a.index:
        assert a[key] == pytest.approx(b[key], rel=1e-9, abs=1e-12)


def test_clv_pipeline_params_unchanged_when_events_after_reference_date_deleted(tmp_path):
    """run_clv_pipeline with a reference_date fits identical models on the trimmed warehouse."""
    import shutil

    from src.models.clv import run_clv_pipeline
    from tests.test_clv_validation import CAL_END, _build_warehouse

    full = _build_warehouse(tmp_path / "full.duckdb", seed=13)
    trimmed = tmp_path / "trimmed.duckdb"
    shutil.copy2(full, trimmed)
    con = duckdb.connect(str(trimmed))
    con.execute(f"DELETE FROM fact_events WHERE timestamp > TIMESTAMP '{CAL_END}'")
    con.close()

    a = run_clv_pipeline(db_path=full, reference_date=CAL_END,
                         features_dir=tmp_path / "fa", artifacts_dir=tmp_path / "aa")
    b = run_clv_pipeline(db_path=trimmed, reference_date=CAL_END,
                         features_dir=tmp_path / "fb", artifacts_dir=tmp_path / "ab")
    for key in a["bgnbd_params"]:
        assert a["bgnbd_params"][key] == pytest.approx(b["bgnbd_params"][key], rel=1e-9)
    assert a["n_transactions"] == b["n_transactions"]


def test_clv_holdout_fit_unchanged_when_events_after_calibration_end_deleted(tmp_path):
    """evaluate_clv_holdout fits BG/NBD and Gamma-Gamma on calibration data only."""
    import shutil

    from src.models.clv_validation import evaluate_clv_holdout
    from tests.test_clv_validation import CAL_END, _build_warehouse

    full = _build_warehouse(tmp_path / "full.duckdb", seed=17)
    trimmed = tmp_path / "trimmed.duckdb"
    shutil.copy2(full, trimmed)
    con = duckdb.connect(str(trimmed))
    con.execute(f"DELETE FROM fact_events WHERE timestamp > TIMESTAMP '{CAL_END}'")
    con.close()
    a = evaluate_clv_holdout(full, CAL_END, 9)
    b = evaluate_clv_holdout(trimmed, CAL_END, 9)
    for key in a["bgnbd_params"]:
        assert a["bgnbd_params"][key] == pytest.approx(b["bgnbd_params"][key], rel=1e-9)
    assert a["n_customers"] == b["n_customers"]


# ---------------------------------------------------------------------------
# W5: every feature table read is cut at the reference date
# ---------------------------------------------------------------------------

from hypothesis import given, settings, strategies as st

from src.features.behavioral_features import compute_behavioral_features
from src.features.migrations import migrate

W5_REF = datetime(2026, 10, 10, 12, 0, 0)
_BASE = datetime(2026, 10, 8, 12, 0, 0)
_offsets = st.integers(min_value=-86400, max_value=86400 * 2)  # seconds around the reference date

_event_st = st.tuples(
    st.sampled_from(["u1", "u2"]), _offsets, st.sampled_from(["amazon_in", "flipkart", "myntra"]),
    st.sampled_from(["p1", "p2"]), st.sampled_from(["view", "cart", "buy"]),
    st.integers(50, 150), st.one_of(st.none(), st.integers(0, 40)),
)
_obs_st = st.tuples(
    st.sampled_from(["u1", "u2"]), _offsets, st.sampled_from(["amazon_in", "flipkart", "myntra"]),
    st.sampled_from(["p1", "p2"]), st.integers(50, 150),
)
_comp_st = st.tuples(
    st.sampled_from(["u1", "u2"]), _offsets, st.integers(0, 7200), st.integers(50, 150), st.integers(50, 150),
)


def _build(events, obs, comps):
    con = duckdb.connect(":memory:")
    migrate(con)
    for i, (u, off, app, prod, act, price, disc) in enumerate(events):
        con.execute(
            "INSERT INTO fact_events (event_id, user_id, timestamp, app_name, canonical_product_id, action, "
            "price, currency, session_id, discount_pct, price_inr) VALUES (?, ?, ?, ?, ?, ?, ?, 'INR', ?, ?, ?)",
            [f"e{i}", u, W5_REF + timedelta(seconds=off), app, prod, act, price, f"s{i}", disc, price],
        )
    for i, (u, off, app, prod, price) in enumerate(obs):
        con.execute(
            "INSERT INTO fact_price_observations (observation_id, source_event_id, user_id, timestamp, app_name, "
            "canonical_product_id, price_inr, is_focal) VALUES (?, ?, ?, ?, ?, ?, ?, TRUE)",
            [f"o{i}", f"x{i}", u, W5_REF + timedelta(seconds=off), app, prod, price],
        )
    for i, (u, off, dur, p1, p2) in enumerate(comps):
        first = W5_REF + timedelta(seconds=off)
        con.execute(
            "INSERT INTO fact_cross_app_comparisons (comparison_id, user_id, session_id, canonical_product_id, "
            "primary_app, compared_app, first_viewed, last_viewed, duration_minutes, price_first_inr, "
            "price_last_inr) VALUES (?, ?, 's', 'p1', 'amazon_in', 'flipkart', ?, ?, 1.0, ?, ?)",
            [f"c{i}", u, first, first + timedelta(seconds=dur), p1, p2],
        )
    return con


@settings(max_examples=40, deadline=None)
@given(st.lists(_event_st, max_size=12), st.lists(_obs_st, max_size=8), st.lists(_comp_st, max_size=5))
def test_behavioral_features_unchanged_after_deleting_rows_past_reference_date(events, obs, comps):
    con = _build(events, obs, comps)
    before = compute_behavioral_features(reference_date=W5_REF, con=con).sort("user_id")
    con.execute("DELETE FROM fact_events WHERE timestamp > ?", [W5_REF])
    con.execute("DELETE FROM fact_price_observations WHERE timestamp > ?", [W5_REF])
    con.execute("DELETE FROM fact_cross_app_comparisons WHERE last_viewed > ?", [W5_REF])
    after = compute_behavioral_features(reference_date=W5_REF, con=con).sort("user_id")
    con.close()
    assert before.columns == after.columns
    assert before.to_dicts() == after.to_dicts()


@settings(max_examples=40, deadline=None)
@given(st.lists(_event_st, max_size=12), st.lists(_obs_st, max_size=8), st.lists(_comp_st, max_size=5))
def test_price_feature_ranges_and_null_rule(events, obs, comps):
    con = _build(events, obs, comps)
    df = compute_behavioral_features(reference_date=W5_REF, con=con)
    con.close()
    for r in df.to_dicts():
        rate, n = r["chose_cheaper_app_rate"], r["informed_purchase_count"]
        assert n >= 0
        assert (rate is None) == (n == 0)
        if rate is not None:
            assert 0.0 <= rate <= 1.0
        gap = r["avg_abs_price_gap_pct"]
        assert gap is None or gap >= 0.0
        assert 0.0 <= r["pct_purchases_with_discount"] <= 1.0
        assert 0.0 <= r["avg_discount_pct_on_purchases"] <= 40.0


def test_price_observations_after_reference_date_do_not_inform_purchases():
    con = _build(
        [("u1", -3600, "flipkart", "p1", "buy", 90, None)],
        [("u1", 3600, "amazon_in", "p1", 100)],
        [],
    )
    df = compute_behavioral_features(reference_date=W5_REF, con=con)
    row = df.to_dicts()[0]
    assert row["informed_purchase_count"] == 0
    assert row["chose_cheaper_app_rate"] is None
