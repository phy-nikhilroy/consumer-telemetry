"""
Unit tests for unified feature store assembly, forward targets, and temporal splitting.
"""

from datetime import datetime, timedelta
from pathlib import Path
import duckdb
import polars as pl
import pytest

from src.features.database import DatabaseManager
from src.features.feature_store import (
    compute_target_variables,
    build_feature_matrix,
    generate_train_test_datasets,
    get_modeling_matrices,
)


def test_compute_target_variables_temporal_boundary():
    """Verifies forward targets count purchases strictly in (reference_date, reference_date + 7d]."""
    db = DatabaseManager(":memory:")
    db.init_schema()
    con = db.get_connection()

    try:
        ref_date = datetime(2026, 10, 10, 0, 0, 0)
        t_before = ref_date - timedelta(days=2)
        t_target_window = ref_date + timedelta(days=3)
        t_beyond_window = ref_date + timedelta(days=10)

        events = [
            # Event before ref: used for dominant app
            ("evt_1", "user_target_test", t_before, "amazon_in", "p1", "view", 100.0, "INR", "s1"),
            # Purchase within 7d window: MUST be counted in targets
            ("evt_2", "user_target_test", t_target_window, "flipkart", "p1", "buy", 200.0, "INR", "s2"),
            # Purchase beyond 7d window: MUST NOT be counted
            ("evt_3", "user_target_test", t_beyond_window, "flipkart", "p1", "buy", 300.0, "INR", "s3"),
        ]

        con.executemany("""
            INSERT INTO fact_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, events)

        targets_df = compute_target_variables(
            reference_date=ref_date,
            prediction_window_days=7,
            con=con,
        )

        assert targets_df.height == 1
        t_row = targets_df.to_dicts()[0]

        assert t_row["will_purchase_next_7d"] == 1
        assert t_row["num_purchases_next_7d"] == 1  # Only evt_2 within 7d
        assert t_row["will_switch_app_next_7d"] == 1  # Bought on flipkart, dominant was amazon_in
    finally:
        con.close()


def test_build_feature_matrix_warehouse():
    """Verifies unified feature matrix has 30+ features and all 3 forward targets."""
    db_path = Path("data/processed/consumer.duckdb")
    if not db_path.exists():
        pytest.skip("Warehouse database not available")

    df = build_feature_matrix(reference_date="2026-10-18 00:00:00", prediction_window_days=7, db_path=db_path)

    # One row per user in the warehouse; the population size is a simulation parameter.
    with duckdb.connect(str(db_path), read_only=True) as con:
        expected_users = con.execute("SELECT COUNT(*) FROM dim_users").fetchone()[0]
    assert df.height == expected_users
    assert df.width >= 35

    # Check 3 targets
    targets = ["will_purchase_next_7d", "num_purchases_next_7d", "will_switch_app_next_7d"]
    for t in targets:
        assert t in df.columns

    # Check sample features
    assert "days_since_last_purchase" in df.columns
    assert "app_entropy" in df.columns
    assert "total_spend_30d" in df.columns


def test_train_test_split_datasets(tmp_path):
    """Verifies train and test datasets are created with valid schemas."""
    train_file = tmp_path / "train.parquet"
    test_file = tmp_path / "test.parquet"

    db_path = Path("data/processed/consumer.duckdb")
    if not db_path.exists():
        pytest.skip("Warehouse database not available")

    from src.config import TEST_REFERENCE_DATE as _TE, TRAIN_REFERENCE_DATES as _TR

    train_df, test_df = generate_train_test_datasets(
        train_ref_dates=_TR,
        test_ref_date=_TE,
        train_path=train_file,
        test_path=test_file,
        db_path=db_path,
    )

    assert train_file.exists()
    assert test_file.exists()
    with duckdb.connect(str(db_path), read_only=True) as con:
        expected_users = con.execute("SELECT COUNT(*) FROM dim_users").fetchone()[0]
    assert train_df.height == len(_TR) * expected_users
    assert test_df.height == expected_users
    assert set(train_df.columns) == set(test_df.columns)


def test_get_modeling_matrices_train_and_test():
    """Verifies get_modeling_matrices loads features and binary targets with selected features."""
    db_path = Path("data/processed/consumer.duckdb")
    if not db_path.exists():
        pytest.skip("Warehouse database not available")

    with duckdb.connect(str(db_path), read_only=True) as con:
        expected_users = con.execute("SELECT COUNT(*) FROM dim_users").fetchone()[0]

    X_train, y_train, feat_names_train = get_modeling_matrices(split="train", target_col="will_purchase_next_7d")
    assert X_train.shape[0] == expected_users
    assert X_train.shape[1] == len(feat_names_train)
    assert len(feat_names_train) == 18
    assert len(y_train) == expected_users
    assert set(pl.Series(y_train).unique().to_list()).issubset({0, 1})

    X_test, y_test, feat_names_test = get_modeling_matrices(split="test", target_col="will_purchase_next_7d")
    assert X_test.shape[0] == expected_users
    assert feat_names_train == feat_names_test



import json

import numpy as np

from src.config import RECENCY_FEATURES, RECENCY_NULL_SENTINEL_DAYS
from src.features.feature_store import fill_missing_features, print_target_distributions


def test_fill_missing_features_never_reads_a_missing_recency_as_today():
    """
    Verifies a null recency becomes the never-happened sentinel while other nulls are
    zero.

    Zero filling told the model that a user who had never purchased had purchased that
    very day; on real data that cost 61 percent of propensity PR-AUC.
    """
    df = pl.DataFrame(
        {
            "days_since_last_purchase": [None, 2.5],
            "days_since_last_cart_add": [None, None],
            "avg_order_value": [None, 120.0],
            "total_purchases_30d": [0, 3],
        }
    )
    filled = fill_missing_features(df)
    assert filled["days_since_last_purchase"].to_list() == [RECENCY_NULL_SENTINEL_DAYS, 2.5]
    assert filled["days_since_last_cart_add"].to_list() == [RECENCY_NULL_SENTINEL_DAYS] * 2
    assert filled["avg_order_value"].to_list() == [0.0, 120.0]
    assert filled["total_purchases_30d"].to_list() == [0.0, 3.0]
    assert "days_since_last_event" in RECENCY_FEATURES


def test_sentinel_exceeds_every_observed_recency():
    """Verifies the sentinel sorts after any recency the warehouse can produce."""
    path = Path("data/features/train_features.parquet")
    if not path.exists():
        pytest.skip("Feature matrix not available")
    df = pl.read_parquet(path)
    for column in RECENCY_FEATURES:
        observed = df[column].drop_nulls()
        if observed.len():
            assert float(observed.max()) < RECENCY_NULL_SENTINEL_DAYS


def test_modeling_matrix_marks_never_purchased_users_with_sentinel(tmp_path):
    """Verifies users with no purchase history carry the sentinel in the model input."""
    train_path = Path("data/features/train_features.parquet")
    if not train_path.exists():
        pytest.skip("Feature matrix not available")

    selection = tmp_path / "selected_features.json"
    with open(selection, "w", encoding="utf-8") as f:
        json.dump({"selected_features": ["days_since_last_purchase", "total_events_30d"]}, f)

    df = pl.read_parquet(train_path)
    X, _, names = get_modeling_matrices(
        split="train", target_col="will_purchase_next_7d", selected_features_path=selection
    )
    never = df["days_since_last_purchase"].is_null().to_numpy()
    assert never.any(), "the synthetic population is expected to contain never-purchasers"

    column = X[:, names.index("days_since_last_purchase")]
    assert np.all(column[never] == RECENCY_NULL_SENTINEL_DAYS)
    assert np.all(column[~never] < RECENCY_NULL_SENTINEL_DAYS)
    assert not np.isnan(X).any()


def test_get_modeling_matrices_reads_from_features_dir(tmp_path):
    """Verifies a features directory redirects both the matrix and the feature selection."""
    frame = pl.DataFrame(
        {
            "user_id": ["u1", "u2", "u3"],
            "days_since_last_purchase": [None, 1.0, 4.0],
            "total_events_30d": [3, 8, 1],
            "unused_feature": [1.0, 2.0, 3.0],
            "will_purchase_next_7d": [0, 1, 0],
        }
    )
    frame.write_parquet(tmp_path / "train_features.parquet")
    with open(tmp_path / "selected_features.json", "w", encoding="utf-8") as f:
        json.dump({"selected_features": ["days_since_last_purchase", "total_events_30d"]}, f)

    X, y, names = get_modeling_matrices(split="train", features_dir=tmp_path)
    assert names == ["days_since_last_purchase", "total_events_30d"]
    assert X.shape == (3, 2)
    assert X[0, 0] == RECENCY_NULL_SENTINEL_DAYS
    assert y.tolist() == [0, 1, 0]


def test_target_distribution_headers_follow_reference_dates(capsys):
    """
    Verifies the printed split headers come from the reference dates actually used.

    They were hardcoded as simulation day numbers, which mislabelled a 2015 run.
    """
    frame = pl.DataFrame(
        {
            "will_purchase_next_7d": [0, 1],
            "will_switch_app_next_7d": [0, 0],
            "num_purchases_next_7d": [0, 2],
        }
    )
    print_target_distributions(
        frame, frame,
        train_ref_date="2015-09-04 00:00:00",
        test_ref_date="2015-09-07 00:00:00",
        prediction_window_days=7,
    )
    out = capsys.readouterr().out
    assert "2015-09-04 -> 2015-09-11" in out
    assert "2015-09-07 -> 2015-09-14" in out
    assert "Day 20" not in out


from src.features.feature_store import censored_recency_fill


def test_censored_recency_fill_uses_longest_observed_gap():
    """Verifies the censoring value is the largest observed recency, per column."""
    df = pl.DataFrame(
        {
            "days_since_last_purchase": [None, 2.0, 11.5],
            "days_since_last_cart_add": [None, None, None],
            "total_events_30d": [1, 2, 3],
        }
    )
    fills = censored_recency_fill(df)
    assert fills == {"days_since_last_purchase": 11.5, "days_since_last_cart_add": 0.0}

    filled = fill_missing_features(df, ["days_since_last_purchase"], recency_fill=fills)
    assert filled["days_since_last_purchase"].to_list() == [11.5, 2.0, 11.5]


# ---------------------------------------------------------------------------
# E1: disjoint reference dates with a validator
# ---------------------------------------------------------------------------

import subprocess
import sys

import numpy as np
from hypothesis import HealthCheck, given, settings, strategies as st

from src.config import PREDICTION_WINDOW_DAYS, TEST_REFERENCE_DATE, TRAIN_REFERENCE_DATES

_FMT = "%Y-%m-%d %H:%M:%S"
_E1_USERS = ["u1", "u2", "u3", "u4"]


def _validate_split():
    from src.features.feature_store import validate_split

    return validate_split


def _build_warehouse(path: Path, start: datetime, end: datetime) -> Path:
    """A small deterministic warehouse with one event per user per day in [start, end]."""
    db = DatabaseManager(str(path))
    db.init_schema()
    con = db.get_connection()
    try:
        con.execute(
            "INSERT INTO dim_products VALUES ('p1','b','t','c1','c2',100.0), ('p2','b','t2','c1','c2',200.0)"
        )
        rows = []
        users = []
        n = 0
        for ui, uid in enumerate(_E1_USERS):
            users.append((uid, start, "amazon_in", 0))
            day = 0
            while start + timedelta(days=day, hours=ui) <= end:
                ts = start + timedelta(days=day, hours=ui + 1)
                if ts <= end:
                    action = "buy" if (day + ui) % 3 == 0 else "view"
                    app = "amazon_in" if (day + ui) % 2 == 0 else "flipkart"
                    rows.append((f"e{n}", uid, ts, app, "p1" if day % 2 else "p2", action, 100.0, "INR", f"s{uid}{day}"))
                    n += 1
                day += 1
        con.executemany("INSERT INTO dim_users VALUES (?,?,?,?)", users)
        con.executemany("INSERT INTO fact_events VALUES (?,?,?,?,?,?,?,?,?)", rows)
    finally:
        con.close()
    return path


@pytest.fixture(scope="module")
def e1_db(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("e1") / "w.duckdb"
    return _build_warehouse(path, datetime(2026, 9, 22), datetime(2026, 10, 28, 12))


def test_validate_split_rejects_overlapping_windows():
    with pytest.raises(ValueError):
        _validate_split()(["2026-10-18 00:00:00"], "2026-10-21 00:00:00")


def test_validate_split_accepts_adjacent_windows():
    _validate_split()(["2026-10-13 00:00:00"], "2026-10-20 00:00:00")


def test_validate_split_rejects_test_window_past_data_end():
    test_ref = datetime(2026, 10, 20)
    end = test_ref + timedelta(days=7)
    with pytest.raises(ValueError):
        _validate_split()(
            ["2026-10-13 00:00:00"], "2026-10-20 00:00:00",
            max_event_timestamp=end - timedelta(seconds=1),
        )


def test_validate_split_accepts_test_window_ending_exactly_at_data_end():
    end = datetime(2026, 10, 20) + timedelta(days=7)
    _validate_split()(["2026-10-13 00:00:00"], "2026-10-20 00:00:00", max_event_timestamp=end)
    _validate_split()(
        ["2026-10-13 00:00:00"], "2026-10-20 00:00:00",
        max_event_timestamp=end + timedelta(seconds=1),
    )


def test_config_split_is_valid():
    _validate_split()(TRAIN_REFERENCE_DATES, TEST_REFERENCE_DATE)


def test_config_constants_have_spec_values():
    assert PREDICTION_WINDOW_DAYS == 7
    assert tuple(TRAIN_REFERENCE_DATES) == ("2026-10-06 00:00:00", "2026-10-13 00:00:00")
    assert TEST_REFERENCE_DATE == "2026-10-20 00:00:00"


def test_validate_split_rejects_empty_train_refs():
    with pytest.raises(ValueError):
        _validate_split()([], "2026-10-20 00:00:00")


def test_validate_split_rejects_when_any_one_train_ref_overlaps():
    with pytest.raises(ValueError):
        _validate_split()(
            ["2026-10-06 00:00:00", "2026-10-14 00:00:00", "2026-10-13 00:00:00"],
            "2026-10-20 00:00:00",
        )


def test_validate_split_rejects_train_ref_equal_to_test_ref():
    with pytest.raises(ValueError):
        _validate_split()(["2026-10-20 00:00:00"], "2026-10-20 00:00:00")


def test_validate_split_rejects_train_ref_after_test_ref():
    with pytest.raises(ValueError):
        _validate_split()(["2026-10-27 00:00:00"], "2026-10-20 00:00:00")


def test_validate_split_honours_custom_window():
    # 3-day window: a train ref 3 days before test is fine, 7-day window would reject it.
    _validate_split()(["2026-10-17 00:00:00"], "2026-10-20 00:00:00", prediction_window_days=3)
    with pytest.raises(ValueError):
        _validate_split()(["2026-10-17 00:00:00"], "2026-10-20 00:00:00", prediction_window_days=4)


@settings(max_examples=50, deadline=None)
@given(
    train_offsets=st.lists(st.integers(-20 * 86400, 20 * 86400), min_size=1, max_size=4),
    window=st.integers(1, 14),
)
def test_validate_split_matches_definition(train_offsets, window):
    test_ref = datetime(2026, 10, 20)
    refs = [(test_ref + timedelta(seconds=o)).strftime(_FMT) for o in train_offsets]
    ok = all(
        datetime.strptime(r, _FMT) + timedelta(days=window) <= test_ref for r in refs
    )
    validate_split = _validate_split()
    if ok:
        validate_split(refs, test_ref.strftime(_FMT), prediction_window_days=window)
    else:
        with pytest.raises(ValueError):
            validate_split(refs, test_ref.strftime(_FMT), prediction_window_days=window)


@settings(max_examples=50, deadline=None)
@given(delta=st.integers(-3 * 86400, 3 * 86400), window=st.integers(1, 14))
def test_validate_split_data_end_boundary_is_inclusive(delta, window):
    test_ref = datetime(2026, 10, 20)
    max_ts = test_ref + timedelta(days=window, seconds=delta)
    validate_split = _validate_split()
    args = (["2026-09-01 00:00:00"], test_ref.strftime(_FMT))
    if delta >= 0:
        validate_split(*args, prediction_window_days=window, max_event_timestamp=max_ts)
    else:
        with pytest.raises(ValueError):
            validate_split(*args, prediction_window_days=window, max_event_timestamp=max_ts)


def test_train_matrix_stacks_reference_dates(e1_db, tmp_path):
    train_df, test_df = generate_train_test_datasets(
        train_path=tmp_path / "tr.parquet", test_path=tmp_path / "te.parquet", db_path=e1_db,
    )
    expected = 0
    for ref in TRAIN_REFERENCE_DATES:
        expected += build_feature_matrix(reference_date=ref, db_path=e1_db).height
    assert train_df.height == expected
    assert train_df["reference_date"].n_unique() == len(TRAIN_REFERENCE_DATES)
    assert train_df.select(["user_id", "reference_date"]).is_duplicated().sum() == 0
    assert test_df.select(["user_id", "reference_date"]).is_duplicated().sum() == 0
    assert test_df["reference_date"].n_unique() == 1


def test_reference_date_column_is_string_and_matches_config(e1_db, tmp_path):
    train_df, test_df = generate_train_test_datasets(
        train_path=tmp_path / "tr.parquet", test_path=tmp_path / "te.parquet", db_path=e1_db,
    )
    assert train_df["reference_date"].dtype == pl.Utf8
    assert test_df["reference_date"].dtype == pl.Utf8
    assert set(train_df["reference_date"].to_list()) == set(TRAIN_REFERENCE_DATES)
    assert set(test_df["reference_date"].to_list()) == {TEST_REFERENCE_DATE}
    # Persisted copies carry the column too.
    assert "reference_date" in pl.read_parquet(tmp_path / "tr.parquet").columns
    assert "reference_date" in pl.read_parquet(tmp_path / "te.parquet").columns


def test_train_block_for_each_ref_equals_standalone_build(e1_db, tmp_path):
    train_df, _ = generate_train_test_datasets(
        train_path=tmp_path / "tr.parquet", test_path=tmp_path / "te.parquet", db_path=e1_db,
    )
    for ref in TRAIN_REFERENCE_DATES:
        block = train_df.filter(pl.col("reference_date") == ref).drop("reference_date").sort("user_id")
        alone = build_feature_matrix(reference_date=ref, db_path=e1_db).sort("user_id")
        assert block.columns == alone.columns
        assert block.equals(alone)


def test_train_rows_unchanged_when_events_after_window_are_deleted(e1_db, tmp_path):
    """Leakage: nothing after the last train window may influence the train matrix."""
    full, _ = generate_train_test_datasets(
        train_path=tmp_path / "a.parquet", test_path=tmp_path / "b.parquet", db_path=e1_db,
    )
    cut = datetime.fromisoformat(max(TRAIN_REFERENCE_DATES)) + timedelta(days=PREDICTION_WINDOW_DAYS)
    trimmed_db = tmp_path / "trim.duckdb"
    import shutil

    shutil.copy2(e1_db, trimmed_db)
    with duckdb.connect(str(trimmed_db)) as con:
        con.execute("DELETE FROM fact_events WHERE timestamp > ?", [cut])
    ref = max(TRAIN_REFERENCE_DATES)
    a = full.filter(pl.col("reference_date") == ref).drop("reference_date").sort("user_id")
    b = build_feature_matrix(reference_date=ref, db_path=trimmed_db).sort("user_id")
    assert a.equals(b)


def test_generate_rejects_overlapping_split_before_writing(e1_db, tmp_path):
    train_p, test_p = tmp_path / "tr.parquet", tmp_path / "te.parquet"
    with pytest.raises(ValueError):
        generate_train_test_datasets(
            train_ref_dates=["2026-10-18 00:00:00"], test_ref_date="2026-10-21 00:00:00",
            train_path=train_p, test_path=test_p, db_path=e1_db,
        )
    assert not train_p.exists() and not test_p.exists()


def test_generate_rejects_test_window_past_warehouse_end(tmp_path):
    db = _build_warehouse(tmp_path / "short.duckdb", datetime(2026, 9, 22), datetime(2026, 10, 26, 23))
    train_p, test_p = tmp_path / "tr.parquet", tmp_path / "te.parquet"
    with pytest.raises(ValueError):
        generate_train_test_datasets(train_path=train_p, test_path=test_p, db_path=db)
    assert not train_p.exists() and not test_p.exists()


def test_generate_accepts_explicit_refs(e1_db, tmp_path):
    train_df, test_df = generate_train_test_datasets(
        train_ref_dates=["2026-10-13 00:00:00"], test_ref_date="2026-10-20 00:00:00",
        train_path=tmp_path / "tr.parquet", test_path=tmp_path / "te.parquet", db_path=e1_db,
    )
    assert set(train_df["reference_date"].to_list()) == {"2026-10-13 00:00:00"}
    assert set(test_df["reference_date"].to_list()) == {"2026-10-20 00:00:00"}


def test_get_modeling_groups_returns_user_ids_in_row_order(e1_db, tmp_path):
    from src.features.feature_store import get_modeling_groups

    train_df, test_df = generate_train_test_datasets(db_path=e1_db, features_dir=tmp_path)
    g_train = get_modeling_groups(split="train", features_dir=tmp_path)
    g_test = get_modeling_groups(split="test", features_dir=tmp_path)
    assert isinstance(g_train, np.ndarray)
    assert g_train.tolist() == train_df["user_id"].to_list()
    assert g_test.tolist() == test_df["user_id"].to_list()
    # Default split is train.
    assert get_modeling_groups(features_dir=tmp_path).tolist() == g_train.tolist()


def test_cli_accepts_multiple_train_refs_and_one_test_ref():
    root = Path(__file__).resolve().parent.parent
    out = subprocess.run(
        [sys.executable, "src/features/feature_store.py", "--help"],
        cwd=root, capture_output=True, text=True, timeout=60,
    ).stdout
    flat = " ".join(out.split())
    assert "--train-ref TRAIN_REF [TRAIN_REF ...]" in flat
    assert "--test-ref TEST_REF [TEST_REF ...]" not in flat
    assert "--test-ref" in flat


def test_reference_date_is_in_every_non_feature_ignore_list():
    """Every src file that ignores top_category_l1 as a non-feature must also ignore reference_date."""
    root = Path(__file__).resolve().parent.parent / "src"
    offenders = []
    for f in root.rglob("*.py"):
        text = f.read_text()
        if '"top_category_l1"' in text and '"reference_date"' not in text:
            offenders.append(str(f))
    assert offenders == []
