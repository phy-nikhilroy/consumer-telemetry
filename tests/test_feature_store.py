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

    train_df, test_df = generate_train_test_datasets(
        train_ref_date="2026-10-18 00:00:00",
        test_ref_date="2026-10-21 00:00:00",
        train_path=train_file,
        test_path=test_file,
        db_path=db_path,
    )

    assert train_file.exists()
    assert test_file.exists()
    assert train_df.shape == test_df.shape
    with duckdb.connect(str(db_path), read_only=True) as con:
        expected_users = con.execute("SELECT COUNT(*) FROM dim_users").fetchone()[0]
    assert train_df.height == expected_users


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
