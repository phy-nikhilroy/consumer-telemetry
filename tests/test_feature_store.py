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

    # Check shape: 50 users
    assert df.height == 50
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
    assert train_df.height == 50


def test_get_modeling_matrices_train_and_test():
    """Verifies get_modeling_matrices loads features and binary targets with selected features."""
    db_path = Path("data/processed/consumer.duckdb")
    if not db_path.exists():
        pytest.skip("Warehouse database not available")

    X_train, y_train, feat_names_train = get_modeling_matrices(split="train", target_col="will_purchase_next_7d")
    assert X_train.shape[0] == 50
    assert X_train.shape[1] == len(feat_names_train)
    assert len(feat_names_train) == 18
    assert len(y_train) == 50
    assert set(pl.Series(y_train).unique().to_list()).issubset({0, 1})

    X_test, y_test, feat_names_test = get_modeling_matrices(split="test", target_col="will_purchase_next_7d")
    assert X_test.shape[0] == 50
    assert feat_names_train == feat_names_test

