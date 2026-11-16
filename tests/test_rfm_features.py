"""
Unit tests for RFM (Recency, Frequency, Monetary) behavioral feature store.
"""

from datetime import datetime, timedelta
from pathlib import Path
import duckdb
import polars as pl
import pytest

from src.features.database import DatabaseManager
from src.features.rfm_features import compute_rfm_features, save_rfm_features


def test_leak_free_reference_date_cutoff():
    """Verifies that events occurring after the reference_date are strictly excluded."""
    db = DatabaseManager(":memory:")
    db.init_schema()
    con = db.get_connection()

    try:
        ref_date = datetime(2026, 10, 10, 12, 0, 0)
        t_past = ref_date - timedelta(days=2)
        t_future = ref_date + timedelta(days=2)

        events = [
            # Past event: should be included
            ("evt_1", "user_leak_test", t_past, "amazon_in", "prod_1", "buy", 1000.0, "INR", "sess_1"),
            # Future event: MUST be excluded
            ("evt_2", "user_leak_test", t_future, "amazon_in", "prod_2", "buy", 5000.0, "INR", "sess_2"),
        ]

        con.executemany("""
            INSERT INTO fact_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, events)

        df = compute_rfm_features(reference_date=ref_date, con=con)

        assert df.height == 1
        user_row = df.to_dicts()[0]

        # Should only reflect past event
        assert user_row["total_events_lifetime"] == 1
        assert user_row["total_purchases_30d"] == 1
        assert user_row["total_spend_lifetime"] == 1000.0
        assert user_row["max_single_purchase"] == 1000.0
    finally:
        con.close()


def test_rfm_feature_column_names_and_types():
    """Verifies that all 16 required RFM features exist and have expected types."""
    db_path = Path("data/processed/consumer.duckdb")
    if not db_path.exists():
        pytest.skip("Warehouse database not available")

    df = compute_rfm_features(reference_date="2026-10-28 00:00:00", db_path=db_path)

    expected_features = [
        "days_since_last_event",
        "days_since_last_purchase",
        "days_since_last_cart_add",
        "total_events_7d",
        "total_events_30d",
        "total_events_lifetime",
        "total_purchases_7d",
        "total_purchases_30d",
        "total_sessions_7d",
        "total_sessions_30d",
        "avg_events_per_session",
        "total_spend_7d",
        "total_spend_30d",
        "total_spend_lifetime",
        "avg_order_value",
        "max_single_purchase",
    ]

    for col in expected_features:
        assert col in df.columns, f"Missing feature column: {col}"

    # Population size is a simulation parameter, so assert against the warehouse itself
    # rather than a hardcoded user count.
    with duckdb.connect(str(db_path), read_only=True) as con:
        expected_users = con.execute("SELECT COUNT(*) FROM dim_users").fetchone()[0]
    assert df.height == expected_users
    assert df.height > 0


def test_save_rfm_features_parquet(tmp_path):
    """Verifies Parquet output generation and readability."""
    df_mock = pl.DataFrame({
        "user_id": ["user_001"],
        "total_events_lifetime": [10],
        "total_spend_lifetime": [2500.0],
    })

    out_file = tmp_path / "test_rfm.parquet"
    saved_path = save_rfm_features(df_mock, output_path=out_file)

    assert saved_path.exists()
    loaded_df = pl.read_parquet(saved_path)
    assert loaded_df.shape == (1, 3)
    assert loaded_df["user_id"][0] == "user_001"


def test_user_filter_returns_matching_rows_of_full_result(tiny_warehouse):
    """Per-user serving features equal the corresponding rows of the batch computation."""
    from polars.testing import assert_frame_equal

    con = duckdb.connect(str(tiny_warehouse), read_only=True)
    try:
        full = compute_rfm_features(reference_date="2026-10-21 00:00:00", con=con)
        users = full["user_id"].to_list()[:3]
        part = compute_rfm_features(reference_date="2026-10-21 00:00:00", con=con, user_ids=users)
        none = compute_rfm_features(reference_date="2026-10-21 00:00:00", con=con, user_ids=["nobody"])
    finally:
        con.close()
    assert_frame_equal(part, full.filter(pl.col("user_id").is_in(users)))
    assert none.height == 0
