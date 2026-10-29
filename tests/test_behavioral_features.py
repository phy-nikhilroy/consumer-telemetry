"""
Unit tests for cross-app behavioral and entropy feature store calculations.
"""

from datetime import datetime, timedelta
import math
from pathlib import Path
import duckdb
import polars as pl
import pytest

from src.features.database import DatabaseManager
from src.features.behavioral_features import compute_behavioral_features, save_behavioral_features


def test_entropy_and_dominant_share_isolated():
    """Verifies Shannon entropy and dominant share on known probability distributions."""
    db = DatabaseManager(":memory:")
    db.init_schema()
    con = db.get_connection()

    try:
        t0 = datetime(2026, 10, 1, 10, 0, 0)
        # User Single: 4 events strictly on amazon_in -> entropy = 0.0, dominant_app_share = 1.0
        # User Dual: 2 on amazon_in, 2 on flipkart -> entropy = 1.0, dominant_app_share = 0.5
        events = [
            ("evt_s1", "user_single", t0, "amazon_in", "p1", "view", 100.0, "INR", "s1"),
            ("evt_s2", "user_single", t0 + timedelta(minutes=5), "amazon_in", "p1", "cart", 100.0, "INR", "s1"),
            ("evt_s3", "user_single", t0 + timedelta(minutes=10), "amazon_in", "p1", "buy", 100.0, "INR", "s1"),
            ("evt_s4", "user_single", t0 + timedelta(minutes=15), "amazon_in", "p1", "view", 100.0, "INR", "s1"),

            ("evt_d1", "user_dual", t0, "amazon_in", "p1", "view", 100.0, "INR", "sd1"),
            ("evt_d2", "user_dual", t0 + timedelta(minutes=5), "amazon_in", "p1", "view", 100.0, "INR", "sd1"),
            ("evt_d3", "user_dual", t0 + timedelta(minutes=10), "flipkart", "p1", "view", 100.0, "INR", "sd1"),
            ("evt_d4", "user_dual", t0 + timedelta(minutes=15), "flipkart", "p1", "view", 100.0, "INR", "sd1"),
        ]

        con.executemany("""
            INSERT INTO fact_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, events)

        df = compute_behavioral_features(reference_date=t0 + timedelta(days=1), con=con)

        rows = {r["user_id"]: r for r in df.to_dicts()}

        # Single app validation
        single = rows["user_single"]
        assert single["num_apps_used"] == 1
        assert single["app_entropy"] == 0.0
        assert single["dominant_app_share"] == 1.0

        # Dual app validation
        dual = rows["user_dual"]
        assert dual["num_apps_used"] == 2
        assert dual["app_entropy"] == 1.0
        assert dual["dominant_app_share"] == 0.5
    finally:
        con.close()


def test_behavioral_features_warehouse_schema():
    """Verifies that all 16 behavioral features exist on warehouse dataset."""
    db_path = Path("data/processed/consumer.duckdb")
    if not db_path.exists():
        pytest.skip("Warehouse database not available")

    df = compute_behavioral_features(reference_date="2026-10-28 00:00:00", db_path=db_path)

    expected_features = [
        "num_apps_used",
        "app_entropy",
        "dominant_app_share",
        "cross_app_comparison_count",
        "chose_cheaper_app_rate",
        "view_to_cart_ratio",
        "cart_to_purchase_ratio",
        "avg_session_duration_minutes",
        "avg_time_between_sessions_hours",
        "avg_discount_pct_on_purchases",
        "pct_purchases_with_discount",
        "price_range_browsed",
        "top_category_l1",
        "category_diversity",
        "weekend_event_share",
        "evening_event_share",
    ]

    for feat in expected_features:
        assert feat in df.columns, f"Missing behavioral feature: {feat}"

    assert df.height == 50


def test_save_behavioral_features_parquet(tmp_path):
    """Verifies Parquet output writing and schema persistence."""
    df_mock = pl.DataFrame({
        "user_id": ["user_001"],
        "num_apps_used": [3],
        "app_entropy": [1.584],
    })

    out_file = tmp_path / "test_behavioral.parquet"
    saved = save_behavioral_features(df_mock, output_path=out_file)

    assert saved.exists()
    loaded = pl.read_parquet(saved)
    assert loaded.shape == (1, 3)
    assert loaded["user_id"][0] == "user_001"
