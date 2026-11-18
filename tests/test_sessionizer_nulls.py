"""Null-product events must never produce cross-app comparisons."""

from datetime import datetime, timedelta

import duckdb

from src.features.sessionizer import detect_cross_app_comparisons


def test_null_product_events_give_no_comparison():
    con = duckdb.connect(":memory:")
    con.execute(
        "CREATE TABLE fact_events (event_id VARCHAR, user_id VARCHAR, timestamp TIMESTAMP, "
        "app_name VARCHAR, canonical_product_id VARCHAR, session_id VARCHAR, price_inr DECIMAL(18,2))"
    )
    t0 = datetime(2026, 1, 1, 10, 0)
    con.executemany(
        "INSERT INTO fact_events VALUES (?, 'u', ?, ?, NULL, 'u_1', NULL)",
        [("e1", t0, "flipkart"), ("e2", t0 + timedelta(minutes=5), "amazon_in")],
    )
    result = detect_cross_app_comparisons(con=con, max_window_minutes=60.0)
    assert result["total_comparisons"] == 0
