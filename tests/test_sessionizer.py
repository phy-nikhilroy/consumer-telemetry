"""
Unit tests for event sessionization and cross-app shopping journey detection.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import duckdb
import pytest

from src.features.database import DatabaseManager
from src.features.sessionizer import sessionize_events, detect_cross_app_comparisons


def test_sessionize_events_inactivity_window():
    """Verifies that gaps <= 30 min share a session, while gaps > 30 min start a new session."""
    db = DatabaseManager(":memory:")
    db.init_schema()
    con = db.get_connection()

    try:
        t0 = datetime(2026, 10, 1, 10, 0, 0)
        t1 = t0 + timedelta(minutes=15)  # within 30 min -> session 1
        t2 = t1 + timedelta(minutes=45)  # 45 min gap -> session 2
        t3 = t2 + timedelta(minutes=10)  # within 30 min -> session 2

        events = [
            ("evt_1", "user_1", t0, "amazon_in", "prod_1", "view", 100.0, "INR", None),
            ("evt_2", "user_1", t1, "amazon_in", "prod_1", "cart", 100.0, "INR", None),
            ("evt_3", "user_1", t2, "flipkart", "prod_2", "view", 200.0, "INR", None),
            ("evt_4", "user_1", t3, "flipkart", "prod_2", "buy", 200.0, "INR", None),
        ]

        con.executemany("""
            INSERT INTO fact_events (event_id, user_id, timestamp, app_name, canonical_product_id, action, price, currency, session_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, events)

        sessionize_events(con=con, inactivity_minutes=30.0)

        rows = con.execute("SELECT event_id, session_id FROM fact_events ORDER BY event_id").fetchall()
        session_map = dict(rows)

        assert session_map["evt_1"] == session_map["evt_2"]
        assert session_map["evt_3"] == session_map["evt_4"]
        assert session_map["evt_1"] != session_map["evt_3"]
        assert session_map["evt_1"] == "user_1_1"
        assert session_map["evt_3"] == "user_1_2"
    finally:
        con.close()


def test_detect_cross_app_comparisons():
    """Verifies detection of cross-app views within 60 minutes and exclusion of views beyond window."""
    db = DatabaseManager(":memory:")
    db.init_schema()
    con = db.get_connection()

    try:
        t0 = datetime(2026, 10, 1, 12, 0, 0)
        t_comp_fast = t0 + timedelta(minutes=20)  # 20 min gap: valid cross-app comparison
        t_comp_slow = t0 + timedelta(minutes=90)  # 90 min gap: exceeds 60-minute cutoff

        events = [
            # User A: Cross-app comparison within 20 min
            ("evt_10", "user_A", t0, "amazon_in", "prod_common", "view", 100.0, "INR", "user_A_1"),
            ("evt_11", "user_A", t_comp_fast, "flipkart", "prod_common", "view", 95.0, "INR", "user_A_1"),
            # User B: Different apps but 90 min apart (> 60 min)
            ("evt_20", "user_B", t0, "walmart", "prod_slow", "view", 50.0, "USD", "user_B_1"),
            ("evt_21", "user_B", t_comp_slow, "amazon_us", "prod_slow", "view", 48.0, "USD", "user_B_2"),
        ]

        con.executemany("""
            INSERT INTO fact_events (event_id, user_id, timestamp, app_name, canonical_product_id, action, price, currency, session_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, events)

        results = detect_cross_app_comparisons(con=con, max_window_minutes=60.0)

        assert results["total_comparisons"] == 1
        assert results["cross_app_users"] == 1

        comp_row = con.execute("SELECT user_id, canonical_product_id, primary_app, compared_app, duration_minutes FROM fact_cross_app_comparisons").fetchone()
        assert comp_row[0] == "user_A"
        assert comp_row[1] == "prod_common"
        assert comp_row[2] == "amazon_in"
        assert comp_row[3] == "flipkart"
        assert float(comp_row[4]) == 20.0
    finally:
        con.close()


def test_warehouse_sessionization_integrity():
    """Verifies that all events in persistent warehouse have non-null session_id."""
    db_path = Path("data/processed/consumer.duckdb")
    if not db_path.exists():
        pytest.skip("Persistent warehouse not created yet")

    con = duckdb.connect(str(db_path))
    try:
        null_sessions = con.execute("SELECT COUNT(*) FROM fact_events WHERE session_id IS NULL").fetchone()[0]
        assert null_sessions == 0

        cross_app_count = con.execute("SELECT COUNT(*) FROM fact_cross_app_comparisons").fetchone()[0]
        assert cross_app_count > 0
    finally:
        con.close()
