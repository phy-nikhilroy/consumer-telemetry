"""
Unit tests for DuckDB database manager, schema DDL, and telemetry loader.
"""

from pathlib import Path
import duckdb
import pytest

from src.features.database import DatabaseManager, get_db_session
from src.features.loader import load_telemetry_to_warehouse


def test_init_schema_in_memory():
    """Verifies that schema.sql creates all 3 star schema tables properly."""
    db = DatabaseManager(":memory:")
    db.init_schema()

    con = db.get_connection()
    try:
        tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
        assert "fact_events" in tables
        assert "dim_products" in tables
        assert "dim_users" in tables

        # Verify fact_events columns
        fact_cols = [row[0] for row in con.execute("DESCRIBE fact_events").fetchall()]
        expected_cols = [
            "event_id",
            "user_id",
            "timestamp",
            "app_name",
            "canonical_product_id",
            "action",
            "price",
            "currency",
            "session_id",
        ]
        for col in expected_cols:
            assert col in fact_cols
    finally:
        con.close()


def test_load_telemetry_to_warehouse(tmp_path):
    """Tests end-to-end telemetry loading into an isolated DuckDB instance."""
    db_file = tmp_path / "test_consumer.duckdb"
    parquet_path = Path("data/processed/extracted_products.parquet")

    result = load_telemetry_to_warehouse(
        parquet_path=parquet_path,
        db_path=db_file,
        num_users=50,
        seed=123
    )

    assert result["fact_events"] == 250
    assert result["dim_products"] > 0
    assert result["dim_users"] > 0

    con = duckdb.connect(str(db_file))
    try:
        # Check that user IDs follow simulated pattern
        user_ids = [row[0] for row in con.execute("SELECT DISTINCT user_id FROM fact_events").fetchall()]
        assert all(uid.startswith("user_") for uid in user_ids)

        # Check product dimension
        sample_prod = con.execute("SELECT canonical_product_id, brand, product_title FROM dim_products LIMIT 1").fetchone()
        assert sample_prod[0] is not None
    finally:
        con.close()


def test_get_db_session_context_manager(tmp_path):
    """Verifies get_db_session opens connection, executes queries, and manages cleanup."""
    db_file = tmp_path / "test_session.duckdb"

    with get_db_session(db_path=db_file) as con:
        con.execute("CREATE TABLE test_tbl (id INT, val VARCHAR)")
        con.execute("INSERT INTO test_tbl VALUES (1, 'alpha')")
        row = con.execute("SELECT val FROM test_tbl WHERE id = 1").fetchone()
        assert row[0] == "alpha"



def test_price_columns_are_wide_enough_for_real_data():
    """
    Verifies the warehouse price columns can hold real-world magnitudes.

    A real clickstream aborted the loader with a DECIMAL(10,2) overflow at 1.25e8.
    Synthetic prices peak near 1.3e5, three orders of magnitude below that ceiling, so
    the constraint was invisible until real data hit it.
    """
    db_mgr = DatabaseManager(db_path=":memory:")
    db_mgr.init_schema()
    con = db_mgr.get_connection()

    large_price = 125_399_880.00
    con.execute(
        """
        INSERT INTO fact_events (
            event_id, user_id, timestamp, app_name,
            canonical_product_id, action, price, currency, session_id
        ) VALUES (
            'evt_wide', 'user_wide', TIMESTAMP '2015-09-04 00:00:00', 'test_app',
            'prod_wide', 'buy', ?, 'INR', 'sess_wide'
        )
        """,
        [large_price],
    )
    stored = con.execute(
        "SELECT price FROM fact_events WHERE event_id = 'evt_wide'"
    ).fetchone()[0]
    assert float(stored) == pytest.approx(large_price, rel=1e-9)

    con.execute(
        """
        INSERT INTO dim_products (
            canonical_product_id, brand, product_title,
            category_l1, category_l2, avg_price
        ) VALUES ('prod_wide', NULL, NULL, 'cat_1', NULL, ?)
        """,
        [large_price],
    )
    stored_avg = con.execute(
        "SELECT avg_price FROM dim_products WHERE canonical_product_id = 'prod_wide'"
    ).fetchone()[0]
    assert float(stored_avg) == pytest.approx(large_price, rel=1e-9)
