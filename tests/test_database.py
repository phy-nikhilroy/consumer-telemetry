"""
Unit tests for DuckDB database manager, schema DDL, and telemetry loader.
"""

from pathlib import Path
import duckdb
import pytest

from src.features.database import DatabaseManager
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
