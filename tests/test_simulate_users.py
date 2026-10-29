"""
Unit tests for user behavioral telemetry simulation.
"""

from datetime import datetime, timezone
from pathlib import Path
import duckdb
import pytest

from src.ingestion.simulate_users import (
    ARCHETYPES,
    create_user_profiles,
    simulate_events_dataset,
    reload_duckdb_warehouse,
)


def test_create_user_profiles():
    """Validates user profile creation across all 5 archetypes."""
    profiles = create_user_profiles(num_users=50, seed=42)
    assert len(profiles) == 50

    # Ensure all 5 archetypes exist
    archetypes_present = {p.archetype for p in profiles}
    assert archetypes_present == set(ARCHETYPES)

    # Check user ID formatting
    assert profiles[0].user_id == "user_001"
    assert profiles[-1].user_id == "user_050"


def test_simulate_events_dataset():
    """Validates simulation event generation, volume, and archetype behavior."""
    # Test on a smaller 3-day slice for fast test execution
    events, profiles = simulate_events_dataset(
        num_users=10,
        days=3,
        seed=100,
        start_date=datetime(2026, 10, 1, tzinfo=timezone.utc),
    )

    assert len(events) >= 150
    assert len(profiles) == 10

    # Verify event structure
    first_event = events[0]
    required_keys = [
        "event_id",
        "user_id",
        "archetype",
        "timestamp",
        "app_name",
        "canonical_product_id",
        "action",
        "price",
        "currency",
    ]
    for key in required_keys:
        assert key in first_event

    # Test archetype behavior: window_shopper should rarely buy
    window_events = [e for e in events if e["archetype"] == "window_shopper"]
    window_buys = [e for e in window_events if e["action"] == "buy"]
    assert len(window_buys) <= max(1, len(window_events) * 0.05)


def test_reload_duckdb_warehouse(tmp_path):
    """Verifies that the generated simulation loads into DuckDB cleanly."""
    db_file = tmp_path / "test_sim_warehouse.duckdb"

    events, profiles = simulate_events_dataset(
        num_users=5,
        days=2,
        seed=42,
    )

    counts = reload_duckdb_warehouse(events, profiles, db_path=str(db_file))

    assert counts["fact_events"] == len(events)
    assert counts["dim_users"] == 5
    assert counts["dim_products"] > 0

    con = duckdb.connect(str(db_file))
    try:
        fact_rows = con.execute("SELECT COUNT(*) FROM fact_events").fetchone()[0]
        assert fact_rows == len(events)
    finally:
        con.close()
