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


def test_user_profiles_carry_latent_traits():
    """Validates that latent behavioral traits are drawn within their clamped ranges."""
    profiles = create_user_profiles(num_users=200, seed=7)

    for p in profiles:
        assert 0.45 <= p.activity_level <= 2.2
        assert 0.25 <= p.buy_propensity <= 2.6
        assert 0.40 <= p.price_sensitivity <= 2.5
        assert 0.25 <= p.app_loyalty <= 0.95
        assert p.dormant_from_day is None or 12 <= p.dormant_from_day <= 26

    # Traits must actually vary across the population, otherwise there is no signal
    # for the downstream models to learn.
    assert len({round(p.buy_propensity, 4) for p in profiles}) > 100
    assert len({round(p.activity_level, 4) for p in profiles}) > 100

    # A minority of the population goes dormant partway through the window.
    dormant = [p for p in profiles if p.dormant_from_day is not None]
    assert 0 < len(dormant) < len(profiles) * 0.3


def test_dormant_users_stop_generating_events():
    """Verifies that is_active_on_day gates event generation after dormancy onset."""
    profiles = create_user_profiles(num_users=100, seed=11)
    dormant = next(p for p in profiles if p.dormant_from_day is not None)

    assert dormant.is_active_on_day(dormant.dormant_from_day - 1) is True
    assert dormant.is_active_on_day(dormant.dormant_from_day) is False
    assert dormant.is_active_on_day(dormant.dormant_from_day + 3) is False

    always_active = next(p for p in profiles if p.dormant_from_day is None)
    assert always_active.is_active_on_day(0) is True
    assert always_active.is_active_on_day(29) is True


def test_effective_rates_respond_to_discount_depth():
    """Verifies deeper discounts lift conversion, scaled by price sensitivity."""
    profiles = create_user_profiles(num_users=50, seed=3)
    bargain = next(p for p in profiles if p.archetype == "bargain_hunter")

    assert bargain.effective_buy_rate(50) > bargain.effective_buy_rate(0)
    assert bargain.effective_cart_rate(50) > bargain.effective_cart_rate(0)

    # Rates stay valid probabilities even at maximum discount and propensity.
    for pct in (0, 10, 25, 50):
        for p in profiles:
            assert 0.0 <= p.effective_buy_rate(pct) <= 1.0
            assert 0.0 <= p.effective_cart_rate(pct) <= 1.0
            assert p.effective_buy_rate(pct) + p.effective_cart_rate(pct) <= 1.0


def test_archetype_conversion_ordering():
    """Verifies loyalists convert far more often than window shoppers in simulation."""
    events, profiles = simulate_events_dataset(
        num_users=50,
        days=10,
        seed=21,
        start_date=datetime(2026, 10, 1, tzinfo=timezone.utc),
    )

    by_archetype: dict[str, dict[str, int]] = {}
    for e in events:
        bucket = by_archetype.setdefault(e["archetype"], {"events": 0, "buys": 0})
        bucket["events"] += 1
        if e["action"] == "buy":
            bucket["buys"] += 1

    loyalist_rate = by_archetype["loyalist"]["buys"] / by_archetype["loyalist"]["events"]
    window_rate = by_archetype["window_shopper"]["buys"] / by_archetype["window_shopper"]["events"]

    assert loyalist_rate > window_rate
    # Purchases must stay a realistic minority of all recorded interactions.
    total_buys = sum(b["buys"] for b in by_archetype.values())
    assert 0.0 < total_buys / len(events) < 0.10
