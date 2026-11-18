"""
Unit tests for user behavioral telemetry simulation.
"""

from datetime import datetime, timezone
from pathlib import Path
import duckdb
import pytest

from src.ingestion.simulate_users import (
    ARCHETYPES,
    CATALOG_BY_TIER,
    DORMANCY_ONSET_RANGE,
    DORMANCY_SHARE,
    PRICE_TIERS,
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
        "raw_text",
        "true_sku_key",
        "true_action",
        "true_price",
        "true_currency",
        "true_app",
    ]
    for key in required_keys:
        assert key in first_event

    # Test archetype behavior: window_shopper should rarely buy
    window_events = [e for e in events if e["archetype"] == "window_shopper"]
    window_buys = [e for e in window_events if e["true_action"] == "buy"]
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
        assert p.price_tier in PRICE_TIERS
        # Bounds come from the module constants rather than literals, so retuning the
        # simulation does not silently break this test.
        assert (
            p.dormant_from_day is None
            or DORMANCY_ONSET_RANGE[0] <= p.dormant_from_day <= DORMANCY_ONSET_RANGE[1]
        )

    # Traits must actually vary across the population, otherwise there is no signal
    # for the downstream models to learn.
    assert len({round(p.buy_propensity, 4) for p in profiles}) > 100
    assert len({round(p.activity_level, 4) for p in profiles}) > 100

    # A minority of the population goes dormant partway through the window.
    dormant = [p for p in profiles if p.dormant_from_day is not None]
    assert len(dormant) == int(len(profiles) * DORMANCY_SHARE)
    assert 0 < len(dormant) < len(profiles) * 0.5

    # Every price tier must be represented, or Gamma-Gamma loses the between-customer
    # spend heterogeneity it needs.
    assert {p.price_tier for p in profiles} == set(PRICE_TIERS)


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
        if e["true_action"] == "buy":
            bucket["buys"] += 1

    loyalist_rate = by_archetype["loyalist"]["buys"] / by_archetype["loyalist"]["events"]
    window_rate = by_archetype["window_shopper"]["buys"] / by_archetype["window_shopper"]["events"]

    assert loyalist_rate > window_rate
    # Purchases must stay a realistic minority of all recorded interactions.
    total_buys = sum(b["buys"] for b in by_archetype.values())
    assert 0.0 < total_buys / len(events) < 0.10


def test_catalog_tiers_partition_the_product_range():
    """Verifies every catalog item lands in exactly one non-empty price tier."""
    total = sum(len(items) for items in CATALOG_BY_TIER.values())
    assert set(CATALOG_BY_TIER) == set(PRICE_TIERS)
    assert all(len(items) > 0 for items in CATALOG_BY_TIER.values())

    # Tiers must be disjoint and together cover the whole catalog.
    ids = [id(item) for items in CATALOG_BY_TIER.values() for item in items]
    assert len(ids) == len(set(ids))
    assert total == len(ids)

    # Each tier's prices must respect its own INR bounds.
    for tier, (low, high) in PRICE_TIERS.items():
        for item in CATALOG_BY_TIER[tier]:
            assert low <= float(item["base_inr"]) < high


def test_price_tier_affinity_stabilizes_order_values():
    """
    Verifies tier affinity keeps per-user spend levels distinguishable.

    Gamma-Gamma models each customer as having a stable mean order value. If users drew
    uniformly from a catalog spanning 1,099 to 129,999 INR, within-customer variance
    would swamp between-customer variance and the model could not be identified.
    """
    profiles = create_user_profiles(num_users=60, seed=5)

    tier_means = {}
    for tier in PRICE_TIERS:
        tier_profiles = [p for p in profiles if p.price_tier == tier]
        if not tier_profiles:
            continue
        picks = [float(p.choose_product()["base_inr"]) for p in tier_profiles for _ in range(300)]
        tier_means[tier] = sum(picks) / len(picks)

    # Mean selected price must increase from budget through premium.
    assert tier_means["budget"] < tier_means["mid"] < tier_means["premium"]
