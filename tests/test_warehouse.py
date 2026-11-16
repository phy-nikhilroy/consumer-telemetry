"""
Tests for the serving warehouse: bootstrap, the live write path, and read queries.

The central property is that a live event is stored exactly as the batch pipeline would
store it. `test_live_writes_match_batch_sessionizer` replays a simulated event log one
event at a time through `record_event` and asserts that every session ID and every
cross-app comparison matches what the batch sessionizer produced for the same events.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pytest

from src.api.warehouse import (
    CLOCK_SIMULATED,
    CLOCK_WALL,
    ServingClock,
    ServingWarehouse,
    ensure_serving_warehouse,
    normalized_price_inr,
    to_naive_utc,
    warehouse_action,
)
from src.config import SCHEMA_SQL_PATH, USD_TO_INR_RATE


def _product(product_id: str = "prod_test", action: str = "view", price: float = 100.0, currency: str = "INR") -> dict:
    return {
        "canonical_product_id": product_id,
        "canonical_name": "Test Brand Test Product",
        "brand": "Test Brand",
        "category_l1": "Electronics",
        "category_l2": "Smartphones",
        "action": action,
        "price": price,
        "currency": currency,
    }


def test_helpers():
    assert warehouse_action("buy") == "buy"
    assert warehouse_action("wishlist") == "view"
    assert warehouse_action("search") == "view"
    assert normalized_price_inr(10.0, "USD") == pytest.approx(10.0 * USD_TO_INR_RATE)
    assert normalized_price_inr(10.0, "INR") == 10.0
    assert normalized_price_inr(None, "USD") is None
    aware = datetime(2026, 10, 1, 12, 0, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    assert to_naive_utc(aware) == datetime(2026, 10, 1, 6, 30)
    assert to_naive_utc(None) is None


def test_ensure_serving_warehouse_copies_then_reuses(tiny_warehouse: Path, tmp_path: Path):
    target = tmp_path / "copy.duckdb"
    assert ensure_serving_warehouse(db_path=target, source_db_path=tiny_warehouse) == "copied"
    assert ensure_serving_warehouse(db_path=target, source_db_path=tiny_warehouse) == "existing"

    source = duckdb.connect(str(tiny_warehouse), read_only=True)
    copy = duckdb.connect(str(target), read_only=True)
    try:
        for table in ("fact_events", "dim_products", "dim_users", "fact_cross_app_comparisons"):
            query = f"SELECT COUNT(*) FROM {table}"
            assert copy.execute(query).fetchone() == source.execute(query).fetchone()
        indexes = {row[0] for row in copy.execute("SELECT index_name FROM duckdb_indexes()").fetchall()}
        assert "idx_fact_user_ts" in indexes
    finally:
        source.close()
        copy.close()


def test_ensure_serving_warehouse_refuses_batch_path(tiny_warehouse: Path):
    with pytest.raises(ValueError):
        ensure_serving_warehouse(db_path=tiny_warehouse, source_db_path=tiny_warehouse)


def test_ensure_serving_warehouse_reset_rebuilds(tiny_warehouse: Path, tmp_path: Path):
    target = tmp_path / "reset.duckdb"
    ensure_serving_warehouse(db_path=target, source_db_path=tiny_warehouse)
    warehouse = ServingWarehouse(target)
    warehouse.record_event("reset_user", "flipkart", _product())
    warehouse.close()
    assert ensure_serving_warehouse(db_path=target, source_db_path=tiny_warehouse, reset=True) == "copied"
    con = duckdb.connect(str(target), read_only=True)
    try:
        assert con.execute("SELECT COUNT(*) FROM fact_events WHERE user_id = 'reset_user'").fetchone()[0] == 0
    finally:
        con.close()


def test_serving_clock_modes():
    anchor = datetime(2026, 10, 27, 22, 0, 0)
    simulated = ServingClock(anchor, mode=CLOCK_SIMULATED)
    first = simulated.now()
    assert anchor <= first < anchor + timedelta(seconds=5)
    assert simulated.now() >= first

    wall = ServingClock(anchor, mode=CLOCK_WALL)
    assert abs((wall.now() - datetime.now(timezone.utc).replace(tzinfo=None)).total_seconds()) < 5

    with pytest.raises(ValueError):
        ServingClock(anchor, mode="lunar")


def test_live_writes_match_batch_sessionizer(tiny_warehouse: Path, tmp_path: Path):
    """Replaying events one by one reproduces the batch sessions and comparisons."""
    source = duckdb.connect(str(tiny_warehouse), read_only=True)
    try:
        users = [r[0] for r in source.execute(
            "SELECT user_id FROM fact_cross_app_comparisons GROUP BY user_id ORDER BY COUNT(*) DESC LIMIT 3"
        ).fetchall()]
        params = {"users": users}
        events = source.execute(
            """
            SELECT fe.user_id, fe.timestamp, fe.app_name, fe.canonical_product_id, fe.action,
                   CAST(fe.price AS DOUBLE), fe.currency, fe.session_id
            FROM fact_events fe WHERE fe.user_id IN (SELECT UNNEST($users))
            ORDER BY fe.timestamp, fe.event_id
            """,
            params,
        ).fetchall()
        batch_comparisons = set(source.execute(
            """
            SELECT user_id, canonical_product_id, primary_app, compared_app, first_viewed, last_viewed, session_id
            FROM fact_cross_app_comparisons WHERE user_id IN (SELECT UNNEST($users))
            """,
            params,
        ).fetchall())
    finally:
        source.close()
    assert batch_comparisons, "fixture should contain cross-app comparisons to check"

    empty = tmp_path / "replay.duckdb"
    con = duckdb.connect(str(empty))
    con.execute(SCHEMA_SQL_PATH.read_text(encoding="utf-8"))
    con.close()

    warehouse = ServingWarehouse(empty)
    try:
        for user_id, ts, app, product_id, action, price, currency, batch_session in events:
            stored = warehouse.record_event(
                user_id, app, _product(product_id, action, price, currency), timestamp=ts
            )
            assert stored["session_id"] == batch_session, (user_id, ts)

        live_comparisons = set(warehouse.con.execute(
            """
            SELECT user_id, canonical_product_id, primary_app, compared_app, first_viewed, last_viewed, session_id
            FROM fact_cross_app_comparisons
            """
        ).fetchall())
    finally:
        warehouse.close()
    assert live_comparisons == batch_comparisons


def test_record_event_upserts_dimensions(serving_db: Path):
    warehouse = ServingWarehouse(serving_db)
    try:
        before = warehouse.counts()
        first = warehouse.record_event("brand_new", "amazon_us", _product("new_prod", "wishlist", 10.0, "USD"))
        assert first["is_new_user"] is True
        assert first["stored_action"] == "view"
        assert first["session_id"] == "brand_new_1"

        second = warehouse.record_event(
            "brand_new", "flipkart", _product("new_prod", "buy", 900.0, "INR"),
            timestamp=first["timestamp"] + timedelta(minutes=10),
        )
        assert second["is_new_user"] is False
        assert second["session_id"] == "brand_new_1"
        assert second["cross_app_comparisons_detected"] == 1

        third = warehouse.record_event(
            "brand_new", "flipkart", _product("new_prod", "view"),
            timestamp=first["timestamp"] + timedelta(hours=3),
        )
        assert third["session_id"] == "brand_new_2"

        after = warehouse.counts()
        assert after["events"] == before["events"] + 3
        assert after["users"] == before["users"] + 1
        assert after["products"] == before["products"] + 1
        assert after["live_events"] == before["live_events"] + 3

        total, first_seen = warehouse.con.execute(
            "SELECT total_events, first_seen FROM dim_users WHERE user_id = 'brand_new'"
        ).fetchone()
        assert total == 3 and first_seen == first["timestamp"]
        avg_price = warehouse.scalar("SELECT CAST(avg_price AS DOUBLE) FROM dim_products WHERE canonical_product_id = 'new_prod'")
        assert avg_price == pytest.approx(10.0 * USD_TO_INR_RATE)
    finally:
        warehouse.close()


def test_record_event_rolls_back_on_failure(serving_db: Path, monkeypatch: pytest.MonkeyPatch):
    warehouse = ServingWarehouse(serving_db)
    try:
        before = warehouse.counts()

        def boom(*args, **kwargs):
            raise RuntimeError("simulated failure")

        monkeypatch.setattr(warehouse, "_record_comparisons", boom)
        with pytest.raises(RuntimeError):
            warehouse.record_event("rollback_user", "flipkart", _product("rollback_prod"))
        assert warehouse.counts() == before
        assert not warehouse.user_exists("rollback_user")
    finally:
        warehouse.close()


def test_reference_date_covers_future_event(serving_db: Path):
    warehouse = ServingWarehouse(serving_db)
    try:
        future = warehouse.clock.now() + timedelta(days=2)
        warehouse.record_event("time_traveller", "walmart", _product(), timestamp=future)
        assert warehouse.reference_date() >= future
        assert warehouse.latest_event_timestamp() == future
    finally:
        warehouse.close()


def test_read_queries(serving_db: Path):
    warehouse = ServingWarehouse(serving_db)
    try:
        users = warehouse.list_users(limit=5)
        assert len(users) == 5
        assert users[0]["last_seen"] >= users[-1]["last_seen"]

        recent = warehouse.recent_events(limit=3)
        assert len(recent) == 3
        assert recent[0]["timestamp"] >= recent[-1]["timestamp"]

        activity = warehouse.activity_summary(window_hours=24)
        assert activity["events_in_window"] > 0
        assert activity["active_users_in_window"] <= activity["users"]
        assert activity["last_purchase"] is not None

        user_id = users[0]["user_id"]
        profile = warehouse.user_activity(user_id, warehouse.reference_date())
        assert profile["total_events"] == sum(a["events"] for a in profile["app_usage"])
        assert sum(a["share"] for a in profile["app_usage"]) == pytest.approx(1.0, abs=1e-3)
        assert 1 <= len(profile["top_categories"]) <= 3
        assert warehouse.user_exists(user_id)
        assert not warehouse.user_exists("nobody")
    finally:
        warehouse.close()
