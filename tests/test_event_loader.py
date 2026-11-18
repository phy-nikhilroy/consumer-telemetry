"""
Tests for the raw-event loader: extraction in the loop, one per-event write function shared
with the serving API, and a warehouse that never holds simulator ground truth.
"""

from __future__ import annotations

import random
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import duckdb
import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from src.config import USD_TO_INR_RATE, WAREHOUSE_ACTIONS
from src.features.event_loader import load_raw_events, warehouse_action
from src.features.migrations import migrate
from src.ingestion.screen_generator import render_listing, render_screen
from src.ingestion.synthetic_data import CATALOG

T0 = datetime(2026, 9, 1, 10, 0, 0)
FOUR_TABLES = ("raw_events", "fact_events", "fact_price_observations", "dim_products")


def _screen(items: List[int], app: str, action: str, seed: int = 1, discount: int = 10) -> str:
    """A rendered screen: the first item is the focal listing, the rest are context."""
    rng = random.Random(seed)
    listings = []
    for position, index in enumerate(items):
        listing_action = action if position == 0 else "view"
        listings.append(render_listing(CATALOG[index], app, listing_action, discount, 4.4, rng))
    return render_screen(listings, rng)


def _events(rows: List[Dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "event_id": pl.Utf8,
            "user_id": pl.Utf8,
            "timestamp": pl.Datetime,
            "app_name": pl.Utf8,
            "raw_text": pl.Utf8,
        },
    )


def _row(event_id: str, text: str, app: str = "amazon_in", user: str = "u1", minutes: int = 0) -> Dict[str, Any]:
    return {
        "event_id": event_id,
        "user_id": user,
        "timestamp": T0 + timedelta(minutes=minutes),
        "app_name": app,
        "raw_text": text,
    }


@pytest.fixture(autouse=True)
def _fresh_screen_memo():
    """Extraction must actually run in every test, so the loader's screen memo starts empty."""
    from src.features import event_loader

    snapshot = list(event_loader._SCREEN_MEMO.items())
    event_loader._SCREEN_MEMO.clear()
    yield
    event_loader._SCREEN_MEMO.clear()
    event_loader._SCREEN_MEMO.update(snapshot)


@pytest.fixture
def con():
    connection = duckdb.connect(":memory:")
    migrate(connection)
    yield connection
    connection.close()


def _rows(connection, sql: str) -> list:
    return connection.execute(sql).fetchall()


def _snapshot(connection) -> Dict[str, list]:
    """All four loaded tables in a comparable form, ignoring generated observation ids."""
    return {
        "raw_events": _rows(connection, "SELECT * FROM raw_events ORDER BY event_id"),
        "fact_events": _rows(
            connection,
            "SELECT event_id, user_id, timestamp, app_name, canonical_product_id, action, price, currency, "
            "mrp, discount_pct, rating, extraction_confidence, price_inr FROM fact_events ORDER BY event_id",
        ),
        "fact_price_observations": _rows(
            connection,
            "SELECT source_event_id, user_id, timestamp, app_name, canonical_product_id, price_inr, "
            "discount_pct, is_focal FROM fact_price_observations "
            "ORDER BY source_event_id, canonical_product_id, is_focal",
        ),
        "dim_products": _rows(connection, "SELECT * FROM dim_products ORDER BY canonical_product_id"),
    }


def _counts(connection) -> Dict[str, int]:
    return {t: connection.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in FOUR_TABLES}


# ---------------------------------------------------------------------------
# One hand-written event produces the expected rows in all four tables.
# ---------------------------------------------------------------------------

def test_one_raw_event_populates_all_four_tables(con):
    text = _screen([0, 1, 2], "amazon_in", "buy")
    result = load_raw_events(con, _events([_row("e1", text)]))
    assert isinstance(result, dict)
    assert all(isinstance(v, int) for v in result.values())

    raw = _rows(con, "SELECT event_id, user_id, timestamp, app_name, raw_text, extractor_version FROM raw_events")
    assert len(raw) == 1
    assert raw[0][:5] == ("e1", "u1", T0, "amazon_in", text)
    assert raw[0][5]

    fact = _rows(
        con,
        "SELECT user_id, timestamp, app_name, canonical_product_id, action, price, currency, price_inr, "
        "mrp, discount_pct, rating, extraction_confidence FROM fact_events",
    )
    assert len(fact) == 1
    user, ts, app, product_id, action, price, currency, price_inr, mrp, discount, rating, confidence = fact[0]
    assert (user, ts, app) == ("u1", T0, "amazon_in")
    assert action == "buy"
    assert currency == "INR"
    assert float(price) == pytest.approx(float(CATALOG[0]["base_inr"]), rel=0.15)
    assert float(price_inr) == pytest.approx(float(price))
    assert mrp is not None and float(mrp) > float(price)
    assert discount == 10
    assert rating == pytest.approx(4.4, abs=0.05)
    assert 0.0 <= confidence <= 1.0

    obs = _rows(
        con,
        "SELECT canonical_product_id, price_inr, is_focal, source_event_id, user_id, app_name "
        "FROM fact_price_observations",
    )
    assert len(obs) == 3
    focal = [o for o in obs if o[2]]
    assert len(focal) == 1
    assert focal[0][0] == product_id
    assert float(focal[0][1]) == pytest.approx(float(price_inr))
    assert {o[3] for o in obs} == {"e1"}
    assert {o[4] for o in obs} == {"u1"}

    dim = _rows(con, "SELECT canonical_product_id, brand, product_title, category_l1, first_seen FROM dim_products")
    assert {d[0] for d in dim} == {o[0] for o in obs}
    assert all(d[1] and d[3] and d[3] != "Unknown" for d in dim)
    assert all(d[4] == T0 for d in dim)


def test_usd_price_is_converted_for_price_inr(con):
    text = _screen([0], "amazon_us", "buy", discount=0)
    load_raw_events(con, _events([_row("e1", text, app="amazon_us")]))
    price, currency, price_inr = _rows(con, "SELECT price, currency, price_inr FROM fact_events")[0]
    assert currency == "USD"
    assert float(price_inr) == pytest.approx(float(price) * USD_TO_INR_RATE, rel=1e-4)
    obs = _rows(con, "SELECT price_inr FROM fact_price_observations")
    assert float(obs[0][0]) == pytest.approx(float(price) * USD_TO_INR_RATE, rel=1e-4)


def test_focal_row_matches_the_focal_product_among_context(con):
    text = _screen([3, 4, 5], "flipkart", "cart")
    load_raw_events(con, _events([_row("e1", text, app="flipkart")]))
    assert _rows(con, "SELECT action FROM fact_events") == [("cart",)]
    focal_ids = _rows(con, "SELECT canonical_product_id FROM fact_price_observations WHERE is_focal")
    fact_ids = _rows(con, "SELECT canonical_product_id FROM fact_events")
    assert focal_ids == fact_ids


def test_first_seen_is_earliest_event_for_a_product(con):
    text = _screen([0], "amazon_in", "view", seed=3)
    load_raw_events(con, _events([_row("e2", text, minutes=60), _row("e1", text, minutes=5)]))
    first_seen = _rows(con, "SELECT first_seen FROM dim_products")
    assert first_seen == [(T0 + timedelta(minutes=5),)]
    assert _counts(con)["dim_products"] == 1


def test_first_seen_keeps_earliest_across_separate_loads(con):
    text = _screen([0], "amazon_in", "view", seed=3)
    load_raw_events(con, _events([_row("e1", text, minutes=60)]))
    load_raw_events(con, _events([_row("e2", text, minutes=5)]))
    load_raw_events(con, _events([_row("e3", text, minutes=90)]))
    assert _rows(con, "SELECT first_seen FROM dim_products") == [(T0 + timedelta(minutes=5),)]


def test_registry_is_rebuilt_from_dim_products(con):
    """A product already in dim_products keeps its id when a later load sees it again."""
    text = _screen([0], "amazon_in", "buy", seed=4)
    load_raw_events(con, _events([_row("e1", text)]))
    first = _rows(con, "SELECT canonical_product_id FROM fact_events WHERE event_id = 'e1'")
    load_raw_events(con, _events([_row("e2", text, minutes=10)]))
    second = _rows(con, "SELECT canonical_product_id FROM fact_events WHERE event_id = 'e2'")
    assert first == second
    assert _counts(con)["dim_products"] == 1


# ---------------------------------------------------------------------------
# Screens without a price, and non-warehouse actions.
# ---------------------------------------------------------------------------

def test_screen_without_price_still_writes_a_fact_event(con):
    load_raw_events(con, _events([_row("e1", "Settings\nBattery 80%\nWiFi")]))
    counts = _counts(con)
    assert counts["raw_events"] == 1
    assert counts["fact_events"] == 1
    assert counts["fact_price_observations"] == 0
    price, price_inr, currency, action = _rows(con, "SELECT price, price_inr, currency, action FROM fact_events")[0]
    assert price is None and price_inr is None
    assert action in WAREHOUSE_ACTIONS


@pytest.mark.parametrize("extracted", ["search", "wishlist"])
def test_actions_outside_the_warehouse_funnel_are_stored_as_view(con, extracted):
    text = _screen([0], "amazon_in", extracted)
    load_raw_events(con, _events([_row("e1", text)]))
    assert _rows(con, "SELECT action FROM fact_events") == [("view",)]


def test_warehouse_action_is_shared_with_the_api_module():
    from src.api import warehouse as api_warehouse
    from src.features import event_loader

    assert api_warehouse.warehouse_action is event_loader.warehouse_action


@pytest.mark.parametrize("action", WAREHOUSE_ACTIONS)
def test_warehouse_action_keeps_funnel_actions(action):
    assert warehouse_action(action) == action


@given(st.text(max_size=20))
@settings(max_examples=50, deadline=None)
def test_warehouse_action_always_lands_in_the_funnel(value):
    out = warehouse_action(value)
    assert out in WAREHOUSE_ACTIONS
    assert out == (value if value in WAREHOUSE_ACTIONS else "view")


# ---------------------------------------------------------------------------
# Workers and ordering.
# ---------------------------------------------------------------------------

def _mixed_events() -> pl.DataFrame:
    rows = []
    for i in range(6):
        app = ["amazon_in", "flipkart", "amazon_us"][i % 3]
        action = ["buy", "view", "cart"][i % 3]
        rows.append(_row(f"e{i}", _screen([i, i + 1], app, action, seed=i), app=app, user=f"u{i % 2}", minutes=i * 7))
    rows.append(_row("e6", "no price here at all", minutes=50))
    return _events(rows)


def test_workers_one_and_two_produce_identical_tables(tmp_path):
    frame = _mixed_events()
    snapshots = []
    orders = []
    from src.features import event_loader

    for workers in (1, 2):
        event_loader._SCREEN_MEMO.clear()  # a memo hit would skip the worker pool entirely
        connection = duckdb.connect(str(tmp_path / f"w{workers}.duckdb"))
        migrate(connection)
        load_raw_events(connection, frame, workers=workers)
        snapshots.append(_snapshot(connection))
        orders.append(
            {
                "raw": [r[0] for r in _rows(connection, "SELECT event_id FROM raw_events ORDER BY rowid")],
                "fact": [r[0] for r in _rows(connection, "SELECT event_id FROM fact_events ORDER BY rowid")],
            }
        )
        connection.close()
    assert snapshots[0] == snapshots[1]
    assert orders[0] == orders[1]
    assert orders[1]["fact"] == frame["event_id"].to_list()
    assert orders[1]["raw"] == frame["event_id"].to_list()


def test_empty_frame_writes_nothing(con):
    load_raw_events(con, _events([]))
    assert _counts(con) == {t: 0 for t in FOUR_TABLES}


def test_single_row_with_many_workers(con):
    text = _screen([0], "amazon_in", "buy")
    load_raw_events(con, _events([_row("e1", text)]), workers=2)
    assert _counts(con)["fact_events"] == 1


# ---------------------------------------------------------------------------
# Idempotence: the same event_id is never duplicated.
# ---------------------------------------------------------------------------

def _load_ignoring_duplicate_error(connection, frame) -> None:
    try:
        load_raw_events(connection, frame)
    except Exception:  # raising on a duplicate is allowed; leaving partial rows is not
        pass


def test_loading_the_same_event_id_twice_is_never_duplicated(con):
    frame = _events([_row("e1", _screen([0, 1], "amazon_in", "buy"))])
    load_raw_events(con, frame)
    before = _snapshot(con)
    _load_ignoring_duplicate_error(con, frame)
    assert _snapshot(con) == before
    assert _counts(con)["fact_events"] == 1
    assert _counts(con)["raw_events"] == 1


def test_duplicate_event_id_inside_one_frame_is_never_duplicated(con):
    text = _screen([0], "amazon_in", "buy")
    frame = _events([_row("e1", text), _row("e1", text, minutes=5)])
    _load_ignoring_duplicate_error(con, frame)
    for table in ("raw_events", "fact_events"):
        assert con.execute(f"SELECT COUNT(*) - COUNT(DISTINCT event_id) FROM {table}").fetchone()[0] == 0
    n = con.execute("SELECT COUNT(*) - COUNT(DISTINCT observation_id) FROM fact_price_observations").fetchone()[0]
    assert n == 0


@given(repeats=st.integers(min_value=1, max_value=3))
@settings(max_examples=6, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_repeated_loads_leave_one_copy_of_each_event(repeats):
    connection = duckdb.connect(":memory:")
    migrate(connection)
    frame = _events([_row("e1", _screen([0, 1], "amazon_in", "cart", seed=9))])
    load_raw_events(connection, frame)
    expected = _snapshot(connection)
    for _ in range(repeats):
        _load_ignoring_duplicate_error(connection, frame)
    assert _snapshot(connection) == expected
    connection.close()


# ---------------------------------------------------------------------------
# Ground truth never enters the warehouse.
# ---------------------------------------------------------------------------

def _truth_columns(connection) -> list:
    return connection.execute(
        "SELECT table_name, column_name FROM information_schema.columns "
        "WHERE lower(column_name) = 'archetype' OR lower(column_name) LIKE 'true\\_%' ESCAPE '\\'"
    ).fetchall()


def test_warehouse_holds_no_ground_truth(con):
    load_raw_events(con, _events([_row("e1", _screen([0, 1], "amazon_in", "buy"))]))
    assert _truth_columns(con) == []


def test_warehouse_holds_no_ground_truth_after_simulation(tiny_warehouse):
    connection = duckdb.connect(str(tiny_warehouse), read_only=True)
    try:
        assert _truth_columns(connection) == []
    finally:
        connection.close()


def test_loader_ignores_extra_truth_columns_in_input(con):
    frame = _events([_row("e1", _screen([0], "amazon_in", "buy"))]).with_columns(
        pl.lit("bargain_hunter").alias("archetype"), pl.lit(1.0).alias("true_price")
    )
    load_raw_events(con, frame)
    assert _truth_columns(con) == []


# ---------------------------------------------------------------------------
# The simulated warehouse goes through the loader.
# ---------------------------------------------------------------------------

def test_tiny_warehouse_events_come_from_raw_screens(tiny_warehouse):
    connection = duckdb.connect(str(tiny_warehouse), read_only=True)
    try:
        raw, fact, orphan, unversioned, priced = connection.execute(
            "SELECT (SELECT COUNT(*) FROM raw_events), (SELECT COUNT(*) FROM fact_events), "
            "(SELECT COUNT(*) FROM fact_events f WHERE NOT EXISTS "
            "  (SELECT 1 FROM raw_events r WHERE r.event_id = f.event_id)), "
            "(SELECT COUNT(*) FROM raw_events WHERE extractor_version IS NULL OR extractor_version = ''), "
            "(SELECT COUNT(*) FROM fact_price_observations)"
        ).fetchone()
        assert raw == fact > 0
        assert orphan == 0
        assert unversioned == 0
        assert priced > 0
        discount, conf, price_inr = connection.execute(
            "SELECT COUNT(discount_pct), COUNT(extraction_confidence), COUNT(price_inr) FROM fact_events"
        ).fetchone()
        assert conf == fact
        assert price_inr > 0.9 * fact
        focal = connection.execute(
            "SELECT COUNT(*) FROM (SELECT source_event_id FROM fact_price_observations "
            "GROUP BY 1 HAVING SUM(CASE WHEN is_focal THEN 1 ELSE 0 END) > 1)"
        ).fetchone()[0]
        assert focal == 0
    finally:
        connection.close()


def test_simulated_events_use_the_loader_registry(tiny_warehouse):
    connection = duckdb.connect(str(tiny_warehouse), read_only=True)
    try:
        missing = connection.execute(
            "SELECT COUNT(*) FROM fact_events f WHERE canonical_product_id IS NOT NULL AND NOT EXISTS "
            "(SELECT 1 FROM dim_products d WHERE d.canonical_product_id = f.canonical_product_id)"
        ).fetchone()[0]
        assert missing == 0
        assert connection.execute("SELECT COUNT(*) FROM dim_products WHERE first_seen IS NULL").fetchone()[0] == 0
    finally:
        connection.close()


def test_simulation_ocr_noise_default():
    from src.config import SIMULATION_OCR_NOISE

    assert SIMULATION_OCR_NOISE == 0.01


# ---------------------------------------------------------------------------
# Batch and serving writes use the same function and cannot diverge.
# ---------------------------------------------------------------------------

def test_write_events_is_the_shared_write_function():
    from src.features import event_loader

    assert callable(event_loader.write_events)


def test_ingest_and_loader_write_identical_rows(tiny_warehouse, tmp_path):
    from fastapi.testclient import TestClient

    from src.api.main import create_app

    api_db = tmp_path / "api.duckdb"
    batch_db = tmp_path / "batch.duckdb"
    shutil.copy2(tiny_warehouse, api_db)
    shutil.copy2(tiny_warehouse, batch_db)
    text = _screen([0, 1, 2], "amazon_in", "buy", seed=21)
    stamp = datetime(2027, 1, 5, 12, 30, 0)

    app = create_app(db_path=api_db, bootstrap=False, clock_mode="simulated")
    with TestClient(app) as client:
        response = client.post(
            "/ingest",
            json={"raw_text": text, "user_id": "parity_user", "app_name": "amazon_in", "timestamp": stamp.isoformat()},
        )
        assert response.status_code == 200
        event_id = response.json()["event_id"]

    connection = duckdb.connect(str(batch_db))
    load_raw_events(connection, _events([_row(event_id, text, user="parity_user")]).with_columns(
        pl.lit(stamp).alias("timestamp")
    ))
    connection.close()

    tables = {
        "raw_events": "SELECT * FROM raw_events WHERE event_id = ?",
        "fact_events": (
            "SELECT event_id, user_id, timestamp, app_name, canonical_product_id, action, price, currency, "
            "mrp, discount_pct, rating, extraction_confidence, price_inr FROM fact_events WHERE event_id = ?"
        ),
        "fact_price_observations": (
            "SELECT source_event_id, user_id, timestamp, app_name, canonical_product_id, price_inr, "
            "discount_pct, is_focal FROM fact_price_observations WHERE source_event_id = ? "
            "ORDER BY canonical_product_id, is_focal"
        ),
    }
    a = duckdb.connect(str(api_db), read_only=True)
    b = duckdb.connect(str(batch_db), read_only=True)
    try:
        for name, sql in tables.items():
            left = a.execute(sql, [event_id]).fetchall()
            right = b.execute(sql, [event_id]).fetchall()
            assert left, name
            assert left == right, name
        product_ids = [r[0] for r in b.execute(
            "SELECT canonical_product_id FROM fact_price_observations WHERE source_event_id = ?", [event_id]
        ).fetchall()]
        for pid in product_ids:
            sql = "SELECT * FROM dim_products WHERE canonical_product_id = ?"
            assert a.execute(sql, [pid]).fetchall() == b.execute(sql, [pid]).fetchall()
    finally:
        a.close()
        b.close()


def test_new_events_beside_a_duplicate_are_loaded_or_nothing_is(con):
    """Ignoring a duplicate must not drop the new events sent with it; raising must leave no partial rows."""
    old = _events([_row("e1", _screen([0], "amazon_in", "buy", seed=2))])
    load_raw_events(con, old)
    mixed = _events(
        [
            _row("e2", _screen([1], "amazon_in", "cart", seed=5), minutes=10),
            _row("e1", _screen([0], "amazon_in", "buy", seed=2)),
        ]
    )
    raised = False
    try:
        load_raw_events(con, mixed)
    except Exception:
        raised = True
    raw_ids = [r[0] for r in _rows(con, "SELECT event_id FROM raw_events ORDER BY event_id")]
    fact_ids = [r[0] for r in _rows(con, "SELECT event_id FROM fact_events ORDER BY event_id")]
    assert raw_ids == fact_ids
    assert raw_ids.count("e1") == 1
    if not raised:
        assert raw_ids == ["e1", "e2"]


def test_run_simulation_writes_raw_and_truth_files_apart(tmp_path):
    from src.ingestion.simulate_users import TRUTH_COLUMNS, default_truth_path, run_simulation

    raw_path = tmp_path / "simulated_events.parquet"
    db_path = tmp_path / "sim.duckdb"
    run_simulation(num_users=3, days=2, db_path=str(db_path), output_parquet=str(raw_path), workers=1)

    truth_path = default_truth_path(str(raw_path))
    assert Path(truth_path).name == "simulation_truth.parquet"
    raw = pl.read_parquet(raw_path)
    truth = pl.read_parquet(truth_path)
    assert tuple(truth.columns) == tuple(TRUTH_COLUMNS)
    assert set(raw.columns) == {"event_id", "user_id", "timestamp", "app_name", "raw_text"}
    assert not [c for c in raw.columns if c == "archetype" or c.startswith("true_")]
    assert raw.height > 0
    assert set(raw["event_id"]) == set(truth["event_id"])
    connection = duckdb.connect(str(db_path), read_only=True)
    try:
        warehouse_ids = {r[0] for r in connection.execute("SELECT event_id FROM fact_events").fetchall()}
        assert _truth_columns(connection) == []
    finally:
        connection.close()
    assert warehouse_ids == set(raw["event_id"])
