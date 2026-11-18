"""
Tests for the re-extraction job: extracted tables rebuilt from raw_events equal the originals.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import random
from typing import List

import duckdb
import polars as pl
import pytest

from src.features.event_loader import load_raw_events
from src.features.migrations import migrate
from src.features.sessionizer import detect_cross_app_comparisons, sessionize_events
from src.ingestion.reextract import reextract
from src.ingestion.screen_generator import render_listing, render_screen
from src.ingestion.synthetic_data import CATALOG

T0 = datetime(2026, 9, 1, 10, 0, 0)
TABLES = {
    "fact_events": "event_id",
    "fact_price_observations": "observation_id",
    "dim_products": "canonical_product_id",
    "dim_users": "user_id",
    "product_aliases": "alias_key",
    "fact_cross_app_comparisons": "comparison_id",
}


def _screen(items: List[int], app: str, action: str, seed: int) -> str:
    rng = random.Random(seed)
    listings = [
        render_listing(CATALOG[i], app, action if p == 0 else "view", 10, 4.4, rng)
        for p, i in enumerate(items)
    ]
    return render_screen(listings, rng)


@pytest.fixture
def con(tmp_path):
    connection = duckdb.connect(str(tmp_path / "w.duckdb"))
    migrate(connection)
    rows = [
        ("e1", "u1", 0, "amazon_in", [0, 1], "view"),
        ("e2", "u1", 10, "flipkart", [0], "view"),
        ("e3", "u1", 20, "amazon_in", [1], "buy"),
        ("e4", "u2", 5, "amazon_in", [2, 0], "cart"),
        ("e5", "u1", 600, "amazon_in", [0], "view"),
    ]
    frame = pl.DataFrame(
        [
            {"event_id": e, "user_id": u, "timestamp": T0 + timedelta(minutes=m), "app_name": a,
             "raw_text": _screen(items, a, act, seed=i)}
            for i, (e, u, m, a, items, act) in enumerate(rows)
        ],
        schema={"event_id": pl.Utf8, "user_id": pl.Utf8, "timestamp": pl.Datetime("us"),
                "app_name": pl.Utf8, "raw_text": pl.Utf8},
    )
    load_raw_events(connection, frame)
    sessionize_events(con=connection)
    detect_cross_app_comparisons(con=connection)
    yield connection
    connection.close()


def _snapshot(con):
    return {t: con.execute(f"SELECT * FROM {t} ORDER BY {k}").fetchall() for t, k in TABLES.items()}


def test_reextract_restores_nulled_price_and_product_id(con):
    original = _snapshot(con)
    assert original["fact_events"]
    con.execute("UPDATE fact_events SET price = NULL, price_inr = NULL, canonical_product_id = NULL")
    con.execute("DELETE FROM fact_price_observations")
    con.execute("UPDATE dim_products SET avg_price = NULL")
    counts = reextract(con)
    assert _snapshot(con) == original
    assert counts["fact_events"] == len(original["fact_events"])
    assert counts["comparisons"] == len(original["fact_cross_app_comparisons"])


def test_reextract_is_idempotent_and_keeps_identity_columns(con):
    before = con.execute(
        "SELECT event_id, user_id, timestamp, app_name FROM fact_events ORDER BY event_id"
    ).fetchall()
    reextract(con)
    first = _snapshot(con)
    reextract(con)
    assert _snapshot(con) == first
    after = con.execute(
        "SELECT event_id, user_id, timestamp, app_name FROM fact_events ORDER BY event_id"
    ).fetchall()
    assert after == before
    assert con.execute("SELECT COUNT(*) FROM raw_events").fetchone()[0] == len(before)


def test_reextract_drops_stale_derived_rows(con):
    con.execute("INSERT INTO dim_products (canonical_product_id, brand) VALUES ('stale', 'x')")
    reextract(con)
    assert con.execute("SELECT COUNT(*) FROM dim_products WHERE canonical_product_id = 'stale'").fetchone()[0] == 0
    assert con.execute("SELECT COUNT(*) FROM fact_events WHERE session_id IS NULL").fetchone()[0] == 0


def test_reextract_on_empty_warehouse(tmp_path):
    connection = duckdb.connect(str(tmp_path / "e.duckdb"))
    migrate(connection)
    counts = reextract(connection)
    assert counts["fact_events"] == 0 and counts["comparisons"] == 0
    connection.close()
