"""Tests for src/features/data_checks.py: each check fails on one planted violation."""
from __future__ import annotations

import re

import duckdb
import pytest

from src.config import SCHEMA_SQL_PATH
from src.features.data_checks import run_checks


def _build() -> duckdb.DuckDBPyConnection:
    """Clean in-memory warehouse; key constraints are stripped so violations can be planted."""
    con = duckdb.connect(":memory:")
    ddl = re.sub(r"\s+PRIMARY KEY", "", SCHEMA_SQL_PATH.read_text())
    con.execute(ddl)
    con.execute("INSERT INTO dim_users VALUES ('u1', '2026-01-01', 'a', 1)")
    con.execute("INSERT INTO dim_products (canonical_product_id) VALUES ('p1')")
    con.execute("INSERT INTO raw_events VALUES ('e1','u1','2026-01-01','a','t','v')")
    con.execute(
        "INSERT INTO fact_events (event_id,user_id,timestamp,app_name,canonical_product_id,action,"
        "session_id,discount_pct,price_inr) VALUES ('e1','u1','2026-01-01','a','p1','view','s1',10,100)"
    )
    con.execute(
        "INSERT INTO fact_cross_app_comparisons VALUES ('c1','u1','s1','p1','a','b',"
        "'2026-01-01 00:00:00','2026-01-01 00:10:00',10,1,1,'a')"
    )
    return con


def _failed(con: duckdb.DuckDBPyConnection) -> set[str]:
    return {r.name for r in run_checks(con) if not r.passed}


def test_clean_database_passes() -> None:
    results = run_checks(_build())
    assert results and all(r.passed and r.failing_rows == 0 for r in results)


@pytest.mark.parametrize("sql,expected", [
    ("INSERT INTO dim_users VALUES ('u1', NULL, NULL, 0)", "pk_unique:dim_users"),
    ("INSERT INTO dim_products (canonical_product_id) VALUES ('p1')", "pk_unique:dim_products"),
    ("INSERT INTO raw_events VALUES ('e1','u1','2026-01-01','a','t','v')", "pk_unique:raw_events"),
    ("INSERT INTO fact_events (event_id,user_id,timestamp,app_name,action,session_id) "
     "VALUES ('e1','u1','2026-01-01','a','view','s1')", "pk_unique:fact_events"),
    ("UPDATE fact_events SET action='refund'", "action_in_warehouse_actions"),
    ("INSERT INTO fact_events (event_id,user_id,timestamp,app_name,action,session_id) "
     "VALUES ('e2','u1','2026-01-01','a','view','s1')", "event_id_in_raw_events"),
    ("UPDATE fact_events SET canonical_product_id='ghost'", "product_id_in_dim_products"),
    ("UPDATE fact_events SET user_id='ghost'", "user_id_in_dim_users"),
    ("UPDATE fact_events SET price_inr=0", "price_inr_positive"),
    ("UPDATE fact_events SET discount_pct=100", "discount_pct_in_range"),
    ("UPDATE fact_events SET discount_pct=-1", "discount_pct_in_range"),
    ("UPDATE fact_events SET session_id=NULL", "session_id_not_null"),
    ("UPDATE fact_cross_app_comparisons SET first_viewed='2026-01-02'", "comparison_order"),
    ("UPDATE fact_cross_app_comparisons SET duration_minutes=9999", "comparison_duration_in_window"),
])
def test_planted_violation_fails(sql: str, expected: str) -> None:
    con = _build()
    con.execute(sql)
    assert expected in _failed(con)


def test_empty_raw_events_skips_lineage_check() -> None:
    con = _build()
    con.execute("DELETE FROM raw_events")
    names = {r.name for r in run_checks(con)}
    assert "event_id_in_raw_events" not in names
    assert not _failed(con)


def test_null_values_are_allowed() -> None:
    con = _build()
    con.execute("UPDATE fact_events SET price_inr=NULL, discount_pct=NULL, canonical_product_id=NULL")
    assert not _failed(con)
