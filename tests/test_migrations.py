"""Tests for schema version 2 and the idempotent migration."""

from __future__ import annotations

from pathlib import Path

import duckdb
from hypothesis import given, settings, strategies as st

from src import config
from src.features import migrations
from src.features.database import DatabaseManager
from src.features.migrations import migrate

V1_DDL = Path(__file__).parent / "fixtures" / "schema_v1.sql"
SCHEMA_SQL = Path(config.SCHEMA_SQL_PATH)

NEW_FACT_COLS = {"mrp", "discount_pct", "rating", "extraction_confidence", "price_inr"}
ALL_TABLES = {
    "schema_info", "raw_events", "fact_events", "fact_price_observations",
    "fact_cross_app_comparisons", "dim_products", "dim_users",
}


def _tables(con) -> set[str]:
    return {r[0] for r in con.execute("SELECT table_name FROM information_schema.tables").fetchall()}


def _cols(con, table: str) -> dict[str, str]:
    rows = con.execute(
        "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = ?", [table]
    ).fetchall()
    return {r[0]: r[1] for r in rows}


def _v1_con(rows=()):
    con = duckdb.connect(":memory:")
    con.execute(V1_DDL.read_text())
    for i, (price, cur) in enumerate(rows):
        con.execute(
            "INSERT INTO fact_events VALUES (?, 'u1', TIMESTAMP '2026-01-01 00:00:00', 'a', 'p', 'view', ?, ?, 's')",
            [f"e{i}", price, cur],
        )
    return con


def _snapshot(con) -> dict:
    out = {}
    for t in sorted(_tables(con)):
        out[t] = (sorted(_cols(con, t).items()), sorted(map(str, con.execute(f"SELECT * FROM {t}").fetchall())))
    return out


def test_version_constants():
    assert config.EXTRACTOR_VERSION == "2.0.0"
    assert config.SCHEMA_VERSION == "2"


def test_migrate_empty_database_creates_every_table():
    con = duckdb.connect(":memory:")
    assert migrate(con) == "2"
    assert ALL_TABLES <= _tables(con)
    assert NEW_FACT_COLS <= set(_cols(con, "fact_events"))
    assert {"price_first_inr", "price_last_inr", "cheaper_app"} <= set(_cols(con, "fact_cross_app_comparisons"))
    assert "first_seen" in _cols(con, "dim_products")


def test_migrate_sets_schema_version_row():
    con = duckdb.connect(":memory:")
    migrate(con)
    assert con.execute("SELECT value FROM schema_info WHERE key = 'schema_version'").fetchall() == [("2",)]


def test_new_table_columns_and_types():
    con = duckdb.connect(":memory:")
    migrate(con)
    raw = _cols(con, "raw_events")
    assert set(raw) == {"event_id", "user_id", "timestamp", "app_name", "raw_text", "extractor_version"}
    obs = _cols(con, "fact_price_observations")
    assert set(obs) == {
        "observation_id", "source_event_id", "user_id", "timestamp", "app_name",
        "canonical_product_id", "price_inr", "discount_pct", "is_focal",
    }
    assert obs["is_focal"] == "BOOLEAN"
    assert obs["price_inr"] == "DECIMAL(18,2)"
    fe = _cols(con, "fact_events")
    assert fe["price_inr"] == "DECIMAL(18,2)" and fe["mrp"] == "DECIMAL(18,2)"
    assert fe["discount_pct"] == "INTEGER" and fe["rating"] == "DOUBLE"
    assert fe["extraction_confidence"] == "DOUBLE"


def test_migrate_v1_warehouse_adds_columns_and_backfills():
    con = _v1_con([(100.00, "INR"), (10.00, "USD"), (None, "INR")])
    assert not NEW_FACT_COLS & set(_cols(con, "fact_events"))
    assert migrate(con) == "2"
    assert NEW_FACT_COLS <= set(_cols(con, "fact_events"))
    assert ALL_TABLES <= _tables(con)
    got = dict(con.execute("SELECT event_id, price_inr FROM fact_events").fetchall())
    assert float(got["e0"]) == 100.0
    assert float(got["e1"]) == 830.0
    assert got["e2"] is None
    for c in ("mrp", "discount_pct", "rating", "extraction_confidence"):
        assert con.execute(f"SELECT COUNT({c}) FROM fact_events").fetchone()[0] == 0


def test_price_inr_equals_price_times_83_for_usd_row():
    con = _v1_con([(19.99, "USD")])
    migrate(con)
    price, inr = con.execute("SELECT price, price_inr FROM fact_events").fetchone()
    assert float(inr) == round(float(price) * 83, 2)
    assert config.USD_TO_INR_RATE == 83.0


def test_backfill_uses_config_rate_not_hardcoded(monkeypatch):
    monkeypatch.setattr(config, "USD_TO_INR_RATE", 90.0)
    monkeypatch.setattr(migrations, "USD_TO_INR_RATE", 90.0, raising=False)
    con = _v1_con([(10.00, "USD")])
    migrate(con)
    assert float(con.execute("SELECT price_inr FROM fact_events").fetchone()[0]) == 900.0


def test_backfill_does_not_overwrite_existing_price_inr():
    con = _v1_con([(10.00, "USD")])
    migrate(con)
    con.execute("UPDATE fact_events SET price_inr = 1.00")
    migrate(con)
    assert float(con.execute("SELECT price_inr FROM fact_events").fetchone()[0]) == 1.0


def test_backfill_fills_only_null_rows_after_new_inserts():
    con = _v1_con([(10.00, "USD")])
    migrate(con)
    con.execute(
        "INSERT INTO fact_events (event_id, user_id, timestamp, app_name, action, price, currency)"
        " VALUES ('n', 'u', TIMESTAMP '2026-01-02 00:00:00', 'a', 'view', 5.00, 'USD')"
    )
    migrate(con)
    got = dict(con.execute("SELECT event_id, price_inr FROM fact_events").fetchall())
    assert float(got["n"]) == 415.0 and float(got["e0"]) == 830.0


def test_migrate_twice_changes_nothing():
    con = _v1_con([(100.00, "INR"), (10.00, "USD")])
    migrate(con)
    first = _snapshot(con)
    assert migrate(con) == "2"
    assert _snapshot(con) == first


def test_migrate_preserves_existing_rows():
    con = _v1_con([(100.00, "INR"), (10.00, "USD")])
    con.execute("INSERT INTO dim_products VALUES ('p', 'b', 't', 'c1', 'c2', 50.00)")
    migrate(con)
    assert con.execute("SELECT COUNT(*) FROM fact_events").fetchone()[0] == 2
    assert con.execute("SELECT brand, avg_price FROM dim_products").fetchone() == ("b", 50)
    assert con.execute("SELECT first_seen FROM dim_products").fetchone() == (None,)


def test_migrate_upgrades_stale_schema_info_version():
    con = duckdb.connect(":memory:")
    migrate(con)
    con.execute("UPDATE schema_info SET value = '1' WHERE key = 'schema_version'")
    assert migrate(con) == "2"
    assert con.execute("SELECT COUNT(*) FROM schema_info WHERE key='schema_version'").fetchone()[0] == 1
    assert con.execute("SELECT value FROM schema_info WHERE key='schema_version'").fetchone()[0] == "2"


def test_init_schema_calls_migrate(tmp_path):
    db = tmp_path / "w.duckdb"
    DatabaseManager(db).init_schema()
    con = duckdb.connect(str(db))
    assert ALL_TABLES <= _tables(con)
    assert con.execute("SELECT value FROM schema_info WHERE key='schema_version'").fetchone() == ("2",)
    assert "price_inr" in _cols(con, "fact_events")
    con.close()


def test_init_schema_in_memory_has_version():
    mgr = DatabaseManager(":memory:")
    mgr.init_schema()
    assert mgr.execute_query("SELECT value FROM schema_info WHERE key='schema_version'") == [("2",)]


def test_schema_sql_alone_defines_v2_tables():
    con = duckdb.connect(":memory:")
    con.execute(SCHEMA_SQL.read_text())
    assert {"schema_info", "raw_events", "fact_price_observations"} <= _tables(con)


def test_schema_sql_documents_primary_app_meaning():
    text = SCHEMA_SQL.read_text().lower()
    comment_lines = [ln for ln in text.splitlines() if "--" in ln]
    joined = " ".join(comment_lines)
    assert "primary_app" in joined and "compared_app" in joined
    assert "earlier" in joined and "later" in joined


def _make_v1_file(path: Path, rows) -> None:
    con = _v1_con(rows)
    con.execute(f"ATTACH '{path}' AS d")
    for t in ("fact_events", "dim_products", "dim_users", "fact_cross_app_comparisons"):
        con.execute(f"CREATE TABLE d.{t} AS SELECT * FROM {t}")
    con.execute("DETACH d")
    con.close()


def test_ensure_serving_warehouse_migrates_copied_source(tmp_path):
    from src.api.warehouse import ensure_serving_warehouse

    src_db = tmp_path / "batch.duckdb"
    _make_v1_file(src_db, [(10.00, "USD"), (7.00, "INR")])
    target = tmp_path / "serving.duckdb"
    assert ensure_serving_warehouse(db_path=target, source_db_path=src_db) == "copied"
    con = duckdb.connect(str(target), read_only=True)
    assert con.execute("SELECT value FROM schema_info WHERE key='schema_version'").fetchone() == ("2",)
    got = sorted(float(r[0]) for r in con.execute("SELECT price_inr FROM fact_events").fetchall())
    assert got == [7.0, 830.0]
    con.close()


def test_ensure_serving_warehouse_migrates_existing_v1_target(tmp_path):
    from src.api.warehouse import ensure_serving_warehouse

    target = tmp_path / "serving.duckdb"
    _make_v1_file(target, [(10.00, "USD")])
    src_db = tmp_path / "batch.duckdb"
    _make_v1_file(src_db, [])
    assert ensure_serving_warehouse(db_path=target, source_db_path=src_db) == "existing"
    con = duckdb.connect(str(target), read_only=True)
    assert "price_inr" in _cols(con, "fact_events")
    assert float(con.execute("SELECT price_inr FROM fact_events").fetchone()[0]) == 830.0
    con.close()


def test_retailrocket_adapter_fills_price_inr():
    import inspect
    from src.ingestion import retailrocket_adapter

    src = inspect.getsource(retailrocket_adapter)
    assert "price_inr" in src


@settings(max_examples=50, deadline=None)
@given(
    rows=st.lists(
        st.tuples(st.one_of(st.none(), st.integers(0, 10_000_000).map(lambda c: c / 100)),
                  st.sampled_from(["INR", "USD"])),
        max_size=8,
    )
)
def test_backfill_matches_formula_and_is_idempotent(rows):
    con = _v1_con(rows)
    migrate(con)
    snap = _snapshot(con)
    migrate(con)
    assert _snapshot(con) == snap
    got = dict(con.execute("SELECT event_id, price_inr FROM fact_events").fetchall())
    for i, (price, cur) in enumerate(rows):
        if price is None:
            assert got[f"e{i}"] is None
        else:
            want = price * 83.0 if cur == "USD" else price
            assert abs(float(got[f"e{i}"]) - want) < 0.011
