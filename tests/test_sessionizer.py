"""
Unit tests for event sessionization and cross-app shopping journey detection.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import duckdb
import pytest

from src.features.database import DatabaseManager
from src.features.sessionizer import sessionize_events, detect_cross_app_comparisons


def test_sessionize_events_inactivity_window():
    """Verifies that gaps <= 30 min share a session, while gaps > 30 min start a new session."""
    db = DatabaseManager(":memory:")
    db.init_schema()
    con = db.get_connection()

    try:
        t0 = datetime(2026, 10, 1, 10, 0, 0)
        t1 = t0 + timedelta(minutes=15)  # within 30 min -> session 1
        t2 = t1 + timedelta(minutes=45)  # 45 min gap -> session 2
        t3 = t2 + timedelta(minutes=10)  # within 30 min -> session 2

        events = [
            ("evt_1", "user_1", t0, "amazon_in", "prod_1", "view", 100.0, "INR", None),
            ("evt_2", "user_1", t1, "amazon_in", "prod_1", "cart", 100.0, "INR", None),
            ("evt_3", "user_1", t2, "flipkart", "prod_2", "view", 200.0, "INR", None),
            ("evt_4", "user_1", t3, "flipkart", "prod_2", "buy", 200.0, "INR", None),
        ]

        con.executemany("""
            INSERT INTO fact_events (event_id, user_id, timestamp, app_name, canonical_product_id, action, price, currency, session_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, events)

        sessionize_events(con=con, inactivity_minutes=30.0)

        rows = con.execute("SELECT event_id, session_id FROM fact_events ORDER BY event_id").fetchall()
        session_map = dict(rows)

        assert session_map["evt_1"] == session_map["evt_2"]
        assert session_map["evt_3"] == session_map["evt_4"]
        assert session_map["evt_1"] != session_map["evt_3"]
        assert session_map["evt_1"] == "user_1_1"
        assert session_map["evt_3"] == "user_1_2"
    finally:
        con.close()


def test_detect_cross_app_comparisons():
    """Verifies detection of cross-app views within 60 minutes and exclusion of views beyond window."""
    db = DatabaseManager(":memory:")
    db.init_schema()
    con = db.get_connection()

    try:
        t0 = datetime(2026, 10, 1, 12, 0, 0)
        t_comp_fast = t0 + timedelta(minutes=20)  # 20 min gap: valid cross-app comparison
        t_comp_slow = t0 + timedelta(minutes=90)  # 90 min gap: exceeds 60-minute cutoff

        events = [
            # User A: Cross-app comparison within 20 min
            ("evt_10", "user_A", t0, "amazon_in", "prod_common", "view", 100.0, "INR", "user_A_1"),
            ("evt_11", "user_A", t_comp_fast, "flipkart", "prod_common", "view", 95.0, "INR", "user_A_1"),
            # User B: Different apps but 90 min apart (> 60 min)
            ("evt_20", "user_B", t0, "walmart", "prod_slow", "view", 50.0, "USD", "user_B_1"),
            ("evt_21", "user_B", t_comp_slow, "amazon_us", "prod_slow", "view", 48.0, "USD", "user_B_2"),
        ]

        con.executemany("""
            INSERT INTO fact_events (event_id, user_id, timestamp, app_name, canonical_product_id, action, price, currency, session_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, events)

        results = detect_cross_app_comparisons(con=con, max_window_minutes=60.0)

        assert results["total_comparisons"] == 1
        assert results["cross_app_users"] == 1

        comp_row = con.execute("SELECT user_id, canonical_product_id, primary_app, compared_app, duration_minutes FROM fact_cross_app_comparisons").fetchone()
        assert comp_row[0] == "user_A"
        assert comp_row[1] == "prod_common"
        assert comp_row[2] == "amazon_in"
        assert comp_row[3] == "flipkart"
        assert float(comp_row[4]) == 20.0
    finally:
        con.close()


def test_warehouse_sessionization_integrity():
    """Verifies that all events in persistent warehouse have non-null session_id."""
    db_path = Path("data/processed/consumer.duckdb")
    if not db_path.exists():
        pytest.skip("Persistent warehouse not created yet")

    con = duckdb.connect(str(db_path))
    try:
        null_sessions = con.execute("SELECT COUNT(*) FROM fact_events WHERE session_id IS NULL").fetchone()[0]
        assert null_sessions == 0

        cross_app_count = con.execute("SELECT COUNT(*) FROM fact_cross_app_comparisons").fetchone()[0]
        assert cross_app_count > 0
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Bidirectional comparison detector (W4)
# ---------------------------------------------------------------------------
import inspect
import random

from hypothesis import given, settings, strategies as st

import src.features.sessionizer as sessionizer_module

_T0 = datetime(2026, 10, 1, 10, 0, 0)
_CMP_COLS = (
    "user_id, canonical_product_id, primary_app, compared_app, first_viewed, last_viewed, "
    "session_id, CAST(price_first_inr AS DOUBLE), CAST(price_last_inr AS DOUBLE), cheaper_app"
)


def _cmp_con():
    db = DatabaseManager(":memory:")
    db.init_schema()
    return db.get_connection()


def _add(con, rows):
    """rows: (event_id, user, ts, app, product, price_inr, session_id)."""
    if not rows:
        return
    con.executemany(
        "INSERT INTO fact_events (event_id, user_id, timestamp, app_name, canonical_product_id, action, "
        "price, currency, session_id, price_inr) VALUES (?, ?, ?, ?, ?, 'view', ?, 'INR', ?, ?)",
        [(e, u, t, a, p, pr, s, pr) for e, u, t, a, p, pr, s in rows],
    )


def _comparisons(con):
    return con.execute(f"SELECT {_CMP_COLS} FROM fact_cross_app_comparisons ORDER BY first_viewed, last_viewed").fetchall()


def _run(rows, window=60.0):
    con = _cmp_con()
    _add(con, rows)
    detect_cross_app_comparisons(con=con, max_window_minutes=window)
    return con


def test_comparison_a_then_b_primary_is_a():
    con = _run([
        ("e1", "u", _T0, "amazon_in", "p", 100.0, "u_1"),
        ("e2", "u", _T0 + timedelta(minutes=5), "flipkart", "p", 90.0, "u_1"),
    ])
    rows = _comparisons(con)
    assert len(rows) == 1
    assert rows[0][2] == "amazon_in" and rows[0][3] == "flipkart"


def test_comparison_b_then_a_primary_is_b():
    """The v1.0 bug: primary_app was the alphabetically smaller app, not the earlier one."""
    con = _run([
        ("e1", "u", _T0, "flipkart", "p", 100.0, "u_1"),
        ("e2", "u", _T0 + timedelta(minutes=5), "amazon_in", "p", 90.0, "u_1"),
    ])
    rows = _comparisons(con)
    assert len(rows) == 1
    assert rows[0][2] == "flipkart" and rows[0][3] == "amazon_in"
    assert rows[0][4] == _T0 and rows[0][5] == _T0 + timedelta(minutes=5)


def test_same_app_twice_gives_no_comparison():
    con = _run([
        ("e1", "u", _T0, "amazon_in", "p", 100.0, "u_1"),
        ("e2", "u", _T0 + timedelta(minutes=5), "amazon_in", "p", 90.0, "u_1"),
    ])
    assert _comparisons(con) == []


def test_gap_one_second_over_window_gives_none():
    con = _run([
        ("e1", "u", _T0, "amazon_in", "p", 100.0, "u_1"),
        ("e2", "u", _T0 + timedelta(minutes=60, seconds=1), "flipkart", "p", 90.0, "u_1"),
    ])
    assert _comparisons(con) == []


def test_gap_exactly_at_window_gives_one():
    con = _run([
        ("e1", "u", _T0, "amazon_in", "p", 100.0, "u_1"),
        ("e2", "u", _T0 + timedelta(minutes=60), "flipkart", "p", 90.0, "u_1"),
    ])
    assert len(_comparisons(con)) == 1


def test_custom_window_is_respected():
    rows = [
        ("e1", "u", _T0, "amazon_in", "p", 100.0, "u_1"),
        ("e2", "u", _T0 + timedelta(minutes=10, seconds=1), "flipkart", "p", 90.0, "u_1"),
    ]
    assert _comparisons(_run(rows, window=10.0)) == []
    assert len(_comparisons(_run(rows, window=11.0))) == 1


def test_same_timestamp_ordered_by_event_id():
    con = _run([
        ("e2", "u", _T0, "amazon_in", "p", 100.0, "u_1"),
        ("e1", "u", _T0, "flipkart", "p", 90.0, "u_1"),
    ])
    rows = _comparisons(con)
    assert len(rows) == 1
    assert rows[0][2] == "flipkart" and rows[0][3] == "amazon_in"


def test_three_apps_in_sequence_give_two_rows():
    con = _run([
        ("e1", "u", _T0, "amazon_in", "p", 100.0, "u_1"),
        ("e2", "u", _T0 + timedelta(minutes=10), "flipkart", "p", 90.0, "u_1"),
        ("e3", "u", _T0 + timedelta(minutes=20), "walmart", "p", 80.0, "u_1"),
    ])
    rows = _comparisons(con)
    assert [(r[2], r[3]) for r in rows] == [("amazon_in", "flipkart"), ("flipkart", "walmart")]


def test_comparison_is_between_consecutive_events_only():
    """A, B, B gives one row (A to the first B); the second B has a same-app predecessor."""
    con = _run([
        ("e1", "u", _T0, "amazon_in", "p", 100.0, "u_1"),
        ("e2", "u", _T0 + timedelta(minutes=10), "flipkart", "p", 90.0, "u_1"),
        ("e3", "u", _T0 + timedelta(minutes=20), "flipkart", "p", 90.0, "u_1"),
    ])
    assert len(_comparisons(con)) == 1


def test_return_trip_a_b_a_gives_two_rows():
    con = _run([
        ("e1", "u", _T0, "amazon_in", "p", 100.0, "u_1"),
        ("e2", "u", _T0 + timedelta(minutes=10), "flipkart", "p", 90.0, "u_1"),
        ("e3", "u", _T0 + timedelta(minutes=20), "amazon_in", "p", 95.0, "u_1"),
    ])
    assert [(r[2], r[3]) for r in _comparisons(con)] == [("amazon_in", "flipkart"), ("flipkart", "amazon_in")]


def test_users_and_products_are_isolated():
    con = _run([
        ("e1", "u1", _T0, "amazon_in", "p", 100.0, "u1_1"),
        ("e2", "u2", _T0 + timedelta(minutes=1), "flipkart", "p", 90.0, "u2_1"),
        ("e3", "u1", _T0 + timedelta(minutes=2), "flipkart", "q", 90.0, "u1_1"),
    ])
    assert _comparisons(con) == []


def test_prices_cheaper_app_and_session_come_from_the_right_events():
    con = _run([
        ("e1", "u", _T0, "flipkart", "p", 120.0, "u_1"),
        ("e2", "u", _T0 + timedelta(minutes=5), "amazon_in", "p", 100.0, "u_2"),
    ])
    (row,) = _comparisons(con)
    assert row[6] == "u_1"  # the previous event's session
    assert row[7] == 120.0 and row[8] == 100.0
    assert row[9] == "amazon_in"
    duration = con.execute("SELECT CAST(duration_minutes AS DOUBLE) FROM fact_cross_app_comparisons").fetchone()[0]
    assert duration == pytest.approx(5.0)


def test_cheaper_app_is_first_app_when_first_is_cheaper():
    con = _run([
        ("e1", "u", _T0, "flipkart", "p", 80.0, "u_1"),
        ("e2", "u", _T0 + timedelta(minutes=5), "amazon_in", "p", 100.0, "u_1"),
    ])
    assert _comparisons(con)[0][9] == "flipkart"


def test_cheaper_app_null_on_tie_and_on_missing_price():
    tie = _run([
        ("e1", "u", _T0, "flipkart", "p", 100.0, "u_1"),
        ("e2", "u", _T0 + timedelta(minutes=5), "amazon_in", "p", 100.0, "u_1"),
    ])
    assert _comparisons(tie)[0][9] is None
    missing = _run([
        ("e1", "u", _T0, "flipkart", "p", None, "u_1"),
        ("e2", "u", _T0 + timedelta(minutes=5), "amazon_in", "p", 100.0, "u_1"),
    ])
    row = _comparisons(missing)[0]
    assert row[7] is None and row[8] == 100.0 and row[9] is None


def test_detection_is_idempotent_and_replaces_stale_rows():
    con = _run([
        ("e1", "u", _T0, "flipkart", "p", 100.0, "u_1"),
        ("e2", "u", _T0 + timedelta(minutes=5), "amazon_in", "p", 90.0, "u_1"),
    ])
    first = _comparisons(con)
    detect_cross_app_comparisons(con=con)
    assert _comparisons(con) == first


@pytest.mark.parametrize("restricted", [False, True])
def test_comparison_sql_uses_lag_and_no_self_join(restricted):
    sql = " ".join(
        sessionizer_module.comparison_insert_sql(60.0, restrict_to_user_product=restricted).upper().split()
    )
    assert "LAG(" in sql
    assert "PARTITION BY USER_ID, CANONICAL_PRODUCT_ID ORDER BY TIMESTAMP, EVENT_ID" in sql
    assert " JOIN " not in sql
    assert sql.count("FACT_EVENTS") == 1


_SHARED = []


def _shared_con():
    if not _SHARED:
        _SHARED.append(_cmp_con())
    return _SHARED[0]


_APPS = ["amazon_in", "flipkart", "walmart"]


@settings(max_examples=25, deadline=None)
@given(
    events=st.lists(
        st.tuples(st.integers(0, 150), st.sampled_from(_APPS), st.sampled_from(["p", "q"]),
                  st.one_of(st.none(), st.integers(1, 5))),
        min_size=0, max_size=8,
    ),
    window=st.sampled_from([10.0, 60.0]),
)
def test_batch_matches_reference_model_for_any_sequence(events, window):
    rows = [(f"e{i:02d}", "u", _T0 + timedelta(minutes=m), app, prod, None if pr is None else pr * 10.0, "u_1")
            for i, (m, app, prod, pr) in enumerate(events)]
    con = _shared_con()
    con.execute("DELETE FROM fact_cross_app_comparisons")
    con.execute("DELETE FROM fact_events")
    _add(con, rows)
    detect_cross_app_comparisons(con=con, max_window_minutes=window)
    expected = []
    for prod in {r[4] for r in rows}:
        seq = sorted((r for r in rows if r[4] == prod), key=lambda r: (r[2], r[0]))
        for prev, cur in zip(seq, seq[1:]):
            minutes = (cur[2] - prev[2]).total_seconds() / 60.0
            if prev[3] != cur[3] and 0 <= minutes <= window:
                cheaper = None
                if prev[5] is not None and cur[5] is not None and prev[5] != cur[5]:
                    cheaper = prev[3] if prev[5] < cur[5] else cur[3]
                expected.append((prod, prev[3], cur[3], prev[2], cur[2], prev[5], cur[5], cheaper))
    got = [(r[1], r[2], r[3], r[4], r[5], r[7], r[8], r[9]) for r in _comparisons(con)]
    assert sorted(got, key=str) == sorted(expected, key=str)
