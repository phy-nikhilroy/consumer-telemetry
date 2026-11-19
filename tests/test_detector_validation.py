"""Tests for src/experiments/detector_validation.py."""

from datetime import datetime, timedelta

import duckdb
import polars as pl
import pytest

from src.experiments import detector_validation as dv


def _con(rows):
    """In-memory fact_events from (event_id, user, minute, app, product)."""
    con = duckdb.connect()
    con.execute("""CREATE TABLE fact_events (event_id VARCHAR, user_id VARCHAR, timestamp TIMESTAMP,
        app_name VARCHAR, canonical_product_id VARCHAR, price_inr DOUBLE, session_id VARCHAR)""")
    t0 = datetime(2026, 1, 1)
    for eid, u, m, app, p in rows:
        con.execute("INSERT INTO fact_events VALUES (?,?,?,?,?,?,NULL)", [eid, u, t0 + timedelta(minutes=m), app, p, 100.0])
    return con


def _truth(rows):
    return pl.DataFrame({"event_id": [r[0] for r in rows], "true_session_id": [r[1] for r in rows],
                         "is_deliberate_comparison": [r[2] for r in rows]})


def test_prf_counts():
    r = dv.prf(2, 2, 0)
    assert r["precision"] == 0.5 and r["recall"] == 1.0
    assert r["f1"] == pytest.approx(2 / 3)


def test_prf_zero_denominators():
    assert dv.prf(0, 0, 0)["f1"] == 0.0


def test_sessionizer_perfect_and_over_split():
    con = _con([("a", "u", 0, "x", "p"), ("b", "u", 10, "x", "p"), ("c", "u", 100, "x", "p")])
    truth = _truth([("a", "s1", False), ("b", "s1", False), ("c", "s2", False)])
    ok = dv.score_sessionizer(con, truth, 30)
    assert ok["f1"] == 1.0
    tight = dv.score_sessionizer(con, truth, 5)
    assert tight["precision"] == 0.5 and tight["recall"] == 1.0
    loose = dv.score_sessionizer(con, truth, 120)
    assert loose["recall"] == 0.0


def test_comparison_scores_later_event():
    con = _con([("a", "u", 0, "x", "p"), ("b", "u", 20, "y", "p"), ("c", "u", 500, "x", "p")])
    truth = _truth([("a", "s1", False), ("b", "s1", True), ("c", "s2", True)])
    r30 = dv.score_comparisons(con, truth, 30)
    assert r30["precision"] == 1.0 and r30["recall"] == 0.5
    r10 = dv.score_comparisons(con, truth, 10)
    assert r10["tp"] == 0 and r10["recall"] == 0.0


def test_choose_best_mean_f1_ties_go_smaller():
    def blk(f):
        return {"f1": f}
    per = {1: {"sessionizer": {"5": blk(0.5), "10": blk(0.9), "20": blk(0.9)}},
           2: {"sessionizer": {"5": blk(0.5), "10": blk(0.9), "20": blk(0.9)}}}
    assert dv.choose(per, "sessionizer", [1, 2], prefer=0) == 10.0
    assert dv.choose(per, "sessionizer", [1, 2], prefer=25) == 20.0


def test_grids_match_spec():
    assert dv.INACTIVITY_GRID == (5, 10, 20, 30, 45, 60)
    assert dv.WINDOW_GRID == (10, 30, 60, 120)
    assert dv.SELECTION_SEEDS == (1, 2, 3) and dv.REPORT_SEEDS == (4, 5)
