"""Tests for the v1.4 behaviour engine: funnel, sessions, comparison, churn, drift, parity."""
from __future__ import annotations

import ast
import random
import statistics
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from hypothesis import given, settings, strategies as st

from src.simulation.behavior import APPS, TRUTH_COLUMNS, BehaviorEngine, UserProfile, UserState
from src.simulation.pricebook import PriceBook
from src.ingestion.catalog import build_catalog

START = datetime(2026, 9, 28, 8, 0)
DAYS = 21
USERS = 60
RAW_COLUMNS = ["event_id", "user_id", "timestamp", "app_name", "raw_text"]
SPEC_TRUTH = [
    "event_id", "user_id", "archetype", "true_session_id", "intent_family", "state",
    "true_sku_id", "true_action", "true_price_inr", "true_app", "is_deliberate_comparison",
    "chose_cheapest_seen",
]
CATALOG = build_catalog()


def make_engine(seed: int = 7, days: int = DAYS, **kw) -> BehaviorEngine:
    return BehaviorEngine(CATALOG, PriceBook(CATALOG, APPS, days, seed), seed=seed, **kw)


def run(seed: int = 7, users: int = USERS, days: int = DAYS, **kw):
    eng = make_engine(seed, days, **kw)
    profiles = eng.create_profiles(users)
    raw, truth = eng.simulate(profiles, START, days)
    return eng, profiles, raw, truth


@pytest.fixture(scope="module")
def sim():
    return run()


def _sessions(truth: pl.DataFrame, raw: pl.DataFrame) -> pl.DataFrame:
    return truth.join(raw.select("event_id", "timestamp"), on="event_id").sort(["user_id", "timestamp", "event_id"])


def test_truth_columns_constant_has_spec_columns():
    assert set(SPEC_TRUTH) <= set(TRUTH_COLUMNS)


def test_simulate_returns_spec_columns(sim):
    _, _, raw, truth = sim
    assert raw.columns == RAW_COLUMNS
    assert set(SPEC_TRUTH) <= set(truth.columns)
    assert raw.height > USERS
    assert raw["timestamp"].dtype == pl.Datetime
    assert truth["state"].is_in(["browse", "consider", "compare", "cart", "buy"]).all()
    assert truth["true_action"].is_in(["view", "cart", "buy"]).all()
    assert truth["true_app"].is_in(list(APPS)).all()
    assert raw["app_name"].to_list() == truth["true_app"].to_list()


def test_same_seed_same_output(sim):
    _, _, raw, truth = sim
    _, _, raw2, truth2 = run()
    assert raw.equals(raw2)
    assert truth.equals(truth2)


def test_different_seed_differs(sim):
    _, _, raw, _ = sim
    _, _, raw2, _ = run(seed=8)
    assert not raw.equals(raw2)


def test_engine_does_not_use_module_random():
    random.seed(1)
    _, _, raw_a, _ = run(users=15, days=14)
    random.seed(999)
    random.random()
    _, _, raw_b, _ = run(users=15, days=14)
    assert raw_a.equals(raw_b)


def test_behavior_source_never_calls_module_random():
    src = (Path(__file__).resolve().parents[1] / "src/simulation/behavior.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "random":
                assert node.func.attr == "Random", f"module-level random.{node.func.attr} used"


def test_every_buy_follows_a_view_of_the_same_sku(sim):
    _, _, raw, truth = sim
    df = _sessions(truth, raw)
    buys = df.filter(pl.col("true_action") == "buy")
    assert buys.height > 0
    views = df.filter(pl.col("true_action") == "view").select(
        "user_id", "true_sku_id", pl.col("timestamp").alias("vt"))
    j = buys.join(views, on=["user_id", "true_sku_id"], how="left").filter(
        (pl.col("vt") < pl.col("timestamp")) & (pl.col("timestamp") - pl.col("vt") <= timedelta(days=7)))
    ok = j.select("event_id").unique().height
    assert ok == buys.height


def test_sessions_are_bursty(sim):
    _, _, raw, truth = sim
    sizes = truth.group_by(["user_id", "true_session_id"]).len()["len"].to_list()
    assert statistics.median(sizes) >= 3
    assert sum(1 for s in sizes if s == 1) / len(sizes) <= 0.25


def test_within_session_gaps_are_short(sim):
    _, _, raw, truth = sim
    df = _sessions(truth, raw)
    df = df.with_columns(
        gap=(pl.col("timestamp") - pl.col("timestamp").shift(1).over("user_id")).dt.total_seconds(),
        prev_sess=pl.col("true_session_id").shift(1).over("user_id"),
    ).filter(pl.col("gap").is_not_null())
    within = df.filter(pl.col("prev_sess") == pl.col("true_session_id"))
    between = df.filter(pl.col("prev_sess") != pl.col("true_session_id"))
    assert within.height > 0 and between.height > 0
    assert within.filter(~pl.col("is_deliberate_comparison"))["gap"].max() <= 180
    assert between["gap"].min() > 30 * 60


def test_within_session_gap_bounds_exact(sim):
    _, _, raw, truth = sim
    df = _sessions(truth, raw).with_columns(
        gap=(pl.col("timestamp") - pl.col("timestamp").shift(1).over("user_id")).dt.total_seconds(),
        prev_sess=pl.col("true_session_id").shift(1).over("user_id"),
    ).filter(pl.col("gap").is_not_null() & (pl.col("prev_sess") == pl.col("true_session_id")))
    normal = df.filter(~pl.col("is_deliberate_comparison"))
    comp = df.filter(pl.col("is_deliberate_comparison"))
    assert normal["gap"].min() >= 20 and normal["gap"].max() <= 180
    assert comp.height > 0
    assert comp["gap"].min() >= 60 and comp["gap"].max() <= 600


def test_sessions_do_not_cross_midnight(sim):
    _, _, raw, truth = sim
    df = _sessions(truth, raw).group_by(["user_id", "true_session_id"]).agg(
        a=pl.col("timestamp").min().dt.date(), b=pl.col("timestamp").max().dt.date())
    assert (df["a"] == df["b"]).all()


def test_events_within_window(sim):
    _, _, raw, _ = sim
    assert raw["timestamp"].min() >= START
    assert raw["timestamp"].max() < START + timedelta(days=DAYS + 1)


def test_deliberate_comparison_changes_app_within_ten_minutes(sim):
    _, _, raw, truth = sim
    df = _sessions(truth, raw).with_columns(
        pa=pl.col("true_app").shift(1).over("user_id"),
        ps=pl.col("true_sku_id").shift(1).over("user_id"),
        pt=pl.col("timestamp").shift(1).over("user_id"),
        psess=pl.col("true_session_id").shift(1).over("user_id"),
    )
    comp = df.filter(pl.col("is_deliberate_comparison"))
    assert comp.height > 0
    assert (comp["true_app"] != comp["pa"]).all()
    assert (comp["true_sku_id"] == comp["ps"]).all()
    assert (comp["true_session_id"] == comp["psess"]).all()
    gaps = (comp["timestamp"] - comp["pt"]).dt.total_seconds()
    assert gaps.min() >= 60 and gaps.max() <= 600


def test_comparison_events_are_in_compare_state(sim):
    _, _, _, truth = sim
    comp = truth.filter(pl.col("is_deliberate_comparison"))
    assert (comp["state"] == "compare").all()
    assert (comp["true_action"] == "view").all()


def test_chose_cheapest_seen_only_on_buys_after_comparison(sim):
    _, _, raw, truth = sim
    df = _sessions(truth, raw)
    assert df.filter(pl.col("true_action") != "buy")["chose_cheapest_seen"].is_null().all()
    had_comp = df.with_columns(
        c=pl.col("is_deliberate_comparison").cum_sum().over(["user_id", "true_session_id"]))
    buys = had_comp.filter(pl.col("true_action") == "buy")
    assert buys.filter(pl.col("c") == 0)["chose_cheapest_seen"].is_null().all()
    after = buys.filter(pl.col("c") > 0)
    assert after.height > 0
    assert after["chose_cheapest_seen"].is_not_null().all()


def test_buy_without_comparison_is_on_current_app(sim):
    _, _, raw, truth = sim
    df = _sessions(truth, raw).with_columns(
        c=pl.col("is_deliberate_comparison").cum_sum().over(["user_id", "true_session_id"]),
        pa=pl.col("true_app").shift(1).over("user_id"),
        psess=pl.col("true_session_id").shift(1).over("user_id"))
    b = df.filter((pl.col("true_action") == "buy") & (pl.col("c") == 0) & (pl.col("psess") == pl.col("true_session_id")))
    assert b.height > 0
    assert (b["true_app"] == b["pa"]).all()


def _cheap_mean(seed, beta, users=120):
    _, _, _, truth = run(seed=seed, users=users, price_beta=beta)
    col = truth["chose_cheapest_seen"].drop_nulls()
    assert len(col) >= 20
    return col.cast(pl.Float64).mean()


def test_price_beta_zero_ignores_price():
    m0 = _cheap_mean(7, 0.0)
    ref = _cheap_mean(1234, 0.0)
    assert abs(m0 - ref) <= 0.10
    m8 = _cheap_mean(7, 8.0)
    assert m8 >= m0 + 0.15


def test_churned_users_stay_churned(sim):
    eng, _, raw, truth = sim
    churned = {u: s.churned_day for u, s in eng.states.items() if s.churned_day is not None}
    assert churned
    cut = _sessions(truth, raw)
    for u, d in churned.items():
        assert d >= 7
        last = cut.filter(pl.col("user_id") == u)["timestamp"]
        if len(last):
            assert last.max() < START + timedelta(days=d)


def test_no_churn_before_day_seven():
    eng, _, raw, _ = run(users=40, days=7)
    assert all(s.churned_day is None for s in eng.states.values())


def test_churn_share_in_range_over_sixty_days():
    eng, _, _, _ = run(seed=3, users=200, days=60)
    share = sum(s.churned_day is not None for s in eng.states.values()) / 200
    assert 0.20 <= share <= 0.35


def test_continue_user_does_not_revive_churned(sim):
    eng, profiles, _, _ = sim
    p = next(p for p in profiles if eng.states[p.user_id].churned_day is not None)
    st_ = eng.states[p.user_id]
    out = eng.continue_user(p, st_, START + timedelta(days=DAYS), START + timedelta(days=DAYS + 7))
    assert out == []


def test_truth_and_raw_events_align(sim):
    _, _, raw, truth = sim
    assert raw["event_id"].to_list() == truth["event_id"].to_list()
    assert raw["event_id"].n_unique() == raw.height
    assert raw["user_id"].to_list() == truth["user_id"].to_list()


def test_raw_text_non_empty(sim):
    _, _, raw, _ = sim
    assert (raw["raw_text"].str.len_chars() > 0).all()


def test_screens_quote_focal_with_pricebook():
    _, _, raw, truth = run(users=20, days=7, ocr_noise=0.0)
    sku = {s["sku_id"]: s for s in CATALOG}
    assert raw.height >= 50
    for k in range(50):
        brand = sku[truth["true_sku_id"][k]]["brand"].lower()
        assert brand in raw["raw_text"][k].lower()


def test_intent_family_matches_focal_family_mostly(sim):
    _, _, _, truth = sim
    fam = {s["sku_id"]: s["family_id"] for s in CATALOG}
    non_comp = truth.filter(~pl.col("is_deliberate_comparison"))
    share = sum(fam[s] == f for s, f in zip(non_comp["true_sku_id"], non_comp["intent_family"])) / non_comp.height
    assert share >= 0.5


def test_one_intent_family_per_session(sim):
    _, _, _, truth = sim
    n = truth.group_by(["user_id", "true_session_id"]).agg(pl.col("intent_family").n_unique())["intent_family"]
    assert n.max() == 1


def test_archetype_recorded_and_drifts():
    _, _, _, truth = run(seed=5, users=120, days=42)
    assert truth["archetype"].null_count() == 0
    per_user = truth.group_by("user_id").agg(pl.col("archetype").n_unique())["archetype"]
    assert per_user.max() <= 4
    assert (per_user == 1).sum() > 0


def test_archetype_drift_constant():
    from src.config import ARCHETYPE_DRIFT_WEEKLY
    assert ARCHETYPE_DRIFT_WEEKLY == 0.03


def test_config_defaults():
    from src import config
    assert config.SIMULATION_USERS == 1000
    assert config.SIMULATION_DAYS == 60
    assert config.SIMULATION_START == "2026-09-28 08:00:00"
    assert config.PRICE_BETA == 4.0
    assert config.TRAIN_REFERENCE_DATES == (
        "2026-10-12 00:00:00", "2026-10-19 00:00:00", "2026-10-26 00:00:00",
        "2026-11-02 00:00:00", "2026-11-09 00:00:00")
    assert config.TEST_REFERENCE_DATE == "2026-11-16 00:00:00"


def test_clv_dates_live_in_config():
    from src import config
    assert config.CLV_CALIBRATION_END == "2026-11-02 00:00:00"
    assert config.CLV_HOLDOUT_DAYS == 21


def test_create_profiles_deterministic_and_sized():
    a = make_engine().create_profiles(25)
    b = make_engine().create_profiles(25)
    assert len(a) == 25
    assert [p.user_id for p in a] == [p.user_id for p in b]
    assert len({p.user_id for p in a}) == 25
    assert all(isinstance(p, UserProfile) for p in a)


def test_continue_user_matches_simulate():
    seed = 11
    eng, profiles, raw, truth = run(seed=seed, users=USERS, days=DAYS)
    eng14 = make_engine(seed, DAYS)
    profs14 = eng14.create_profiles(USERS)
    eng14.simulate(profs14, START, 14)
    full = raw.join(truth, on=["event_id", "user_id"])
    t0 = START + timedelta(days=14)
    t1 = START + timedelta(days=21)
    checked = 0
    for p in profs14:
        state = eng14.states[p.user_id]
        got = eng14.continue_user(p, state, t0, t1)
        want = full.filter((pl.col("user_id") == p.user_id) & (pl.col("timestamp") >= t0)).sort(
            ["timestamp", "event_id"]).to_dicts()
        keys = [(g["timestamp"], g["event_id"]) for g in got]
        assert keys == sorted(keys)
        got = sorted(got, key=lambda g: (g["timestamp"], g["event_id"]))
        assert len(got) == len(want)
        for g, w in zip(got, want):
            for k in RAW_COLUMNS + SPEC_TRUTH:
                assert g[k] == w[k], (p.user_id, k)
        checked += len(want)
    assert checked > 0


def test_continue_user_empty_when_until_equals_now():
    eng, profiles, _, _ = run(users=10, days=14)
    p = profiles[0]
    now = START + timedelta(days=14)
    assert eng.continue_user(p, eng.states[p.user_id], now, now) == []


@settings(max_examples=8, deadline=None)
@given(seed=st.integers(0, 10_000))
def test_property_seed_determinism_and_alignment(seed):
    _, _, r1, t1 = run(seed=seed, users=8, days=8)
    _, _, r2, t2 = run(seed=seed, users=8, days=8)
    assert r1.equals(r2) and t1.equals(t2)
    assert r1["event_id"].to_list() == t1["event_id"].to_list()
    assert t1["true_action"].is_in(["view", "cart", "buy"]).all()


@settings(max_examples=8, deadline=None)
@given(seed=st.integers(0, 10_000))
def test_property_buys_have_prior_view(seed):
    _, _, raw, truth = run(seed=seed, users=10, days=14)
    df = _sessions(truth, raw)
    seen = {}
    for r in df.iter_rows(named=True):
        key = (r["user_id"], r["true_sku_id"])
        if r["true_action"] == "view":
            seen.setdefault(key, []).append(r["timestamp"])
        if r["true_action"] == "buy":
            assert any(0 < (r["timestamp"] - t).total_seconds() <= 7 * 86400 for t in seen.get(key, []))


def test_engine_leaves_global_random_state_untouched():
    random.seed(4242)
    before = random.getstate()
    eng = make_engine()
    profiles = eng.create_profiles(10)
    eng.simulate(profiles, START, 7)
    assert random.getstate() == before
