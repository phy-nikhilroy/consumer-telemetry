"""
Unit tests for cross-app behavioral and entropy feature store calculations.
"""

from datetime import datetime, timedelta
import math
from pathlib import Path
import duckdb
import polars as pl
import pytest

from src.features.database import DatabaseManager
from src.features.behavioral_features import compute_behavioral_features, save_behavioral_features


def test_entropy_and_dominant_share_isolated():
    """Verifies Shannon entropy and dominant share on known probability distributions."""
    db = DatabaseManager(":memory:")
    db.init_schema()
    con = db.get_connection()

    try:
        t0 = datetime(2026, 10, 1, 10, 0, 0)
        # User Single: 4 events strictly on amazon_in -> entropy = 0.0, dominant_app_share = 1.0
        # User Dual: 2 on amazon_in, 2 on flipkart -> entropy = 1.0, dominant_app_share = 0.5
        events = [
            ("evt_s1", "user_single", t0, "amazon_in", "p1", "view", 100.0, "INR", "s1"),
            ("evt_s2", "user_single", t0 + timedelta(minutes=5), "amazon_in", "p1", "cart", 100.0, "INR", "s1"),
            ("evt_s3", "user_single", t0 + timedelta(minutes=10), "amazon_in", "p1", "buy", 100.0, "INR", "s1"),
            ("evt_s4", "user_single", t0 + timedelta(minutes=15), "amazon_in", "p1", "view", 100.0, "INR", "s1"),

            ("evt_d1", "user_dual", t0, "amazon_in", "p1", "view", 100.0, "INR", "sd1"),
            ("evt_d2", "user_dual", t0 + timedelta(minutes=5), "amazon_in", "p1", "view", 100.0, "INR", "sd1"),
            ("evt_d3", "user_dual", t0 + timedelta(minutes=10), "flipkart", "p1", "view", 100.0, "INR", "sd1"),
            ("evt_d4", "user_dual", t0 + timedelta(minutes=15), "flipkart", "p1", "view", 100.0, "INR", "sd1"),
        ]

        con.executemany("""
            INSERT INTO fact_events (event_id, user_id, timestamp, app_name, canonical_product_id, action, price, currency, session_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, events)

        df = compute_behavioral_features(reference_date=t0 + timedelta(days=1), con=con)

        rows = {r["user_id"]: r for r in df.to_dicts()}

        # Single app validation
        single = rows["user_single"]
        assert single["num_apps_used"] == 1
        assert single["app_entropy"] == 0.0
        assert single["dominant_app_share"] == 1.0

        # Dual app validation
        dual = rows["user_dual"]
        assert dual["num_apps_used"] == 2
        assert dual["app_entropy"] == 1.0
        assert dual["dominant_app_share"] == 0.5
    finally:
        con.close()


def test_behavioral_features_warehouse_schema():
    """Verifies that all 16 behavioral features exist on warehouse dataset."""
    db_path = Path("data/processed/consumer.duckdb")
    if not db_path.exists():
        pytest.skip("Warehouse database not available")

    df = compute_behavioral_features(reference_date="2026-10-28 00:00:00", db_path=db_path)

    expected_features = [
        "num_apps_used",
        "app_entropy",
        "dominant_app_share",
        "cross_app_comparison_count",
        "chose_cheaper_app_rate",
        "view_to_cart_ratio",
        "cart_to_purchase_ratio",
        "avg_session_duration_minutes",
        "avg_time_between_sessions_hours",
        "avg_discount_pct_on_purchases",
        "pct_purchases_with_discount",
        "price_range_browsed",
        "top_category_l1",
        "category_diversity",
        "weekend_event_share",
        "evening_event_share",
    ]

    for feat in expected_features:
        assert feat in df.columns, f"Missing behavioral feature: {feat}"

    # Population size is a simulation parameter; compare against the warehouse.
    with duckdb.connect(str(db_path), read_only=True) as con:
        expected_users = con.execute("SELECT COUNT(*) FROM dim_users").fetchone()[0]
    assert df.height == expected_users
    assert df.height > 0


def test_save_behavioral_features_parquet(tmp_path):
    """Verifies Parquet output writing and schema persistence."""
    df_mock = pl.DataFrame({
        "user_id": ["user_001"],
        "num_apps_used": [3],
        "app_entropy": [1.584],
    })

    out_file = tmp_path / "test_behavioral.parquet"
    saved = save_behavioral_features(df_mock, output_path=out_file)

    assert saved.exists()
    loaded = pl.read_parquet(saved)
    assert loaded.shape == (1, 3)
    assert loaded["user_id"][0] == "user_001"


def test_user_filter_returns_matching_rows_of_full_result(tiny_warehouse):
    """Per-user serving features, cross-app comparisons included, equal the batch rows."""
    from polars.testing import assert_frame_equal

    con = duckdb.connect(str(tiny_warehouse), read_only=True)
    try:
        full = compute_behavioral_features(reference_date="2026-10-21 00:00:00", con=con)
        users = full.sort("cross_app_comparison_count", descending=True)["user_id"].to_list()[:3]
        part = compute_behavioral_features(reference_date="2026-10-21 00:00:00", con=con, user_ids=users)
    finally:
        con.close()
    expected = full.filter(pl.col("user_id").is_in(users))
    assert expected["cross_app_comparison_count"].sum() > 0
    assert_frame_equal(part, expected)


# ---------------------------------------------------------------------------
# W5: features that mean what they say
# ---------------------------------------------------------------------------

from src.config import MISSING_AS_SENTINEL_FEATURES, CROSS_APP_PRICE_LOOKBACK_HOURS
from src.features.feature_store import fill_missing_features
from src.features.migrations import migrate

W5_REF = datetime(2026, 10, 10, 12, 0, 0)
W5_BUY = datetime(2026, 10, 9, 12, 0, 0)


def _w5_con():
    con = duckdb.connect(":memory:")
    migrate(con)
    return con


def _event(con, eid, user, ts, app, product, action, price_inr, discount_pct=None, price=None, currency="INR"):
    con.execute(
        "INSERT INTO fact_events (event_id, user_id, timestamp, app_name, canonical_product_id, "
        "action, price, currency, session_id, discount_pct, price_inr) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [eid, user, ts, app, product, action, price if price is not None else price_inr,
         currency, f"s_{eid}", discount_pct, price_inr],
    )


def _obs(con, oid, user, ts, app, product, price_inr):
    con.execute(
        "INSERT INTO fact_price_observations (observation_id, source_event_id, user_id, timestamp, "
        "app_name, canonical_product_id, price_inr, is_focal) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [oid, f"src_{oid}", user, ts, app, product, price_inr, True],
    )


def _comp(con, cid, user, first, last, p_first, p_last, product="p1"):
    con.execute(
        "INSERT INTO fact_cross_app_comparisons (comparison_id, user_id, session_id, canonical_product_id, "
        "primary_app, compared_app, first_viewed, last_viewed, duration_minutes, price_first_inr, "
        "price_last_inr, cheaper_app) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [cid, user, "s", product, "amazon_in", "flipkart", first, last, 5.0, p_first, p_last, None],
    )


def _row(con, user="u1"):
    df = compute_behavioral_features(reference_date=W5_REF, con=con)
    rows = [r for r in df.to_dicts() if r["user_id"] == user]
    assert len(rows) == 1
    return rows[0]


def test_config_constants_for_price_features():
    assert CROSS_APP_PRICE_LOOKBACK_HOURS == 24
    assert MISSING_AS_SENTINEL_FEATURES == {"chose_cheaper_app_rate": -1.0, "avg_abs_price_gap_pct": -1.0}


def test_chose_cheaper_app_rate_one_when_bought_below_other_app_price():
    con = _w5_con()
    _obs(con, "o1", "u1", W5_BUY - timedelta(hours=2), "amazon_in", "p1", 100.0)
    _event(con, "e1", "u1", W5_BUY, "flipkart", "p1", "buy", 90.0)
    row = _row(con)
    assert row["informed_purchase_count"] == 1
    assert row["chose_cheaper_app_rate"] == 1.0


def test_chose_cheaper_app_rate_zero_when_bought_above_other_app_price():
    con = _w5_con()
    _obs(con, "o1", "u1", W5_BUY - timedelta(hours=2), "amazon_in", "p1", 90.0)
    _event(con, "e1", "u1", W5_BUY, "flipkart", "p1", "buy", 100.0)
    row = _row(con)
    assert row["informed_purchase_count"] == 1
    assert row["chose_cheaper_app_rate"] == 0.0


def test_chose_cheaper_app_rate_equal_price_counts_as_cheaper():
    con = _w5_con()
    _obs(con, "o1", "u1", W5_BUY - timedelta(hours=2), "amazon_in", "p1", 100.0)
    _event(con, "e1", "u1", W5_BUY, "flipkart", "p1", "buy", 100.0)
    assert _row(con)["chose_cheaper_app_rate"] == 1.0


def test_chose_cheaper_compares_against_minimum_other_app_price():
    con = _w5_con()
    _obs(con, "o1", "u1", W5_BUY - timedelta(hours=3), "amazon_in", "p1", 100.0)
    _obs(con, "o2", "u1", W5_BUY - timedelta(hours=1), "myntra", "p1", 85.0)
    _event(con, "e1", "u1", W5_BUY, "flipkart", "p1", "buy", 90.0)
    row = _row(con)
    assert row["informed_purchase_count"] == 1
    assert row["chose_cheaper_app_rate"] == 0.0


def test_chose_cheaper_uses_price_inr_not_raw_price():
    con = _w5_con()
    _obs(con, "o1", "u1", W5_BUY - timedelta(hours=2), "amazon_in", "p1", 8000.0)
    _event(con, "e1", "u1", W5_BUY, "flipkart", "p1", "buy", 7500.0, price=90.0, currency="USD")
    assert _row(con)["chose_cheaper_app_rate"] == 1.0


@pytest.mark.parametrize("case", ["no_obs", "same_app", "too_old", "after_buy", "other_product", "other_user"])
def test_uninformed_purchase_gives_null_rate_and_zero_count(case):
    con = _w5_con()
    ts, app, product, user = W5_BUY - timedelta(hours=2), "amazon_in", "p1", "u1"
    if case == "same_app":
        app = "flipkart"
    elif case == "too_old":
        ts = W5_BUY - timedelta(hours=24, seconds=1)
    elif case == "after_buy":
        ts = W5_BUY + timedelta(seconds=1)
    elif case == "other_product":
        product = "p2"
    elif case == "other_user":
        user = "u2"
        _event(con, "e2", "u2", W5_BUY - timedelta(days=1), "amazon_in", "p9", "view", 10.0)
    if case != "no_obs":
        _obs(con, "o1", user, ts, app, product, 100.0)
    _event(con, "e1", "u1", W5_BUY, "flipkart", "p1", "buy", 90.0)
    row = _row(con)
    assert row["informed_purchase_count"] == 0
    assert row["chose_cheaper_app_rate"] is None


@pytest.mark.parametrize("delta", [timedelta(hours=24), timedelta(0)])
def test_lookback_window_is_closed_at_both_ends(delta):
    con = _w5_con()
    _obs(con, "o1", "u1", W5_BUY - delta, "amazon_in", "p1", 100.0)
    _event(con, "e1", "u1", W5_BUY, "flipkart", "p1", "buy", 90.0)
    row = _row(con)
    assert row["informed_purchase_count"] == 1
    assert row["chose_cheaper_app_rate"] == 1.0


def test_rate_denominator_counts_only_informed_purchases():
    con = _w5_con()
    # Two informed buys (one cheaper, one dearer) and one uninformed buy.
    for i, (prod, buy_price) in enumerate([("p1", 90.0), ("p2", 120.0)]):
        t = W5_BUY - timedelta(days=i)
        _obs(con, f"o{i}", "u1", t - timedelta(hours=1), "amazon_in", prod, 100.0)
        _event(con, f"e{i}", "u1", t, "flipkart", prod, "buy", buy_price)
    _event(con, "e9", "u1", W5_BUY - timedelta(days=5), "flipkart", "p3", "buy", 50.0)
    row = _row(con)
    assert row["informed_purchase_count"] == 2
    assert row["chose_cheaper_app_rate"] == 0.5


def test_informed_purchase_ignores_view_events_and_post_reference_obs():
    con = _w5_con()
    _event(con, "e1", "u1", W5_BUY, "flipkart", "p1", "view", 90.0)
    _obs(con, "o1", "u1", W5_BUY - timedelta(hours=1), "amazon_in", "p1", 100.0)
    row = _row(con)
    assert row["informed_purchase_count"] == 0
    assert row["chose_cheaper_app_rate"] is None


def test_discount_features_use_stored_discount_pct_not_dim_products_avg_price():
    con = _w5_con()
    # avg_price far above the buy prices would fabricate a discount under the old definition.
    con.execute(
        "INSERT INTO dim_products (canonical_product_id, brand, product_title, category_l1, category_l2, avg_price) "
        "VALUES ('p1', 'b', 't', 'c1', 'c2', 100000.00)"
    )
    _event(con, "e1", "u1", W5_BUY, "amazon_in", "p1", "buy", 100.0, discount_pct=0)
    _event(con, "e2", "u1", W5_BUY + timedelta(hours=1), "amazon_in", "p1", "buy", 100.0, discount_pct=10)
    _event(con, "e3", "u1", W5_BUY + timedelta(hours=2), "amazon_in", "p1", "buy", 100.0, discount_pct=30)
    _event(con, "e4", "u1", W5_BUY + timedelta(hours=3), "amazon_in", "p1", "view", 100.0, discount_pct=70)
    row = _row(con)
    assert row["avg_discount_pct_on_purchases"] == pytest.approx(40.0 / 3, abs=0.01)
    assert row["pct_purchases_with_discount"] == pytest.approx(2 / 3, abs=1e-3)


def test_discount_features_zero_with_no_buys():
    con = _w5_con()
    _event(con, "e1", "u1", W5_BUY, "amazon_in", "p1", "view", 100.0, discount_pct=50)
    row = _row(con)
    assert row["avg_discount_pct_on_purchases"] == 0.0
    assert row["pct_purchases_with_discount"] == 0.0


def test_discount_features_zero_when_buys_have_no_discount():
    con = _w5_con()
    _event(con, "e1", "u1", W5_BUY, "amazon_in", "p1", "buy", 100.0, discount_pct=0)
    row = _row(con)
    assert row["avg_discount_pct_on_purchases"] == 0.0
    assert row["pct_purchases_with_discount"] == 0.0


def test_avg_abs_price_gap_pct_mean_of_absolute_gaps():
    con = _w5_con()
    t = W5_BUY
    _event(con, "e1", "u1", t, "amazon_in", "p1", "view", 100.0)
    _comp(con, "c1", "u1", t, t + timedelta(hours=1), 100.0, 120.0)
    _comp(con, "c2", "u1", t, t + timedelta(hours=1), 200.0, 150.0, product="p2")
    assert _row(con)["avg_abs_price_gap_pct"] == pytest.approx(22.5, abs=0.01)


def test_avg_abs_price_gap_pct_null_without_comparisons():
    con = _w5_con()
    _event(con, "e1", "u1", W5_BUY, "amazon_in", "p1", "view", 100.0)
    assert _row(con)["avg_abs_price_gap_pct"] is None


def test_avg_abs_price_gap_pct_requires_last_viewed_at_or_before_reference():
    con = _w5_con()
    _event(con, "e1", "u1", W5_BUY, "amazon_in", "p1", "view", 100.0)
    _comp(con, "c_in", "u1", W5_REF - timedelta(hours=1), W5_REF, 100.0, 110.0)
    _comp(con, "c_out", "u1", W5_REF - timedelta(hours=1), W5_REF + timedelta(seconds=1), 100.0, 300.0, product="p2")
    row = _row(con)
    assert row["avg_abs_price_gap_pct"] == pytest.approx(10.0, abs=0.01)
    assert row["cross_app_comparison_count"] == 1


def test_new_features_for_restricted_users_match_unrestricted_rows():
    con = _w5_con()
    for u, price in (("u1", 90.0), ("u2", 120.0)):
        _obs(con, f"o_{u}", u, W5_BUY - timedelta(hours=2), "amazon_in", "p1", 100.0)
        _event(con, f"e_{u}", u, W5_BUY, "flipkart", "p1", "buy", price, discount_pct=5)
        _comp(con, f"c_{u}", u, W5_BUY, W5_BUY + timedelta(hours=1), 100.0, 90.0)
    full = compute_behavioral_features(reference_date=W5_REF, con=con)
    part = compute_behavioral_features(reference_date=W5_REF, con=con, user_ids=["u2"])
    assert part.to_dicts() == [r for r in full.to_dicts() if r["user_id"] == "u2"]


def test_fill_missing_features_uses_sentinels_for_nullable_price_features():
    df = pl.DataFrame({
        "chose_cheaper_app_rate": [None, 0.25],
        "avg_abs_price_gap_pct": [None, 3.0],
        "informed_purchase_count": [None, 2],
    })
    out = fill_missing_features(df)
    assert out["chose_cheaper_app_rate"].to_list() == [-1.0, 0.25]
    assert out["avg_abs_price_gap_pct"].to_list() == [-1.0, 3.0]
    assert out["informed_purchase_count"].to_list() == [0.0, 2.0]


def test_fill_missing_features_sentinel_never_zero_or_half():
    df = pl.DataFrame({"chose_cheaper_app_rate": [None], "avg_abs_price_gap_pct": [None]})
    out = fill_missing_features(df)
    for name in df.columns:
        assert out[name][0] not in (0.0, 0.5)


def test_no_feature_sql_references_dim_products_avg_price():
    root = Path(__file__).resolve().parent.parent / "src" / "features"
    for name in ("behavioral_features.py", "rfm_features.py", "feature_store.py"):
        assert "avg_price" not in (root / name).read_text(encoding="utf-8"), name
