"""Tests for the deterministic price book."""
from __future__ import annotations

import pytest

from src.config import USD_TO_INR_RATE
from src.ingestion.catalog import build_catalog
from src.simulation.pricebook import PROMO_DISCOUNTS, PriceBook

APPS = ["amazon_in", "flipkart", "amazon_us"]


@pytest.fixture(scope="module")
def catalog() -> list[dict]:
    return build_catalog()


def test_determinism(catalog):
    a, b = PriceBook(catalog, APPS, 30, 7), PriceBook(catalog, APPS, 30, 7)
    for it in catalog[:40]:
        for app in APPS:
            for d in range(30):
                assert a.quote(it["sku_id"], app, d) == b.quote(it["sku_id"], app, d)


def test_seed_changes_quotes(catalog):
    a, b = PriceBook(catalog, APPS, 30, 1), PriceBook(catalog, APPS, 30, 2)
    sku = catalog[0]["sku_id"]
    assert any(a.quote(sku, "flipkart", d) != b.quote(sku, "flipkart", d) for d in range(30))


def test_order_independence(catalog):
    keys = [(it["sku_id"], app, d) for it in catalog[:20] for app in APPS for d in (0, 5, 29)]
    fwd = PriceBook(catalog, APPS, 30, 3)
    rev = PriceBook(catalog, APPS, 30, 3)
    f = {k: fwd.quote(*k) for k in keys}
    r = {k: rev.quote(*k) for k in reversed(keys)}
    assert f == r


def test_price_le_mrp_and_discount_set(catalog):
    pb = PriceBook(catalog, APPS, 60, 11)
    seen = set()
    for it in catalog[:60]:
        for app in APPS:
            for d in range(60):
                q = pb.quote(it["sku_id"], app, d)
                assert q.price <= q.mrp
                assert q.discount_pct in (0, *PROMO_DISCOUNTS)
                seen.add(q.discount_pct)
    assert seen == {0, *PROMO_DISCOUNTS}


def test_currency_and_price_inr(catalog):
    pb = PriceBook(catalog, APPS, 10, 5)
    sku = catalog[0]["sku_id"]
    for app in ("amazon_in", "flipkart"):
        q = pb.quote(sku, app, 0)
        assert q.currency == "INR" and q.price_inr == q.price and q.price == round(q.price)
    q = pb.quote(sku, "amazon_us", 0)
    assert q.currency == "USD"
    assert q.price_inr == pytest.approx(q.price * USD_TO_INR_RATE)
    assert q.price == round(q.price, 2)


def test_promotion_lengths(catalog):
    pb = PriceBook(catalog, APPS, 200, 4)
    for it in catalog[:30]:
        ds = [pb.quote(it["sku_id"], "flipkart", d).discount_pct for d in range(200)]
        run, runs = 0, []
        for i, x in enumerate(ds):
            if x:
                run += 1
            if run and (not x or i == len(ds) - 1):
                runs.append(run)
                run = 0
        # adjacent episodes can merge, so only the lower bound is strict
        assert all(r >= 2 for r in runs[:-1] or runs)


def test_prices_differ_across_apps_day0(catalog):
    pb = PriceBook(catalog, APPS, 30, 0)
    assert len(catalog) >= 300
    diff = sum(
        len({pb.quote(it["sku_id"], a, 0).price_inr for a in APPS}) >= 2 for it in catalog[:300]
    )
    assert diff / 300 >= 0.9


def test_prefix_stable_and_beyond_horizon(catalog):
    short, long_ = PriceBook(catalog, APPS, 14, 9), PriceBook(catalog, APPS, 21, 9)
    for it in catalog[:50]:
        for app in APPS:
            for d in range(14):
                assert short.quote(it["sku_id"], app, d) == long_.quote(it["sku_id"], app, d)
            assert short.quote(it["sku_id"], app, 20) == long_.quote(it["sku_id"], app, 20)
            assert short.quote(it["sku_id"], app, 90) == PriceBook(catalog, APPS, 100, 9).quote(
                it["sku_id"], app, 90
            )


def test_bad_inputs(catalog):
    pb = PriceBook(catalog, APPS, 5, 0)
    sku = catalog[0]["sku_id"]
    with pytest.raises(ValueError):
        pb.quote(sku, "flipkart", -1)
    with pytest.raises(ValueError):
        pb.quote("nope", "flipkart", 0)
    with pytest.raises(ValueError):
        pb.quote(sku, "nope", 0)
