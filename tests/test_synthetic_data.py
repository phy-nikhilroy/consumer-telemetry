"""Tests for the shared screen-text generator used by the simulator and the live stream."""

from __future__ import annotations

import random

from src.ingestion.synthetic_data import CATALOG, generate_app_screen_text
from src.ingestion.pipeline import process


def test_generate_app_screen_text_is_seedable():
    item = CATALOG[0]
    first = generate_app_screen_text(item, "flipkart", "cart", 10, 4.2, rng=random.Random(9))
    second = generate_app_screen_text(item, "flipkart", "cart", 10, 4.2, rng=random.Random(9))
    assert first == second
    assert "Add to Cart" in first and "M.R.P" in first and "(10% off)" in first


def test_generate_app_screen_text_currency_by_app():
    item = CATALOG[0]
    assert "₹" in generate_app_screen_text(item, "amazon_in", "view", 0, 4.0, rng=random.Random(1))
    usd = generate_app_screen_text(item, "walmart", "buy", 0, 4.0, rng=random.Random(1))
    assert "$" in usd and "Buy Now" in usd


def test_generated_text_extracts_to_catalog_price():
    """Every catalog item's generated USD text extracts back to its exact price."""
    for item in CATALOG:
        text = generate_app_screen_text(item, "amazon_us", "view", 0, 4.5, rng=random.Random(2))
        assert process(text).price == round(item["base_usd"], 2), text
