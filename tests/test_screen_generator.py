"""Tests for the multi-listing screen generator."""

from __future__ import annotations

import random

import pytest

from src.ingestion.screen_generator import (
    OCR_CONFUSIONS,
    apply_ocr_noise,
    render_listing,
    render_screen,
)
from src.ingestion.segmenter import is_end_marker
from src.ingestion.synthetic_data import ACTION_SNIPPETS, CATALOG

APPS = list(ACTION_SNIPPETS)
ACTIONS = ["view", "cart", "buy", "search", "wishlist"]


def _random_listing(rng: random.Random, **kwargs) -> list:
    item = rng.choice(CATALOG)
    return render_listing(
        item,
        app=kwargs.get("app", rng.choice(APPS)),
        action=kwargs.get("action", rng.choice(ACTIONS)),
        discount_pct=rng.choice([0, 10, 25]),
        rating=round(rng.uniform(3.5, 4.9), 1),
        rng=rng,
    )


def test_listing_has_two_to_four_lines_ending_on_a_marker() -> None:
    rng = random.Random(0)
    for _ in range(400):
        lines = _random_listing(rng)
        assert 2 <= len(lines) <= 4
        assert is_end_marker(lines[-1])
        assert not any(is_end_marker(line) for line in lines[:-1])


@pytest.mark.parametrize("action", ["buy", "cart"])
def test_terminal_action_is_the_last_line(action: str) -> None:
    rng = random.Random(1)
    for app in APPS:
        lines = _random_listing(rng, app=app, action=action)
        assert lines[-1] in ACTION_SNIPPETS[app][action]


def test_brand_and_title_are_never_altered() -> None:
    rng = random.Random(2)
    for item in CATALOG:
        for _ in range(10):
            text = " ".join(render_listing(item, "amazon_us", "view", 0, 4.2, rng))
            assert f"{item['brand']} {item['title']}" in text


def test_title_variant_false_keeps_every_spec_in_order() -> None:
    rng = random.Random(3)
    item = next(i for i in CATALOG if len(i["specs"]) > 1)
    lines = render_listing(item, "amazon_us", "buy", 0, 4.0, rng, title_variant=False)
    assert " ".join(item["specs"]) in " ".join(lines)


def test_title_variant_switches_capacity_spacing() -> None:
    rng = random.Random(4)
    item = {"brand": "Samsung", "title": "Galaxy S24", "specs": ["256GB"], "base_inr": 80000, "base_usd": 900}
    texts = {" ".join(render_listing(item, "amazon_us", "buy", 0, 4.0, rng)) for _ in range(50)}
    assert any("256 GB" in t for t in texts)
    assert any("256GB" in t for t in texts)


def test_explicit_price_is_rendered() -> None:
    rng = random.Random(5)
    lines = render_listing(CATALOG[0], "amazon_us", "buy", 0, 4.0, rng, price=1234.5)
    assert "$1,234.50" in " ".join(lines)


def test_unknown_app_or_action_raises() -> None:
    rng = random.Random(6)
    with pytest.raises(ValueError):
        render_listing(CATALOG[0], "ebay", "buy", 0, 4.0, rng)
    with pytest.raises(ValueError):
        render_listing(CATALOG[0], "amazon_us", "steal", 0, 4.0, rng)


def test_noise_rate_zero_is_identity() -> None:
    rng = random.Random(7)
    text = "Samsung Galaxy S24 Ultra 256GB\n$1,299.99\nBuy Now"
    assert apply_ocr_noise(text, 0.0, rng) == text


def test_noise_rate_one_swaps_every_confusable_and_keeps_newlines() -> None:
    rng = random.Random(8)
    text = "SOB 015\nl8"
    noisy = apply_ocr_noise(text, 1.0, rng)
    assert "\n" in noisy
    expected = "".join(OCR_CONFUSIONS.get(c, c) for c in text if c != " ")
    assert noisy.replace(" ", "") == expected


def test_noise_is_seeded() -> None:
    text = "Samsung Galaxy S24 Ultra 256GB Buy Now " * 5
    assert apply_ocr_noise(text, 0.1, random.Random(9)) == apply_ocr_noise(text, 0.1, random.Random(9))


def test_same_seed_renders_the_same_screen() -> None:
    def build(seed: int) -> str:
        rng = random.Random(seed)
        return render_screen([_random_listing(rng) for _ in range(4)], rng)

    assert build(10) == build(10)
    assert build(10) != build(11)


def test_render_screen_joins_with_newlines() -> None:
    rng = random.Random(12)
    listings = [["a", "Buy Now"], ["b", "Add to Cart"]]
    screen = render_screen(listings, rng)
    assert [line for line in screen.split("\n") if line] == ["a", "Buy Now", "b", "Add to Cart"]
    assert not screen.endswith("\n")
