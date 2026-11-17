"""Tests for the screen segmenter."""

from __future__ import annotations

import random

import pytest

from src.ingestion.screen_generator import apply_ocr_noise, render_listing, render_screen
from src.ingestion.segmenter import is_end_marker, segment_screen
from src.ingestion.synthetic_data import ACTION_SNIPPETS, CATALOG, generate_single_sample

APPS = list(ACTION_SNIPPETS)
ACTIONS = ["view", "cart", "buy", "search", "wishlist"]


def test_one_line_v1_string_is_returned_unchanged() -> None:
    raw = "Samsung Galaxy S24 Ultra 256GB Titanium $1299.99 4.7 Add to Cart"
    assert segment_screen(raw) == [raw]


def test_every_v1_sample_is_one_segment() -> None:
    random.seed(0)
    for i in range(100):
        raw = generate_single_sample(i)["raw_text"]
        assert segment_screen(raw) == [raw]


def test_empty_string_is_one_segment() -> None:
    assert segment_screen("") == [""]


def test_blank_multiline_returns_no_segments() -> None:
    assert segment_screen("\n  \n") == []


@pytest.mark.parametrize("n", range(1, 7))
def test_rendered_screen_segments_into_n_parts(n: int) -> None:
    for seed in range(30):
        rng = random.Random(seed * 100 + n)
        items = [rng.choice(CATALOG) for _ in range(n)]
        listings = [
            render_listing(item, rng.choice(APPS), rng.choice(ACTIONS), 0, 4.1, rng) for item in items
        ]
        segments = segment_screen(render_screen(listings, rng))
        assert len(segments) == n
        for item, segment in zip(items, segments):
            assert item["title"] in segment


def test_trailing_lines_join_the_last_listing() -> None:
    screen = "Apple iPhone 15\n$799.00\nBuy Now\nSamsung Galaxy S24\nAdd to Cart\nSponsored footer"
    assert segment_screen(screen) == [
        "Apple iPhone 15 $799.00 Buy Now",
        "Samsung Galaxy S24 Add to Cart Sponsored footer",
    ]


def test_screen_without_markers_is_one_segment() -> None:
    assert segment_screen("Apple iPhone 15\n$799.00") == ["Apple iPhone 15 $799.00"]


@pytest.mark.parametrize("line", ["Buy Now", "ADD TO CART", "In stock", "GO TO CART", "Buy now with 1-click"])
def test_clean_end_markers(line: str) -> None:
    assert is_end_marker(line)


@pytest.mark.parametrize("line", ["8uy Now", "BuyNow", "AddtoCart", "In5tock", "8uyNow"])
def test_noisy_end_markers(line: str) -> None:
    assert is_end_marker(line)


@pytest.mark.parametrize("line", ["Apple iPhone 15 128GB", "$799.00 M.R.P: $899.00 (11% off)", "Add to Wish List"])
def test_non_markers(line: str) -> None:
    assert not is_end_marker(line)


def test_light_noise_keeps_segment_count() -> None:
    hits = 0
    for seed in range(200):
        rng = random.Random(seed)
        n = rng.randint(2, 5)
        listings = [render_listing(rng.choice(CATALOG), rng.choice(APPS), "buy", 0, 4.0, rng) for _ in range(n)]
        screen = apply_ocr_noise(render_screen(listings, rng), 0.02, rng)
        hits += len(segment_screen(screen)) == n
    assert hits / 200 >= 0.95
