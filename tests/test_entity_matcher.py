"""Tests for wording-tolerant product matching."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from src.ingestion.entity_matcher import (
    ProductRegistry,
    build_labelled_pairs,
    capacity_tokens,
    family_split,
    model_tokens,
    pair_score,
    record_name,
    tune_threshold,
    variant_conflict,
)


class BagOfWordsEncoder:
    """Deterministic stand-in for the sentence encoder: hashed bag of words, L2-normalised."""

    dim = 64

    def encode(self, texts, normalize_embeddings=True, batch_size=32):
        out = []
        for text in texts:
            vec = np.zeros(self.dim, dtype=np.float32)
            for tok in text.split():
                vec[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
            norm = np.linalg.norm(vec)
            out.append(vec / norm if norm else vec)
        return np.stack(out)


@pytest.fixture
def encoder() -> BagOfWordsEncoder:
    return BagOfWordsEncoder()


def rec(brand, title, specs=(), price=None, currency="INR", key=None) -> dict:
    record = {"brand": brand, "product_title": title, "specs": list(specs), "price": price, "currency": currency}
    if key:
        record["canonical_product_id"] = key
    return record


@pytest.mark.parametrize("text", ["256 GB", "256GB", "256gb", "Galaxy 256 Gb Titanium"])
def test_capacity_normalisation(text: str) -> None:
    assert capacity_tokens(text) == frozenset({"256gb"})


def test_capacity_terabytes_and_none() -> None:
    assert capacity_tokens("1 TB SSD") == frozenset({"1tb"})
    assert capacity_tokens("Rockerz 450 Black") == frozenset()


def test_model_tokens_contain_a_digit() -> None:
    assert model_tokens("Sony WH-1000XM5 Black 256 GB") == frozenset({"wh-1000xm5", "256gb"})
    assert model_tokens("Bose QuietComfort") == frozenset()


def test_different_capacity_scores_zero(encoder) -> None:
    a = rec("Samsung", "Galaxy S24", ["128GB", "Onyx Black"])
    b = rec("Samsung", "Galaxy S24", ["256 GB", "Onyx Black"])
    assert pair_score(a, b, encoder) == 0.0


def test_different_brand_scores_zero(encoder) -> None:
    assert pair_score(rec("Sony", "WH-1000XM5"), rec("Bose", "WH-1000XM5"), encoder) == 0.0


def test_brand_comparison_ignores_case(encoder) -> None:
    assert pair_score(rec("boAt", "Rockerz 450"), rec("BOAT", "Rockerz 450"), encoder) > 0.9


def test_identical_records_score_one(encoder) -> None:
    a = rec("Apple", "iPhone 15", ["128GB", "Blue"], price=79900)
    assert pair_score(a, dict(a), encoder) == pytest.approx(1.0)


def test_score_in_unit_interval(encoder) -> None:
    pairs = [
        (rec("Apple", "iPhone 15", ["Blue"]), rec("Apple", "AirPods Pro", ["White"])),
        (rec("Nike", "Pegasus 41", ["UK 7"]), rec("Nike", "Pegasus 41", ["UK 8"])),
        (rec("JBL", "Flip 6"), rec("JBL", "Flip 6 Pro")),
    ]
    for a, b in pairs:
        assert 0.0 <= pair_score(a, b, encoder) <= 1.0


def test_incompatible_price_halves_score(encoder) -> None:
    a = rec("Apple", "iPhone 15", ["128GB"], price=79900)
    near = rec("Apple", "iPhone 15", ["128GB"], price=82000)
    far = rec("Apple", "iPhone 15", ["128GB"], price=20000)
    assert pair_score(a, far, encoder) == pytest.approx(pair_score(a, near, encoder) * 0.5)


def test_usd_price_is_converted_before_comparison(encoder) -> None:
    inr = rec("Apple", "iPhone 15", ["128GB"], price=79900, currency="INR")
    usd = rec("Apple", "iPhone 15", ["128GB"], price=950.0, currency="USD")
    assert pair_score(inr, usd, encoder) == pytest.approx(1.0)


def test_variant_conflict_rules() -> None:
    assert variant_conflict("sony wh-1000xm5 black", "sony wh-1000xm5 silver")
    assert variant_conflict("samsung galaxy s24 256gb", "samsung galaxy s24 ultra 256gb")
    assert not variant_conflict("apple iphone 15 black", "apple iphone 15 128gb")
    assert not variant_conflict("apple iphone 15 black 128gb", "apple iphone 15 128gb")


def test_record_name_drops_interface_words_and_split_capacity() -> None:
    name = record_name(rec("Sony", "WH-1000XM5 Black View Details", ["Black"]))
    assert name == "sony wh-1000xm5 black"
    assert record_name(rec("Apple", "iPhone 15 Pro 128", ["128 GB"])) == "apple iphone 15 pro 128gb"


def test_resolve_same_id_for_two_wordings(encoder) -> None:
    registry = ProductRegistry(threshold=0.8, encoder=encoder)
    first = registry.resolve(rec("Samsung", "Galaxy S24 Ultra 256 GB", ["Titanium Gray"], price=129999))
    second = registry.resolve(rec("Samsung", "Galaxy S24 Ultra Titanium Gray", ["256GB"], price=131000))
    assert first == second
    assert len(registry) == 1


def test_resolve_different_ids_for_128_and_256(encoder) -> None:
    registry = ProductRegistry(threshold=0.5, encoder=encoder)
    small = registry.resolve(rec("Apple", "iPhone 15", ["128GB", "Black"]))
    large = registry.resolve(rec("Apple", "iPhone 15", ["256 GB", "Black"]))
    assert small != large
    assert len(registry) == 2


def test_resolve_exact_key_hit_returns_that_id(encoder) -> None:
    registry = ProductRegistry(encoder=encoder)
    first = registry.resolve(rec("Apple", "iPhone 15", ["128GB"], key="k1"))
    assert first == "k1"
    assert registry.resolve(rec("Something", "else", key="k1")) == "k1"


def test_resolve_never_crosses_brands(encoder) -> None:
    registry = ProductRegistry(threshold=0.5, encoder=encoder)
    a = registry.resolve(rec("Sony", "WH-1000XM5"))
    b = registry.resolve(rec("Bose", "WH-1000XM5"))
    assert a != b


def test_from_records_round_trip_resolves_identically(encoder) -> None:
    records = [
        rec("Samsung", "Galaxy S24 Ultra 256 GB", ["Titanium Gray"]),
        rec("Samsung", "Galaxy S24 Ultra Titanium Gray", ["256GB"]),
        rec("Apple", "iPhone 15", ["128GB", "Black"]),
        rec("Apple", "iPhone 15", ["256GB", "Black"]),
        rec("Sony", "WH-1000XM5 Black"),
    ]
    original = ProductRegistry(threshold=0.8, encoder=encoder)
    ids = [original.resolve(r) for r in records]
    copy = ProductRegistry.from_records(original.to_records(), threshold=0.8, encoder=encoder)
    assert len(copy) == len(original)
    assert [copy.resolve(r) for r in records] == ids
    probes = [rec("Apple", "iPhone 15 Black", ["128 GB"]), rec("Sony", "WH-1000XM5", ["Black"])]
    fresh = ProductRegistry.from_records(original.to_records(), threshold=0.8, encoder=encoder)
    assert [original.resolve(p) for p in probes] == [fresh.resolve(p) for p in probes]


def test_to_records_is_plain_data(encoder) -> None:
    import json

    registry = ProductRegistry(threshold=0.8, encoder=encoder)
    registry.resolve(rec("Apple", "iPhone 15", ["128GB"], price=79900))
    registry.resolve(rec("Apple", "iPhone 15 128 GB", [], price=79000))
    json.dumps(registry.to_records())


def test_tune_threshold_picks_best_f1_from_grid() -> None:
    scores = [0.95, 0.9, 0.86, 0.7, 0.6, 0.55]
    labels = [1, 1, 1, 0, 0, 0]
    assert tune_threshold(scores, labels) in (0.75, 0.8, 0.85)


def test_family_split_keeps_siblings_together() -> None:
    from src.ingestion.catalog import build_catalog

    split = family_split(build_catalog())
    assert split["samsung-galaxy-s24"] == split["samsung-galaxy-s24-ultra"]
    assert split["apple-iphone-15"] == split["apple-iphone-15-pro"]
    assert set(split.values()) == {"tune", "test"}


@pytest.fixture(scope="module")
def small_pairs():
    from src.ingestion.catalog import build_catalog

    return build_labelled_pairs(build_catalog(), seed=7, n_pairs=120)


def test_tune_and_test_pairs_share_no_family(small_pairs) -> None:
    tune = {f for p in small_pairs if p["split"] == "tune" for f in p["families"]}
    test = {f for p in small_pairs if p["split"] == "test" for f in p["families"]}
    assert tune and test
    assert not tune & test


def test_pair_composition(small_pairs) -> None:
    positives = [p for p in small_pairs if p["label"] == 1]
    negatives = [p for p in small_pairs if p["label"] == 0]
    hard = [p for p in negatives if p["kind"] == "hard"]
    assert len(positives) == 60
    assert len(hard) >= 0.4 * len(negatives)
    assert all(p["skus"][0] == p["skus"][1] for p in positives)
    assert all(p["skus"][0] != p["skus"][1] for p in negatives)


def test_labelled_pairs_are_deterministic() -> None:
    from src.ingestion.catalog import build_catalog

    a = build_labelled_pairs(build_catalog(), seed=3, n_pairs=20)
    b = build_labelled_pairs(build_catalog(), seed=3, n_pairs=20)
    assert [(p["skus"], p["a"], p["b"], p["label"]) for p in a] == [(p["skus"], p["a"], p["b"], p["label"]) for p in b]


def test_process_without_registry_is_unchanged() -> None:
    from datetime import datetime, timezone

    from src.ingestion.pipeline import process

    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    raw = "Samsung Galaxy S24 Ultra 256GB Titanium $1299.99 4.7 Add to Cart"
    assert process(raw, timestamp=ts).model_dump() == process(raw, timestamp=ts, registry=None).model_dump()


def test_process_with_registry_merges_wordings(encoder) -> None:
    from src.ingestion.pipeline import process, process_screen

    registry = ProductRegistry(threshold=0.8, encoder=encoder)
    a = process("Samsung Galaxy S24 Ultra 256GB Titanium ₹1,29,999 Buy Now", registry=registry)
    b = process("Samsung Galaxy S24 Ultra Titanium 256 GB ₹1,31,000 Add to Cart", registry=registry)
    assert a.canonical_product_id == b.canonical_product_id
    screen = process_screen("Samsung Galaxy S24 Ultra 256 GB Titanium\n₹1,30,500\nBuy Now", registry=registry)
    assert screen[0].canonical_product_id == a.canonical_product_id


def test_real_encoder_separates_storage_and_merges_wording() -> None:
    from src.config import ENTITY_MATCH_THRESHOLD

    registry = ProductRegistry()
    a = registry.resolve(rec("Apple", "iPhone 15", ["128GB", "Blue"], price=79900))
    b = registry.resolve(rec("Apple", "iPhone 15 Blue", ["128 GB"], price=81000))
    c = registry.resolve(rec("Apple", "iPhone 15", ["256GB", "Blue"], price=89900))
    assert a == b
    assert c != a
    assert 0.5 <= ENTITY_MATCH_THRESHOLD <= 0.95


@pytest.mark.parametrize("raw, expected", [
    ("Puma Smash v2 White UK 7 ₹4,499 M.R.P: ₹5,999 (25% off) 4.3 ★ (2,847) FREE delivery Thu, 18 Jan In stock",
     "puma smash v2 white uk 7"),
    ("Logitech MX Master 3S Graphite $99.99 4.5 out of 5 stars Prime FREE One-Day Buy Now",
     "logitech mx master 3s graphite"),
    ("Apple iPhone 15 128 GB Blue $799.00 Free 90-day returns Buy now with 1-click", "apple iphone 15 128gb blue"),
    ("Motorola Edge 50 Jungle Green ₹29,732 (20% off) 3.7 out of 5 stars Buy Now", "motorola edge 50 jungle green"),
])
def test_record_name_from_listing_text(raw: str, expected: str) -> None:
    assert record_name({"raw_text": raw, "brand": "ignored", "product_title": "ignored"}) == expected


def test_listing_text_keeps_variants_the_extractor_lost(encoder) -> None:
    a = {"raw_text": "Puma Smash v2 White UK 7 ₹4,499 In stock", "brand": "Puma", "product_title": "Smash v2 White UK"}
    b = {"raw_text": "Puma Smash v2 White UK 8 ₹4,679 Buy Now", "brand": "Puma", "product_title": "Smash v2 White UK"}
    assert record_name({k: v for k, v in a.items() if k != "raw_text"}) == record_name(
        {k: v for k, v in b.items() if k != "raw_text"})
    assert pair_score(a, b, encoder) < pair_score(a, dict(a), encoder) * 0.6
