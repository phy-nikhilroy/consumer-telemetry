from datetime import datetime, timezone
import polars as pl
import pytest
from src.ingestion.pipeline import process, run_pipeline, ExtractedProduct


def test_process_structure():
    raw_text = "Sponsored 4.7 out of 5 stars Apple iPhone 15 (128 GB) ₹68,330 M.R.P: ₹75,920 (10% off) FREE delivery Buy Now"
    prod = process(raw_text, app_source="amazon_in")

    assert isinstance(prod, ExtractedProduct)
    assert prod.brand == "Apple"
    assert "iPhone 15" in prod.product_title
    assert prod.canonical_product_id is not None
    assert prod.price == 68330.0
    assert prod.currency == "INR"
    assert prod.action == "buy"
    assert prod.category_l2 != "Unknown"
    bare = process("Sponsored 4.7 out of 5 stars iPhone 15 (128 GB) FREE delivery Buy Now", app_source="amazon_in")
    assert prod.extraction_confidence > bare.extraction_confidence
    assert 0.0 < prod.extraction_confidence < 1.0


def test_process_empty():
    prod = process("")
    assert isinstance(prod, ExtractedProduct)
    assert prod.brand is None
    assert prod.price is None
    assert prod.action == "view"
    assert prod.extraction_confidence == 0.0


def test_run_pipeline_end_to_end(tmp_path):
    input_file = tmp_path / "test_input.jsonl"
    output_parquet = tmp_path / "test_output.parquet"

    input_file.write_text(
        '{"id": 1, "app": "walmart", "raw_text": "boAt Airdopes 141 $22.90 ★★★★☆ (43,981) Add to Cart", "timestamp": "2026-01-27T11:22:54"}\n'
    )

    df = run_pipeline(str(input_file), str(output_parquet))

    assert output_parquet.exists()
    assert df.shape[0] == 1
    assert "canonical_product_id" in df.columns
    assert "price" in df.columns
    assert df["brand"][0] == "boAt"
    assert df["action"][0] == "cart"


from src.ingestion.ner_extractor import BRAND_SOURCE_LEADING_MATCH, BRAND_SOURCE_MATCH
from src.ingestion.pipeline import calculate_extraction_confidence, category_evidence
from src.ingestion.synthetic_data import CATALOG


@pytest.mark.parametrize("item", CATALOG, ids=lambda item: f"{item['brand']} {item['title']}")
def test_catalog_products_keep_brand_and_category(item):
    """
    Verifies every synthetic catalog product still resolves to its brand and a real
    category under the open-vocabulary brand path and the taxonomy abstain floor.

    The simulator derives warehouse products from exactly this call, so an abstention
    here would silently change the synthetic warehouse.
    """
    text = f"{item['brand']} {item['title']} {' '.join(item.get('specs', []))} $99.99 4.5★ Add to Cart"
    prod = process(text)
    assert prod.brand == item["brand"]
    assert prod.category_l2 != "Unknown"


def test_confidence_reflects_evidence_not_population():
    """
    Verifies two records with the same populated fields score differently when the
    evidence behind them differs.

    The previous score counted populated fields, so it averaged 0.90 on real listings
    whose brand was right 1 percent of the time.
    """
    base = {
        "brand": "Samsung",
        "product_title": "Galaxy S6 Case",
        "price": 9.99,
        "price_verified": True,
        "currency": "USD",
        "rating": 4.2,
        "category_l2": "Smartphones",
        "category_similarity": 0.40,
    }
    strong = calculate_extraction_confidence({**base, "brand_source": BRAND_SOURCE_LEADING_MATCH})
    weak = calculate_extraction_confidence({**base, "brand_source": BRAND_SOURCE_MATCH})
    assert strong > weak

    abstained = calculate_extraction_confidence(
        {**base, "brand_source": BRAND_SOURCE_LEADING_MATCH, "category_l2": "Unknown",
         "category_abstained": True}
    )
    assert abstained < strong

    unverified = calculate_extraction_confidence(
        {**base, "brand_source": BRAND_SOURCE_LEADING_MATCH, "price_verified": False}
    )
    assert unverified < strong


def test_category_evidence_is_zero_at_floor_and_capped_at_one():
    """Verifies category evidence rises from zero at the floor to one."""
    assert category_evidence(0.9, abstained=True) == 0.0
    assert category_evidence(0.0, abstained=False) == 0.0
    assert category_evidence(0.99, abstained=False) == 1.0
    assert 0.0 < category_evidence(0.30, abstained=False) < 1.0


def test_accessory_listing_scores_below_clean_listing():
    """Verifies an accessory listing with a compatibility brand mention scores lower."""
    clean = process("Sponsored 4.7 out of 5 stars Apple iPhone 15 (128 GB) ₹68,330 FREE delivery Buy Now")
    accessory = process("4.1★ (88) Case for Samsung Galaxy S6 Slim Translucent Mirror Flip $9.99 Add to Cart")
    assert accessory.brand != "Samsung"
    assert accessory.extraction_confidence < clean.extraction_confidence
    assert clean.brand_source == BRAND_SOURCE_LEADING_MATCH


def test_process_screen_one_line_matches_process() -> None:
    from datetime import datetime, timezone
    from src.ingestion.pipeline import process_screen

    raw = "Samsung Galaxy S24 Ultra 256GB Titanium $1299.99 4.7 Add to Cart"
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    products = process_screen(raw, app_source="amazon_us", timestamp=ts)
    assert len(products) == 1
    single = process(raw, app_source="amazon_us", timestamp=ts)
    assert products[0].model_dump(exclude={"is_focal"}) == single.model_dump(exclude={"is_focal"})
    assert products[0].is_focal


def test_process_screen_marks_exactly_one_focal_product() -> None:
    import random as _random
    from src.ingestion.pipeline import process_screen
    from src.ingestion.screen_generator import render_listing, render_screen
    from src.ingestion.synthetic_data import CATALOG

    rng = _random.Random(3)
    for actions, focal_index in ((["view", "search", "cart", "buy"], 2), (["view", "wishlist"], 0)):
        listings = [render_listing(CATALOG[i], "amazon_us", a, 0, 4.2, rng) for i, a in enumerate(actions)]
        products = process_screen(render_screen(listings, rng), app_source="amazon_us")
        assert len(products) == len(actions)
        assert [p.is_focal for p in products].count(True) == 1
        assert products[focal_index].is_focal
        assert len({p.timestamp for p in products}) == 1


def test_process_defaults_is_focal_false() -> None:
    assert process("Apple iPhone 15 $799 Buy Now").is_focal is False


def _catalog_category(item: dict) -> tuple:
    text = f"{item['brand']} {item['title']} {' '.join(item.get('specs', []))} $99.99 4.5★ Add to Cart"
    prod = process(text)
    return prod.category_l1, prod.category_l2


def test_catalog_categories_are_structurally_consistent():
    """
    Verifies the catalog's categories keep their structure under the supervised taxonomy,
    whose label names differ from v1.0: all smartphones share one category_l2, all shoes
    share one category_l1, and the two differ.
    """
    phones = {_catalog_category(i) for i in CATALOG if i["category"] == "Smartphones"}
    shoes = {_catalog_category(i) for i in CATALOG if i["category"] == "Footwear"}
    assert sum(1 for i in CATALOG if i["category"] == "Smartphones") == 6
    assert sum(1 for i in CATALOG if i["category"] == "Footwear") == 3
    assert len({l2 for _, l2 in phones}) == 1
    assert len({l1 for l1, _ in shoes}) == 1
    assert next(iter(phones))[1] not in {l2 for _, l2 in shoes}
    assert all("Unknown" not in pair for pair in phones | shoes)
