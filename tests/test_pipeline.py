from datetime import datetime, timezone
import polars as pl
import pytest
from src.ingestion.pipeline import process_record, run_pipeline, ExtractedProduct


def test_process_record_structure():
    raw_text = "Sponsored 4.7 out of 5 stars Apple iPhone 15 (128 GB) ₹68,330 M.R.P: ₹75,920 (10% off) FREE delivery Buy Now"
    prod = process_record(raw_text, app_source="amazon_in")

    assert isinstance(prod, ExtractedProduct)
    assert prod.brand == "Apple"
    assert "iPhone 15" in prod.product_title
    assert prod.canonical_product_id is not None
    assert prod.price == 68330.0
    assert prod.currency == "INR"
    assert prod.action == "buy"
    assert prod.category_l1 == "Electronics"
    assert prod.category_l2 == "Smartphones"
    assert prod.extraction_confidence >= 0.80


def test_process_record_empty():
    prod = process_record("")
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
