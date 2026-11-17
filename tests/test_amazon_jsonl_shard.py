"""Tests for the Amazon metadata JSONL to parquet shard converter."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from src.ingestion.amazon_jsonl_shard import CONVERTERS, convert

STR_LIST = pl.List(pl.String)
REFERENCE_SCHEMA = pl.Schema({
    "main_category": pl.String, "title": pl.String, "average_rating": pl.Float64,
    "rating_number": pl.Int64, "features": STR_LIST, "description": STR_LIST, "price": pl.String,
    "images": pl.Struct({"hi_res": STR_LIST, "large": STR_LIST, "thumb": STR_LIST, "variant": STR_LIST}),
    "videos": pl.Struct({"title": STR_LIST, "url": STR_LIST, "user_id": STR_LIST}),
    "store": pl.String, "categories": STR_LIST, "details": pl.String, "parent_asin": pl.String,
    "bought_together": pl.String, "subtitle": pl.String, "author": pl.String,
})


@pytest.fixture
def reference(tmp_path: Path) -> Path:
    path = tmp_path / "reference.parquet"
    pl.DataFrame(schema=REFERENCE_SCHEMA).write_parquet(path)
    return path


def _write_jsonl(path: Path, records: list, extra_lines: tuple = ()) -> Path:
    lines = [json.dumps(r) for r in records] + list(extra_lines)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_output_schema_equals_reference(tmp_path: Path, reference: Path) -> None:
    src = _write_jsonl(tmp_path / "in.jsonl", [{
        "main_category": "AMAZON FASHION", "title": "Levi's 501 Jeans", "average_rating": 4.5,
        "rating_number": "120", "features": ["cotton"], "description": [], "price": 59.5,
        "images": [{"hi_res": "a.jpg", "large": "b.jpg", "thumb": None, "variant": "MAIN"}],
        "videos": [], "store": "Levi's", "categories": [], "details": {"Material": "Cotton"},
        "parent_asin": "B000TEST01", "bought_together": None,
    }])
    out = tmp_path / "out.parquet"
    assert convert(src, reference, out) == 1
    frame = pl.read_parquet(out)
    assert frame.schema == REFERENCE_SCHEMA
    row = frame.row(0, named=True)
    assert row["price"] == "59.5"
    assert row["rating_number"] == 120
    assert row["images"]["large"] == ["b.jpg"]
    assert json.loads(row["details"]) == {"Material": "Cotton"}
    assert row["subtitle"] is None


@pytest.mark.parametrize("price", ["None", "", "—", None, "from $5"])
def test_non_numeric_price_becomes_null(tmp_path: Path, reference: Path, price) -> None:
    src = _write_jsonl(tmp_path / "in.jsonl", [{"title": "Shirt", "parent_asin": "B1", "price": price}])
    out = tmp_path / "out.parquet"
    convert(src, reference, out)
    assert pl.read_parquet(out)["price"][0] is None


def test_unparseable_and_blank_lines_are_skipped(tmp_path: Path, reference: Path) -> None:
    src = _write_jsonl(tmp_path / "in.jsonl", [{"title": "A", "parent_asin": "B1"}], ("{broken", "", "  "))
    out = tmp_path / "out.parquet"
    assert convert(src, reference, out) == 1


def test_unknown_reference_column_raises(tmp_path: Path) -> None:
    ref = tmp_path / "ref.parquet"
    pl.DataFrame({"not_a_meta_column": ["x"]}).write_parquet(ref)
    src = _write_jsonl(tmp_path / "in.jsonl", [{"title": "A"}])
    with pytest.raises(ValueError):
        convert(src, ref, tmp_path / "out.parquet")


def test_every_reference_column_has_a_converter() -> None:
    assert set(REFERENCE_SCHEMA.names()) <= set(CONVERTERS)


def test_reference_schema_matches_the_real_shards_when_present() -> None:
    from src.config import AMAZON_SHARDS

    for name in ("meta_cell_shard1", "meta_fashion_shard0"):
        path = AMAZON_SHARDS[name]
        if not path.exists():
            pytest.skip(f"{name} not downloaded")
        assert pl.scan_parquet(path).collect_schema() == REFERENCE_SCHEMA
