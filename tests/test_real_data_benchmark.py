"""Tests for the held-out benchmark: gap arithmetic, report merge, and synthetic screens."""

from __future__ import annotations

import json
import random

import polars as pl

from src.ingestion import real_data_benchmark as rdb


def test_split_gap_is_tune_minus_heldout() -> None:
    tune = {k: 0.9 for k in rdb.SPLIT_METRICS}
    held = {k: 0.6 for k in rdb.SPLIT_METRICS}
    gap = rdb.split_gap(tune, held)
    assert abs(gap["brand_accuracy"] - 0.3) < 1e-9
    assert gap["final_category_accuracy"] is None


def test_merge_results_keeps_throughput_and_envelope(tmp_path) -> None:
    out = tmp_path / "v1_2_extraction.json"
    out.write_text(json.dumps({"results": {"throughput": {"x": 1}}}))
    rdb.merge_results({"real": {"a": 1}}, output=out)
    saved = json.loads(out.read_text())
    assert saved["results"] == {"throughput": {"x": 1}, "real": {"a": 1}}
    assert saved["release"] == "1.2" and saved["schema_version"] == 1


def test_synthetic_screen_truth_matches_listing_count() -> None:
    from src.ingestion.catalog import build_catalog

    catalog = build_catalog()
    rng = random.Random(3)
    text, truth = rdb._synthetic_screen(catalog, rng)
    assert 1 <= len(truth) <= 3 and text


def test_run_synthetic_clean_screens_mostly_segment() -> None:
    res = rdb.run_synthetic(n_screens=20, rates=(0.0,))
    block = res["by_noise"]["0.00"]
    assert block["segment_count_accuracy"] >= 0.8
    assert 0.0 <= block["price_accuracy"] <= 1.0


def test_score_rows_empty_frame_has_no_division_error() -> None:
    df = pl.DataFrame({"title": [], "true_brand": [], "true_price": [], "true_rating": [],
                       "true_num_ratings": [], "true_label": []},
                      schema={"title": pl.String, "true_brand": pl.String, "true_price": pl.Float64,
                              "true_rating": pl.Float64, "true_num_ratings": pl.Int64, "true_label": pl.String})
    res = rdb.score_rows(df)
    assert res["n"] == 0 and res["abstain_rate"] is None
