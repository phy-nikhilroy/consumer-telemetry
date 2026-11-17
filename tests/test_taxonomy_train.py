"""Tests for the supervised taxonomy trainer."""

from __future__ import annotations

import hashlib

import joblib
import numpy as np
import polars as pl
import pytest

from src.ingestion import real_splits
from src.ingestion.taxonomy_train import (
    build_label_frame,
    choose_tau,
    product_type_label,
    sample_training_rows,
    select_labels,
    train_taxonomy,
)


class HashEncoder:
    """Deterministic stand-in encoder: hashed bag of words, L2-normalised."""

    def encode(self, texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False):
        out = np.zeros((len(texts), 32), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in str(text).lower().split():
                out[i, int(hashlib.md5(tok.encode()).hexdigest(), 16) % 32] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


@pytest.mark.parametrize("path, expected", [
    (["Clothing, Shoes & Jewelry", "Men", "Shoes", "Athletic"], ("Clothing, Shoes & Jewelry", "Shoes")),
    (["Clothing, Shoes & Jewelry", "Women", "Girls", "Clothing"], ("Clothing, Shoes & Jewelry", "Clothing")),
    (["Electronics", "Camera & Photo", "Lenses"], ("Electronics", "Camera & Photo")),
    (["Cell Phones & Accessories", "Wireless Trade-In Buy Box Widget"], None),
    (["Clothing, Shoes & Jewelry", "Women"], None),
    (["Electronics"], None),
    ([], None),
    (None, None),
])
def test_product_type_label(path, expected) -> None:
    assert product_type_label(path) == expected


def test_label_frame_on_hand_written_rows() -> None:
    df = pl.DataFrame({
        "title": ["Running shoe", "Phone case", "", None, "Lens", "Dress"],
        "categories": [
            ["Clothing, Shoes & Jewelry", "Men", "Shoes"],
            ["Cell Phones & Accessories", "Cases, Holsters & Sleeves"],
            ["Electronics", "Camera & Photo"],
            ["Electronics", "Camera & Photo"],
            ["Electronics"],
            ["Clothing, Shoes & Jewelry", "Women"],
        ],
    })
    out = build_label_frame(df)
    assert out["title"].to_list() == ["Running shoe", "Phone case"]
    assert out["label"].to_list() == [
        "Clothing, Shoes & Jewelry > Shoes",
        "Cell Phones & Accessories > Cases, Holsters & Sleeves",
    ]
    assert out.columns == ["title", "category_l1", "category_l2", "label"]


def test_select_labels_applies_minimum() -> None:
    df = pl.DataFrame({"label": ["a"] * 5 + ["b"] * 3 + ["c"]})
    assert select_labels(df, min_rows=3) == ["a", "b"]


def test_sample_training_rows_caps_per_label_and_total() -> None:
    df = pl.DataFrame({"title": [f"t{i}" for i in range(100)], "label": ["a"] * 70 + ["b"] * 25 + ["c"] * 5})
    out = sample_training_rows(df, ["a", "b"], per_label=30, total=50, seed=1)
    assert out.height == 50
    assert set(out["label"]) <= {"a", "b"}
    assert out["label"].value_counts().filter(pl.col("label") == "a")["count"][0] <= 30
    assert out.equals(sample_training_rows(df, ["a", "b"], per_label=30, total=50, seed=1))


def test_choose_tau_smallest_qualifying() -> None:
    max_prob = np.array([0.95, 0.9, 0.8, 0.7, 0.6, 0.4, 0.3])
    correct = np.array([1, 1, 1, 1, 0, 0, 0], dtype=bool)
    assert choose_tau(max_prob, correct, grid=[0.1, 0.5, 0.65, 0.9], target=0.85) == 0.65
    assert choose_tau(max_prob, correct, grid=[0.1, 0.5], target=0.6) == 0.5


def test_choose_tau_falls_back_when_none_qualifies() -> None:
    assert choose_tau(np.array([0.9, 0.8]), np.array([False, False]), grid=[0.1, 0.5], target=0.9, fallback=0.95) == 0.95


def test_choose_tau_ignores_grid_values_with_no_confident_rows() -> None:
    assert choose_tau(np.array([0.3, 0.2]), np.array([True, False]), grid=[0.25, 0.9], target=0.9, fallback=0.77) == 0.25


def _write_shard(path, n_per_label: int, offset: int) -> None:
    titles, cats, asins = [], [], []
    for i in range(n_per_label):
        titles.append(f"running shoe sneaker size {i}")
        cats.append(["Clothing, Shoes & Jewelry", "Men", "Shoes"])
        titles.append(f"camera lens zoom photo {i}")
        cats.append(["Electronics", "Camera & Photo"])
        asins += [f"S{offset}{i}", f"C{offset}{i}"]
    pl.DataFrame({"parent_asin": asins, "title": titles, "categories": cats}).write_parquet(path)


def test_artifact_round_trips(tmp_path, monkeypatch) -> None:
    shard = tmp_path / "shard.parquet"
    _write_shard(shard, 900, 0)
    monkeypatch.setattr(real_splits, "AMAZON_SHARDS", {"tiny": shard})
    out = tmp_path / "taxonomy_lr.joblib"
    summary = train_taxonomy(["tiny"], out, seed=3, encoder=HashEncoder(), tune_rows=500)
    artifact = joblib.load(out)
    assert set(artifact) == {"model", "labels", "tau", "encoder_name", "trained_on_rows"}
    assert artifact["labels"] == ["Clothing, Shoes & Jewelry > Shoes", "Electronics > Camera & Photo"]
    assert 0.0 < artifact["tau"] <= 0.95
    assert artifact["trained_on_rows"] == summary["trained_on_rows"] > 0
    probs = artifact["model"].predict_proba(HashEncoder().encode(["camera lens zoom photo 5"]))
    assert artifact["labels"][int(probs.argmax())] == "Electronics > Camera & Photo"
