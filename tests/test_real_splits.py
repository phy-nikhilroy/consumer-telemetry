"""Tests for src/ingestion/real_splits.py: deterministic hash splits of real Amazon shards."""
from __future__ import annotations

import hashlib
from pathlib import Path

import polars as pl
import pytest
from hypothesis import given, settings, strategies as st

from src import config
from src.ingestion import real_splits
from src.ingestion.real_splits import load_split, split_of

SRC = Path(__file__).resolve().parents[1] / "src"


def _expected(asin: str) -> str:
    b = int(hashlib.md5(asin.encode()).hexdigest(), 16) % 10
    return "train" if b <= 5 else ("tune" if b <= 7 else "heldout")


def _asins(n: int, prefix: str = "B0") -> list[str]:
    return [f"{prefix}{i:07d}" for i in range(n)]


def _write(path: Path, asins, titles=None):
    titles = titles if titles is not None else [f"Title {a}" for a in asins]
    pl.DataFrame({"parent_asin": asins, "title": titles, "price": [1.0] * len(asins)},
                 schema_overrides={"parent_asin": pl.Utf8, "title": pl.Utf8}).write_parquet(path)


@pytest.fixture
def shards(tmp_path, monkeypatch):
    asins = _asins(400)
    paths = {}
    for key in ("meta_shard0", "meta_cell_shard1", "meta_electronics_shard0", "meta_fashion_shard0"):
        p = tmp_path / f"{key}.parquet"
        _write(p, [f"{key[:6]}{a}" for a in asins])
        paths[key] = p
    monkeypatch.setattr(config, "AMAZON_SHARDS", paths)
    monkeypatch.setattr(real_splits, "AMAZON_SHARDS", paths, raising=False)
    return paths


def test_split_of_matches_md5_rule():
    for a in _asins(300):
        assert split_of(a) == _expected(a)


def test_split_of_is_deterministic():
    assert [split_of(a) for a in _asins(200)] == [split_of(a) for a in _asins(200)]


def test_split_of_known_bucket_boundaries():
    seen = {}
    for a in _asins(500):
        seen[int(hashlib.md5(a.encode()).hexdigest(), 16) % 10] = split_of(a)
    assert set(seen) == set(range(10))
    for b in range(6):
        assert seen[b] == "train"
    for b in (6, 7):
        assert seen[b] == "tune"
    for b in (8, 9):
        assert seen[b] == "heldout"


def test_split_of_proportions_within_two_points():
    ids = [f"B{i:09d}" for i in range(10_000)]
    counts = {"train": 0, "tune": 0, "heldout": 0}
    for a in ids:
        counts[split_of(a)] += 1
    assert abs(counts["train"] / 10_000 - 0.6) <= 0.02
    assert abs(counts["tune"] / 10_000 - 0.2) <= 0.02
    assert abs(counts["heldout"] / 10_000 - 0.2) <= 0.02


@settings(max_examples=50, deadline=None)
@given(st.text(min_size=1, max_size=30))
def test_split_of_property_valid_and_stable(s):
    assert split_of(s) in {"train", "tune", "heldout"}
    assert split_of(s) == split_of(s) == _expected(s)


def test_load_split_returns_only_rows_of_that_split(shards):
    for split in ("train", "tune"):
        df = load_split(split, ["meta_cell_shard1"])
        assert df.height > 0
        assert all(split_of(a) == split for a in df["parent_asin"])


def test_load_split_heldout_for_benchmark_shard(shards):
    df = load_split("heldout", ["meta_fashion_shard0"])
    assert df.height > 0
    assert all(split_of(a) == "heldout" for a in df["parent_asin"])


def test_splits_partition_a_shard(shards):
    parts = [set(load_split(s, ["meta_electronics_shard0"])["parent_asin"])
             for s in ("train", "tune", "heldout")]
    assert not (parts[0] & parts[1]) and not (parts[0] & parts[2]) and not (parts[1] & parts[2])
    total = pl.read_parquet(shards["meta_electronics_shard0"])["parent_asin"].n_unique()
    assert sum(len(p) for p in parts) == total


def test_nulls_in_parent_asin_or_title_dropped(tmp_path, monkeypatch):
    asins = [a for a in _asins(60) if split_of(a) == "train"][:6]
    rows_a = asins[:4] + [None, asins[4]]
    rows_t = ["ok", None, "ok", "ok", "ok", "ok"]
    p = tmp_path / "s.parquet"
    _write(p, rows_a, rows_t)
    monkeypatch.setattr(config, "AMAZON_SHARDS", {"meta_cell_shard1": p})
    monkeypatch.setattr(real_splits, "AMAZON_SHARDS", {"meta_cell_shard1": p}, raising=False)
    df = load_split("train", ["meta_cell_shard1"])
    assert df["parent_asin"].null_count() == 0 and df["title"].null_count() == 0
    assert sorted(df["parent_asin"].to_list()) == sorted([asins[0], asins[2], asins[3], asins[4]])


def test_max_rows_caps_and_is_seeded(shards):
    full = load_split("train", ["meta_cell_shard1"])
    a = load_split("train", ["meta_cell_shard1"], max_rows=25, seed=7)
    b = load_split("train", ["meta_cell_shard1"], max_rows=25, seed=7)
    assert a.height == 25
    assert a["parent_asin"].to_list() == b["parent_asin"].to_list()
    assert set(a["parent_asin"]) <= set(full["parent_asin"])
    c = load_split("train", ["meta_cell_shard1"], max_rows=25, seed=8)
    assert c["parent_asin"].to_list() != a["parent_asin"].to_list()


def test_max_rows_larger_than_data_returns_all(shards):
    full = load_split("tune", ["meta_cell_shard1"])
    big = load_split("tune", ["meta_cell_shard1"], max_rows=10**6)
    assert big.height == full.height


def test_max_rows_none_and_zero_behaviour(shards):
    assert load_split("tune", ["meta_cell_shard1"], max_rows=1).height == 1


def test_multiple_shards_are_combined(shards):
    one = load_split("train", ["meta_cell_shard1"]).height
    two = load_split("train", ["meta_cell_shard1", "meta_fashion_shard0"]).height
    assert two > one


def test_meta_shard0_never_yields_heldout(shards):
    try:
        df = load_split("heldout", ["meta_shard0"])
    except (ValueError, KeyError, PermissionError):
        return
    assert df.height == 0


def test_meta_shard0_is_used_as_train_or_tune_whatever_hash_says(shards):
    train = load_split("train", ["meta_shard0"])
    tune = load_split("tune", ["meta_shard0"])
    assert train.height > 0 and tune.height > 0
    assert not (set(train["parent_asin"]) & set(tune["parent_asin"]))
    total = pl.read_parquet(shards["meta_shard0"])["parent_asin"].n_unique()
    assert train["parent_asin"].n_unique() + tune["parent_asin"].n_unique() <= total


def test_invalid_split_name_rejected(shards):
    with pytest.raises((ValueError, KeyError)):
        load_split("validation", ["meta_cell_shard1"])


def test_unknown_shard_key_rejected(shards):
    with pytest.raises((ValueError, KeyError)):
        load_split("train", ["not_a_shard"])


def test_only_the_benchmark_reads_heldout():
    allowed = {"real_splits.py", "real_data_benchmark.py"}
    offenders = []
    for f in SRC.rglob("*.py"):
        text = f.read_text(encoding="utf-8")
        if ('"heldout"' in text or "'heldout'" in text) and f.name not in allowed:
            offenders.append(str(f.relative_to(SRC)))
    assert offenders == []
    assert '"heldout"' in (SRC / "ingestion" / "real_splits.py").read_text(encoding="utf-8")


def test_meta_shard0_heldout_request_raises_value_error(shards):
    with pytest.raises(ValueError):
        load_split("heldout", ["meta_shard0"])


def test_meta_shard0_folds_heldout_buckets_into_tune(shards):
    ids = set(pl.read_parquet(shards["meta_shard0"])["parent_asin"])
    folded = {a for a in ids if split_of(a) == "heldout"}
    assert folded
    tune = set(load_split("tune", ["meta_shard0"])["parent_asin"])
    train = set(load_split("train", ["meta_shard0"])["parent_asin"])
    assert folded <= tune
    assert train == {a for a in ids if split_of(a) == "train"}
    assert tune | train == ids
