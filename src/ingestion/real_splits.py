"""Deterministic hash splits (train/tune/heldout) of the real Amazon metadata shards.

The split of a row depends only on its ``parent_asin``: ``md5 % 10`` with buckets 0-5 train,
6-7 tune and 8-9 heldout. Heldout rows of the benchmark shards are read only by
``real_data_benchmark.py``. The tuned shard (``config.AMAZON_TUNED_SHARD``) is never used as
heldout: its buckets 8-9 are folded into tune, so its tune split is buckets 6-9.
"""
from __future__ import annotations

import argparse
import hashlib
from typing import Optional, Sequence

import polars as pl

from src import config
from src.config import AMAZON_SHARDS, AMAZON_TUNED_SHARD, RANDOM_SEED

SPLITS = ("train", "tune", "heldout")


def _bucket(parent_asin: str) -> int:
    return int(hashlib.md5(parent_asin.encode("utf-8")).hexdigest(), 16) % 10


def split_of(parent_asin: str) -> str:
    """Return "train", "tune" or "heldout" for a parent_asin (md5 bucket 0-5, 6-7, 8-9)."""
    b = _bucket(parent_asin)
    return "train" if b <= 5 else ("tune" if b <= 7 else "heldout")


def load_split(split: str, shards: Sequence[str], max_rows: Optional[int] = None,
               seed: int = RANDOM_SEED) -> pl.DataFrame:
    """Load the rows of ``split`` from the named shards, dropping null parent_asin or title.

    On the tuned shard, buckets 8-9 count as tune and a heldout request raises ValueError.
    ``max_rows`` draws a seeded random sample when the split has more rows.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    registry = AMAZON_SHARDS  # module attribute, read at call time (tests monkeypatch it)
    frames = []
    for key in shards:
        if key not in registry:
            raise ValueError(f"unknown shard {key!r}")
        tuned = key == AMAZON_TUNED_SHARD
        if tuned and split == "heldout":
            raise ValueError(f"{key} is the tuned shard and is never used as heldout")
        df = pl.read_parquet(registry[key]).drop_nulls(["parent_asin", "title"])
        names = [split_of(a) for a in df["parent_asin"].to_list()]
        if tuned:
            names = ["tune" if n == "heldout" else n for n in names]
        frames.append(df.filter(pl.Series(names) == split) if df.height else df)
    out = pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()
    if max_rows is not None and out.height > max_rows:
        out = out.sample(n=max_rows, seed=seed)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show split sizes of the Amazon shards.")
    parser.add_argument("--shards", nargs="+", default=list(config.AMAZON_SHARDS))
    args = parser.parse_args()
    for shard in args.shards:
        for sp in ("train", "tune") if shard == AMAZON_TUNED_SHARD else SPLITS:
            print(shard, sp, load_split(sp, [shard]).height)
