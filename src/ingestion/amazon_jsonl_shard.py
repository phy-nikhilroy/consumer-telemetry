"""
Converts an Amazon-Reviews-2023 metadata JSONL file into a parquet shard.

The dataset publishes parquet folders for only some categories; Clothing, Shoes and Jewelry
exists only as JSONL. This module builds `meta_fashion_shard0.parquet` from the first lines
of that file (see the README reproduce section) with the column names, order, and dtypes of
a reference parquet shard, so every shard reads the same way. Non-numeric prices become
null, and nested image and video fields are normalised to the reference struct types.

Run: .venv/bin/python -m src.ingestion.amazon_jsonl_shard --input <jsonl> \
         --reference-shard <parquet> --output <parquet>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import polars as pl

IMG_KEYS = ("hi_res", "large", "thumb", "variant")
VID_KEYS = ("title", "url", "user_id")


def _to_str(value: Any) -> str | None:
    """Return a string, JSON-encoding non-string values."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _flat_strings(value: Any) -> list[str]:
    """Flatten a str, list, or None into a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    out: list[str] = []
    for item in value:
        if item is None:
            continue
        if isinstance(item, list):
            out.extend(str(x) for x in item if x is not None)
        else:
            out.append(str(item))
    return out


def _to_str_list(value: Any) -> list[str] | None:
    """Return a list of strings, or None when the field is missing."""
    if value is None:
        return None
    return _flat_strings(value)


def _to_float(value: Any) -> float | None:
    """Return a float, or None when it cannot be parsed."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    """Return an int, or None when it cannot be parsed."""
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _to_price(value: Any) -> str | None:
    """Keep numeric prices as text and map anything non-numeric to null."""
    if value is None:
        return None
    try:
        float(value)
    except (TypeError, ValueError):
        return None
    return str(value)


def _to_images(value: Any) -> dict[str, list[str]] | None:
    """Normalise the images field to a struct of four string lists."""
    if value is None:
        return None
    if isinstance(value, dict):
        return {k: _flat_strings(value.get(k)) for k in IMG_KEYS}
    if isinstance(value, list):
        items = [x for x in value if isinstance(x, dict)]
        return {k: [s for x in items for s in _flat_strings(x.get(k))] for k in IMG_KEYS}
    return None


def _to_videos(value: Any) -> dict[str, list[str]] | None:
    """Normalise the videos field to a struct of three string lists."""
    if value is None:
        return None
    if isinstance(value, dict):
        return {k: _flat_strings(value.get(k)) for k in VID_KEYS}
    if isinstance(value, list):
        items = [x for x in value if isinstance(x, dict)]
        return {k: [s for x in items for s in _flat_strings(x.get(k))] for k in VID_KEYS}
    return None


CONVERTERS: dict[str, Callable[[Any], Any]] = {
    "main_category": _to_str,
    "title": _to_str,
    "average_rating": _to_float,
    "rating_number": _to_int,
    "features": _to_str_list,
    "description": _to_str_list,
    "price": _to_price,
    "images": _to_images,
    "videos": _to_videos,
    "store": _to_str,
    "categories": _to_str_list,
    "details": _to_str,
    "parent_asin": _to_str,
    "bought_together": _to_str,
    "subtitle": _to_str,
    "author": _to_str,
}


def convert(jsonl_path: Path, reference_path: Path, out_path: Path) -> int:
    """Writes the converted shard with the reference schema and returns its row count."""
    target = pl.scan_parquet(reference_path).collect_schema()
    missing = [name for name in target.names() if name not in CONVERTERS]
    if missing:
        raise ValueError(f"no converter for reference columns: {missing}")

    columns: dict[str, list[Any]] = {name: [] for name in target.names()}
    skipped = 0
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            for name in target.names():
                columns[name].append(CONVERTERS[name](record.get(name)))

    frame = pl.DataFrame(columns, schema=target)
    frame.write_parquet(out_path)
    print(f"skipped_unparseable_lines={skipped}")
    return frame.height


def main() -> None:
    """Parse arguments and run the conversion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference-shard", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = convert(args.input, args.reference_shard, args.output)
    print(f"rows_written={rows}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
