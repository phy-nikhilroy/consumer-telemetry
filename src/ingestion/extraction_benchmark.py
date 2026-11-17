"""
Extraction throughput benchmark: events per second for `process` and `process_batch`.

Renders listings from the catalog with the screen generator, times both paths on the same
texts, and merges the result into `reports/v1_2_extraction.json` under `results.throughput`,
leaving every other key of an existing report untouched.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import RANDOM_SEED, REPORTS_DIR
from src.ingestion.catalog import build_catalog
from src.ingestion.pipeline import process, process_batch
from src.ingestion.screen_generator import render_listing
from src.ingestion import taxonomy_classifier
from src.ingestion.synthetic_data import ACTION_SNIPPETS
from src.release.reporting import write_report

REPORT_NAME = "v1_2_extraction"
BATCH_ACTIONS = ("view", "cart", "buy")


def render_texts(n: int, seed: int = RANDOM_SEED) -> List[str]:
    """Renders `n` listings (newline-joined lines) from the catalog, deterministically."""
    rng = random.Random(seed)
    catalog = build_catalog()
    apps = sorted(ACTION_SNIPPETS)
    texts: List[str] = []
    for _ in range(n):
        item = catalog[rng.randrange(len(catalog))]
        app = apps[rng.randrange(len(apps))]
        actions = [a for a in BATCH_ACTIONS if a in ACTION_SNIPPETS[app]]
        lines = render_listing(item, app, rng.choice(actions), rng.choice([0, 10, 25]), round(rng.uniform(3, 5), 1), rng)
        texts.append("\n".join(lines))
    return texts


def _clear_query_cache() -> None:
    """Empties the taxonomy query cache so each path is timed cold."""
    taxonomy_classifier._encode_query.cache_clear()
    taxonomy_classifier._SEEN.clear()


def run_benchmark(n: int = 5000, seed: int = RANDOM_SEED, batch_size: int = 256) -> dict:
    """Times `process` and `process_batch` on `n` rendered listings; returns events per second."""
    texts = render_texts(n, seed)
    process_batch(texts[:32])  # load models outside the timed region
    _clear_query_cache()
    start = time.perf_counter()
    for text in texts:
        process(text)
    single = time.perf_counter() - start
    _clear_query_cache()
    start = time.perf_counter()
    process_batch(texts, batch_size=batch_size)
    batch = time.perf_counter() - start
    return {
        "n_events": n,
        "batch_size": batch_size,
        "process_events_per_s": round(n / single, 1),
        "process_batch_events_per_s": round(n / batch, 1),
        "speedup": round(single / batch, 2),
    }


def merge_report(throughput: dict, seed: int = RANDOM_SEED, reports_dir: Optional[Path] = None) -> Path:
    """Writes `throughput` into the report, keeping any other results already there."""
    path = Path(reports_dir or REPORTS_DIR) / f"{REPORT_NAME}.json"
    results: dict = {}
    if path.exists():
        results = dict(json.loads(path.read_text()).get("results", {}))
    results["throughput"] = throughput
    return write_report(REPORT_NAME, "1.2", results, seed=seed)


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)
    throughput = run_benchmark(args.n, args.seed, args.batch_size)
    print(json.dumps(throughput, indent=2))
    if not args.no_write:
        print(merge_report(throughput, args.seed))


if __name__ == "__main__":
    main()
