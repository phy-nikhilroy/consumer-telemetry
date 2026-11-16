"""
API latency benchmark: p50, p95, and p99 per endpoint over repeated requests.

Two modes:

  In process (default). Builds the app against a throwaway copy of the warehouse and
  calls it through FastAPI's TestClient. This measures the service's own work
  (extraction, SQL, scoring) without network overhead, and never touches the real
  serving warehouse.

      .venv/bin/python src/api/benchmark.py --requests 100

  Against a running server. Measures what a client sees, network included.

      .venv/bin/python src/api/benchmark.py --url http://localhost:8000 --requests 100

Write the results to a file with `--output path.json`; nothing is written by default, so
a benchmark run cannot overwrite a committed artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional

# Ensure consumer-telemetry root is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.simulation.stream_generator import build_event

# Latency targets from the project specification, in milliseconds.
TARGETS_MS = {"ingest": 500.0, "propensity": 200.0}


def latency_summary(samples_ms: List[float]) -> Dict[str, float]:
    """Returns count, mean, min, max, and the 50th, 95th, and 99th percentiles."""
    if not samples_ms:
        raise ValueError("No latency samples to summarize.")
    values = np.asarray(samples_ms, dtype=float)
    return {
        "requests": int(values.size),
        "mean_ms": round(float(values.mean()), 2),
        "min_ms": round(float(values.min()), 2),
        "p50_ms": round(float(np.percentile(values, 50)), 2),
        "p95_ms": round(float(np.percentile(values, 95)), 2),
        "p99_ms": round(float(np.percentile(values, 99)), 2),
        "max_ms": round(float(values.max()), 2),
    }


def _timed(call: Callable[[], Any]) -> float:
    start = time.perf_counter()
    response = call()
    elapsed = (time.perf_counter() - start) * 1000.0
    if response.status_code >= 400:
        raise RuntimeError(f"Benchmark request failed with {response.status_code}: {response.text[:200]}")
    return elapsed


def run_benchmark(client: Any, requests: int = 100, seed: int = 42) -> Dict[str, Any]:
    """
    Times every main endpoint `requests` times through `client`, which may be an
    `httpx.Client` or a FastAPI `TestClient`.

    Ingest runs first, so the prediction endpoints are measured against a warehouse
    that already contains the new events. The segments endpoint caches its result
    until the data changes, so its first (cold) call is reported separately.
    """
    rng = random.Random(seed)
    users = [row["user_id"] for row in client.get("/users", params={"limit": 1000}).json()]
    if not users:
        raise RuntimeError("The warehouse has no users to benchmark against.")
    pool = list(users)

    results: Dict[str, Any] = {}
    results["health"] = latency_summary([_timed(lambda: client.get("/health")) for _ in range(requests)])
    results["ingest"] = latency_summary(
        [_timed(lambda: client.post("/ingest", json=build_event(rng, pool))) for _ in range(requests)]
    )
    results["propensity"] = latency_summary(
        [_timed(lambda: client.get(f"/user/{rng.choice(users)}/propensity")) for _ in range(requests)]
    )
    results["profile"] = latency_summary(
        [_timed(lambda: client.get(f"/user/{rng.choice(users)}/profile")) for _ in range(requests)]
    )
    cold = _timed(lambda: client.get("/analytics/segments"))
    results["segments"] = latency_summary([_timed(lambda: client.get("/analytics/segments")) for _ in range(requests)])
    results["segments"]["cold_ms"] = round(cold, 2)

    results["targets"] = {
        name: {"target_p95_ms": target, "p95_ms": results[name]["p95_ms"], "met": results[name]["p95_ms"] <= target}
        for name, target in TARGETS_MS.items()
    }
    return results


def print_results(results: Dict[str, Any], mode: str) -> None:
    """Prints a latency table."""
    print(f"\nLatency over {results['health']['requests']} requests per endpoint ({mode})")
    print(f"{'Endpoint':<12} {'p50 ms':>9} {'p95 ms':>9} {'p99 ms':>9} {'max ms':>9}")
    for name in ("health", "ingest", "propensity", "profile", "segments"):
        row = results[name]
        print(f"{name:<12} {row['p50_ms']:>9.1f} {row['p95_ms']:>9.1f} {row['p99_ms']:>9.1f} {row['max_ms']:>9.1f}")
    print(f"segments cold call: {results['segments']['cold_ms']:.1f} ms (later calls are cached until data changes)")
    for name, target in results["targets"].items():
        verdict = "met" if target["met"] else "MISSED"
        print(f"target {name}: p95 {target['p95_ms']:.1f} ms against {target['target_p95_ms']:.0f} ms, {verdict}")


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark API latency per endpoint.")
    parser.add_argument("--url", type=str, default=None, help="Benchmark a running server instead of in process")
    parser.add_argument("--requests", type=int, default=100, help="Requests per endpoint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None, help="Optional JSON file for the results")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    """Runs the benchmark in the requested mode and prints, and optionally saves, the results."""
    args = _parse_args(argv)
    if args.url:
        import httpx

        with httpx.Client(base_url=args.url, timeout=60.0) as client:
            results = run_benchmark(client, requests=args.requests, seed=args.seed)
        mode = f"server at {args.url}"
    else:
        from fastapi.testclient import TestClient

        from src.api.main import create_app

        with tempfile.TemporaryDirectory() as scratch:
            app = create_app(db_path=Path(scratch) / "benchmark.duckdb", clock_mode="simulated", bootstrap=True)
            with TestClient(app) as client:
                results = run_benchmark(client, requests=args.requests, seed=args.seed)
        mode = "in process, no network"

    results["mode"] = mode
    print_results(results, mode)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Saved results to {out}")
    return results


if __name__ == "__main__":
    main()
