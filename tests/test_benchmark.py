"""Tests for the API latency benchmark."""

from __future__ import annotations

import pytest

from src.api.benchmark import TARGETS_MS, latency_summary, run_benchmark


def test_latency_summary():
    summary = latency_summary([float(v) for v in range(1, 101)])
    assert summary["requests"] == 100
    assert summary["min_ms"] == 1.0 and summary["max_ms"] == 100.0
    assert summary["p50_ms"] == pytest.approx(50.5)
    assert summary["p95_ms"] == pytest.approx(95.05)
    assert summary["p99_ms"] == pytest.approx(99.01)
    with pytest.raises(ValueError):
        latency_summary([])


def test_run_benchmark_covers_every_endpoint(api_client):
    results = run_benchmark(api_client, requests=3, seed=1)
    for name in ("health", "ingest", "propensity", "profile", "segments"):
        assert results[name]["requests"] == 3
        assert results[name]["p50_ms"] <= results[name]["p99_ms"]
    assert "cold_ms" in results["segments"]
    assert set(results["targets"]) == set(TARGETS_MS)
