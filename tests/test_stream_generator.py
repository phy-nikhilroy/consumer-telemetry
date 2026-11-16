"""Tests for the live stream generator, using a mocked API transport."""

from __future__ import annotations

import json
import random

import httpx
import pytest

from src.simulation.stream_generator import APPS, build_event, describe_response, run_stream


def _mock_api(fail_every: int = 0):
    calls = {"ingest": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/users":
            return httpx.Response(200, json=[{"user_id": "user_001"}, {"user_id": "user_002"}])
        if request.url.path == "/ingest":
            calls["ingest"] += 1
            if fail_every and calls["ingest"] % fail_every == 0:
                return httpx.Response(500, text="boom")
            body = json.loads(request.content)
            return httpx.Response(200, json={
                "stored_action": "view", "session_id": f"{body['user_id']}_1", "is_new_user": False,
                "cross_app_comparisons_detected": 0,
                "product": {"canonical_name": "Apple iPhone 15", "price": 68330.0, "currency": "INR"},
            })
        return httpx.Response(404)

    return httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler)), calls


def test_build_event_is_reproducible_and_valid():
    first = [build_event(random.Random(3), ["user_001"]) for _ in range(1)]
    second = [build_event(random.Random(3), ["user_001"]) for _ in range(1)]
    assert first == second
    event = build_event(random.Random(5), ["user_001", "user_002"], new_user_rate=0.0)
    assert event["user_id"] in {"user_001", "user_002"}
    assert event["app_name"] in APPS
    assert event["raw_text"]


def test_build_event_creates_new_users():
    pool: list = []
    event = build_event(random.Random(1), pool)
    assert event["user_id"].startswith("live_user_")
    assert pool == [event["user_id"]]
    always_new = build_event(random.Random(1), ["user_001"], new_user_rate=1.0)
    assert always_new["user_id"].startswith("live_user_")


def test_run_stream_sends_requested_count():
    client, calls = _mock_api()
    sleeps: list = []
    lines: list = []
    tallies = run_stream(count=5, seed=1, client=client, sleep=sleeps.append, log=lines.append,
                         min_interval=1.0, max_interval=2.0)
    assert tallies == {"sent": 5, "ok": 5, "failed": 0}
    assert calls["ingest"] == 5
    assert len(sleeps) == 4 and all(1.0 <= s <= 2.0 for s in sleeps)
    assert any("Apple iPhone 15" in line for line in lines)


def test_run_stream_survives_failures():
    client, _ = _mock_api(fail_every=2)
    tallies = run_stream(count=4, seed=1, client=client, sleep=lambda s: None, log=lambda s: None)
    assert tallies == {"sent": 4, "ok": 2, "failed": 2}


def test_run_stream_rejects_bad_intervals():
    with pytest.raises(ValueError):
        run_stream(count=1, min_interval=5, max_interval=1)


def test_describe_response_flags():
    line = describe_response(
        {"user_id": "u1", "app_name": "flipkart"},
        {"stored_action": "buy", "session_id": "u1_1", "is_new_user": True, "cross_app_comparisons_detected": 2,
         "product": {"canonical_name": "X", "price": 10.0, "currency": "USD"}},
    )
    assert "new user" in line and "2 cross-app comparison(s)" in line and "USD 10.00" in line
