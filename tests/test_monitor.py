"""Tests for the terminal activity monitor."""

from __future__ import annotations

import httpx

from src.simulation.monitor import format_activity_line, run_monitor

ACTIVITY = {
    "events_in_window": 847,
    "active_users_in_window": 23,
    "live_events": 5,
    "last_purchase": {"user_id": "user_42", "product_title": "iPhone 15", "app_name": "amazon_in"},
}


def test_format_activity_line():
    line = format_activity_line(ACTIVITY, stamp="12:34:56")
    assert line == (
        '[12:34:56] Events today: 847 | Users active: 23 | Live events: 5 | '
        'Last purchase: user_42 bought "iPhone 15" on amazon_in'
    )
    assert "none yet" in format_activity_line({**ACTIVITY, "last_purchase": None}, stamp="x")


def test_run_monitor_polls_and_reports_errors():
    responses = iter([httpx.Response(200, json=ACTIVITY), httpx.Response(503, text="down")])
    client = httpx.Client(base_url="http://test", transport=httpx.MockTransport(lambda request: next(responses)))
    sleeps: list = []
    lines = run_monitor(iterations=2, interval=3.0, client=client, sleep=sleeps.append, log=lambda s: None)
    assert "Events today: 847" in lines[0]
    assert "API unavailable" in lines[1]
    assert sleeps == [3.0]
