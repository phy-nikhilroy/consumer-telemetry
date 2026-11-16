"""
End-to-end integration test: raw screen text in, live predictions out.

Follows the specification's checklist: send raw text to /ingest, verify the event is in
DuckDB, then verify the propensity and profile endpoints answer for that user with a
valid probability, a lifetime value, and a segment.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
from fastapi.testclient import TestClient

from src.api.main import create_app


def test_full_pipeline(serving_db: Path):
    app = create_app(db_path=serving_db, bootstrap=False)
    with TestClient(app) as client:
        user_id = client.get("/users", params={"limit": 1}).json()[0]["user_id"]

        # 1. Send raw screen text to /ingest.
        ingest = client.post(
            "/ingest",
            json={
                "raw_text": "Sponsored 4.6★ (2,847) Apple iPhone 15 (128 GB) ₹68,330 M.R.P: ₹75,920 (10% off) Buy Now",
                "user_id": user_id,
                "app_name": "amazon_in",
            },
        )
        assert ingest.status_code == 200
        event = ingest.json()
        assert event["product"]["brand"] == "Apple"
        assert event["product"]["price"] == 68330.0
        assert event["stored_action"] == "buy"

        # 2. Verify the event appears in DuckDB, read through the API's own connection.
        warehouse = app.state.service.warehouse
        with warehouse.lock:
            stored = warehouse.con.execute(
                "SELECT user_id, app_name, action, CAST(price AS DOUBLE) FROM fact_events WHERE event_id = ?",
                [event["event_id"]],
            ).fetchone()
        assert stored == (user_id, "amazon_in", "buy", 68330.0)

        # 3 and 4. Propensity responds with a valid probability.
        propensity = client.get(f"/user/{user_id}/propensity").json()
        assert 0.0 <= propensity["purchase_probability"] <= 1.0

        # 5 and 6. The profile carries a lifetime value and a segment, and reflects the purchase.
        profile = client.get(f"/user/{user_id}/profile").json()
        assert profile["lifetime_value"] is not None
        assert profile["lifetime_value"]["predicted_clv_inr"] > 0
        assert profile["segment"]["segment_name"]
        assert profile["last_seen"].startswith(event["timestamp"][:19])

    # The write is durable: it survives the API shutting down.
    con = duckdb.connect(str(serving_db), read_only=True)
    try:
        assert con.execute("SELECT COUNT(*) FROM fact_events WHERE event_id = ?", [event["event_id"]]).fetchone()[0] == 1
    finally:
        con.close()
