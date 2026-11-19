"""
Shared fixtures for the serving-layer tests.

`tiny_warehouse` simulates a small population once per test session into a temporary
directory, using the same seeded simulator and sessionizer as the real pipeline. Tests
that write to a warehouse take `serving_db`, a private copy, so no test can affect
another or touch the real batch and serving warehouses under data/processed.
"""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Iterator

import pytest

TINY_USERS = 40
TINY_DAYS = 30


@pytest.fixture(scope="session")
def tiny_warehouse(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A sessionized, simulated warehouse of 40 users over 30 days."""
    from src.api.warehouse import ensure_serving_warehouse

    directory = tmp_path_factory.mktemp("tiny_warehouse")
    target = directory / "tiny.duckdb"
    outcome = ensure_serving_warehouse(
        db_path=target,
        source_db_path=directory / "no_batch_warehouse.duckdb",
        num_users=TINY_USERS,
        days=TINY_DAYS,
    )
    assert outcome == "simulated"
    return target


@pytest.fixture
def serving_db(tiny_warehouse: Path, tmp_path: Path) -> Path:
    """A private, writable copy of the tiny warehouse."""
    copy = tmp_path / "serving.duckdb"
    shutil.copy2(tiny_warehouse, copy)
    return copy


@pytest.fixture
def api_client(serving_db: Path) -> Iterator["TestClient"]:  # noqa: F821
    """A TestClient for an app bound to a private copy of the tiny warehouse."""
    from fastapi.testclient import TestClient

    from src.api.main import create_app

    app = create_app(db_path=serving_db, bootstrap=False, clock_mode="simulated")
    with TestClient(app) as client:
        yield client
