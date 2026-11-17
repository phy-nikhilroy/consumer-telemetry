"""Writes versioned JSON reports under reports/ with the standard envelope."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.config import PROJECT_ROOT, RANDOM_SEED, REPORTS_DIR

SCHEMA_VERSION = 1


def _git_short_sha() -> str:
    """Return the short git commit sha, or 'unknown' if unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10,
        )
        sha = out.stdout.strip()
        return sha if out.returncode == 0 and sha else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def write_report(name: str, release: str, results: dict, seed: Optional[int] = None) -> Path:
    """Write reports/<name>.json with the standard envelope and return its path."""
    if not name or "/" in name or name.startswith("."):
        raise ValueError(f"invalid report name: {name!r}")
    name = name[:-5] if name.endswith(".json") else name
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "release": release,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_short_sha(),
        "seed": RANDOM_SEED if seed is None else seed,
        "results": results,
        "tripwire_notes": [],
    }
    path = REPORTS_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write a sample report envelope.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--release", required=True)
    args = parser.parse_args()
    print(write_report(args.name, args.release, {}))
