"""Release gate runner. Gates are small read-only functions registered per version."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

GATE_ID_PATTERN = re.compile(r"^G\d\.\d-\d{2}$")


@dataclass(frozen=True)
class GateResult:
    """Outcome of one gate: id, kind ("H" hard or "T" tripwire), verdict, one-line detail."""

    gate_id: str
    kind: str
    passed: bool
    detail: str


# Each gate function takes no arguments and returns GateResult. Later releases append here.
GATES: Dict[str, List[Callable[[], GateResult]]] = {"1.1": []}


def run_gates(version: str) -> List[GateResult]:
    """Run every gate of a version (or all versions) and return results; never raises per gate."""
    if version == "all":
        versions = list(GATES)
    elif version in GATES:
        versions = [version]
    else:
        raise ValueError(f"unknown version: {version!r}")
    results: List[GateResult] = []
    for v in versions:
        for fn in GATES[v]:
            gate_id = getattr(fn, "gate_id", fn.__name__)
            kind = getattr(fn, "kind", "H")
            try:
                results.append(fn())
            except Exception as exc:  # a failing gate must not crash the runner
                results.append(GateResult(gate_id, kind, False, f"exception: {exc}"))
    return results


def format_result(r: GateResult) -> str:
    """Format one result as 'PASS G1.1-03 [H] detail'."""
    return f"{'PASS' if r.passed else 'FAIL'} {r.gate_id} [{r.kind}] {r.detail}"


def main(argv: List[str] | None = None) -> int:
    """CLI entry point; returns 0 only if every gate passes."""
    parser = argparse.ArgumentParser(description="Run release gates.")
    parser.add_argument("--version", required=True, help="e.g. 1.1, or 'all'")
    args = parser.parse_args(argv)
    try:
        results = run_gates(args.version)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    for r in results:
        print(format_result(r))
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
