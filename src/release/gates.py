"""Release gate runner. Gates are small read-only functions registered per version."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

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


def gate(version: str, gate_id: str, kind: str) -> Callable:
    """
    Registers a check under `version`. The check returns `(passed, detail)`; any exception
    it raises becomes a failed result carrying the exception text, so a gate never crashes
    the runner or a test that calls it directly.
    """

    def register(check: Callable[[], Tuple[bool, str]]) -> Callable[[], GateResult]:
        def run() -> GateResult:
            try:
                passed, detail = check()
            except Exception as exc:  # report, never raise
                first_line = (str(exc).strip().splitlines() or [""])[0][:200]
                return GateResult(gate_id, kind, False, f"{type(exc).__name__}: {first_line}")
            return GateResult(gate_id, kind, bool(passed), detail)

        run.gate_id = gate_id  # type: ignore[attr-defined]
        run.kind = kind  # type: ignore[attr-defined]
        run.__name__ = check.__name__
        run.__doc__ = check.__doc__
        GATES.setdefault(version, []).append(run)
        return run

    return register


def _json(path: Path) -> Any:
    """Reads a JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _report(name: str) -> Dict[str, Any]:
    """Returns the `results` block of reports/<name>.json."""
    from src.config import REPORTS_DIR

    return _json(REPORTS_DIR / f"{name}.json")["results"]


def _flatten(value: Any, prefix: str = "") -> Dict[str, Any]:
    """Flattens nested dicts to dotted keys."""
    if not isinstance(value, dict):
        return {prefix: value}
    out: Dict[str, Any] = {}
    for key, item in value.items():
        out.update(_flatten(item, f"{prefix}.{key}" if prefix else str(key)))
    return out


# ------------------------------------------------------------------ v1.1 honest evaluation


@gate("1.1", "G1.1-01", "H")
def split_is_valid_against_warehouse() -> Tuple[bool, str]:
    """validate_split passes for the configured dates and the warehouse's last event."""
    import duckdb

    from src.config import DEFAULT_DB_PATH, TEST_REFERENCE_DATE, TRAIN_REFERENCE_DATES
    from src.features.feature_store import validate_split

    con = duckdb.connect(str(DEFAULT_DB_PATH), read_only=True)
    try:
        max_ts = con.execute("SELECT MAX(timestamp) FROM fact_events").fetchone()[0]
    finally:
        con.close()
    detail = f"train {list(TRAIN_REFERENCE_DATES)}, test {TEST_REFERENCE_DATE}, data end {max_ts}"
    try:
        validate_split(TRAIN_REFERENCE_DATES, TEST_REFERENCE_DATE, max_event_timestamp=max_ts)
    except ValueError as exc:
        return False, f"{detail}: {exc}"
    return True, detail


def _train_matrix():
    import polars as pl

    from src.config import TRAIN_FEATURES_PATH

    return pl.read_parquet(TRAIN_FEATURES_PATH, columns=["user_id", "reference_date"])


@gate("1.1", "G1.1-02", "H")
def train_matrix_stacks_reference_dates() -> Tuple[bool, str]:
    """The train matrix has one block per train reference date and unique (user, date) rows."""
    from src.config import TRAIN_REFERENCE_DATES

    df = _train_matrix()
    counts = dict(df.group_by("reference_date").len().sort("reference_date").iter_rows())
    n_dates = df["reference_date"].n_unique()
    duplicates = int(df.is_duplicated().sum())
    ok = n_dates == len(TRAIN_REFERENCE_DATES) and duplicates == 0
    return ok, f"rows per reference date {counts}, duplicate (user, date) rows {duplicates}"


@gate("1.1", "G1.1-03", "H")
def train_windows_end_before_test_reference() -> Tuple[bool, str]:
    """Every train row's target window ends at or before the test reference date."""
    from src.config import PREDICTION_WINDOW_DAYS, TEST_REFERENCE_DATE

    df = _train_matrix()
    test_ref = datetime.fromisoformat(TEST_REFERENCE_DATE)
    window = timedelta(days=PREDICTION_WINDOW_DAYS)
    ends = {r: datetime.fromisoformat(r) + window for r in df["reference_date"].unique().to_list()}
    late = {r: str(e) for r, e in ends.items() if e > test_ref}
    return not late, (f"train target windows end before test reference {TEST_REFERENCE_DATE}"
                      if not late else f"windows past the test reference: {late}")


@gate("1.1", "G1.1-04", "H")
def selection_keeps_cross_app_features() -> Tuple[bool, str]:
    """selected_features.json is schema 2 with the cross-app features in both binary lists."""
    from src.config import SELECTED_FEATURES_PATH

    data = _json(SELECTED_FEATURES_PATH)
    needed = ("cross_app_comparison_count", "app_entropy", "num_apps_used")
    missing = {
        target: [f for f in needed if f not in data.get("features_by_target", {}).get(target, [])]
        for target in ("will_purchase_next_7d", "will_switch_app_next_7d")
    }
    missing = {t: m for t, m in missing.items() if m}
    ok = data.get("schema_version") == 2 and not missing
    return ok, f"schema_version {data.get('schema_version')}, missing {missing or 'none'}"


@gate("1.1", "G1.1-05", "H")
def propensity_metrics_have_intervals_and_baselines() -> Tuple[bool, str]:
    """propensity_metrics.json test block has ci, both baselines, and both deltas."""
    from src.config import ARTIFACTS_DIR

    test = _json(ARTIFACTS_DIR / "propensity_metrics.json")["test"]
    baselines = test.get("baselines") or {}
    checks = {
        "ci": bool(test.get("ci")),
        "baselines.single_feature": bool(baselines.get("single_feature")),
        "baselines.logistic": bool(baselines.get("logistic")),
        "delta_vs_single_feature": bool(test.get("delta_vs_single_feature")),
        "delta_vs_logistic": bool(test.get("delta_vs_logistic")),
    }
    absent = [k for k, ok in checks.items() if not ok]
    return not absent, f"missing {absent}" if absent else "ci, both baselines, both deltas present"


@gate("1.1", "G1.1-06", "H")
def conditional_switching_has_enough_buyers() -> Tuple[bool, str]:
    """switching_metrics.json has conditional_test with at least 100 test buyers."""
    from src.config import ARTIFACTS_DIR

    block = _json(ARTIFACTS_DIR / "switching_metrics.json")["conditional_test"]
    n = int(block.get("n_samples", 0))
    return n >= 100, f"conditional_test n_samples {n} (need >= 100), roc_auc {block.get('roc_auc')}"


def _propensity_test() -> Dict[str, Any]:
    from src.config import ARTIFACTS_DIR

    return _json(ARTIFACTS_DIR / "propensity_metrics.json")["test"]


@gate("1.1", "G1.1-07", "T")
def propensity_auc_in_expected_range() -> Tuple[bool, str]:
    """Propensity test ROC-AUC lies in [0.78, 0.94]."""
    auc = float(_propensity_test()["roc_auc"])
    return 0.78 <= auc <= 0.94, f"propensity test ROC-AUC {auc:.4f}, expected [0.78, 0.94]"


@gate("1.1", "G1.1-08", "T")
def propensity_beats_single_feature() -> Tuple[bool, str]:
    """Propensity test ROC-AUC is at least the single-feature baseline minus 0.01."""
    test = _propensity_test()
    auc = float(test["roc_auc"])
    base = float(test["baselines"]["single_feature"]["roc_auc"])
    return auc >= base - 0.01, f"model {auc:.4f} vs single-feature {base:.4f} (allowed 0.01 below)"


@gate("1.1", "G1.1-09", "H")
def seeds_report_covers_five_seeds() -> Tuple[bool, str]:
    """reports/v1_1_seeds.json covers 5+ seeds with mean and std for both ROC-AUCs."""
    results = _report("v1_1_seeds")
    seeds = results.get("seeds") or []
    flat = _flatten(results)

    def has(model: str) -> bool:
        keys = [k for k in flat if model in k and "roc_auc" in k]
        return any(k.endswith(".mean") for k in keys) and any(k.endswith(".std") for k in keys)

    ok = len(seeds) >= 5 and has("propensity") and has("switching")
    return ok, (f"seeds {seeds}, propensity mean/std {has('propensity')}, "
                f"switching mean/std {has('switching')}")


@gate("1.1", "G1.1-10", "H")
def clv_report_complete() -> Tuple[bool, str]:
    """reports/v1_1_clv.json has every E7 key and bucket counts summing to n_customers."""
    results = _report("v1_1_clv")
    keys = ("n_customers", "actual_total_purchases", "predicted_total_purchases",
            "mae_purchases_bgnbd", "mae_purchases_naive", "spearman_spend_bgnbd_gg",
            "spearman_spend_naive", "by_frequency_bucket", "order_value_mae")
    missing = [k for k in keys if k not in results]
    estimators = set((results.get("order_value_mae") or {}))
    missing += [f"order_value_mae.{e}" for e in ("gamma_gamma", "observed_mean", "shrunk")
                if e not in estimators]
    total = sum(int(b["n"]) for b in (results.get("by_frequency_bucket") or {}).values())
    ok = not missing and total == results.get("n_customers")
    return ok, f"missing {missing or 'none'}, bucket total {total} vs n_customers {results.get('n_customers')}"


@gate("1.1", "G1.1-11", "H")
def segmentation_summary_validated() -> Tuple[bool, str]:
    """segmentation_summary.json carries ari_vs_archetype, bootstrap_stability, gmm_bic_by_k."""
    from src.config import ARTIFACTS_DIR

    summary = _json(ARTIFACTS_DIR / "segmentation_summary.json")
    keys = ("ari_vs_archetype", "bootstrap_stability", "gmm_bic_by_k")
    missing = [k for k in keys if k not in summary]
    return not missing, (f"missing {missing}" if missing else
                         f"ari {summary['ari_vs_archetype']}, stability {summary['bootstrap_stability']}")


@gate("1.1", "G1.1-12", "H")
def retailrocket_grid_concluded() -> Tuple[bool, str]:
    """reports/v1_1_retailrocket.json has 8 grid cells and a non-empty conclusion."""
    results = _report("v1_1_retailrocket")
    cells = results.get("grid") or []
    conclusion = str(results.get("conclusion") or "").strip()
    return len(cells) == 8 and bool(conclusion), f"{len(cells)} grid cells, conclusion {'set' if conclusion else 'empty'}"


@gate("1.1", "G1.1-13", "H")
def suite_passes_with_enough_tests() -> Tuple[bool, str]:
    """At least 300 tests are collected and the full suite passes."""
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return False, "not run inside pytest (it would run the suite recursively)"
    pytest_bin = str(_ROOT / ".venv" / "bin" / "pytest")
    args = [pytest_bin, "-q", "-p", "no:cacheprovider"]
    collected = subprocess.run(args + ["--collect-only"], cwd=_ROOT, capture_output=True, text=True)
    match = re.search(r"(\d+) tests? collected", collected.stdout)
    count = int(match.group(1)) if match else 0
    run = subprocess.run(args, cwd=_ROOT, capture_output=True, text=True)
    summary = run.stdout.strip().splitlines()[-1] if run.stdout.strip() else "no output"
    return count >= 300 and run.returncode == 0, f"{count} tests collected; {summary}"


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
