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


_SUITE_CACHE: Dict[str, Tuple[int, int, str]] = {}


def _suite_result() -> Tuple[int, int, str]:
    """
    Collected test count, pytest exit code, and summary line of one full suite run. The run
    is shared by every gate in the process, so `--version all` runs the suite once.
    """
    if "run" not in _SUITE_CACHE:
        pytest_bin = str(_ROOT / ".venv" / "bin" / "pytest")
        args = [pytest_bin, "-q", "-p", "no:cacheprovider"]
        collected = subprocess.run(args + ["--collect-only"], cwd=_ROOT, capture_output=True, text=True)
        match = re.search(r"(\d+) tests? collected", collected.stdout)
        count = int(match.group(1)) if match else 0
        run = subprocess.run(args, cwd=_ROOT, capture_output=True, text=True)
        summary = run.stdout.strip().splitlines()[-1] if run.stdout.strip() else "no output"
        _SUITE_CACHE["run"] = (count, run.returncode, summary)
    return _SUITE_CACHE["run"]


@gate("1.1", "G1.1-13", "H")
def suite_passes_with_enough_tests() -> Tuple[bool, str]:
    """At least 300 tests are collected and the full suite passes."""
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return False, "not run inside pytest (it would run the suite recursively)"
    count, code, summary = _suite_result()
    return count >= 300 and code == 0, f"{count} tests collected; {summary}"


# ---------------------------------------------------------------------------------------
# v1.2: extraction measured on unseen data
# ---------------------------------------------------------------------------------------

def _get(data: Dict[str, Any], dotted: str) -> Any:
    """Reads a dotted key path from nested dicts; raises KeyError naming the missing path."""
    value: Any = data
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(dotted)
        value = value[part]
    return value


def _run_pytest(*node_ids: str) -> Tuple[bool, str]:
    """Runs the named tests in a subprocess; returns (passed, summary line)."""
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return False, "not run inside pytest"
    pytest_bin = str(_ROOT / ".venv" / "bin" / "pytest")
    run = subprocess.run([pytest_bin, "-q", "-p", "no:cacheprovider", *node_ids], cwd=_ROOT,
                         capture_output=True, text=True)
    summary = run.stdout.strip().splitlines()[-1] if run.stdout.strip() else "no output"
    return run.returncode == 0, summary


@gate("1.2", "G1.2-01", "H")
def extraction_artifacts_load() -> Tuple[bool, str]:
    """Both committed extraction artifacts exist and load."""
    import joblib

    from src.config import CONFIDENCE_ARTIFACT_PATH, TAXONOMY_ARTIFACT_PATH

    taxonomy = joblib.load(TAXONOMY_ARTIFACT_PATH)
    confidence = joblib.load(CONFIDENCE_ARTIFACT_PATH)
    ok = {"model", "labels", "tau"} <= set(taxonomy) and "model" in confidence
    return ok, f"taxonomy {len(taxonomy['labels'])} labels tau {taxonomy['tau']}; confidence loaded"


@gate("1.2", "G1.2-02", "H")
def extraction_report_complete() -> Tuple[bool, str]:
    """The extraction report has the tune, held-out, and gap blocks, synthetic, and throughput."""
    results = _report("v1_2_extraction")
    needed = ["real.tune", f"real.{_HELD_OUT}", "real.gap", "synthetic", "throughput"]
    missing = []
    for path in needed:
        try:
            _get(results, path)
        except KeyError:
            missing.append(path)
    return not missing, f"missing {missing or 'none'}"


# The held-out split's name comes from real_splits, the one module (with the benchmark)
# allowed to spell it; gate G1.2-10 checks that no other module does.
from src.ingestion.real_splits import SPLITS as _SPLITS  # noqa: E402

_HELD_OUT = _SPLITS[2]
HELD_OUT_CATEGORY_ACCURACY_KEY = f"real.{_HELD_OUT}.category_accuracy_non_abstained"
HELD_OUT_ABSTAIN_RATE_KEY = f"real.{_HELD_OUT}.abstain_rate"


@gate("1.2", "G1.2-03", "T")
def held_out_taxonomy_accuracy_and_abstention() -> Tuple[bool, str]:
    """Held-out taxonomy accuracy among non-abstained rows >= 0.80 and abstain rate <= 0.50."""
    results = _report("v1_2_extraction")
    accuracy = float(_get(results, HELD_OUT_CATEGORY_ACCURACY_KEY))
    abstain = float(_get(results, HELD_OUT_ABSTAIN_RATE_KEY))
    return accuracy >= 0.80 and abstain <= 0.50, f"accuracy {accuracy:.4f} (>= 0.80), abstain {abstain:.4f} (<= 0.50)"


# results.synthetic.by_noise is keyed by the noise rate as text ("0.00"), so it is read
# directly rather than through a dotted path.
SYNTHETIC_NOISE0_FLOORS = {"price_accuracy": 0.99, "brand_accuracy": 0.95, "segmentation_accuracy": 0.95}


@gate("1.2", "G1.2-04", "T")
def synthetic_accuracy_at_zero_noise() -> Tuple[bool, str]:
    """At OCR noise 0.0: price >= 0.99, brand >= 0.95, segmentation >= 0.95."""
    by_noise = _get(_report("v1_2_extraction"), "synthetic.by_noise")
    zero = by_noise["0.00"]
    parts, ok = [], True
    for key, floor in SYNTHETIC_NOISE0_FLOORS.items():
        value = float(zero[key])
        ok = ok and value >= floor
        parts.append(f"{key} {value:.4f} (>= {floor})")
    return ok, ", ".join(parts)


@gate("1.2", "G1.2-05", "T")
def matcher_f1_and_hard_negatives() -> Tuple[bool, str]:
    """Matcher on test pairs: F1 >= 0.90 and hard-negative false-match rate <= 0.05."""
    matcher = _report("v1_2_entity_resolution")["test"]["matcher"]
    f1, fmr = float(matcher["f1"]), float(matcher["hard_negative_false_match_rate"])
    return f1 >= 0.90 and fmr <= 0.05, f"F1 {f1:.4f} (>= 0.90), hard false-match {fmr:.4f} (<= 0.05)"


@gate("1.2", "G1.2-06", "H")
def matcher_recall_beats_exact_key() -> Tuple[bool, str]:
    """The v1.0 exact key's recall is reported and the matcher's recall is higher."""
    test = _report("v1_2_entity_resolution")["test"]
    exact, matcher = float(test["v1_exact_key"]["recall"]), float(test["matcher"]["recall"])
    return matcher > exact, f"matcher recall {matcher:.4f} vs exact key {exact:.4f}"


@gate("1.2", "G1.2-07", "T")
def learned_confidence_is_calibrated() -> Tuple[bool, str]:
    """Learned confidence on tune: ECE <= 0.08 and AUC >= the hand-weighted AUC."""
    tune = _get(_report("v1_2_extraction"), "confidence.tune")
    ece = float(tune["learned"]["ece"])
    auc, hand_auc = float(tune["learned"]["auc"]), float(tune["hand_weighted"]["auc"])
    return ece <= 0.08 and auc >= hand_auc, f"ECE {ece:.4f} (<= 0.08), AUC {auc:.4f} vs hand {hand_auc:.4f}"


@gate("1.2", "G1.2-08", "T")
def batch_throughput() -> Tuple[bool, str]:
    """process_batch reaches at least 250 events per second on this machine."""
    rate = float(_get(_report("v1_2_extraction"), "throughput.process_batch_events_per_s"))
    return rate >= 250.0, f"{rate:.1f} events per second (>= 250)"


@gate("1.2", "G1.2-09", "H")
def batch_equals_process_and_catalog_classified() -> Tuple[bool, str]:
    """process_batch equals process, and no catalog product is Unknown (the two tests pass)."""
    return _run_pytest(
        "tests/test_process_batch.py::test_process_batch_equals_process_on_200_listings",
        "tests/test_pipeline.py::test_catalog_products_keep_brand_and_category",
        "tests/test_pipeline.py::test_catalog_categories_are_structurally_consistent",
    )


@gate("1.2", "G1.2-10", "H")
def held_out_split_named_only_by_benchmark() -> Tuple[bool, str]:
    """The held-out split's name appears in no module except real_splits.py and real_data_benchmark.py."""
    return _run_pytest(f"tests/test_real_splits.py::test_only_the_benchmark_reads_{_HELD_OUT}")


@gate("1.2", "G1.2-11", "H")
def suite_passes_with_350_tests() -> Tuple[bool, str]:
    """At least 350 tests are collected and the full suite passes."""
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return False, "not run inside pytest (it would run the suite recursively)"
    count, code, summary = _suite_result()
    return count >= 350 and code == 0, f"{count} tests collected; {summary}"


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
