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
        run.__wrapped__ = check  # type: ignore[attr-defined]
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


# ---------------------------------------------------------------------------------------
# v1.3: an extraction-driven warehouse
# ---------------------------------------------------------------------------------------

def _warehouse() -> Any:
    """A read-only connection to the batch warehouse."""
    import duckdb

    from src.config import DEFAULT_DB_PATH

    return duckdb.connect(str(DEFAULT_DB_PATH), read_only=True)


def _scalar(sql: str) -> Any:
    """The first value of a read-only warehouse query."""
    con = _warehouse()
    try:
        return con.execute(sql).fetchone()[0]
    finally:
        con.close()


@gate("1.3", "G1.3-01", "H")
def warehouse_is_schema_version_2() -> Tuple[bool, str]:
    """schema_info records schema version 2 in the batch warehouse."""
    version = _scalar("SELECT value FROM schema_info WHERE key = 'schema_version'")
    return version == "2", f"schema_version {version!r}"


@gate("1.3", "G1.3-02", "H")
def raw_and_fact_event_ids_match() -> Tuple[bool, str]:
    """raw_events and fact_events hold the same set of event ids."""
    only_raw = _scalar("SELECT COUNT(*) FROM raw_events r ANTI JOIN fact_events f USING (event_id)")
    only_fact = _scalar("SELECT COUNT(*) FROM fact_events f ANTI JOIN raw_events r USING (event_id)")
    return only_raw == 0 and only_fact == 0, f"{only_raw} only in raw_events, {only_fact} only in fact_events"


@gate("1.3", "G1.3-03", "H")
def data_checks_pass() -> Tuple[bool, str]:
    """src/features/data_checks.py exits 0 on the batch warehouse."""
    run = subprocess.run([str(_ROOT / ".venv" / "bin" / "python"), "src/features/data_checks.py"],
                         cwd=_ROOT, capture_output=True, text=True)
    failing = [line for line in run.stdout.splitlines() if line.startswith("FAIL")]
    return run.returncode == 0, f"exit {run.returncode}; {len(failing)} failing" + (f": {failing[0]}" if failing else "")


@gate("1.3", "G1.3-04", "H")
def no_ground_truth_columns_in_warehouse() -> Tuple[bool, str]:
    """No warehouse column is named archetype or starts with true_."""
    con = _warehouse()
    try:
        bad = con.execute(
            "SELECT table_name || '.' || column_name FROM information_schema.columns "
            "WHERE column_name = 'archetype' OR column_name LIKE 'true\\_%' ESCAPE '\\'"
        ).fetchall()
    finally:
        con.close()
    return not bad, "none" if not bad else ", ".join(r[0] for r in bad)


@gate("1.3", "G1.3-05", "T")
def comparison_direction_is_balanced() -> Tuple[bool, str]:
    """The share of comparisons whose primary app sorts after the compared app is in [0.35, 0.65]."""
    share = _scalar(
        "SELECT AVG(CASE WHEN primary_app > compared_app THEN 1.0 ELSE 0.0 END) FROM fact_cross_app_comparisons"
    )
    share = float(share) if share is not None else float("nan")
    return 0.35 <= share <= 0.65, f"share {share:.4f} (in [0.35, 0.65])"


@gate("1.3", "G1.3-06", "T")
def extraction_accuracy_at_default_noise() -> Tuple[bool, str]:
    """Against the simulator truth: action >= 0.97, price >= 0.95, product identity >= 0.95."""
    from src.config import DEFAULT_DB_PATH, SIMULATION_TRUTH_PATH
    from src.experiments.noise_propagation import event_accuracy

    acc = event_accuracy(DEFAULT_DB_PATH, SIMULATION_TRUTH_PATH)
    ok = acc["action"] >= 0.97 and acc["price"] >= 0.95 and acc["identity"] >= 0.95
    return ok, f"action {acc['action']:.4f}, price {acc['price']:.4f}, identity {acc['identity']:.4f} over {acc['events']} events"


@gate("1.3", "G1.3-07", "H")
def noise_propagation_report_complete() -> Tuple[bool, str]:
    """reports/v1_3_noise_propagation.json has three noise levels with accuracies, ROC-AUC, CI and delta."""
    levels = _report("v1_3_noise_propagation")["levels"]
    missing = []
    for key, level in levels.items():
        for path in ("accuracy.action", "accuracy.price", "accuracy.identity", "propensity_test.roc_auc",
                     "propensity_test.ci", "delta_vs_baseline.delta"):
            try:
                _get(level, path)
            except KeyError:
                missing.append(f"{key}:{path}")
    return len(levels) == 3 and not missing, f"{len(levels)} levels {sorted(levels)}" + (f"; missing {missing[:3]}" if missing else "")


@gate("1.3", "G1.3-08", "H")
def feature_sql_has_no_avg_price() -> Tuple[bool, str]:
    """No feature module (src/features/*_features.py) mentions avg_price."""
    hits = [f"{path.name}:{n}" for path in sorted((_ROOT / "src" / "features").glob("*_features.py"))
            for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1) if "avg_price" in line]
    return not hits, "no references" if not hits else ", ".join(hits)


@gate("1.3", "G1.3-09", "H")
def leakage_tests_pass() -> Tuple[bool, str]:
    """tests/test_leakage.py passes, covering events, price observations and comparisons."""
    return _run_pytest("tests/test_leakage.py")


@gate("1.3", "G1.3-10", "H")
def serving_parity_tests_pass() -> Tuple[bool, str]:
    """The serving parity tests pass (tests/test_scoring.py and tests/test_warehouse.py)."""
    return _run_pytest("tests/test_scoring.py", "tests/test_warehouse.py")


@gate("1.3", "G1.3-11", "T")
def default_load_under_fifteen_minutes() -> Tuple[bool, str]:
    """reports/v1_3_load.json: loading the default warehouse with 4 workers took under 15 minutes."""
    load = _report("v1_3_load")
    total = float(load["total_seconds"])
    return int(load["workers"]) == 4 and total < 900.0, (
        f"{total:.0f} s with {load['workers']} workers (extract wait {load['extract_wait_seconds']:.0f} s, "
        f"resolve {load['resolve_seconds']:.0f} s, write {load['write_seconds']:.0f} s; < 900 s)"
    )


# Gates that run every other gate; each skips all of them so that none calls another.
_COMPOSITE_GATES = {"G1.3-12", "G1.4-15"}


def _suite_seconds(summary: str) -> float:
    """Wall-clock seconds from a pytest summary line, or infinity when absent."""
    match = re.search(r" in ([0-9.]+)s", summary)
    return float(match.group(1)) if match else float("inf")


@gate("1.3", "G1.3-12", "H")
def suite_fast_and_every_other_gate_passes() -> Tuple[bool, str]:
    """At least 400 tests, the suite passes in under 5 minutes, and every other gate of every version passes."""
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return False, "not run inside pytest (it would run the suite recursively)"
    count, code, summary = _suite_result()
    seconds = _suite_seconds(summary)
    others = [fn() for version in GATES for fn in GATES[version]
              if getattr(fn, "gate_id", "") not in _COMPOSITE_GATES]
    failed = [r.gate_id for r in others if not r.passed]
    ok = count >= 400 and code == 0 and seconds < 300.0 and not failed
    return ok, f"{count} tests, {seconds:.0f} s (< 300); {summary}; other gates failing: {failed or 'none'}"


# ---------------------------------------------------------------------------------------
# v1.4: a behaviour engine with ground truth
# ---------------------------------------------------------------------------------------

BINARY_TARGETS_V14 = ("will_purchase_next_7d", "will_switch_app_next_7d")


@gate("1.4", "G1.4-01", "H")
def catalog_and_products_are_large_enough() -> Tuple[bool, str]:
    """The catalog has at least 300 SKUs and dim_products at least 250 rows."""
    from src.ingestion.catalog import build_catalog

    skus = len(build_catalog())
    products = _scalar("SELECT COUNT(*) FROM dim_products")
    return skus >= 300 and products >= 250, f"{skus} catalog SKUs (>= 300), {products} dim_products rows (>= 250)"


@gate("1.4", "G1.4-02", "H")
def split_valid_with_v14_constants() -> Tuple[bool, str]:
    """validate_split passes for the configured dates and the warehouse's last event."""
    return split_is_valid_against_warehouse.__wrapped__()  # type: ignore[attr-defined]


@gate("1.4", "G1.4-03", "H")
def train_matrix_has_4000_rows() -> Tuple[bool, str]:
    """The train matrix has at least 4,000 rows."""
    frame = _train_matrix()
    per_date = {str(k): v for k, v in frame.group_by("reference_date").len().sort("reference_date").iter_rows()}
    return frame.height >= 4000, f"{frame.height} rows (>= 4000); per date {per_date}"


@gate("1.4", "G1.4-04", "H")
def truth_matches_raw_events_and_stays_out() -> Tuple[bool, str]:
    """The truth file and raw_events hold the same event ids, and no truth column is in the warehouse."""
    import polars as pl

    from src.config import PROCESSED_DATA_DIR

    truth = set(pl.read_parquet(PROCESSED_DATA_DIR / "simulation_truth.parquet", columns=["event_id"])["event_id"])
    con = _warehouse()
    try:
        raw = {r[0] for r in con.execute("SELECT event_id FROM raw_events").fetchall()}
    finally:
        con.close()
    clean, columns = no_ground_truth_columns_in_warehouse.__wrapped__()  # type: ignore[attr-defined]
    ok = truth == raw and clean
    return ok, f"{len(truth - raw)} only in truth, {len(raw - truth)} only in raw_events; truth columns: {columns}"


@gate("1.4", "G1.4-05", "T")
def sessions_have_realistic_length() -> Tuple[bool, str]:
    """Warehouse sessions: median events per session at least 3, single-event share at most 0.30."""
    con = _warehouse()
    try:
        median, single = con.execute(
            "SELECT median(n), avg(CASE WHEN n = 1 THEN 1.0 ELSE 0.0 END) FROM "
            "(SELECT session_id, COUNT(*) AS n FROM fact_events WHERE session_id IS NOT NULL GROUP BY session_id)"
        ).fetchone()
    finally:
        con.close()
    return float(median) >= 3 and float(single) <= 0.30, f"median {median} (>= 3), single-event share {float(single):.3f} (<= 0.30)"


@gate("1.4", "G1.4-06", "T")
def binary_targets_between_5_and_70_percent() -> Tuple[bool, str]:
    """Each binary target is between 5% and 70% positive in train and test."""
    import polars as pl

    from src.config import TEST_FEATURES_PATH, TRAIN_FEATURES_PATH

    rates = {}
    for split, path in (("train", TRAIN_FEATURES_PATH), ("test", TEST_FEATURES_PATH)):
        frame = pl.read_parquet(path, columns=list(BINARY_TARGETS_V14))
        for target in BINARY_TARGETS_V14:
            rates[f"{split}.{target}"] = float(frame[target].cast(pl.Float64).mean())
    ok = all(0.05 <= r <= 0.70 for r in rates.values())
    return ok, ", ".join(f"{k} {v:.3f}" for k, v in rates.items()) + " (each in [0.05, 0.70])"


@gate("1.4", "G1.4-07", "T")
def detectors_recover_ground_truth() -> Tuple[bool, str]:
    """Held-out seeds: sessionizer boundary F1 >= 0.90; comparison recall >= 0.85 and precision >= 0.70."""
    report = _report("v1_4_detectors")
    seeds = [str(s) for s in report["report_seeds"]]
    sess = report["sessionizer"]["report_at_chosen"]
    comp = report["comparison"]["report_at_chosen"]
    ok = bool(seeds) and all(sess[s]["f1"] >= 0.90 and comp[s]["recall"] >= 0.85 and comp[s]["precision"] >= 0.70
                             for s in seeds)
    detail = "; ".join(f"seed {s}: session F1 {sess[s]['f1']:.3f}, comparison P {comp[s]['precision']:.3f} "
                       f"R {comp[s]['recall']:.3f}" for s in seeds)
    return ok, detail


def _price_levels() -> Dict[str, Any]:
    levels = _report("v1_4_price_effect")["levels"]
    return {str(int(float(k))): v for k, v in levels.items()}


@gate("1.4", "G1.4-08", "T")
def price_dose_response_rises() -> Tuple[bool, str]:
    """Mean chose_cheaper_app_rate rises by at least 0.05 from price_beta 0 to 4 and again from 4 to 8."""
    levels = _price_levels()
    m0, m4, m8 = (float(levels[b]["chose_cheaper_mean"]) for b in ("0", "4", "8"))
    return m4 - m0 >= 0.05 and m8 - m4 >= 0.05, f"{m0:.3f} -> {m4:.3f} -> {m8:.3f} (steps {m4 - m0:+.3f}, {m8 - m4:+.3f}; >= 0.05)"


@gate("1.4", "G1.4-09", "T")
def price_coefficient_has_expected_sign() -> Tuple[bool, str]:
    """The price coefficient's 95% CI contains 0 at price_beta 0 and is below 0 at 4 and 8."""
    levels = _price_levels()
    ci = {b: (float(levels[b]["regression"]["ci"]["lo"]), float(levels[b]["regression"]["ci"]["hi"])) for b in ("0", "4", "8")}
    ok = ci["0"][0] <= 0.0 <= ci["0"][1] and ci["4"][1] < 0.0 and ci["8"][1] < 0.0
    return ok, ", ".join(f"beta {b} [{lo:.3f}, {hi:.3f}]" for b, (lo, hi) in ci.items())


@gate("1.4", "G1.4-10", "H")
def ablation_report_complete() -> Tuple[bool, str]:
    """The ablation report has models A to D for both targets, five seeds, deltas with CIs, the caveat and RetailRocket."""
    report = _report("v1_4_ablation")
    targets = ("will_purchase_next_7d", "switched_given_purchase")
    problems = []
    if len(report.get("seeds", [])) < 5:
        problems.append(f"{len(report.get('seeds', []))} seeds")
    for seed, block in report["per_seed"].items():
        for t in targets:
            models = set(block[t]["models"])
            if not {"A", "B", "C", "D"} <= models:
                problems.append(f"seed {seed} {t} models {sorted(models)}")
            for name in ("B-A", "C-A", "D-A"):
                d = block[t]["deltas"].get(name, {})
                if not {"delta", "lo", "hi"} <= set(d):
                    problems.append(f"seed {seed} {t} {name} without CI")
    if not str(report.get("caveat") or "").strip():
        problems.append("no caveat")
    if not report.get("retailrocket"):
        problems.append("no retailrocket block")
    return not problems, "complete" if not problems else "; ".join(problems[:4])


@gate("1.4", "G1.4-11", "H")
def learning_curve_and_calibration_complete() -> Tuple[bool, str]:
    """The learning curve has four sizes; the calibration report has both columns and deliberate differences."""
    curve = _report("v1_4_learning_curve")
    calib = _report("v1_4_sim_calibration")
    sizes = [p["users"] for p in curve["points"]]
    columns = all(set(v) == {"simulated", "retailrocket"} for v in calib["quantities"].values())
    diffs = [d for d in calib.get("deliberate_differences", []) if str(d).strip()]
    ok = len(sizes) == 4 and columns and bool(diffs)
    return ok, f"curve sizes {sizes}; calibration columns {'both' if columns else 'missing'}, {len(diffs)} deliberate differences"


@gate("1.4", "G1.4-12", "T")
def propensity_auc_in_v14_range() -> Tuple[bool, str]:
    """Propensity test ROC-AUC in [0.65, 0.93] and at least the single-feature baseline minus 0.01."""
    test = _propensity_test()
    auc = float(test["roc_auc"])
    base = float(test["baselines"]["single_feature"]["roc_auc"])
    return 0.65 <= auc <= 0.93 and auc >= base - 0.01, f"model {auc:.4f} in [0.65, 0.93]; single-feature {base:.4f}"


@gate("1.4", "G1.4-13", "T")
def churn_is_identified() -> Tuple[bool, str]:
    """BG/NBD dropout parameter a above 1e-4; churned share of simulated users between 0.20 and 0.35."""
    import polars as pl

    from src.config import ARTIFACTS_DIR, DEFAULT_DB_PATH, PROCESSED_DATA_DIR
    from src.ingestion.simulate_users import default_profiles_path

    a = float(_json(ARTIFACTS_DIR / "clv_summary.json")["bgnbd_params"]["a"])
    profiles = pl.read_parquet(default_profiles_path(DEFAULT_DB_PATH, PROCESSED_DATA_DIR / "simulation_truth.parquet"),
                               columns=["churned_day"])
    share = float(profiles["churned_day"].is_not_null().mean())
    return a > 1e-4 and 0.20 <= share <= 0.35, f"dropout a {a:.4f} (> 1e-4); churned share {share:.3f} of {profiles.height} users"


@gate("1.4", "G1.4-14", "H")
def simulation_has_no_module_random_or_archetype_loop() -> Tuple[bool, str]:
    """ARCHETYPE_PARAMS is not read in src/simulation/ and no module-level random call appears there."""
    pattern = re.compile(r"random\.(random|choice|choices|randint|uniform|seed)\b")
    hits = []
    for path in sorted((_ROOT / "src" / "simulation").glob("*.py")):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line) or "ARCHETYPE_PARAMS" in line:
                hits.append(f"{path.name}:{n}")
    return not hits, "none" if not hits else ", ".join(hits[:6])


@gate("1.4", "G1.4-15", "H")
def suite_fast_with_450_tests_and_all_gates_pass() -> Tuple[bool, str]:
    """At least 450 tests, the suite passes in under 5 minutes, and every other gate of every version passes."""
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return False, "not run inside pytest (it would run the suite recursively)"
    count, code, summary = _suite_result()
    seconds = _suite_seconds(summary)
    others = [fn() for version in GATES for fn in GATES[version]
              if getattr(fn, "gate_id", "") not in _COMPOSITE_GATES]
    failed = [r.gate_id for r in others if not r.passed]
    ok = count >= 450 and code == 0 and seconds < 300.0 and not failed
    return ok, f"{count} tests, {seconds:.0f} s (< 300); {summary}; other gates failing: {failed or 'none'}"


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
