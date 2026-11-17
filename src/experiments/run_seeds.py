"""Multi-seed harness: runs the whole pipeline per simulation seed, isolated, and aggregates test metrics."""

from __future__ import annotations

import argparse
import contextlib
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import PROJECT_ROOT

DEFAULT_OUT = PROJECT_ROOT / "data" / "experiments" / "v1_1_seeds"


def _numeric_only(obj: Any) -> Any:
    """Keeps nested dicts of numeric scalars, dropping strings, lists and other leaves."""
    if isinstance(obj, dict):
        out = {str(k): _numeric_only(v) for k, v in obj.items()}
        return {k: v for k, v in out.items() if v is not None and v != {}}
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        return float(obj) if isinstance(obj, float) else obj
    return None


@contextlib.contextmanager
def _deterministic_lightgbm() -> Iterator[None]:
    """Single-threaded, deterministic LightGBM for the duration of a run.

    Multithreaded histogram building is not reproducible on small matrices, so identical seeds
    could otherwise give different metrics. Parameter dicts are restored on exit.
    """
    from src.models import propensity, switching

    dicts = [propensity.DEFAULT_PARAMS, switching.BINARY_PARAMS, switching.MULTICLASS_PARAMS]
    saved = [dict(d) for d in dicts]
    for d in dicts:
        d.update({"n_jobs": 1, "deterministic": True, "force_row_wise": True})
    try:
        yield
    finally:
        for d, old in zip(dicts, saved):
            d.clear()
            d.update(old)


def run_one_seed(seed: int, out_dir: Path, users: int, days: int) -> Dict[str, Any]:
    """Runs simulate, sessionize, features, validation, propensity, evaluation and switching under `out_dir`.

    Nothing is written outside `out_dir`. Returns the numeric test metrics of each stage.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with _deterministic_lightgbm():
        return _run_pipeline(seed, out_dir, users, days)


def _run_pipeline(seed: int, out_dir: Path, users: int, days: int) -> Dict[str, Any]:
    """Executes the pipeline stages for one seed and collects numeric test metrics."""
    from src.features.feature_store import generate_train_test_datasets
    from src.features.feature_validation import run_feature_validation_pipeline
    from src.features.sessionizer import run_sessionization_pipeline
    from src.ingestion.simulate_users import run_simulation
    from src.models.evaluation import run_evaluation_pipeline
    from src.models.propensity import run_training_pipeline
    from src.models.switching import run_switching_pipeline

    features_dir = out_dir / "features"
    artifacts_dir = out_dir / "artifacts"
    db_path = out_dir / "warehouse.duckdb"
    sim = run_simulation(num_users=users, days=days, db_path=str(db_path),
                         output_parquet=str(out_dir / "events.parquet"), seed=seed)
    run_sessionization_pipeline(db_path=db_path)
    generate_train_test_datasets(db_path=db_path, features_dir=features_dir)
    run_feature_validation_pipeline(features_dir=features_dir)
    prop = run_training_pipeline(features_dir=features_dir, artifacts_dir=artifacts_dir)
    ev = run_evaluation_pipeline(features_dir=features_dir, artifacts_dir=artifacts_dir)
    sw = run_switching_pipeline(features_dir=features_dir, artifacts_dir=artifacts_dir)

    result = {
        "simulation": {"total_events": sim["total_events"]},
        "propensity": {"test": prop["test_metrics"]},
        "evaluation": {
            "calibrated_test": ev["metrics_after"],
            "uncalibrated_test": ev["metrics_before"],
            "summary": {k: v for k, v in ev["summary"].items() if k != "decile_table"},
        },
        "switching": {
            "test": sw["binary_test_metrics"],
            "test_tuned": sw["binary_test_metrics_tuned"],
            "multiclass_test": sw["multiclass_test_metrics"],
        },
    }
    return _numeric_only(result)


def _flatten(obj: Any, prefix: tuple = ()) -> Dict[tuple, float]:
    """Flattens nested dicts to {key path tuple: number}."""
    flat: Dict[tuple, float] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            flat.update(_flatten(v, prefix + (k,)))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        flat[prefix] = float(obj)
    return flat


def _aggregate(values: List[float]) -> Dict[str, float]:
    """Mean, sample std, min, max and n over the finite values."""
    vals = [v for v in values if math.isfinite(v)]
    n = len(vals)
    if n == 0:
        nan = float("nan")
        return {"mean": nan, "std": nan, "min": nan, "max": nan, "n": 0}
    mean = sum(vals) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    return {"mean": mean, "std": std, "min": min(vals), "max": max(vals), "n": n}


def run_seeds(seeds: Sequence[int], out_root: Path, users: int = 400, days: int = 30) -> Dict[str, Any]:
    """Runs `run_one_seed` per seed in its own directory and aggregates every numeric metric."""
    per_seed: Dict[str, Any] = {}
    for seed in seeds:
        per_seed[str(seed)] = run_one_seed(seed, Path(out_root) / f"seed_{seed}", users, days)

    columns: Dict[tuple, List[float]] = {}
    for result in per_seed.values():
        for path, v in _flatten(result).items():
            columns.setdefault(path, []).append(v)

    aggregate: Dict[str, Any] = {}
    for path, vals in columns.items():
        node = aggregate
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = _aggregate(vals)

    return {"seeds": [int(s) for s in seeds], "users": users, "days": days,
            "aggregate": aggregate, "per_seed": per_seed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the pipeline across simulation seeds and aggregate metrics.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT), help="Root for per-seed isolated outputs")
    parser.add_argument("--users", type=int, default=400)
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()

    from src.release.reporting import write_report

    results = run_seeds(args.seeds, Path(args.out), users=args.users, days=args.days)
    print(f"Wrote {write_report('v1_1_seeds', '1.1', results)}")
