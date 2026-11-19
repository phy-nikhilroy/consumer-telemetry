"""Learning curve: propensity and conditional switching test ROC-AUC (with CI) as the simulated population grows."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import PROJECT_ROOT, RANDOM_SEED

DEFAULT_OUT = PROJECT_ROOT / "data" / "experiments" / "v1_4_learning_curve"
USER_COUNTS = (250, 500, 1000, 2000)
CURVE_DAYS = 60
LOAD_WORKERS = 4
NAN = float("nan")


def _ci(value: Any) -> list:
    """Normalises a CI to a two-element list of floats, [nan, nan] when absent."""
    if isinstance(value, dict):
        value = [value.get("lo"), value.get("hi")]
    if not value or len(value) != 2:
        return [NAN, NAN]
    return [NAN if v is None else float(v) for v in value]


def summarize_point(users: int, propensity_test: Dict[str, Any], conditional_test: Dict[str, Any]) -> Dict[str, Any]:
    """Builds one learning-curve point from the propensity and conditional switching test metric blocks."""
    ci_p = propensity_test.get("ci", {})
    ci_p = ci_p.get("roc_auc") if isinstance(ci_p, dict) else ci_p
    return {
        "users": int(users),
        "propensity": {
            "n_samples": propensity_test.get("n_samples"),
            "roc_auc": float(propensity_test.get("roc_auc", NAN)),
            "ci": _ci(ci_p),
        },
        "conditional_switching": {
            "n_samples": conditional_test.get("n_samples"),
            "roc_auc": float(conditional_test.get("roc_auc", NAN)),
            "ci": _ci(conditional_test.get("ci")),
        },
    }


def run_point(users: int, out_dir: Path, days: int = CURVE_DAYS, seed: int = RANDOM_SEED) -> Dict[str, Any]:
    """Simulates `users` users, builds features and trains both models, all under `out_dir`."""
    from src.experiments.run_seeds import _deterministic_lightgbm
    from src.features.feature_store import generate_train_test_datasets, split_for_warehouse
    from src.features.feature_validation import run_feature_validation_pipeline
    from src.features.sessionizer import run_sessionization_pipeline
    from src.ingestion.simulate_users import run_simulation
    from src.models.propensity import run_training_pipeline
    from src.models.switching import run_conditional_switching

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "warehouse.duckdb"
    features_dir = out_dir / "features"
    artifacts_dir = out_dir / "artifacts"
    with _deterministic_lightgbm():
        run_simulation(num_users=users, days=days, db_path=str(db_path),
                       output_parquet=str(out_dir / "events.parquet"), seed=seed, workers=LOAD_WORKERS)
        run_sessionization_pipeline(db_path=db_path)
        train_refs, test_ref = split_for_warehouse(db_path)
        generate_train_test_datasets(train_ref_dates=train_refs, test_ref_date=test_ref,
                                     db_path=db_path, features_dir=features_dir)
        run_feature_validation_pipeline(features_dir=features_dir)
        prop = run_training_pipeline(features_dir=features_dir, artifacts_dir=artifacts_dir)
        cond = run_conditional_switching(features_dir=features_dir, artifacts_dir=artifacts_dir)
    prop_test = json.loads(Path(prop["metrics_path"]).read_text(encoding="utf-8"))["test"]
    return summarize_point(users, prop_test, cond["metrics"])


def run_learning_curve(
    out_root: Path = DEFAULT_OUT,
    user_counts: Sequence[int] = USER_COUNTS,
    days: int = CURVE_DAYS,
    seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """Runs every population size in `out_root/users_<n>/` and returns the curve plus AUC gain from first to last."""
    points = [run_point(n, Path(out_root) / f"users_{n}", days, seed) for n in user_counts]
    gain: Dict[str, float] = {}
    for key in ("propensity", "conditional_switching"):
        a, b = points[0][key]["roc_auc"], points[-1][key]["roc_auc"]
        gain[key] = b - a if math.isfinite(a) and math.isfinite(b) else NAN
    return {"seed": seed, "days": days, "user_counts": [int(n) for n in user_counts],
            "points": points, "auc_gain_first_to_last": gain}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learning curve over simulated population size.")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT), help="Root for per-size isolated outputs")
    parser.add_argument("--users", type=int, nargs="+", default=list(USER_COUNTS))
    parser.add_argument("--days", type=int, default=CURVE_DAYS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    from src.release.reporting import write_report

    results = run_learning_curve(Path(args.out), args.users, args.days, args.seed)
    print(f"Wrote {write_report('v1_4_learning_curve', '1.4', results, seed=args.seed)}")
