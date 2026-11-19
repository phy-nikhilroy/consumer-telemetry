"""Scores the sessionizer and the cross-app comparison detector against simulator ground truth.

A predicted session boundary is correct when the true session id changes at the same event.
A detected comparison is correct when its later event is flagged `is_deliberate_comparison`.
Config values are chosen by best mean F1 on the selection seeds and reported on the report seeds.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import duckdb
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import (
    DEFAULT_CROSS_APP_WINDOW_MINUTES,
    DEFAULT_SESSION_INACTIVITY_MINUTES,
    PROJECT_ROOT,
)

DEFAULT_OUT = PROJECT_ROOT / "data" / "experiments" / "v1_4_detectors"
REPORT_NAME = "v1_4_detectors"
INACTIVITY_GRID = (5, 10, 20, 30, 45, 60)
WINDOW_GRID = (10, 30, 60, 120)
SELECTION_SEEDS = (1, 2, 3)
REPORT_SEEDS = (4, 5)
LOAD_WORKERS = 4


def prf(tp: int, fp: int, fn: int) -> Dict[str, Any]:
    """Precision, recall and F1 from counts; a zero denominator gives 0.0."""
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}


def score_sessionizer(con: duckdb.DuckDBPyConnection, truth: pl.DataFrame,
                      inactivity_minutes: float) -> Dict[str, Any]:
    """Boundary precision, recall and F1 of `sessionize_events` at one threshold.

    Events are ordered per user by (timestamp, event_id). An event with a predecessor is a
    true boundary when `true_session_id` differs from the predecessor's, and a predicted
    boundary when the assigned `session_id` differs. Mutates fact_events.session_id.
    """
    from src.features.sessionizer import sessionize_events

    sessionize_events(con=con, inactivity_minutes=inactivity_minutes)
    facts = pl.from_arrow(con.execute(
        "SELECT event_id, user_id, timestamp, session_id FROM fact_events").to_arrow_table())
    joined = facts.join(truth.select("event_id", "true_session_id"), on="event_id", how="inner")
    joined = joined.sort(["user_id", "timestamp", "event_id"]).with_columns(
        (pl.col("session_id") != pl.col("session_id").shift(1).over("user_id")).alias("pred"),
        (pl.col("true_session_id") != pl.col("true_session_id").shift(1).over("user_id")).alias("true"),
        pl.col("session_id").shift(1).over("user_id").is_not_null().alias("has_prev"),
    ).filter(pl.col("has_prev"))
    tp = int(joined.filter(pl.col("pred") & pl.col("true")).height)
    fp = int(joined.filter(pl.col("pred") & ~pl.col("true")).height)
    fn = int(joined.filter(~pl.col("pred") & pl.col("true")).height)
    return prf(tp, fp, fn)


def score_comparisons(con: duckdb.DuckDBPyConnection, truth: pl.DataFrame,
                      window_minutes: float) -> Dict[str, Any]:
    """Precision, recall and F1 of `detect_cross_app_comparisons` at one window.

    The later event of each detected comparison is the event of the same user, product,
    app and timestamp as `last_viewed`. It is correct when truth flags it
    `is_deliberate_comparison`. Recall is over every flagged event in the truth.
    """
    from src.features.sessionizer import detect_cross_app_comparisons

    detect_cross_app_comparisons(con=con, max_window_minutes=window_minutes)
    later = pl.from_arrow(con.execute("""
        SELECT DISTINCT e.event_id
        FROM fact_cross_app_comparisons c
        JOIN fact_events e
          ON e.user_id = c.user_id AND e.canonical_product_id = c.canonical_product_id
         AND e.app_name = c.compared_app AND e.timestamp = c.last_viewed
    """).to_arrow_table())
    detected = set(later["event_id"].to_list())
    flagged = set(truth.filter(pl.col("is_deliberate_comparison"))["event_id"].to_list())
    tp = len(detected & flagged)
    return prf(tp, len(detected - flagged), len(flagged - detected))


def score_seed(db_path: Path, truth_path: Path) -> Dict[str, Any]:
    """Scores every grid value on one warehouse and truth file."""
    truth = pl.read_parquet(truth_path)
    con = duckdb.connect(str(db_path))
    try:
        sess = {str(m): score_sessionizer(con, truth, m) for m in INACTIVITY_GRID}
        # Comparisons carry the session id of the earlier event, so restore the default sessions.
        from src.features.sessionizer import sessionize_events
        sessionize_events(con=con, inactivity_minutes=DEFAULT_SESSION_INACTIVITY_MINUTES)
        comp = {str(m): score_comparisons(con, truth, m) for m in WINDOW_GRID}
    finally:
        con.close()
    return {"sessionizer": sess, "comparison": comp}


def mean_f1(per_seed: Dict[int, Dict[str, Any]], detector: str, value: str, seeds: Sequence[int]) -> float:
    """Mean F1 of one grid value across `seeds`."""
    vals = [per_seed[s][detector][value]["f1"] for s in seeds]
    return sum(vals) / len(vals)


def choose(per_seed: Dict[int, Dict[str, Any]], detector: str, seeds: Sequence[int],
           prefer: Optional[float] = None) -> float:
    """Grid value with the best mean F1 over `seeds`.

    Ties go to the value closest to `prefer` (the current default when omitted), then to
    the smaller value, so a tie never moves the config.
    """
    if prefer is None:
        prefer = (DEFAULT_SESSION_INACTIVITY_MINUTES if detector == "sessionizer"
                  else DEFAULT_CROSS_APP_WINDOW_MINUTES)
    values = sorted(per_seed[seeds[0]][detector], key=float)
    best = max(values, key=lambda v: (round(mean_f1(per_seed, detector, v, seeds), 12),
                                      -abs(float(v) - prefer), -float(v)))
    return float(best)


def summarise(per_seed: Dict[int, Dict[str, Any]], selection: Sequence[int], report: Sequence[int]) -> Dict[str, Any]:
    """Chosen values, selection-seed mean F1 per value, and report-seed metrics at the chosen values."""
    out: Dict[str, Any] = {"selection_seeds": list(selection), "report_seeds": list(report)}
    for detector, key in (("sessionizer", "inactivity_minutes"), ("comparison", "window_minutes")):
        chosen = choose(per_seed, detector, selection)
        label = f"{chosen:g}"
        grid = per_seed[selection[0]][detector]
        out[detector] = {
            f"chosen_{key}": chosen,
            "selection_mean_f1": {v: mean_f1(per_seed, detector, v, selection) for v in grid},
            "report_mean_f1": {v: mean_f1(per_seed, detector, v, report) for v in grid},
            "report_at_chosen": {str(s): per_seed[s][detector][label] for s in report},
            "report_at_default": {
                str(s): per_seed[s][detector][
                    f"{DEFAULT_SESSION_INACTIVITY_MINUTES if detector == 'sessionizer' else DEFAULT_CROSS_APP_WINDOW_MINUTES:g}"
                ] for s in report
            },
        }
    return out


def run(out_root: Path = DEFAULT_OUT, users: int = 300, days: int = 30,
        selection: Sequence[int] = SELECTION_SEEDS, report: Sequence[int] = REPORT_SEEDS,
        write: bool = True) -> Dict[str, Any]:
    """Simulates each seed under `out_root`, scores the grids, and writes reports/v1_4_detectors.json."""
    from src.ingestion.simulate_users import run_simulation

    per_seed: Dict[int, Dict[str, Any]] = {}
    for seed in [*selection, *report]:
        d = Path(out_root) / f"seed_{seed}"
        d.mkdir(parents=True, exist_ok=True)
        db, truth = d / "warehouse.duckdb", d / "truth.parquet"
        run_simulation(num_users=users, days=days, db_path=str(db), output_parquet=str(d / "events.parquet"),
                       seed=seed, workers=LOAD_WORKERS, truth_parquet=str(truth))
        per_seed[seed] = score_seed(db, truth)
    results = summarise(per_seed, selection, report)
    results["users"], results["days"] = users, days
    results["per_seed"] = {str(s): v for s, v in per_seed.items()}
    if write:
        from src.release.reporting import write_report
        write_report(REPORT_NAME, "v1.4", results, seed=selection[0])
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--users", type=int, default=300)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args()
    res = run(Path(args.out), args.users, args.days, write=not args.no_report)
    for name in ("sessionizer", "comparison"):
        print(name, {k: v for k, v in res[name].items() if k.startswith("chosen_")})
