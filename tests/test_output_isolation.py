"""
Tests that every orchestrator can be pointed at another dataset's directories without
touching the committed synthetic artifacts.

A real-data run once silently overwrote the committed `selected_features.json`, because
the orchestrators wrote to fixed paths whatever warehouse their inputs came from.
"""

from pathlib import Path
import shutil

import pytest

from src.config import (
    ARTIFACTS_DIR,
    FEATURES_DIR,
    SELECTED_FEATURES_PATH,
    TEST_FEATURES_PATH,
    TRAIN_FEATURES_PATH,
    rebase_path,
)


def test_rebase_path_keeps_default_or_moves_file_name(tmp_path):
    """Verifies the helper leaves defaults alone and re-roots only the directory."""
    assert rebase_path(SELECTED_FEATURES_PATH) == SELECTED_FEATURES_PATH
    assert rebase_path(SELECTED_FEATURES_PATH, tmp_path) == tmp_path / "selected_features.json"
    assert rebase_path(SELECTED_FEATURES_PATH, str(tmp_path)) == tmp_path / "selected_features.json"


def _snapshot(*directories: Path) -> dict:
    """Records size and modification time of every file in the given directories."""
    state = {}
    for directory in directories:
        if not directory.exists():
            continue
        for path in directory.iterdir():
            if path.is_file():
                stat = path.stat()
                state[str(path)] = (stat.st_size, stat.st_mtime_ns)
    return state


@pytest.fixture(scope="module")
def isolated_run(tmp_path_factory):
    """
    Runs the whole modeling chain against a copy of the feature store in a temporary
    directory, recording the committed artifact state before and after.
    """
    for path in (TRAIN_FEATURES_PATH, TEST_FEATURES_PATH, SELECTED_FEATURES_PATH):
        if not path.exists():
            pytest.skip("Feature store not available")

    root = tmp_path_factory.mktemp("second_dataset")
    features_dir = root / "features"
    artifacts_dir = root / "artifacts"
    features_dir.mkdir()
    shutil.copy(TRAIN_FEATURES_PATH, features_dir / TRAIN_FEATURES_PATH.name)
    shutil.copy(TEST_FEATURES_PATH, features_dir / TEST_FEATURES_PATH.name)

    before = _snapshot(FEATURES_DIR, ARTIFACTS_DIR)

    from src.features.feature_validation import run_feature_validation_pipeline
    from src.models.clv import run_clv_pipeline
    from src.models.evaluation import run_evaluation_pipeline
    from src.models.propensity import run_training_pipeline
    from src.models.segmentation import run_segmentation_pipeline
    from src.models.switching import run_switching_pipeline

    run_feature_validation_pipeline(features_dir=features_dir)
    run_training_pipeline(features_dir=features_dir, artifacts_dir=artifacts_dir)
    run_evaluation_pipeline(features_dir=features_dir, artifacts_dir=artifacts_dir)
    run_switching_pipeline(features_dir=features_dir, artifacts_dir=artifacts_dir)
    run_clv_pipeline(features_dir=features_dir, artifacts_dir=artifacts_dir)
    run_segmentation_pipeline(
        k_values=(3, 4), features_dir=features_dir, artifacts_dir=artifacts_dir
    )

    after = _snapshot(FEATURES_DIR, ARTIFACTS_DIR)
    return before, after, features_dir, artifacts_dir


def test_redirected_run_leaves_committed_artifacts_untouched(isolated_run):
    """Verifies no file in the default feature or artifact directory was rewritten."""
    before, after, _, _ = isolated_run
    changed = sorted(path for path in before if before[path] != after.get(path))
    added = sorted(path for path in after if path not in before)
    assert changed == []
    assert added == []


def test_redirected_run_writes_every_output(isolated_run):
    """Verifies each orchestrator's outputs land in the redirected directories."""
    _, _, features_dir, artifacts_dir = isolated_run

    for name in ("selected_features.json", "user_clv.parquet", "user_segments.parquet"):
        assert (features_dir / name).exists(), name

    for name in (
        "propensity_lgbm.joblib",
        "propensity_metrics.json",
        "propensity_feature_importance.png",
        "propensity_lgbm_calibrated.joblib",
        "propensity_evaluation_summary.json",
        "propensity_lift_chart.png",
        "switching_lgbm.joblib",
        "switching_metrics.json",
        "switching_shap_summary.png",
        "clv_bgnbd.joblib",
        "clv_summary.json",
        "clv_distribution.png",
        "segmentation_kmeans.joblib",
        "segmentation_scaler.joblib",
        "segmentation_summary.json",
        "segmentation_pca_scatter.png",
    ):
        assert (artifacts_dir / name).exists(), name
