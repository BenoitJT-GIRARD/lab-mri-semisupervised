"""The summary page is rendered from the run's own artefacts, and says which dataset.

The renderer lives in the package rather than in the script that calls it, so that a change
to the page can be replayed over every past run. These tests pin what the page must carry.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mri_semisupervised.protocol.report import load_result, runs, summarise

ARMS = ("supervised", "permuted_control")


def _write_run(directory: Path, *, folds: int = 4) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    predictions = []
    for arm in ARMS:
        for fold in range(folds):
            rows.append(
                {
                    "arm": arm,
                    "fold": fold,
                    "roc_auc": 0.9 if arm == "supervised" else 0.8,
                    "pr_auc": 0.9,
                    "recall_positive": 0.8,
                    "f1_macro": 0.8,
                    "accuracy": 0.8,
                }
            )
        for index in range(8):
            predictions.append(
                {
                    "arm": arm,
                    "image_id": f"i{index}",
                    "y_true": index % 2,
                    "y_score": 0.2 + 0.1 * (index % 2) + (0.3 if arm == "supervised" else 0.0),
                }
            )
    pd.DataFrame(rows).to_parquet(directory / "per_fold.parquet", index=False)
    pd.DataFrame(predictions).to_parquet(directory / "predictions.parquet", index=False)
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "trial-1",
                "mode": "corrected",
                "dataset_fingerprint": "abc123",
                "evaluation_images": 8,
                "unlabelled_pool": 40,
                "protocol": {"n_splits": 2, "n_repeats": 2, "n_bootstrap": 20},
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture()
def run_directory(tmp_path: Path) -> Path:
    _write_run(tmp_path / "trial")
    return tmp_path / "trial"


def test_the_page_names_the_dataset_it_describes(run_directory: Path) -> None:
    """The manifest beside it carried the fingerprint; the page a reader opens carried none."""
    page = summarise(load_result(run_directory), n_bootstrap=20)

    assert "abc123" in page
    assert "evaluation images: 8" in page
    assert "folds: 4 (2 splits x 2 repeats)" in page


def test_the_page_names_the_script_that_writes_it(run_directory: Path) -> None:
    head = summarise(load_result(run_directory), n_bootstrap=20).splitlines()[:10]

    assert any("scripts/rebuild_summaries.py" in line for line in head)


def test_every_arm_gets_a_row_in_both_tables(run_directory: Path) -> None:
    page = summarise(load_result(run_directory), n_bootstrap=20)

    for arm in ARMS:
        assert page.count(f"| {arm} |") >= 2
    assert "Paired differences across the shared folds" in page


def test_a_run_without_fold_metadata_still_renders(run_directory: Path) -> None:
    """`folds.parquet` is optional: the clustering section disappears, the page does not."""
    result = load_result(run_directory)

    assert result.folds_meta.empty
    assert "Clustering method chosen" not in summarise(result, n_bootstrap=20)


def test_the_published_runs_are_the_ones_on_disk() -> None:
    names = {directory.name for directory in runs()}

    assert {"corrected", "legacy", "corrected-stageb"} <= names
