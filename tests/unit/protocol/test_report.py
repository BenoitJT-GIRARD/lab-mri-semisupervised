"""The summary page is rendered from the run's own artefacts, and says which dataset.

The renderer lives in the package rather than in the script that calls it, so that a change
to the page can be replayed over every past run. These tests pin what the page must carry.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mri_semisupervised.protocol.report import (
    load_result,
    published_arms,
    runs,
    summarise,
)

ARMS = ("supervised", "permuted_control")


def _write_run(
    directory: Path, *, folds: int = 4, arms: tuple[str, ...] = ARMS, mode: str = "corrected"
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    predictions = []
    for arm in arms:
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
                "mode": mode,
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


def test_a_variant_run_is_compared_to_the_reference_fold_by_fold(tmp_path: Path) -> None:
    """The README quoted a difference between two runs that no artefact carried.

    The equalised run is the corrected protocol with one preprocessing change, so the number
    that matters is the paired difference between the two. It belongs on the page of the
    variant, where a reader can see both columns at once.
    """
    _write_run(tmp_path / "corrected")
    _write_run(tmp_path / "equalized", mode="equalized")

    page = summarise(load_result(tmp_path / "equalized"), n_bootstrap=20)

    assert "Against the `corrected` run, paired over the 4 shared folds" in page
    assert "| supervised | 0.900 | 0.900 | +0.000 |" in page


def test_the_reference_run_is_not_compared_to_itself(run_directory: Path) -> None:
    assert "Against the `corrected` run" not in summarise(
        load_result(run_directory), n_bootstrap=20
    )


def test_a_variant_alone_on_disk_renders_without_the_comparison(tmp_path: Path) -> None:
    """A repository that publishes only the variant still gets a page, minus that section."""
    _write_run(tmp_path / "equalized", mode="equalized")

    page = summarise(load_result(tmp_path / "equalized"), n_bootstrap=20)

    assert "Per-arm means across folds" in page
    assert "Against the `corrected` run" not in page


def test_the_published_arm_table_crosses_the_two_runs_that_hold_them() -> None:
    """The eight arms of the README come from two runs, so no run's summary carries them."""
    table = published_arms()

    assert set(table["run"]) == {"corrected", "corrected-stageb"}
    assert len(table) == table["arm"].nunique() == 8
    assert table["mean_roc_auc"].is_monotonic_decreasing
    assert set(table["evaluation_images"]) == {99}


def test_an_arm_measured_in_both_runs_is_published_once(monkeypatch, tmp_path: Path) -> None:
    """`corrected-stageb` re-measures nothing today; the day it does, the table stays sane."""
    _write_run(tmp_path / "corrected")
    _write_run(tmp_path / "corrected-stageb")
    monkeypatch.setattr("mri_semisupervised.protocol.report.EXPERIMENTS_DIR", tmp_path)

    table = published_arms()

    assert list(table["arm"]) == ["supervised", "permuted_control"]
    assert set(table["run"]) == {"corrected"}


def test_the_arm_table_is_empty_when_no_published_run_is_on_disk(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("mri_semisupervised.protocol.report.EXPERIMENTS_DIR", tmp_path)

    assert published_arms().empty
