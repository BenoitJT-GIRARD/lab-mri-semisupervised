"""The one command a reader runs before believing any number: is this the same dataset?

The repository holds no images, so this script is the whole answer to "did I unpack the right
archive". Its comparison is what these tests pin.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from mri_semisupervised.config import PROJECT_ROOT

SPEC = importlib.util.spec_from_file_location(
    "check_data", PROJECT_ROOT / "scripts" / "check_data.py"
)
check_data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check_data)


def _run(directory: Path, fingerprint: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "per_fold.parquet").write_bytes(b"")
    (directory / "manifest.json").write_text(
        json.dumps({"run_id": directory.name, "dataset_fingerprint": fingerprint}),
        encoding="utf-8",
    )


def test_it_reads_the_fingerprint_of_every_published_run(tmp_path: Path, monkeypatch) -> None:
    _run(tmp_path / "corrected", "aaa")
    _run(tmp_path / "legacy", "aaa")
    monkeypatch.setattr(check_data, "runs", lambda: sorted(tmp_path.iterdir()))

    assert check_data.recorded_fingerprints() == {"corrected": "aaa", "legacy": "aaa"}


def test_a_run_without_a_manifest_is_not_counted(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "half-written").mkdir()
    monkeypatch.setattr(check_data, "runs", lambda: sorted(tmp_path.iterdir()))

    assert check_data.recorded_fingerprints() == {}


def test_the_expected_layout_names_the_two_folders_the_archive_unpacks_into() -> None:
    """A reader who cannot name the folders cannot rebuild the tree."""
    assert "labelled/" in check_data.EXPECTED
    assert "unlabelled/" in check_data.EXPECTED
    assert "MRI_DATA_DIR" in check_data.EXPECTED


def test_it_refuses_a_directory_that_holds_no_dataset(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(check_data, "LABELLED_DIR", tmp_path / "labelled")
    monkeypatch.setattr(check_data, "UNLABELLED_DIR", tmp_path / "unlabelled")
    monkeypatch.setattr(check_data, "DATASET_ROOT", tmp_path)

    assert check_data.main() == 1
    assert "does not exist" in capsys.readouterr().out


@pytest.mark.parametrize("published", [{"corrected": "bbb"}, {"corrected": "aaa", "legacy": "bbb"}])
def test_a_fingerprint_that_disagrees_is_a_failure(
    tmp_path: Path, monkeypatch, capsys, published: dict[str, str]
) -> None:
    """The numbers in the README then describe another dataset, and the script says so."""
    from tests.conftest import write_dataset

    dataset = write_dataset(tmp_path / "raw" / "mri_dataset_brain_cancer_oc", corrupt=False)
    monkeypatch.setattr(check_data, "LABELLED_DIR", dataset / "labelled")
    monkeypatch.setattr(check_data, "UNLABELLED_DIR", dataset / "unlabelled")
    monkeypatch.setattr(check_data, "DATASET_ROOT", dataset)
    monkeypatch.setattr(check_data, "recorded_fingerprints", lambda: published)

    assert check_data.main() == 1
    assert "do not describe these images" in capsys.readouterr().out
