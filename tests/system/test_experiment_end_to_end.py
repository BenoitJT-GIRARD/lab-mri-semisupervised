"""One whole run of the protocol, on a synthetic dataset, writing the artefacts a reader opens.

Every other test in this suite replaces the training loop with a spy or a stub. This one does
not: it builds a miniature dataset on disk, runs `run_experiment` over it with two folds and
one epoch, and then opens the three files the README and the figures are built from. It is
the only test that can catch a pipeline that computes correctly and publishes nothing.

It is slow by construction — a real ResNet18 takes real gradient steps — and it needs the
ImageNet weights torchvision caches, so it skips with the command that fetches them rather
than failing on a machine that has never downloaded them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torchvision import models

from mri_semisupervised.config import ProtocolConfig, TrainingConfig
from mri_semisupervised.data.manifest import apply_duplicate_rules, build_manifest
from mri_semisupervised.protocol import experiment as experiment_module
from mri_semisupervised.protocol.experiment import CORRECTED, run_experiment
from tests.conftest import write_dataset

pytestmark = [pytest.mark.system, pytest.mark.slow]

#: The two arms that make the run meaningful with no pseudo-labels involved: the treatment
#: and the control it is read against. The pseudo-label arms need a pool large enough to
#: cluster, which a sixteen-image dataset does not give.
ARMS = ("supervised", "permuted_control")


def run_experiment_script():
    """The entry point the README names, imported by path: it is not an installed module."""
    import importlib.util

    from mri_semisupervised.config import PROJECT_ROOT

    spec = importlib.util.spec_from_file_location(
        "run_experiment_script", PROJECT_ROOT / "scripts" / "run_experiment.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _weights_are_cached() -> bool:
    """Whether torchvision can build the classifier without reaching the network."""
    url = models.ResNet18_Weights.IMAGENET1K_V1.url
    cache = Path(torch.hub.get_dir()) / "checkpoints" / url.rsplit("/", 1)[-1]
    return cache.exists()


@dataclass(frozen=True)
class _Features:
    """Stands in for `FeatureConfig`: the run reads one attribute of it, its cache path."""

    cache_path: Path

    def for_variant(self, **_kwargs) -> _Features:
        return self


def _feature_cache(manifest: pd.DataFrame, path: Path, seed: int = 0) -> None:
    """Two separable clusters, one row per distinct image, in the schema the run expects."""
    rng = np.random.default_rng(seed)
    ids = manifest["image_id"].drop_duplicates().to_list()
    centre = rng.normal(size=(2, 8))
    rows = []
    for index, image_id in enumerate(ids):
        vector = centre[index % 2] + rng.normal(scale=0.05, size=8)
        rows.append({"image_id": image_id, **{f"f_{i}": v for i, v in enumerate(vector)}})
    pd.DataFrame(rows).to_parquet(path, index=False)


@pytest.fixture()
def prepared_run(tmp_path: Path, monkeypatch) -> Path:
    """A manifest and a feature cache on disk, wired into the experiment module.

    Twelve images per labelled class, where the unit tier uses four: two stratified folds
    leave six for training, and the inner validation split has to keep a case of each class.
    """
    dataset = write_dataset(
        tmp_path / "mri_dataset_brain_cancer_oc", labelled_per_class=12, unlabelled_per_class=10
    )
    manifest = apply_duplicate_rules(
        build_manifest(dataset / "labelled", dataset / "unlabelled")
    )
    manifest_path = tmp_path / "dataset_manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)

    features_path = tmp_path / "features.parquet"
    _feature_cache(manifest, features_path)

    monkeypatch.setattr(experiment_module, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(experiment_module, "FeatureConfig", _Features(features_path))
    return tmp_path


@pytest.mark.skipif(not _weights_are_cached(), reason="run: uv run python scripts/warm_cache.py")
def test_a_whole_run_writes_the_three_files_every_figure_reads(prepared_run: Path) -> None:
    protocol = ProtocolConfig(n_splits=2, n_repeats=1, n_bootstrap=50, arms=ARMS)
    training = replace(
        TrainingConfig(),
        image_size=32,
        batch_size=4,
        epochs_weak=1,
        epochs_strong=1,
        early_stopping_patience=1,
    )

    result = run_experiment(
        mode=CORRECTED,
        protocol=protocol,
        training=training,
        output_root=prepared_run,
        run_name="system-tier",
        progress=False,
    )

    directory = prepared_run / "system-tier"
    per_fold = pd.read_parquet(directory / "per_fold.parquet")
    predictions = pd.read_parquet(directory / "predictions.parquet")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))

    # Every arm ran on every fold, and no fold went missing on the way to the file.
    assert set(per_fold["arm"]) == set(ARMS)
    assert len(per_fold) == len(ARMS) * protocol.n_splits * protocol.n_repeats
    # The predictions carry one row per evaluation image per arm, with a usable score.
    assert set(predictions["arm"]) == set(ARMS)
    assert predictions["y_score"].between(0.0, 1.0).all()
    assert set(predictions["y_true"].unique()) <= {0, 1}
    # The manifest is what makes the run quotable six months later.
    assert manifest["dataset_fingerprint"]
    assert manifest["protocol"]["n_splits"] == protocol.n_splits
    assert manifest["versions"]["torch"]
    assert result.mode == CORRECTED

    # And the last mile the script does: the markdown a reader opens first.
    summary = run_experiment_script().summarise(result, n_bootstrap=protocol.n_bootstrap)
    assert manifest["dataset_fingerprint"] in summary
    for arm in ARMS:
        assert arm in summary


@pytest.mark.skipif(not _weights_are_cached(), reason="run: uv run python scripts/warm_cache.py")
def test_the_same_seed_gives_the_same_numbers(prepared_run: Path) -> None:
    """A protocol whose conclusions move between two identical runs concludes nothing."""
    protocol = ProtocolConfig(n_splits=2, n_repeats=1, n_bootstrap=50, arms=("supervised",))
    training = replace(
        TrainingConfig(),
        image_size=32,
        batch_size=4,
        epochs_weak=1,
        epochs_strong=1,
        early_stopping_patience=1,
    )

    first = run_experiment(
        mode=CORRECTED,
        protocol=protocol,
        training=training,
        output_root=prepared_run,
        run_name="seed-a",
        progress=False,
    )
    second = run_experiment(
        mode=CORRECTED,
        protocol=protocol,
        training=training,
        output_root=prepared_run,
        run_name="seed-b",
        progress=False,
    )

    left = pd.read_parquet(first.directory / "per_fold.parquet").sort_values(["arm", "fold"])
    right = pd.read_parquet(second.directory / "per_fold.parquet").sort_values(["arm", "fold"])

    pd.testing.assert_series_equal(
        left["roc_auc"].reset_index(drop=True),
        right["roc_auc"].reset_index(drop=True),
        check_exact=False,
        atol=1e-6,
    )
