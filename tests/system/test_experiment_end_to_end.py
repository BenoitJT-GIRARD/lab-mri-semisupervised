"""One whole run, launched the way the README launches it, on a dataset built for the test.

Every other test in this suite replaces the training loop with a spy or a stub. This one runs
`scripts/run_experiment.py` in its own process, over a synthetic dataset of twenty-four
images, and then opens the four files a reader opens: the per-fold metrics, the raw
predictions, the run manifest and the summary page. It is the only test that can catch a
pipeline that computes correctly and publishes nothing.

It is slow by construction, since a real ResNet18 takes real gradient steps, and it needs the
ImageNet weights torchvision caches: it skips with the command that fetches them rather than
failing on a machine that has never downloaded them.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torchvision import models

from mri_semisupervised.config import PROJECT_ROOT, ProtocolConfig
from mri_semisupervised.data.manifest import apply_duplicate_rules, build_manifest
from mri_semisupervised.protocol.experiment import CORRECTED
from tests.conftest import write_dataset

pytestmark = [pytest.mark.system, pytest.mark.slow]

#: The two arms that make the run meaningful with no pseudo-labels involved: the treatment
#: and the control it is read against. The pseudo-label arms need a pool large enough to
#: cluster, which a twenty-image pool does not give.
ARMS = ("supervised", "permuted_control")


def _weights_are_cached() -> bool:
    """Whether torchvision can build the classifier without reaching the network."""
    url = models.ResNet18_Weights.IMAGENET1K_V1.url
    cache = Path(torch.hub.get_dir()) / "checkpoints" / url.rsplit("/", 1)[-1]
    return cache.exists()


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
def prepared_run(tmp_path: Path) -> Path:
    """A dataset, a manifest and a feature cache laid out the way `MRI_DATA_DIR` expects.

    Twelve images per labelled class, where the unit tier uses four: two stratified folds
    leave six for training, and the inner validation split has to keep a case of each class.
    Everything lands under one directory, so the subprocess needs one variable and no patch.
    """
    dataset = write_dataset(
        tmp_path / "raw" / "mri_dataset_brain_cancer_oc",
        labelled_per_class=12,
        unlabelled_per_class=10,
    )
    manifest = apply_duplicate_rules(
        build_manifest(dataset / "labelled", dataset / "unlabelled")
    )
    processed = tmp_path / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(processed / "dataset_manifest.parquet", index=False)
    _feature_cache(manifest, processed / "features_resnet50.parquet")
    return tmp_path


#: Two folds, one repeat, one epoch and 32-pixel images: the smallest run that still goes
#: through every stage. The published runs use five folds, five repeats and 224 pixels.
PROTOCOL = ProtocolConfig(n_splits=2, n_repeats=1, n_bootstrap=50, arms=ARMS)


def _run(root: Path, *, arms: tuple[str, ...], out_name: str) -> None:
    """Launch the experiment the way the README does, in its own process."""
    done = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_experiment.py"),
            "--folds", str(PROTOCOL.n_splits),
            "--repeats", str(PROTOCOL.n_repeats),
            "--epochs-strong", "1",
            "--image-size", "32",
            "--arms", ",".join(arms),
            "--out-name", out_name,
            "--output-root", str(root),
        ],
        cwd=PROJECT_ROOT,
        env={**os.environ, "MRI_DATA_DIR": str(root), "PYTHONIOENCODING": "utf-8"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, (done.stdout[-2000:] + done.stderr[-2000:])


@pytest.mark.skipif(not _weights_are_cached(), reason="run: uv run python scripts/warm_cache.py")
def test_a_whole_run_writes_the_files_every_figure_reads(prepared_run: Path) -> None:
    """The command the README gives, run as a command: a subprocess, its own environment."""
    _run(prepared_run, arms=ARMS, out_name="system-tier")

    protocol = PROTOCOL
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
    assert manifest["mode"] == CORRECTED

    # And the page the script writes beside them, which is what a reader opens first.
    summary = (directory / "summary.md").read_text(encoding="utf-8")
    assert manifest["dataset_fingerprint"] in summary
    for arm in ARMS:
        assert arm in summary


@pytest.mark.skipif(not _weights_are_cached(), reason="run: uv run python scripts/warm_cache.py")
def test_the_same_seed_gives_the_same_numbers(prepared_run: Path) -> None:
    """A protocol whose conclusions move between two identical runs concludes nothing."""
    for name in ("seed-a", "seed-b"):
        _run(prepared_run, arms=("supervised",), out_name=name)

    left = pd.read_parquet(prepared_run / "seed-a" / "per_fold.parquet")
    right = pd.read_parquet(prepared_run / "seed-b" / "per_fold.parquet")

    pd.testing.assert_series_equal(
        left.sort_values("fold")["roc_auc"].reset_index(drop=True),
        right.sort_values("fold")["roc_auc"].reset_index(drop=True),
        check_exact=False,
        atol=1e-6,
    )
