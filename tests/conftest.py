"""What every tier of this suite shares: where the repository is, and how a test is skipped.

The suite is read by tier — ``unit/``, ``integration/``, ``system/`` — and each tier states
in its own ``conftest.py`` what it forbids. This file holds only what all three need.

**A skip names the command that would run the test.** A skip whose reason is a condition —
``"needs the database"``, ``"no model on disk"`` — teaches a reader that the test is
unrunnable. A skip that says ``run: docker compose up -d postgres`` teaches them how to run
it. Use :func:`skip_unless` and the message writes itself.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mri_semisupervised.config import PROJECT_ROOT as ROOT

# The root is NOT recomputed here: the import above takes it from the package, which already
# decides where the repository is. A second answer to that question is a second answer.


def skip_unless(condition: bool, *, command: str) -> pytest.MarkDecorator:
    """Skip the test unless the condition holds, naming the command that makes it hold.

    @skip_unless(port_is_open(5432), command="docker compose up -d postgres")
    def test_the_api_writes_its_prediction_to_the_database(): ...
    """
    return pytest.mark.skipif(not condition, reason=f"run: {command}")


def port_is_open(port: int, host: str = "127.0.0.1", timeout: float = 0.25) -> bool:
    """Is something listening? Asked once at collection, never retried in a loop."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def env_is_set(name: str) -> bool:
    """Is this environment variable set to something?

    A test gated on a variable that no workflow and no documented command ever sets skips on
    every checkout, and the count of tests it belongs to is a count of tests nobody runs.
    """
    return bool(os.environ.get(name))


@pytest.fixture(scope="session")
def root() -> Path:
    """The repository root, for a test that must open a published artefact."""
    return ROOT

# --- Fixtures of this repository ------------------------------------------


def _write_synthetic_image(path: Path, value: int = 128, size: int = 32, tag: int = 0) -> None:
    """Write a flat image of the given intensity, made unique by ``tag``.

    The four corner pixels carry ``tag``, which keeps the two intensity groups clearly
    apart while giving every file distinct pixels. Flat images would share a content hash,
    and identity is exactly what this suite has to be able to test — the repository's whole
    subject is two files that were treated as two images because their paths differed.
    """
    arr = np.full((size, size, 3), value, dtype=np.uint8)
    arr[0, 0] = arr[0, -1] = arr[-1, 0] = arr[-1, -1] = tag % 256
    Image.fromarray(arr).save(path, format="PNG")


def write_dataset(
    root: Path, *, labelled_per_class: int = 4, unlabelled_per_class: int = 3, corrupt: bool = True
) -> Path:
    """Write a ``labelled/{cancer,normal}`` + ``unlabelled/`` tree, sized by the caller.

    A unit test wants the smallest tree that exercises a rule; the system tier wants one
    large enough for a stratified fold and an inner validation split inside it. Same shape,
    same intensities, one function.
    """
    (root / "labelled" / "cancer").mkdir(parents=True)
    (root / "labelled" / "normal").mkdir(parents=True)
    (root / "unlabelled").mkdir(parents=True)

    for i in range(labelled_per_class):
        _write_synthetic_image(root / "labelled" / "cancer" / f"c_{i}.png", value=200, tag=1 + i)
        _write_synthetic_image(root / "labelled" / "normal" / f"n_{i}.png", value=50, tag=101 + i)
    for i in range(unlabelled_per_class):
        _write_synthetic_image(root / "unlabelled" / f"u_cancer_{i}.png", value=190, tag=51 + i)
        _write_synthetic_image(root / "unlabelled" / f"u_normal_{i}.png", value=60, tag=151 + i)

    if corrupt:
        # A deliberately corrupt file: the inventory must report it, not die on it.
        (root / "unlabelled" / "broken.jpg").write_bytes(b"not a jpeg file")
    return root


@pytest.fixture()
def synthetic_dataset(tmp_path: Path) -> Path:
    """The miniature tree: 4 images per labelled class, 3 per unlabelled group, 1 corrupt."""
    root = tmp_path / "mri_dataset_brain_cancer_oc"
    write_dataset(root)

    return root


@pytest.fixture()
def synthetic_features() -> tuple[np.ndarray, np.ndarray]:
    """Return ``(features, truth)`` for two well-separated clusters.

    Fifty points per cluster in sixteen dimensions. ``truth`` covers only 20% of them; the
    rest are NaN, standing in for the unlabelled pool.
    """
    rng = np.random.default_rng(0)
    cluster_a = rng.normal(loc=-2.0, scale=0.5, size=(50, 16))
    cluster_b = rng.normal(loc=+2.0, scale=0.5, size=(50, 16))
    features = np.vstack([cluster_a, cluster_b]).astype(np.float32)

    truth = np.full(100, np.nan, dtype=float)
    truth[:10] = 0  # cluster_a → label 0
    truth[50:60] = 1  # cluster_b → label 1
    return features, truth
