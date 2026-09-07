"""Fixtures shared across the test suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


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


@pytest.fixture()
def synthetic_dataset(tmp_path: Path) -> Path:
    """Build a miniature ``avec_labels/{cancer,normal}`` + ``sans_label/`` tree.

    - 4 cancer images, intensity 200
    - 4 normal images, intensity 50
    - 6 unlabelled images: 3 cancer-like, 3 normal-like
    - 1 corrupt file, so the loader's reporting path is exercised
    """
    root = tmp_path / "mri_dataset_brain_cancer_oc"
    (root / "avec_labels" / "cancer").mkdir(parents=True)
    (root / "avec_labels" / "normal").mkdir(parents=True)
    (root / "sans_label").mkdir(parents=True)

    for i in range(4):
        _write_synthetic_image(root / "avec_labels" / "cancer" / f"c_{i}.png", value=200, tag=1 + i)
    for i in range(4):
        _write_synthetic_image(root / "avec_labels" / "normal" / f"n_{i}.png", value=50, tag=11 + i)
    for i in range(3):
        _write_synthetic_image(root / "sans_label" / f"u_cancer_{i}.png", value=190, tag=21 + i)
    for i in range(3):
        _write_synthetic_image(root / "sans_label" / f"u_normal_{i}.png", value=60, tag=31 + i)

    # A deliberately corrupt file: the loader must report it, not skip it silently.
    (root / "sans_label" / "broken.jpg").write_bytes(b"not a jpeg file")

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
