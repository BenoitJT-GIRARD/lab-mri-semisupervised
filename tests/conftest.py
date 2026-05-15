"""Fixtures partagées par les tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _write_synthetic_image(path: Path, value: int = 128, size: int = 32) -> None:
    arr = np.full((size, size, 3), value, dtype=np.uint8)
    Image.fromarray(arr).save(path, format="JPEG", quality=80)


@pytest.fixture()
def synthetic_dataset(tmp_path: Path) -> Path:
    """Construit une mini-arborescence ``avec_labels/{cancer,normal}`` + ``sans_label/``.

    - 4 images cancer (intensité 200)
    - 4 images normal (intensité 50)
    - 6 images non labellisées : 3 « cancer-like », 3 « normal-like »
    - 1 fichier corrompu pour vérifier la robustesse
    """
    root = tmp_path / "mri_dataset_brain_cancer_oc"
    (root / "avec_labels" / "cancer").mkdir(parents=True)
    (root / "avec_labels" / "normal").mkdir(parents=True)
    (root / "sans_label").mkdir(parents=True)

    for i in range(4):
        _write_synthetic_image(root / "avec_labels" / "cancer" / f"c_{i}.jpg", value=200)
    for i in range(4):
        _write_synthetic_image(root / "avec_labels" / "normal" / f"n_{i}.jpg", value=50)
    for i in range(3):
        _write_synthetic_image(root / "sans_label" / f"u_cancer_{i}.jpg", value=190)
    for i in range(3):
        _write_synthetic_image(root / "sans_label" / f"u_normal_{i}.jpg", value=60)

    # Fichier corrompu intentionnel
    (root / "sans_label" / "broken.jpg").write_bytes(b"not a jpeg file")

    return root


@pytest.fixture()
def synthetic_features() -> tuple[np.ndarray, np.ndarray]:
    """Renvoie ``(features, truth)`` simulant deux clusters bien séparés.

    50 points par cluster en 16 dimensions ; ``truth`` ne couvre que 20 % des
    points (les autres sont NaN, simulant le jeu non labellisé).
    """
    rng = np.random.default_rng(0)
    cluster_a = rng.normal(loc=-2.0, scale=0.5, size=(50, 16))
    cluster_b = rng.normal(loc=+2.0, scale=0.5, size=(50, 16))
    features = np.vstack([cluster_a, cluster_b]).astype(np.float32)

    truth = np.full(100, np.nan, dtype=float)
    truth[:10] = 0  # cluster_a → label 0
    truth[50:60] = 1  # cluster_b → label 1
    return features, truth
