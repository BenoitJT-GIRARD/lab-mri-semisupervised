"""Configuration centralisée : chemins, graines, hyperparamètres.

Toutes les constantes du projet sont définies ici afin que les notebooks et les
scripts restent fins et reproductibles.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT: Path = _project_root()
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
INFOS_DIR: Path = PROJECT_ROOT / "infos"

DATASET_NAME: str = "mri_dataset_brain_cancer_oc"
DATASET_ZIP: Path = INFOS_DIR / "03b_mri_dataset_brain_cancer_oc.zip"
DATASET_ROOT: Path = RAW_DIR / DATASET_NAME

LABELED_DIR: Path = DATASET_ROOT / "avec_labels"
UNLABELED_DIR: Path = DATASET_ROOT / "sans_label"

CLASSES: tuple[str, ...] = ("normal", "cancer")
CLASS_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(CLASSES)}

SEED: int = 42

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
INPUT_SIZE: int = 224


@dataclass(frozen=True)
class FeatureConfig:
    """Paramètres d'extraction de features."""

    backbone: str = "resnet50"
    output_dim: int = 2048
    batch_size: int = 32
    num_workers: int = 0  # Windows : 0 évite les soucis de pickling avec torch
    cache_path: Path = field(default_factory=lambda: PROCESSED_DIR / "features_resnet50.parquet")


@dataclass(frozen=True)
class ClusteringConfig:
    """Paramètres pour la phase clustering exploratoire."""

    n_clusters: int = 2
    pca_variance: float = 0.95
    pca_max_components: int = 100
    tsne_perplexity: float = 30.0
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    weak_labels_path: Path = field(default_factory=lambda: PROCESSED_DIR / "weak_labels.csv")


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparamètres pour l'entraînement CNN supervisé / semi-supervisé."""

    architecture: str = "resnet18"
    num_classes: int = 2
    image_size: int = INPUT_SIZE
    batch_size: int = 16
    epochs_weak: int = 3
    epochs_strong: int = 6
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    test_size: float = 0.5  # 50/50 sur les 100 IRM fortement labellisees
    val_size: float = 0.0
    early_stopping_patience: int = 4
    cv_folds: int = 5


def ensure_dirs() -> None:
    """Crée les dossiers de sortie s'ils n'existent pas."""
    for directory in (INTERIM_DIR, PROCESSED_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def device() -> str:
    """Renvoie ``"cuda"`` si une GPU NVIDIA est disponible, sinon ``"cpu"``."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_global_seeds(seed: int = SEED) -> None:
    """Fixe les graines numpy / random / torch pour reproductibilité."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
