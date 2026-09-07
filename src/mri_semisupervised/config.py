"""Central configuration: paths, seeds, hyperparameters.

Every constant lives here so that the notebooks and the scripts stay thin and
reproducible.

**The data directory is configurable.** ``MRI_DATA_DIR`` moves the dataset off the
repository tree, which matters when the checkout sits on a synchronised drive: reading
1 400 images per epoch from a sync mount starves the GPU. The default keeps everything
under ``data/`` so a fresh clone works with no configuration at all.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT: Path = _project_root()

_env_data_dir = os.environ.get("MRI_DATA_DIR", "").strip()
DATA_DIR: Path = Path(_env_data_dir) if _env_data_dir else PROJECT_ROOT / "data"

RAW_DIR: Path = DATA_DIR / "raw"
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
EXPERIMENTS_DIR: Path = REPORTS_DIR / "experiments"

DATASET_NAME: str = "mri_dataset_brain_cancer_oc"
DATASET_ROOT: Path = RAW_DIR / DATASET_NAME

LABELED_DIR: Path = DATASET_ROOT / "avec_labels"
UNLABELED_DIR: Path = DATASET_ROOT / "sans_label"

MANIFEST_PATH: Path = PROCESSED_DIR / "dataset_manifest.parquet"

CLASSES: tuple[str, ...] = ("normal", "cancer")
CLASS_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(CLASSES)}
POSITIVE_CLASS: str = "cancer"
POSITIVE_INDEX: int = CLASS_TO_INDEX[POSITIVE_CLASS]

SEED: int = 42

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
INPUT_SIZE: int = 224


@dataclass(frozen=True)
class FeatureConfig:
    """Parameters of the feature extraction."""

    backbone: str = "resnet50"
    output_dim: int = 2048
    batch_size: int = 32
    num_workers: int = 0  # Windows: 0 avoids the torch pickling trouble
    cache_path: Path = field(default_factory=lambda: PROCESSED_DIR / "features_resnet50.parquet")


@dataclass(frozen=True)
class ClusteringConfig:
    """Parameters of the exploratory clustering."""

    n_clusters: int = 2
    pca_variance: float = 0.95
    pca_max_components: int = 100
    tsne_perplexity: float = 30.0
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters of the supervised and semi-supervised arms.

    Every field here is read by the protocol. A field that nothing reads is a
    configuration that lies about the experiment, so it does not stay.
    """

    architecture: str = "resnet18"
    num_classes: int = 2
    image_size: int = INPUT_SIZE
    batch_size: int = 16
    epochs_weak: int = 3
    epochs_strong: int = 20
    learning_rate: float = 1e-4
    finetune_lr_factor: float = 0.5
    weight_decay: float = 1e-4
    # Held out from the training fold to select the checkpoint and the decision
    # threshold. Never taken from the test fold.
    inner_val_fraction: float = 0.25
    early_stopping_patience: int = 5


@dataclass(frozen=True)
class ProtocolConfig:
    """How the evaluation is run."""

    n_splits: int = 5
    n_repeats: int = 5
    seed: int = SEED
    n_bootstrap: int = 2000
    arms: tuple[str, ...] = ("supervised", "semi_supervised", "permuted_control")


def ensure_dirs() -> None:
    """Create the output directories if they do not exist yet."""
    for directory in (INTERIM_DIR, PROCESSED_DIR, FIGURES_DIR, EXPERIMENTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def device() -> str:
    """Return ``"cuda"`` when an NVIDIA GPU is available, else ``"cpu"``."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_global_seeds(seed: int = SEED) -> None:
    """Seed numpy, random and torch, and ask cuDNN for deterministic kernels.

    Determinism costs a little throughput and buys a reproducible number. On an
    experiment whose whole point is that its figures can be checked, that is the right
    trade.
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
