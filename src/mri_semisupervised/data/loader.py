"""Chargement, inventaire et contrôle qualité du dataset BrainScanAI.

Ce module ne dépend que de Pillow / pandas / numpy : il peut être importé sans
PyTorch pour l'exploration initiale.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile, UnidentifiedImageError

from curelyticsia.config import (
    CLASS_TO_INDEX,
    DATASET_ROOT,
    LABELED_DIR,
    UNLABELED_DIR,
)

# Tolère les images JPEG légèrement tronquées (sans masquer une corruption).
ImageFile.LOAD_TRUNCATED_IMAGES = False

VALID_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})


@dataclass(frozen=True)
class ImageRecord:
    """Une ligne de l'inventaire."""

    image_id: str
    path: Path
    split: str  # "labeled" / "unlabeled"
    label_name: str | None  # "cancer" / "normal" / None
    label_index: int | None
    width: int
    height: int
    mode: str  # "L", "RGB", ...
    file_size_bytes: int


def _safe_hash(path: Path) -> str:
    """Hash MD5 court d'un chemin (id stable même si l'utilisateur renomme)."""
    return hashlib.md5(str(path).encode("utf-8"), usedforsecurity=False).hexdigest()[:12]


def _iter_image_paths(directory: Path) -> Iterable[Path]:
    if not directory.exists():
        return iter(())
    return (
        p
        for p in sorted(directory.rglob("*"))
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    )


def _read_image_meta(path: Path) -> tuple[int, int, str] | None:
    """Renvoie ``(width, height, mode)`` ou ``None`` si l'image est illisible."""
    try:
        with Image.open(path) as img:
            img.verify()
    except (UnidentifiedImageError, OSError, SyntaxError):
        return None
    try:
        with Image.open(path) as img:
            return img.width, img.height, img.mode
    except (UnidentifiedImageError, OSError, SyntaxError):
        return None


def discover_images(
    dataset_root: Path = DATASET_ROOT,
) -> tuple[list[ImageRecord], list[Path]]:
    """Scanne le dataset et renvoie ``(records_valides, fichiers_corrompus)``.

    Les images qui ne peuvent pas être ouvertes par Pillow sont écartées et
    listées séparément afin d'être documentées dans le notebook.
    """
    records: list[ImageRecord] = []
    corrupted: list[Path] = []

    labeled_root = dataset_root / "avec_labels"
    if labeled_root.exists():
        for class_dir in sorted(p for p in labeled_root.iterdir() if p.is_dir()):
            class_name = class_dir.name.lower()
            label_idx = CLASS_TO_INDEX.get(class_name)
            for path in _iter_image_paths(class_dir):
                meta = _read_image_meta(path)
                if meta is None:
                    corrupted.append(path)
                    continue
                w, h, mode = meta
                records.append(
                    ImageRecord(
                        image_id=_safe_hash(path),
                        path=path,
                        split="labeled",
                        label_name=class_name,
                        label_index=label_idx,
                        width=w,
                        height=h,
                        mode=mode,
                        file_size_bytes=path.stat().st_size,
                    )
                )

    unlabeled_root = dataset_root / "sans_label"
    if unlabeled_root.exists():
        for path in _iter_image_paths(unlabeled_root):
            meta = _read_image_meta(path)
            if meta is None:
                corrupted.append(path)
                continue
            w, h, mode = meta
            records.append(
                ImageRecord(
                    image_id=_safe_hash(path),
                    path=path,
                    split="unlabeled",
                    label_name=None,
                    label_index=None,
                    width=w,
                    height=h,
                    mode=mode,
                    file_size_bytes=path.stat().st_size,
                )
            )

    return records, corrupted


def records_to_dataframe(records: list[ImageRecord]) -> pd.DataFrame:
    """Convertit l'inventaire en DataFrame typé."""
    if not records:
        return pd.DataFrame(
            columns=[
                "image_id",
                "path",
                "split",
                "label_name",
                "label_index",
                "width",
                "height",
                "mode",
                "file_size_bytes",
            ]
        )
    df = pd.DataFrame([r.__dict__ for r in records])
    df["path"] = df["path"].astype(str)
    df["label_index"] = df["label_index"].astype("Int64")
    return df


def compute_pixel_stats(
    records: list[ImageRecord],
    sample_size: int | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Calcule des statistiques de pixels par image.

    Pour limiter le coût, seules les images d'un échantillon (ou toutes si
    ``sample_size is None``) sont lues. Renvoie un DataFrame avec
    ``image_id``, ``mean``, ``std``, ``min``, ``max``.
    """
    if not records:
        return pd.DataFrame(columns=["image_id", "mean", "std", "min", "max"])

    rng = np.random.default_rng(seed)
    if sample_size is not None and sample_size < len(records):
        idx = rng.choice(len(records), size=sample_size, replace=False)
        sampled = [records[int(i)] for i in idx]
    else:
        sampled = records

    rows = []
    for rec in sampled:
        with Image.open(rec.path) as img:
            arr = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
        rows.append(
            {
                "image_id": rec.image_id,
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "min": float(arr.min()),
                "max": float(arr.max()),
            }
        )
    return pd.DataFrame(rows)


def detect_outlier_ids(
    stats: pd.DataFrame,
    z_threshold: float = 4.0,
) -> list[str]:
    """Renvoie les ``image_id`` dont la moyenne ou l'écart-type s'écarte de plus de
    ``z_threshold`` écarts-types par rapport à la moyenne globale.

    Un seuil élevé (4σ) limite la suppression à des cas réellement aberrants.
    """
    if stats.empty:
        return []
    mean_z = (stats["mean"] - stats["mean"].mean()) / (stats["mean"].std(ddof=0) + 1e-9)
    std_z = (stats["std"] - stats["std"].mean()) / (stats["std"].std(ddof=0) + 1e-9)
    mask = (mean_z.abs() > z_threshold) | (std_z.abs() > z_threshold)
    return stats.loc[mask, "image_id"].tolist()


def summarise_dataset(records: list[ImageRecord]) -> dict[str, object]:
    """Résumé textuel : compteurs par split / classe, résolutions, modes."""
    df = records_to_dataframe(records)
    summary: dict[str, object] = {
        "total": int(len(df)),
        "by_split": df.groupby("split", dropna=False).size().to_dict(),
        "by_label": df.groupby("label_name", dropna=False).size().to_dict(),
        "modes": df["mode"].value_counts().to_dict(),
        "resolutions": df.groupby(["width", "height"]).size().sort_values(ascending=False).to_dict(),
    }
    return summary


__all__ = [
    "ImageRecord",
    "LABELED_DIR",
    "UNLABELED_DIR",
    "VALID_EXTENSIONS",
    "compute_pixel_stats",
    "detect_outlier_ids",
    "discover_images",
    "records_to_dataframe",
    "summarise_dataset",
]
