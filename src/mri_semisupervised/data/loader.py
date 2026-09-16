"""Inventory and quality control of the image set.

Depends on Pillow, pandas and numpy alone, so the exploration can start without importing
PyTorch.

Identity comes from :mod:`mri_semisupervised.data.manifest`: an image is what it contains,
not where it sits. That used to be the other way round, and it hid a leak.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile, UnidentifiedImageError

from mri_semisupervised.config import (
    CLASS_TO_INDEX,
    DATASET_ROOT,
    LABELLED_DIR,
    UNLABELLED_DIR,
)
from mri_semisupervised.data.manifest import content_id

# Do not silently accept truncated JPEGs: a corrupt file should be reported, not padded.
ImageFile.LOAD_TRUNCATED_IMAGES = False

VALID_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})


@dataclass(frozen=True)
class ImageRecord:
    """One row of the inventory."""

    image_id: str
    path: Path
    split: str  # "labeled" / "unlabeled"
    label_name: str | None  # "cancer" / "normal" / None
    label_index: int | None
    width: int
    height: int
    mode: str  # "L", "RGB", ...
    file_size_bytes: int


def _iter_image_paths(directory: Path) -> Iterable[Path]:
    if not directory.exists():
        return iter(())
    return (
        p
        for p in sorted(directory.rglob("*"))
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    )


def _read_image_meta(path: Path) -> tuple[int, int, str] | None:
    """Return ``(width, height, mode)``, or ``None`` when the image cannot be read."""
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
    """Walk the dataset and return ``(valid_records, corrupted_files)``.

    Images Pillow cannot open are set aside and returned separately rather than dropped
    quietly: a file that fails to load is a fact about the dataset, and it gets reported.
    """
    records: list[ImageRecord] = []
    corrupted: list[Path] = []

    labelled_root = dataset_root / LABELLED_DIR.name
    if labelled_root.exists():
        for class_dir in sorted(p for p in labelled_root.iterdir() if p.is_dir()):
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
                        image_id=content_id(path),
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

    unlabelled_root = dataset_root / UNLABELLED_DIR.name
    if unlabelled_root.exists():
        for path in _iter_image_paths(unlabelled_root):
            meta = _read_image_meta(path)
            if meta is None:
                corrupted.append(path)
                continue
            w, h, mode = meta
            records.append(
                ImageRecord(
                    image_id=content_id(path),
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
    """Turn the inventory into a typed frame."""
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
    """Per-image pixel statistics.

    Reads a sample, or everything when ``sample_size`` is ``None``. Returns a frame of
    ``image_id``, ``mean``, ``std``, ``min`` and ``max``.
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
    """Return the ``image_id`` whose mean or standard deviation sits more than
    ``z_threshold`` standard deviations away from the overall one.

    The threshold is deliberately high: at 4 sigma it flags the genuinely aberrant and
    leaves the merely unusual alone.
    """
    if stats.empty:
        return []
    mean_z = (stats["mean"] - stats["mean"].mean()) / (stats["mean"].std(ddof=0) + 1e-9)
    std_z = (stats["std"] - stats["std"].mean()) / (stats["std"].std(ddof=0) + 1e-9)
    mask = (mean_z.abs() > z_threshold) | (std_z.abs() > z_threshold)
    return stats.loc[mask, "image_id"].tolist()


def summarise_dataset(records: list[ImageRecord]) -> dict[str, object]:
    """Counts by pool and class, plus resolutions and colour modes."""
    df = records_to_dataframe(records)
    summary: dict[str, object] = {
        "total": len(df),
        "by_split": df.groupby("split", dropna=False).size().to_dict(),
        "by_label": df.groupby("label_name", dropna=False).size().to_dict(),
        "modes": df["mode"].value_counts().to_dict(),
        "resolutions": df.groupby(["width", "height"]).size().sort_values(ascending=False).to_dict(),
    }
    return summary


__all__ = [
    "LABELLED_DIR",
    "UNLABELLED_DIR",
    "VALID_EXTENSIONS",
    "ImageRecord",
    "compute_pixel_stats",
    "detect_outlier_ids",
    "discover_images",
    "records_to_dataframe",
    "summarise_dataset",
]
