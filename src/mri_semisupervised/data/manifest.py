"""Identity of an image, and what follows from it.

The pipeline used to identify an image by a hash of its **path**. Two byte-identical
files stored in two folders therefore received two different identifiers, and the guard
that was supposed to keep the labelled images out of the unlabelled pool — an exclusion
on the folder name — could not see that 31 of the 99 distinct evaluation images were already
sitting in that pool.

Here an image is identified by its **content**. Everything else in this module is the
consequence: which duplicates exist, which copies may take part in training, and a
fingerprint that ties a published number to the exact data that produced it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from mri_semisupervised.config import CLASS_TO_INDEX, LABELED_DIR, UNLABELED_DIR

VALID_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp"})

LABELLED = "labelled"
UNLABELLED = "unlabelled"


def byte_id(path: Path) -> str:
    """SHA-256 of the file as stored."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def content_id(path: Path) -> str:
    """SHA-256 of the decoded pixels, in a canonical form.

    Greyscale and native size: a re-encoded copy, or the same scan saved by another tool,
    lands on the same identifier as the original. That is what ``byte_id`` cannot do, and
    it is the property the leak detection needs.
    """
    with Image.open(path) as img:
        array = np.asarray(img.convert("L"))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _iter_images(directory: Path) -> Iterable[Path]:
    if not directory.exists():
        return iter(())
    return (
        path
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    )


def build_manifest(
    labelled_dir: Path | None = None,
    unlabelled_dir: Path | None = None,
) -> pd.DataFrame:
    """Inventory every image with its content identity, its pool and its label."""
    labelled_dir = labelled_dir or LABELED_DIR
    unlabelled_dir = unlabelled_dir or UNLABELED_DIR

    rows: list[dict[str, object]] = []
    for pool, directory in ((LABELLED, labelled_dir), (UNLABELLED, unlabelled_dir)):
        for path in _iter_images(directory):
            label = path.parent.name if pool == LABELLED else None
            rows.append(
                {
                    "image_id": content_id(path),
                    "byte_sha256": byte_id(path),
                    "path": str(path),
                    "pool": pool,
                    "label": label,
                    "label_index": CLASS_TO_INDEX.get(label) if label else None,
                }
            )

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        return manifest

    counts = manifest["image_id"].value_counts()
    manifest["duplicate_group_size"] = manifest["image_id"].map(counts).astype(int)
    return manifest.sort_values(["pool", "path"]).reset_index(drop=True)


def apply_duplicate_rules(manifest: pd.DataFrame) -> pd.DataFrame:
    """Decide which rows may take part in training, and say why.

    Three rules, each one a decision rather than a detail:

    * a labelled image that also sits in the unlabelled pool leaves **that pool**, never
      the evaluation set — dropping it from the evaluation would amount to choosing the
      test set after having looked at it;
    * redundant copies inside the unlabelled pool are reduced to one, because otherwise
      they silently weight the pre-training;
    * a repeated image inside the labelled pool is reduced to one as well: kept twice, it
      would sit on both sides of a fold, which no protocol survives. The evaluation set is
      therefore the set of **distinct** labelled images;
    * the same content carrying **two different labels** is fatal, and stops the build.
      That is not a duplicate, it is a contradiction in the annotation, and nothing
      downstream can be trusted while it stands.
    """
    if manifest.empty:
        return manifest.assign(kept_for_training=pd.Series(dtype=bool), exclusion_reason=None)

    labelled = manifest[manifest["pool"] == LABELLED]
    conflicting = labelled.groupby("image_id")["label"].nunique()
    conflicting = conflicting[conflicting > 1]
    if len(conflicting):
        raise ValueError(
            f"{len(conflicting)} image(s) carry two different labels in the labelled pool: "
            f"{list(conflicting.index[:3])}. Annotation contradicts itself."
        )

    labelled_ids = set(labelled["image_id"])
    ruled = manifest.copy()
    ruled["kept_for_training"] = True
    ruled["exclusion_reason"] = None

    repeated = labelled.duplicated(subset="image_id", keep="first")
    ruled.loc[labelled.index[repeated], ["kept_for_training", "exclusion_reason"]] = [
        False,
        "repeated_labelled_copy",
    ]

    leaked = (ruled["pool"] == UNLABELLED) & ruled["image_id"].isin(labelled_ids)
    ruled.loc[leaked, ["kept_for_training", "exclusion_reason"]] = [False, "copy_of_labelled"]

    unlabelled_kept = ruled[(ruled["pool"] == UNLABELLED) & ruled["kept_for_training"]]
    redundant = unlabelled_kept.duplicated(subset="image_id", keep="first")
    redundant_index = unlabelled_kept.index[redundant]
    ruled.loc[redundant_index, ["kept_for_training", "exclusion_reason"]] = [
        False,
        "redundant_copy",
    ]

    return ruled


def dataset_fingerprint(manifest: pd.DataFrame) -> str:
    """One hash standing for the whole dataset, so a report names its data."""
    ordered = sorted(manifest["image_id"].tolist())
    digest = hashlib.sha256()
    for image_id in ordered:
        digest.update(image_id.encode())
    return digest.hexdigest()[:16]


def summarise(manifest: pd.DataFrame) -> dict[str, object]:
    """The numbers worth printing after a build, and worth asserting in a test."""
    ruled = manifest if "kept_for_training" in manifest else apply_duplicate_rules(manifest)
    labelled = ruled[ruled["pool"] == LABELLED]
    labelled_ids = set(labelled["image_id"])
    unlabelled = ruled[ruled["pool"] == UNLABELLED]
    leaked = unlabelled[unlabelled["image_id"].isin(labelled_ids)]

    return {
        "files": len(ruled),
        "distinct_images": int(ruled["image_id"].nunique()),
        "labelled_files": len(labelled),
        "evaluation_images": int(labelled["image_id"].nunique()),
        "repeated_labelled_copies_dropped": int(
            (ruled["exclusion_reason"] == "repeated_labelled_copy").sum()
        ),
        "unlabelled_files": len(unlabelled),
        "labelled_also_in_unlabelled": int(leaked["image_id"].nunique()),
        "leaked_by_class": (
            labelled[labelled["image_id"].isin(set(leaked["image_id"]))]
            .drop_duplicates("image_id")["label"]
            .value_counts()
            .to_dict()
        ),
        "evaluation_class_balance": (
            labelled.drop_duplicates("image_id")["label"].value_counts().to_dict()
        ),
        "redundant_copies_dropped": int((ruled["exclusion_reason"] == "redundant_copy").sum()),
        "unlabelled_kept_for_training": int(
            ((ruled["pool"] == UNLABELLED) & ruled["kept_for_training"]).sum()
        ),
        "fingerprint": dataset_fingerprint(ruled),
    }


__all__ = [
    "LABELLED",
    "UNLABELLED",
    "apply_duplicate_rules",
    "build_manifest",
    "byte_id",
    "content_id",
    "dataset_fingerprint",
    "summarise",
]
