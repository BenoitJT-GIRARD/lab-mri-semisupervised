"""Unit tests of the content identity and the duplicate rules."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from mri_semisupervised.data.manifest import (
    LABELLED,
    UNLABELLED,
    apply_duplicate_rules,
    build_manifest,
    byte_id,
    content_id,
    dataset_fingerprint,
)


def _write(path: Path, value: int, size: tuple[int, int] = (16, 16)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, color=value).save(path)
    return path


def test_two_copies_of_one_image_share_an_id(tmp_path: Path) -> None:
    first = _write(tmp_path / "a.png", 17)
    second = tmp_path / "b.png"
    shutil.copy(first, second)
    assert content_id(first) == content_id(second)
    assert byte_id(first) == byte_id(second)


def test_a_re_encoded_copy_shares_the_content_id_but_not_the_bytes(tmp_path: Path) -> None:
    # This is what a path hash and a byte hash both miss: the same scan saved twice.
    first = _write(tmp_path / "a.png", 17)
    second = tmp_path / "b.jpg"
    with Image.open(first) as img:
        img.convert("L").save(second, quality=100, subsampling=0)

    assert content_id(first) == content_id(second)
    assert byte_id(first) != byte_id(second)


def test_different_images_get_different_ids(tmp_path: Path) -> None:
    assert content_id(_write(tmp_path / "a.png", 17)) != content_id(_write(tmp_path / "b.png", 200))


def _dataset(tmp_path: Path) -> tuple[Path, Path]:
    labelled, unlabelled = tmp_path / "avec_labels", tmp_path / "sans_label"
    _write(labelled / "normal" / "n1.png", 10)
    _write(labelled / "normal" / "n2.png", 20)
    _write(labelled / "cancer" / "c1.png", 30)
    _write(labelled / "cancer" / "c2.png", 40)
    _write(unlabelled / "u1.png", 50)
    _write(unlabelled / "u2.png", 60)
    shutil.copy(labelled / "normal" / "n1.png", unlabelled / "leak.png")  # the real defect
    shutil.copy(unlabelled / "u1.png", unlabelled / "u1_again.png")  # redundant copy
    return labelled, unlabelled


def test_the_manifest_sees_the_pools_and_the_labels(tmp_path: Path) -> None:
    manifest = build_manifest(*_dataset(tmp_path))

    assert len(manifest) == 8
    assert set(manifest["pool"]) == {LABELLED, UNLABELLED}
    assert manifest[manifest["pool"] == LABELLED]["label"].value_counts().to_dict() == {
        "normal": 2,
        "cancer": 2,
    }


def test_a_labelled_image_duplicated_in_the_unlabelled_pool_leaves_that_pool(
    tmp_path: Path,
) -> None:
    ruled = apply_duplicate_rules(build_manifest(*_dataset(tmp_path)))

    leaked = ruled[(ruled["pool"] == UNLABELLED) & (ruled["path"].str.endswith("leak.png"))]
    assert not leaked["kept_for_training"].any()
    assert leaked["exclusion_reason"].tolist() == ["copy_of_labelled"]

    # And the evaluation set is untouched: the test set is not chosen after the fact.
    assert ruled.loc[ruled["pool"] == LABELLED, "kept_for_training"].all()


def test_a_redundant_copy_inside_the_unlabelled_pool_is_dropped_once(tmp_path: Path) -> None:
    ruled = apply_duplicate_rules(build_manifest(*_dataset(tmp_path)))
    kept = ruled[(ruled["pool"] == UNLABELLED) & ruled["kept_for_training"]]

    assert kept["image_id"].is_unique
    assert (ruled["exclusion_reason"] == "redundant_copy").sum() == 1


def test_the_same_image_twice_under_one_label_is_reduced_to_one(tmp_path: Path) -> None:
    # The real dataset does this once: 100 labelled files, 99 distinct images. Kept twice,
    # that image would sit on both sides of a fold.
    labelled, unlabelled = _dataset(tmp_path)
    shutil.copy(labelled / "normal" / "n1.png", labelled / "normal" / "n1_again.png")

    ruled = apply_duplicate_rules(build_manifest(labelled, unlabelled))
    evaluation = ruled[(ruled["pool"] == LABELLED) & ruled["kept_for_training"]]

    assert evaluation["image_id"].is_unique
    assert (ruled["exclusion_reason"] == "repeated_labelled_copy").sum() == 1


def test_the_same_image_under_two_labels_is_fatal(tmp_path: Path) -> None:
    # Not a duplicate: a contradiction. Nothing downstream can be trusted while it stands.
    labelled, unlabelled = _dataset(tmp_path)
    shutil.copy(labelled / "normal" / "n1.png", labelled / "cancer" / "same_image.png")

    with pytest.raises(ValueError, match="two different labels"):
        apply_duplicate_rules(build_manifest(labelled, unlabelled))


def test_the_fingerprint_ignores_order_but_not_content(tmp_path: Path) -> None:
    manifest = build_manifest(*_dataset(tmp_path))
    shuffled = manifest.sample(frac=1.0, random_state=0)

    assert dataset_fingerprint(manifest) == dataset_fingerprint(shuffled)
    assert dataset_fingerprint(manifest) != dataset_fingerprint(manifest.iloc[:-1])


def test_an_empty_manifest_does_not_explode() -> None:
    empty = pd.DataFrame(columns=["image_id", "pool", "label", "path"])
    assert apply_duplicate_rules(empty).empty
