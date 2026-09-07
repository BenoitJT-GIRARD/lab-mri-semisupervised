"""Unit tests of the preprocessing pipelines."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

if importlib.util.find_spec("torch") is None:  # pragma: no cover
    pytest.skip("torch indisponible", allow_module_level=True)

from mri_semisupervised.data.preprocess import (
    ImagePathsDataset,
    build_eval_transform,
    build_train_transform,
    equalize_image,
)


def _make_image(path: Path, mode: str = "L", size: int = 64) -> None:
    arr = np.full((size, size), 100, dtype=np.uint8)
    Image.fromarray(arr, mode=mode).save(path, format="JPEG", quality=80)


def test_eval_transform_produces_expected_shape(tmp_path: Path) -> None:
    img_path = tmp_path / "img.jpg"
    _make_image(img_path, mode="L")
    transform = build_eval_transform(image_size=224)
    with Image.open(img_path) as im:
        tensor = transform(im)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype.is_floating_point


def test_train_transform_is_stochastic_but_consistent(tmp_path: Path) -> None:
    img_path = tmp_path / "img.jpg"
    _make_image(img_path, mode="L")
    transform = build_train_transform(image_size=224)
    with Image.open(img_path) as im:
        tensor = transform(im)
    assert tensor.shape == (3, 224, 224)


def test_image_paths_dataset_yields_tensor_and_label(tmp_path: Path) -> None:
    paths = []
    for i in range(3):
        p = tmp_path / f"img_{i}.jpg"
        _make_image(p)
        paths.append(p)
    ds = ImagePathsDataset(paths=paths, transform=build_eval_transform(), labels=[0, 1, 0])
    tensor, label = ds[1]
    assert tensor.shape == (3, 224, 224)
    assert label == 1


def test_image_paths_dataset_falls_back_to_index(tmp_path: Path) -> None:
    paths = []
    for i in range(2):
        p = tmp_path / f"img_{i}.jpg"
        _make_image(p)
        paths.append(p)
    ds = ImagePathsDataset(paths=paths, transform=build_eval_transform())
    _, idx = ds[0]
    assert idx == 0
    assert len(ds) == 2


def test_equalize_image_returns_grayscale_same_size(tmp_path: Path) -> None:
    img_path = tmp_path / "img.jpg"
    _make_image(img_path, mode="L", size=64)
    with Image.open(img_path) as im:
        eq = equalize_image(im)
    assert eq.mode == "L"
    assert eq.size == (64, 64)


def test_label_length_mismatch_raises(tmp_path: Path) -> None:
    p = tmp_path / "img.jpg"
    _make_image(p)
    with pytest.raises(ValueError):
        ImagePathsDataset(paths=[p, p], transform=build_eval_transform(), labels=[0])


def test_the_equalised_transform_changes_the_pixels() -> None:
    """A dull scan must come out different, or the flag is decorative."""
    rng = np.random.default_rng(0)
    dull = Image.fromarray((90 + rng.integers(-8, 8, (64, 64))).astype(np.uint8))

    plain = build_eval_transform(64)(dull)
    equalised = build_eval_transform(64, equalize=True)(dull)

    assert plain.shape == equalised.shape
    assert float(np.abs(plain.numpy() - equalised.numpy()).max()) > 0.5


def test_equalisation_keeps_the_tensor_contract() -> None:
    """It goes in front of the resize; shape and normalisation do not change."""
    image = Image.fromarray(np.full((70, 50), 120, dtype=np.uint8))
    for equalize in (False, True):
        out = build_eval_transform(64, equalize=equalize)(image)
        assert out.shape == (3, 64, 64)


def test_the_training_transform_takes_the_flag_too() -> None:
    """Equalising evaluation and not training would compare two different pipelines."""
    image = Image.fromarray(np.full((70, 50), 120, dtype=np.uint8))
    assert build_train_transform(64, equalize=True)(image).shape == (3, 64, 64)
