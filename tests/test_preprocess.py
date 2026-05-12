"""Tests pour ``curelyticsia.data.preprocess``."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

if importlib.util.find_spec("torch") is None:  # pragma: no cover
    pytest.skip("torch indisponible", allow_module_level=True)

from curelyticsia.data.preprocess import (
    ImagePathsDataset,
    build_eval_transform,
    build_train_transform,
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


def test_label_length_mismatch_raises(tmp_path: Path) -> None:
    p = tmp_path / "img.jpg"
    _make_image(p)
    with pytest.raises(ValueError):
        ImagePathsDataset(paths=[p, p], transform=build_eval_transform(), labels=[0])
