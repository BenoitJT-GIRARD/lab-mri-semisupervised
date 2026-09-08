"""Unit tests of the feature extractor."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

if importlib.util.find_spec("torch") is None:  # pragma: no cover
    pytest.skip("torch indisponible", allow_module_level=True)

from mri_semisupervised.config import FeatureConfig
from mri_semisupervised.data.loader import discover_images
from mri_semisupervised.features.extractor import (
    CacheMismatchError,
    build_backbone,
    extract_features,
    load_cached_features,
)


def test_build_backbone_resnet18_has_identity_head() -> None:
    net, dim = build_backbone("resnet18", pretrained=False)
    assert dim == 512
    # The classification head must have been replaced by an identity.
    assert net.fc.__class__.__name__ == "Identity"
    # Every layer is frozen: this is a forward pass, not training.
    assert not any(p.requires_grad for p in net.parameters())


def test_build_backbone_unknown_raises() -> None:
    with pytest.raises(ValueError):
        build_backbone("invalid", pretrained=False)


def test_extract_features_round_trip(synthetic_dataset: Path, tmp_path: Path) -> None:
    """Embed a handful of images and check the parquet cache round-trips."""
    records, _ = discover_images(synthetic_dataset)
    cache = tmp_path / "features.parquet"
    cfg = FeatureConfig(
        backbone="resnet18",
        batch_size=4,
        cache_path=cache,
    )

    feats, index_df = extract_features(records, cfg=cfg, use_cache=False, progress=False)
    assert feats.shape == (len(records), 512)
    assert index_df.shape[0] == len(records)
    assert cache.exists()

    # Reading the cache back must give identical arrays.
    feats2, index_df2 = load_cached_features(cache)
    np.testing.assert_array_equal(feats, feats2)
    assert (index_df["image_id"].values == index_df2["image_id"].values).all()


def _write_cache(
    path: Path,
    image_ids: list[str],
    *,
    backbone: str = "resnet18",
    equalize: bool = False,
    dim: int = 512,
) -> None:
    """Write a cache parquet by hand, so a test can make it disagree with the records."""
    import pandas as pd

    frame = pd.DataFrame(
        {
            "image_id": image_ids,
            "path": [f"/nowhere/{i}.jpg" for i in image_ids],
            "split": ["labeled"] * len(image_ids),
            "label_name": ["normal"] * len(image_ids),
            "label_index": [0] * len(image_ids),
            "backbone": [backbone] * len(image_ids),
            "equalize": [equalize] * len(image_ids),
        }
    )
    feats = pd.DataFrame(
        np.zeros((len(image_ids), dim), dtype=np.float32),
        columns=[f"f_{i:04d}" for i in range(dim)],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([frame, feats], axis=1).to_parquet(path, index=False)


def _cfg(cache: Path, backbone: str = "resnet18", *, equalize: bool = False) -> FeatureConfig:
    return FeatureConfig(
        backbone=backbone,
        batch_size=4,
        equalize=equalize,
        cache_path=cache,
    )


def test_a_cache_missing_an_image_is_refused_and_names_it(
    synthetic_dataset: Path, tmp_path: Path
) -> None:
    records, _ = discover_images(synthetic_dataset)
    cache = tmp_path / "features.parquet"
    _write_cache(cache, [r.image_id for r in records[:-1]])

    with pytest.raises(CacheMismatchError) as excinfo:
        extract_features(records, cfg=_cfg(cache), use_cache=True, progress=False)

    message = str(excinfo.value)
    assert "1 missing" in message
    assert records[-1].image_id in message


def test_a_cache_holding_an_unexpected_image_is_refused(
    synthetic_dataset: Path, tmp_path: Path
) -> None:
    records, _ = discover_images(synthetic_dataset)
    cache = tmp_path / "features.parquet"
    _write_cache(cache, [r.image_id for r in records] + ["deadbeef" * 8])

    with pytest.raises(CacheMismatchError, match="1 unexpected"):
        extract_features(records, cfg=_cfg(cache), use_cache=True, progress=False)


def test_a_cache_from_another_backbone_is_refused(synthetic_dataset: Path, tmp_path: Path) -> None:
    """A resnet18 cache read for a resnet50 has the right shape and the wrong contents."""
    records, _ = discover_images(synthetic_dataset)
    cache = tmp_path / "features.parquet"
    _write_cache(cache, [r.image_id for r in records], backbone="resnet18")

    with pytest.raises(CacheMismatchError) as excinfo:
        extract_features(records, cfg=_cfg(cache, "resnet50"), use_cache=True, progress=False)
    message = str(excinfo.value)
    assert "resnet18" in message and "resnet50" in message


def test_a_cache_without_the_backbone_column_is_refused(
    synthetic_dataset: Path, tmp_path: Path
) -> None:
    """Caches written before the guard existed carry no backbone, so they are stale."""
    import pandas as pd

    records, _ = discover_images(synthetic_dataset)
    cache = tmp_path / "features.parquet"
    _write_cache(cache, [r.image_id for r in records])
    frame = pd.read_parquet(cache).drop(columns=["backbone"])
    frame.to_parquet(cache, index=False)

    with pytest.raises(CacheMismatchError, match="no backbone"):
        extract_features(records, cfg=_cfg(cache), use_cache=True, progress=False)


def test_a_matching_cache_is_returned_in_the_order_of_the_records(
    synthetic_dataset: Path, tmp_path: Path
) -> None:
    """Row alignment is with the caller's list, not with the order the file happens to hold."""
    records, _ = discover_images(synthetic_dataset)
    cache = tmp_path / "features.parquet"
    _write_cache(cache, [r.image_id for r in reversed(records)])

    feats, index_df = extract_features(records, cfg=_cfg(cache), use_cache=True, progress=False)

    assert feats.shape == (len(records), 512)
    assert list(index_df["image_id"]) == [r.image_id for r in records]
    assert "backbone" not in index_df.columns


def test_an_equalised_cache_is_refused_for_a_plain_run(
    synthetic_dataset: Path, tmp_path: Path
) -> None:
    """Same shape, different pixels — the failure mode a column count cannot see."""
    records, _ = discover_images(synthetic_dataset)
    cache = tmp_path / "features.parquet"
    _write_cache(cache, [r.image_id for r in records], equalize=True)

    with pytest.raises(CacheMismatchError, match="equalize"):
        extract_features(records, cfg=_cfg(cache), use_cache=True, progress=False)


def test_the_variant_helper_ties_the_flag_to_the_cache_path() -> None:
    """Setting one and forgetting the other is the mistake this helper removes."""
    plain = FeatureConfig.for_variant()
    equalised = FeatureConfig.for_variant(equalize=True)

    assert plain.equalize is False
    assert equalised.equalize is True
    assert plain.cache_path != equalised.cache_path
    assert "equalized" in equalised.cache_path.name
