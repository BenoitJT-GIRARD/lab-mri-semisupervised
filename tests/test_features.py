"""Tests pour ``curelyticsia.features.extractor``."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

if importlib.util.find_spec("torch") is None:  # pragma: no cover
    pytest.skip("torch indisponible", allow_module_level=True)

from curelyticsia.config import FeatureConfig
from curelyticsia.data.loader import discover_images
from curelyticsia.features.extractor import build_backbone, extract_features, load_cached_features


def test_build_backbone_resnet18_has_identity_head() -> None:
    net, dim = build_backbone("resnet18", pretrained=False)
    assert dim == 512
    # La tête de classification doit avoir été remplacée par Identity.
    assert net.fc.__class__.__name__ == "Identity"
    # Toutes les couches sont gelées.
    assert not any(p.requires_grad for p in net.parameters())


def test_build_backbone_unknown_raises() -> None:
    with pytest.raises(ValueError):
        build_backbone("invalid", pretrained=False)


def test_extract_features_round_trip(synthetic_dataset: Path, tmp_path: Path) -> None:
    """Calcule les embeddings sur quelques images et vérifie le cache parquet."""
    records, _ = discover_images(synthetic_dataset)
    cache = tmp_path / "features.parquet"
    cfg = FeatureConfig(
        backbone="resnet18",
        output_dim=512,
        batch_size=4,
        cache_path=cache,
    )

    feats, index_df = extract_features(records, cfg=cfg, use_cache=False, progress=False)
    assert feats.shape == (len(records), 512)
    assert index_df.shape[0] == len(records)
    assert cache.exists()

    # Le rechargement depuis le cache doit produire des tableaux identiques.
    feats2, index_df2 = load_cached_features(cache)
    np.testing.assert_array_equal(feats, feats2)
    assert (index_df["image_id"].values == index_df2["image_id"].values).all()
