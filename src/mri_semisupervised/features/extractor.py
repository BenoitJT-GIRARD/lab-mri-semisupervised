"""Embeddings from an ImageNet-pretrained backbone.

``extract_features(records)`` returns an ``(N, output_dim)`` matrix and the index frame
that names each row. The result is cached as Parquet, because re-encoding fifteen hundred
images to try one clustering parameter is a waste of a GPU.

The extraction is unsupervised and identical for every arm and every fold, which is why it
can be done once, outside the protocol, without leaking anything.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from tqdm.auto import tqdm

from mri_semisupervised.config import FeatureConfig, device
from mri_semisupervised.data.loader import ImageRecord
from mri_semisupervised.data.preprocess import ImagePathsDataset, build_eval_transform


def build_backbone(name: str = "resnet50", pretrained: bool = True) -> tuple[nn.Module, int]:
    """Build a feature extractor: the backbone with its classification head removed.

    Returns ``(model, feature_dim)``. Nothing is trained at this stage: the weights are
    frozen and the pass is forward-only, to turn images into vectors.
    """
    name_lower = name.lower()
    if name_lower == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        net = models.resnet50(weights=weights)
    elif name_lower == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.resnet18(weights=weights)
    else:
        raise ValueError(f"Backbone non supporté : {name}")
    feature_dim = net.fc.in_features
    net.fc = nn.Identity()

    for p in net.parameters():
        p.requires_grad_(False)
    net.eval()
    return net, feature_dim


INDEX_COLUMNS = ["image_id", "path", "split", "label_name", "label_index"]


class CacheMismatchError(RuntimeError):
    """The cached features do not describe the images being asked for.

    This is the same class of fault the repository documents one layer up: an identifier
    that did not identify. A cache computed on one dataset and read back on another pairs
    vectors with labels that are not theirs, and nothing downstream notices — the shapes
    agree, and so do the identifiers if the files have moved.
    """


def _check_cache(cached: pd.DataFrame, records: list[ImageRecord], backbone: str) -> None:
    """Refuse a cache that disagrees with the records, and say how it disagrees.

    The comparison is on the identifiers rather than on an aggregate fingerprint. The
    identifiers are already in the parquet — they are the content hashes — and a digest
    would only say *that* something differs. The point of a guard is to name what.
    """
    if "backbone" not in cached.columns:
        raise CacheMismatchError(
            "the cache carries no backbone column, so it predates this guard and cannot "
            "be trusted. Delete it to recompute."
        )
    stored = sorted(set(cached["backbone"].unique()))
    if stored != [backbone]:
        raise CacheMismatchError(
            f"the cache was built with backbone {stored}, and {backbone!r} is being asked "
            "for. Same shape, different contents. Delete it to recompute."
        )

    # Multisets, not sets. The dataset genuinely holds duplicate content — 31 evaluation
    # images also live in the unlabelled pool — so the same identifier legitimately
    # appears on several rows, and a set comparison would call a lost copy a match.
    want = Counter(r.image_id for r in records)
    have = Counter(cached["image_id"])
    missing = sorted((want - have).elements())
    unexpected = sorted((have - want).elements())
    if missing or unexpected:
        raise CacheMismatchError(
            f"{len(missing)} missing, {len(unexpected)} unexpected. "
            f"missing: {missing[:3]}; unexpected: {unexpected[:3]}. "
            "Delete the cache to recompute."
        )


@torch.inference_mode()
def extract_features(
    records: list[ImageRecord],
    cfg: FeatureConfig | None = None,
    use_cache: bool = True,
    progress: bool = True,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Compute the feature matrix, or read it back from the cache.

    Returns
    -------
    features : np.ndarray
        An ``(N, feature_dim)`` array, row-aligned with ``index_df``.
    index_df : pd.DataFrame
        Columns ``image_id``, ``path``, ``split``, ``label_name``, ``label_index``.
    """
    if cfg is None:
        cfg = FeatureConfig()

    cache_path = Path(cfg.cache_path)

    if use_cache and cache_path.exists():
        cached = pd.read_parquet(cache_path)
        _check_cache(cached, records, cfg.backbone)
        # Row alignment is with the caller's list, not with the order the file happens to
        # hold: every consumer indexes the matrix by the position of its record. Rows that
        # share an identifier share their pixels, so which of them a record receives does
        # not matter — reindexing on the identifier would multiply them instead.
        buckets: dict[str, list[int]] = {}
        for position, image_id in enumerate(cached["image_id"]):
            buckets.setdefault(image_id, []).append(position)
        cached = cached.iloc[[buckets[r.image_id].pop() for r in records]].reset_index(drop=True)
        feature_cols = [c for c in cached.columns if c.startswith("f_")]
        features = cached[feature_cols].to_numpy(dtype=np.float32)
        return features, cached[INDEX_COLUMNS].copy()

    paths = [str(r.path) for r in records]
    transform = build_eval_transform()
    dataset = ImagePathsDataset(paths=paths, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=False,
    )

    dev = torch.device(device())
    backbone, feature_dim = build_backbone(cfg.backbone, pretrained=True)
    backbone.to(dev).eval()

    feats = np.zeros((len(records), feature_dim), dtype=np.float32)
    iterator = tqdm(loader, desc=f"Features {cfg.backbone}", disable=not progress)
    cursor = 0
    for batch, _ in iterator:
        batch = batch.to(dev, non_blocking=True)
        out = backbone(batch).detach().cpu().numpy().astype(np.float32)
        feats[cursor : cursor + out.shape[0]] = out
        cursor += out.shape[0]

    index_df = pd.DataFrame(
        {
            "image_id": [r.image_id for r in records],
            "path": [str(r.path) for r in records],
            "split": [r.split for r in records],
            "label_name": [r.label_name for r in records],
            "label_index": [r.label_index for r in records],
        }
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    feature_cols = [f"f_{i:04d}" for i in range(feature_dim)]
    stamp = pd.DataFrame({"backbone": [cfg.backbone] * len(records)})
    df_cache = pd.concat([index_df, stamp, pd.DataFrame(feats, columns=feature_cols)], axis=1)
    df_cache.to_parquet(cache_path, index=False)

    return feats, index_df


def load_cached_features(cache_path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    """Read a feature parquet back and return ``(features, index_df)``."""
    cached = pd.read_parquet(cache_path)
    feature_cols = [c for c in cached.columns if c.startswith("f_")]
    features = cached[feature_cols].to_numpy(dtype=np.float32)
    index_df = cached[INDEX_COLUMNS].copy()
    return features, index_df


__all__ = [
    "CacheMismatchError",
    "build_backbone",
    "extract_features",
    "load_cached_features",
]
