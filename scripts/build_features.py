"""Extrait les features ResNet50 pour toutes les images et les met en cache."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from curelyticsia.config import FeatureConfig, ensure_dirs, set_global_seeds
from curelyticsia.data.loader import discover_images
from curelyticsia.features.extractor import extract_features


def main() -> None:
    ensure_dirs()
    set_global_seeds()

    records, corrupted = discover_images()
    if corrupted:
        print(f"[warn] {len(corrupted)} images corrompues écartées")
    print(f"[info] {len(records)} images valides à encoder")

    cfg = FeatureConfig()
    feats, index_df = extract_features(records, cfg=cfg, use_cache=True)
    print(f"[ok] features shape={feats.shape} -> {cfg.cache_path}")
    print(index_df["split"].value_counts().to_string())


if __name__ == "__main__":
    main()
