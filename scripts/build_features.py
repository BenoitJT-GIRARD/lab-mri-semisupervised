"""Embed every image with the ResNet50 backbone and cache the result.

Usage:
    uv run python scripts/build_features.py              # the plain pipeline
    uv run python scripts/build_features.py --equalize   # with histogram equalisation
"""

from __future__ import annotations

import argparse

from mri_semisupervised.config import FeatureConfig, ensure_dirs, set_global_seeds
from mri_semisupervised.data.loader import discover_images
from mri_semisupervised.features.extractor import extract_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--equalize",
        action="store_true",
        help="histogram-equalise before the backbone; writes a separate cache",
    )
    args = parser.parse_args()

    ensure_dirs()
    set_global_seeds()

    records, corrupted = discover_images()
    if corrupted:
        print(f"[warn] {len(corrupted)} corrupted images skipped")
    print(f"[info] {len(records)} valid images to encode, equalize={args.equalize}")

    cfg = FeatureConfig.for_variant(equalize=args.equalize)
    feats, index_df = extract_features(records, cfg=cfg, use_cache=True)
    print(f"[ok] features shape={feats.shape} -> {cfg.cache_path}")
    print(index_df["split"].value_counts().to_string())


if __name__ == "__main__":
    main()
