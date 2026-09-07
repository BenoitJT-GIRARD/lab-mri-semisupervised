"""Inventory the dataset by image content, and report what the duplicates cost.

Usage:
    uv run python scripts/build_manifest.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mri_semisupervised.config import LABELED_DIR, MANIFEST_PATH, UNLABELED_DIR, ensure_dirs
from mri_semisupervised.data.manifest import apply_duplicate_rules, build_manifest, summarise


def main() -> None:
    """Build the manifest, apply the duplicate rules, and print the summary."""
    ensure_dirs()
    if not LABELED_DIR.exists():
        print(f"[warn] dataset not found under {LABELED_DIR.parent}.")
        print("       set MRI_DATA_DIR, or run scripts/extract_dataset.py first.")
        return

    print(f"[info] scanning {LABELED_DIR.parent}")
    manifest = apply_duplicate_rules(build_manifest(LABELED_DIR, UNLABELED_DIR))
    manifest.to_parquet(MANIFEST_PATH, index=False)

    facts = summarise(manifest)
    print()
    for key, value in facts.items():
        print(f"  {key:32} {value}")
    print()
    print(f"[ok] manifest written : {MANIFEST_PATH}")
    print(f"[ok] fingerprint      : {facts['fingerprint']}")

    if facts["labelled_also_in_unlabelled"]:
        share = 100 * facts["labelled_also_in_unlabelled"] / max(facts["labelled_files"], 1)
        print(
            f"\n[!!] {facts['labelled_also_in_unlabelled']} evaluation image(s) "
            f"({share:.0f}%) also sit in the unlabelled pool. They are excluded from "
            "training, never from evaluation."
        )

    (MANIFEST_PATH.parent / "dataset_summary.json").write_text(
        json.dumps(facts, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
