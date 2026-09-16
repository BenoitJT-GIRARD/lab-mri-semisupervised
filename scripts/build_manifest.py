"""Inventory the dataset by image content, and report what the duplicates cost.

Usage:
    uv run python scripts/build_manifest.py
"""

from __future__ import annotations

import json

from mri_semisupervised.config import (
    LABELLED_DIR,
    MANIFEST_PATH,
    REPORTS_DIR,
    UNLABELLED_DIR,
    ensure_dirs,
)
from mri_semisupervised.data.manifest import apply_duplicate_rules, build_manifest, summarise


def main() -> None:
    """Build the manifest, apply the duplicate rules, and print the summary."""
    ensure_dirs()
    if not LABELLED_DIR.exists():
        print(f"[warn] dataset not found under {LABELLED_DIR.parent}.")
        print("       point MRI_DATA_DIR at the folder that holds the dataset.")
        return

    print(f"[info] scanning {LABELLED_DIR.parent}")
    manifest = apply_duplicate_rules(build_manifest(LABELLED_DIR, UNLABELLED_DIR))
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

    # Beside the manifest, which lives with the images and is never published, and under
    # reports/, which is. The data card quotes these counts, and a reader without the archive
    # can still check that the page and the inventory agree.
    (MANIFEST_PATH.parent / "dataset_summary.json").write_text(
        json.dumps(facts, indent=2) + "\n", encoding="utf-8", newline=""
    )
    published = REPORTS_DIR / "dataset_summary.json"
    published.parent.mkdir(parents=True, exist_ok=True)
    published.write_text(json.dumps(facts, indent=2) + "\n", encoding="utf-8", newline="")
    print(f"[ok] summary written  : {published}")


if __name__ == "__main__":
    main()
