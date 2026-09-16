"""Say whether the archive on this machine is the one every published number came from.

The repository holds no images, so the first question a reader has is whether what they
unpacked is the right thing. This answers it in one command: it names the tree it expects,
says what it found, and compares the dataset fingerprint with the one recorded in the run
manifests under `reports/experiments/`.

    uv run python scripts/check_data.py

It writes nothing and trains nothing. A mismatch is not an error of this script: it means the
numbers in the README describe another dataset than the one on disk.
"""

from __future__ import annotations

import json
import sys

from mri_semisupervised.config import (
    DATASET_ROOT,
    LABELLED_DIR,
    PROJECT_ROOT,
    UNLABELLED_DIR,
)
from mri_semisupervised.data.manifest import (
    apply_duplicate_rules,
    build_manifest,
    dataset_fingerprint,
    summarise,
)
from mri_semisupervised.protocol.report import runs

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

#: The folders the archive unpacks into, under `<MRI_DATA_DIR>/raw/`. The archive ships them
#: in French; they are renamed on the way in, and this is where a reader finds out.
EXPECTED = """expected layout, under MRI_DATA_DIR (defaults to ./data):

  raw/mri_dataset_brain_cancer_oc/
    labelled/
      cancer/     100 files in total, across the two classes
      normal/
    unlabelled/   1 406 files
"""


def recorded_fingerprints() -> dict[str, str]:
    """The dataset fingerprint each published run was produced against."""
    out = {}
    for directory in runs():
        manifest = directory / "manifest.json"
        if manifest.exists():
            meta = json.loads(manifest.read_text(encoding="utf-8"))
            out[directory.name] = str(meta.get("dataset_fingerprint", ""))
    return out


def main() -> int:
    print(EXPECTED)
    print(f"[info] looking under {DATASET_ROOT}")

    missing = [d for d in (LABELLED_DIR, UNLABELLED_DIR) if not d.is_dir()]
    if missing:
        for directory in missing:
            print(f"[fail] {directory} does not exist")
        print("\nUnpack the archive there, or point MRI_DATA_DIR at where you unpacked it.")
        return 1

    manifest = apply_duplicate_rules(build_manifest(LABELLED_DIR, UNLABELLED_DIR))
    if manifest.empty:
        print("[fail] the folders exist and hold no image this code can read")
        return 1

    counts = summarise(manifest)
    for key in (
        "files",
        "unreadable_files",
        "evaluation_images",
        "labelled_also_in_unlabelled",
        "unlabelled_kept_for_training",
    ):
        if key in counts:
            print(f"  {key:32} {counts[key]}")

    found = dataset_fingerprint(manifest)
    print(f"\n[info] fingerprint on this machine : {found}")

    published = recorded_fingerprints()
    if not published:
        print("[warn] no published run to compare with")
        return 0

    disagreeing = {name: value for name, value in published.items() if value != found}
    for name, value in sorted(published.items()):
        mark = "ok  " if value == found else "DIFF"
        print(f"  [{mark}] {name:20} {value}")

    if disagreeing:
        print(
            "\nThe numbers published in the README were measured on another dataset than the "
            "one on disk. Nothing here is wrong; they simply do not describe these images."
        )
        return 1
    print(f"\n[ok] every run under {PROJECT_ROOT.name}/reports/experiments/ used these images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
