"""Extrait le ZIP fourni dans ``infos/`` vers ``data/raw/``.

Idempotent : ne ré-extrait pas si le dossier cible existe déjà.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mri_semisupervised.config import DATASET_ROOT, DATASET_ZIP, RAW_DIR


def main() -> None:
    if DATASET_ROOT.exists() and any(DATASET_ROOT.iterdir()):
        print(f"[ok] Dataset déjà extrait : {DATASET_ROOT}")
        return
    if not DATASET_ZIP.exists():
        raise FileNotFoundError(f"Archive introuvable : {DATASET_ZIP}")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(DATASET_ZIP) as zf:
        zf.extractall(RAW_DIR)
    print(f"[ok] Extrait vers : {DATASET_ROOT}")


if __name__ == "__main__":
    main()
