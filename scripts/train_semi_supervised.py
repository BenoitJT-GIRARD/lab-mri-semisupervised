"""Entraîne et compare la baseline supervisée et l'approche semi-supervisée."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from curelyticsia.config import (  # noqa: E402
    CLASSES,
    CLASS_TO_INDEX,
    ClusteringConfig,
    FeatureConfig,
    PROCESSED_DIR,
    SEED,
    TrainingConfig,
    ensure_dirs,
    set_global_seeds,
)
from curelyticsia.features.extractor import load_cached_features  # noqa: E402
from curelyticsia.models.semi_supervised import (  # noqa: E402
    train_semi_supervised,
    train_supervised,
)


def main() -> None:
    ensure_dirs()
    set_global_seeds()

    feats_cache = FeatureConfig().cache_path
    weak_path = ClusteringConfig().weak_labels_path

    if not feats_cache.exists():
        raise SystemExit("Lancez d'abord scripts/build_features.py")
    if not weak_path.exists():
        raise SystemExit("Lancez d'abord scripts/run_clustering.py")

    _, index_df = load_cached_features(feats_cache)
    weak = pd.read_csv(weak_path)

    labeled = index_df[index_df["split"] == "labeled"].copy()
    labeled["label_index"] = labeled["label_name"].map(CLASS_TO_INDEX).astype(int)

    train_idx, test_idx = train_test_split(
        labeled.index,
        test_size=0.2,
        stratify=labeled["label_index"],
        random_state=SEED,
    )
    strong_train = labeled.loc[train_idx]
    strong_test = labeled.loc[test_idx]

    cfg = TrainingConfig()

    print(f"[run] Supervised baseline (train={len(strong_train)}, test={len(strong_test)})")
    _, sup_report = train_supervised(
        train_paths=strong_train["path"].tolist(),
        train_labels=strong_train["label_index"].astype(int).tolist(),
        test_paths=strong_test["path"].tolist(),
        test_labels=strong_test["label_index"].astype(int).tolist(),
        class_names=list(CLASSES),
        cfg=cfg,
    )

    print(
        f"[run] Semi-supervised (weak={len(weak)}, train={len(strong_train)}, test={len(strong_test)})"
    )
    _, semi_report = train_semi_supervised(
        weak_paths=weak["path"].tolist(),
        weak_labels=weak["weak_label_index"].astype(int).tolist(),
        strong_train_paths=strong_train["path"].tolist(),
        strong_train_labels=strong_train["label_index"].astype(int).tolist(),
        strong_test_paths=strong_test["path"].tolist(),
        strong_test_labels=strong_test["label_index"].astype(int).tolist(),
        class_names=list(CLASSES),
        cfg=cfg,
    )

    out = {
        "supervised": {k: v for k, v in asdict(sup_report).items() if k != "history"},
        "semi_supervised": {k: v for k, v in asdict(semi_report).items() if k != "history"},
    }
    out_path = PROCESSED_DIR / "training_report.json"
    out_path.write_text(json.dumps(out, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o), encoding="utf-8")
    print(f"[ok] Rapport sauvegardé : {out_path}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
