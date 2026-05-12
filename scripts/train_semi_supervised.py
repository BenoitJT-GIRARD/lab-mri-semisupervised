"""Entraîne et compare la baseline supervisée et l'approche semi-supervisée.

Évaluation en 5-fold stratifiée sur les labels forts pour une comparaison
robuste (les deux modèles vus sur les mêmes folds, mêmes hyperparamètres).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
    aggregate_reports,
    cross_validate,
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

    cfg = TrainingConfig()
    print(f"[run] Cross-validation {cfg.cv_folds}-fold stratifiee")
    print(f"      strong n={len(labeled)} ; weak n={len(weak)}")

    reports = cross_validate(
        strong_paths=labeled["path"].tolist(),
        strong_labels=labeled["label_index"].tolist(),
        weak_paths=weak["path"].tolist(),
        weak_labels=weak["weak_label_index"].astype(int).tolist(),
        class_names=list(CLASSES),
        cfg=cfg,
        seed=SEED,
    )

    out = {
        "supervised": aggregate_reports(reports["supervised"]),
        "semi_supervised": aggregate_reports(reports["semi_supervised"]),
        "training_config": {
            k: (str(v) if hasattr(v, "__fspath__") else v) for k, v in cfg.__dict__.items()
        },
    }

    out_path = PROCESSED_DIR / "training_report.json"
    out_path.write_text(
        json.dumps(out, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o),
        encoding="utf-8",
    )
    print(f"[ok] Rapport sauvegarde : {out_path}")
    for strategy in ("supervised", "semi_supervised"):
        for metric in ("accuracy", "f1_macro", "recall_cancer", "f1_cancer"):
            vals = out[strategy].get(metric)
            if vals:
                print(f"  {strategy:18s} {metric:18s} mean={vals['mean']:.3f} ± {vals['std']:.3f}")


if __name__ == "__main__":
    main()
