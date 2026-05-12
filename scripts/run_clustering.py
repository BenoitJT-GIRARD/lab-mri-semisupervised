"""Lance le clustering exploratoire et exporte les pseudo-labels faibles."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from curelyticsia.config import (  # noqa: E402
    CLASS_TO_INDEX,
    ClusteringConfig,
    FeatureConfig,
    ensure_dirs,
    set_global_seeds,
)
from curelyticsia.features.extractor import load_cached_features  # noqa: E402
from curelyticsia.models.clustering import (  # noqa: E402
    align_cluster_labels,
    assign_weak_labels,
    build_clustering_report,
    export_weak_labels,
    fit_agglomerative,
    fit_dbscan,
    fit_gmm,
    fit_kmeans,
    reduce_pca,
    standardise,
)


def main() -> None:
    ensure_dirs()
    set_global_seeds()

    feats, index_df = load_cached_features(FeatureConfig().cache_path)

    truth = np.where(
        index_df["split"] == "labeled",
        index_df["label_name"].map(lambda v: CLASS_TO_INDEX.get(str(v), np.nan) if v else np.nan),
        np.nan,
    ).astype(float)

    feats_std, _ = standardise(feats)
    feats_red, pca = reduce_pca(feats_std, target_variance=0.95)
    print(f"[info] PCA : {feats_red.shape[1]} composantes (variance cumulée ≥ 0.95)")

    results = [
        fit_kmeans(feats_red, truth, n_clusters=2),
        fit_agglomerative(feats_red, truth, n_clusters=2, linkage="ward"),
        fit_agglomerative(feats_red, truth, n_clusters=2, linkage="average"),
        fit_gmm(feats_red, truth, n_components=2),
        fit_dbscan(feats_red, truth, eps=8.0, min_samples=10),
    ]

    report = build_clustering_report(results)
    print(report.to_string(index=False))

    best = max(
        (r for r in results if r.ari_vs_truth is not None),
        key=lambda r: r.ari_vs_truth,
    )
    print(f"\n[best] {best.name} ARI={best.ari_vs_truth:.4f}")

    aligned = align_cluster_labels(best.labels, truth)
    weak = assign_weak_labels(index_df, aligned)
    out = export_weak_labels(weak, ClusteringConfig())
    print(f"[ok] {len(weak)} pseudo-labels exportes -> {out}")

    pd.DataFrame(report).to_csv(
        ClusteringConfig().weak_labels_path.parent / "clustering_report.csv", index=False
    )


if __name__ == "__main__":
    main()
