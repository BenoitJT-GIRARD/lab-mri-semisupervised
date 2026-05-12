"""Tests pour ``curelyticsia.models.clustering``."""

from __future__ import annotations

import numpy as np
import pandas as pd

from curelyticsia.models.clustering import (
    align_cluster_labels,
    assign_weak_labels,
    build_clustering_report,
    fit_agglomerative,
    fit_dbscan,
    fit_gmm,
    fit_kmeans,
    reduce_pca,
    standardise,
)


def test_standardise_shapes(synthetic_features: tuple[np.ndarray, np.ndarray]) -> None:
    feats, _ = synthetic_features
    scaled, scaler = standardise(feats)
    assert scaled.shape == feats.shape
    np.testing.assert_allclose(scaled.mean(axis=0), 0.0, atol=1e-6)
    assert hasattr(scaler, "mean_")


def test_pca_keeps_target_variance(synthetic_features: tuple[np.ndarray, np.ndarray]) -> None:
    feats, _ = synthetic_features
    feats, _ = standardise(feats)
    reduced, pca = reduce_pca(feats, target_variance=0.95)
    assert reduced.shape[0] == feats.shape[0]
    assert reduced.shape[1] >= 2
    assert pca.explained_variance_ratio_.sum() >= 0.94


def test_kmeans_finds_two_clusters(synthetic_features: tuple[np.ndarray, np.ndarray]) -> None:
    feats, truth = synthetic_features
    res = fit_kmeans(feats, truth, n_clusters=2)
    assert res.n_clusters == 2
    # Les deux clusters sont si nettement séparés que ARI doit être ≈ 1.
    assert res.ari_vs_truth is not None
    assert res.ari_vs_truth > 0.8
    assert res.silhouette is not None and res.silhouette > 0.3


def test_agglomerative_and_gmm_run(synthetic_features: tuple[np.ndarray, np.ndarray]) -> None:
    feats, truth = synthetic_features
    agg = fit_agglomerative(feats, truth, n_clusters=2)
    gmm = fit_gmm(feats, truth, n_components=2)
    for r in (agg, gmm):
        assert r.labels.shape == (feats.shape[0],)
        assert r.ari_vs_truth is not None and r.ari_vs_truth > 0.8


def test_dbscan_handles_outliers(synthetic_features: tuple[np.ndarray, np.ndarray]) -> None:
    feats, truth = synthetic_features
    res = fit_dbscan(feats, truth, eps=2.0, min_samples=5)
    # DBSCAN peut trouver 1 ou 2 clusters avec ce eps. On vérifie juste
    # qu'il renvoie des labels et un noise_ratio mesuré.
    assert "noise_ratio" in res.extras
    assert 0.0 <= res.extras["noise_ratio"] <= 1.0


def test_align_cluster_labels_majority_vote(
    synthetic_features: tuple[np.ndarray, np.ndarray],
) -> None:
    feats, truth = synthetic_features
    res = fit_kmeans(feats, truth, n_clusters=2)
    aligned = align_cluster_labels(res.labels, truth)

    # Pour les indices labellisés, l'alignement doit être (presque) parfait.
    valid = ~np.isnan(truth)
    assert (aligned[valid] == truth[valid].astype(int)).mean() > 0.9


def test_build_clustering_report_sorted(
    synthetic_features: tuple[np.ndarray, np.ndarray],
) -> None:
    feats, truth = synthetic_features
    results = [fit_kmeans(feats, truth), fit_agglomerative(feats, truth)]
    report = build_clustering_report(results)
    assert "method" in report.columns
    aris = report["ari_vs_truth"].dropna().tolist()
    assert aris == sorted(aris, reverse=True)


def test_assign_weak_labels_excludes_strong(
    synthetic_features: tuple[np.ndarray, np.ndarray],
) -> None:
    feats, truth = synthetic_features
    res = fit_kmeans(feats, truth, n_clusters=2)
    aligned = align_cluster_labels(res.labels, truth)

    index_df = pd.DataFrame(
        {
            "image_id": [f"id_{i}" for i in range(len(feats))],
            "path": [f"p_{i}" for i in range(len(feats))],
            "split": ["labeled" if not np.isnan(truth[i]) else "unlabeled" for i in range(len(feats))],
            "label_name": [
                "normal" if truth[i] == 0 else "cancer" if truth[i] == 1 else None for i in range(len(feats))
            ],
            "label_index": [int(t) if not np.isnan(t) else None for t in truth],
        }
    )

    weak = assign_weak_labels(index_df, aligned)
    # Les images labellisées doivent être exclues.
    assert (weak["weak_label_index"] != -1).all()
    assert set(weak["weak_label_name"].unique()).issubset({"normal", "cancer"})
    # Aucune image étiquetée fortement ne doit apparaître.
    strong_ids = set(index_df.loc[index_df["split"] == "labeled", "image_id"])
    assert not (set(weak["image_id"]) & strong_ids)
