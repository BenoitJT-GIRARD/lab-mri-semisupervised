"""Helpers for the exploratory clustering.

The point is to try *several* algorithms on the ResNet embeddings and compare them on
internal metrics (silhouette, Davies-Bouldin, Calinski-Harabasz) alongside the external
ARI.

**Which labels the ARI may read is the caller's business, not this module's.** These
functions take ``truth=None`` in the protocol and are scored afterwards, against the
labels the caller is allowed to see. The cluster-to-class alignment and the pseudo-label
assignment used to live here; they now live in :mod:`mri_semisupervised.protocol`, because
deciding which labels are readable is a matter of protocol.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ClusteringResult:
    """What one clustering algorithm produced."""

    name: str
    labels: np.ndarray
    n_clusters: int
    silhouette: float | None
    davies_bouldin: float | None
    calinski_harabasz: float | None
    ari_vs_truth: float | None
    extras: dict[str, float]


def standardise(features: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    """Centre and scale the features.

    Standardising before clustering is the usual move on embeddings, and it has a cost
    worth knowing: a signal carried by a few dimensions is scaled down to the level of the
    noise in every other one.
    """
    scaler = StandardScaler()
    return scaler.fit_transform(features), scaler


def reduce_pca(
    features: np.ndarray,
    target_variance: float = 0.95,
    max_components: int = 100,
    seed: int = 42,
) -> tuple[np.ndarray, PCA]:
    """PCA keeping ``target_variance`` of the variance, capped at ``max_components``."""
    n_components = min(max_components, features.shape[0], features.shape[1])
    pca = PCA(n_components=n_components, random_state=seed)
    reduced = pca.fit_transform(features)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    keep = int(np.searchsorted(cum_var, target_variance) + 1)
    keep = max(2, min(keep, n_components))
    pca = PCA(n_components=keep, random_state=seed)
    reduced = pca.fit_transform(features)
    return reduced, pca


def _safe_internal_metrics(
    features: np.ndarray, labels: np.ndarray
) -> tuple[float | None, float | None, float | None]:
    unique = set(labels.tolist())
    unique.discard(-1)  # bruit DBSCAN
    if len(unique) < 2:
        return None, None, None
    mask = labels != -1
    sub_feats = features[mask]
    sub_labels = labels[mask]
    if len(sub_feats) < 3:
        return None, None, None
    sil = float(silhouette_score(sub_feats, sub_labels))
    db = float(davies_bouldin_score(sub_feats, sub_labels))
    ch = float(calinski_harabasz_score(sub_feats, sub_labels))
    return sil, db, ch


def _ari_against_truth(
    cluster_labels: np.ndarray,
    truth_labels: np.ndarray | None,
) -> float | None:
    if truth_labels is None:
        return None
    valid = ~pd.isna(truth_labels)
    if int(valid.sum()) < 2:
        return None
    return float(adjusted_rand_score(truth_labels[valid].astype(int), cluster_labels[valid]))


def fit_kmeans(
    features: np.ndarray,
    truth: np.ndarray | None,
    n_clusters: int = 2,
    seed: int = 42,
) -> ClusteringResult:
    """K-Means."""
    model = KMeans(n_clusters=n_clusters, n_init="auto", random_state=seed)
    labels = model.fit_predict(features)
    sil, db, ch = _safe_internal_metrics(features, labels)
    return ClusteringResult(
        name="KMeans",
        labels=labels,
        n_clusters=n_clusters,
        silhouette=sil,
        davies_bouldin=db,
        calinski_harabasz=ch,
        ari_vs_truth=_ari_against_truth(labels, truth),
        extras={"inertia": float(model.inertia_)},
    )


def fit_agglomerative(
    features: np.ndarray,
    truth: np.ndarray | None,
    n_clusters: int = 2,
    linkage: str = "ward",
) -> ClusteringResult:
    """Agglomerative clustering."""
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(features)
    sil, db, ch = _safe_internal_metrics(features, labels)
    return ClusteringResult(
        name=f"Agglomerative({linkage})",
        labels=labels,
        n_clusters=n_clusters,
        silhouette=sil,
        davies_bouldin=db,
        calinski_harabasz=ch,
        ari_vs_truth=_ari_against_truth(labels, truth),
        extras={},
    )


def fit_gmm(
    features: np.ndarray,
    truth: np.ndarray | None,
    n_components: int = 2,
    seed: int = 42,
) -> ClusteringResult:
    """Gaussian Mixture Model.

    ``reg_covar`` is raised well above the scikit-learn default: a full covariance over
    roughly a hundred PCA components, estimated from about fourteen hundred points, is
    routinely ill-conditioned, and the fit then fails outright on some folds.
    """
    model = GaussianMixture(
        n_components=n_components,
        random_state=seed,
        covariance_type="full",
        reg_covar=1e-4,
    )
    labels = model.fit_predict(features)
    sil, db, ch = _safe_internal_metrics(features, labels)
    return ClusteringResult(
        name="GMM",
        labels=labels,
        n_clusters=n_components,
        silhouette=sil,
        davies_bouldin=db,
        calinski_harabasz=ch,
        ari_vs_truth=_ari_against_truth(labels, truth),
        extras={"bic": float(model.bic(features)), "aic": float(model.aic(features))},
    )


def fit_dbscan(
    features: np.ndarray,
    truth: np.ndarray | None,
    eps: float = 5.0,
    min_samples: int = 10,
) -> ClusteringResult:
    """DBSCAN — the one that can refuse to split, and say so by calling everything noise."""
    model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = model.fit_predict(features)
    n_clusters = len(set(labels.tolist()) - {-1})
    sil, db, ch = _safe_internal_metrics(features, labels)
    return ClusteringResult(
        name=f"DBSCAN(eps={eps},min={min_samples})",
        labels=labels,
        n_clusters=n_clusters,
        silhouette=sil,
        davies_bouldin=db,
        calinski_harabasz=ch,
        ari_vs_truth=_ari_against_truth(labels, truth),
        extras={"noise_ratio": float(np.mean(labels == -1))},
    )


def build_clustering_report(results: list[ClusteringResult]) -> pd.DataFrame:
    """Comparison table, sorted by decreasing ARI."""
    rows = []
    for r in results:
        row = {
            "method": r.name,
            "n_clusters": r.n_clusters,
            "silhouette": r.silhouette,
            "davies_bouldin": r.davies_bouldin,
            "calinski_harabasz": r.calinski_harabasz,
            "ari_vs_truth": r.ari_vs_truth,
        }
        row.update(r.extras)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.sort_values("ari_vs_truth", ascending=False, na_position="last").reset_index(
        drop=True
    )


__all__ = [
    "ClusteringResult",
    "build_clustering_report",
    "fit_agglomerative",
    "fit_dbscan",
    "fit_gmm",
    "fit_kmeans",
    "reduce_pca",
    "standardise",
]
