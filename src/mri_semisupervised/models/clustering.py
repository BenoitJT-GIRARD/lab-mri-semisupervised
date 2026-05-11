"""Helpers pour le clustering exploratoire.

L'objectif : tester *plusieurs* algorithmes sur les embeddings ResNet, comparer
des métriques internes (silhouette, Davies-Bouldin, Calinski-Harabasz) et la
métrique externe ARI calculée sur les 100 images labellisées.
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

from curelyticsia.config import ClusteringConfig


@dataclass(frozen=True)
class ClusteringResult:
    """Résultat d'un algorithme de clustering."""

    name: str
    labels: np.ndarray
    n_clusters: int
    silhouette: float | None
    davies_bouldin: float | None
    calinski_harabasz: float | None
    ari_vs_truth: float | None
    extras: dict[str, float]


def standardise(features: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    """Centre-réduit les features (recommandation explicite de l'énoncé)."""
    scaler = StandardScaler()
    return scaler.fit_transform(features), scaler


def reduce_pca(
    features: np.ndarray,
    target_variance: float = 0.95,
    max_components: int = 100,
    seed: int = 42,
) -> tuple[np.ndarray, PCA]:
    """PCA pour conserver ``target_variance`` de variance, plafonnée à ``max_components``."""
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
    """Clustering agglomératif."""
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
    """Gaussian Mixture Model."""
    model = GaussianMixture(n_components=n_components, random_state=seed, covariance_type="full")
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
    """DBSCAN — utile pour détecter du bruit / des sous-structures."""
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


def align_cluster_labels(
    cluster_labels: np.ndarray,
    truth: np.ndarray | None,
) -> np.ndarray:
    """Aligne les indices de cluster sur les labels de vérité par vote majoritaire.

    Les images sans label de vérité ne sont pas utilisées pour l'alignement,
    mais reçoivent une étiquette alignée via la table de mapping construite.
    Les points labellisés par DBSCAN comme bruit (-1) gardent ``-1``.
    """
    if truth is None:
        return cluster_labels.copy()

    aligned = cluster_labels.copy()
    valid = ~pd.isna(truth)
    truth_int = truth[valid].astype(int)
    cluster_int = cluster_labels[valid]

    mapping: dict[int, int] = {}
    for cl in np.unique(cluster_labels):
        if cl == -1:
            mapping[-1] = -1
            continue
        mask = cluster_int == cl
        if mask.sum() == 0:
            mapping[int(cl)] = int(cl)
            continue
        majority = int(np.bincount(truth_int[mask]).argmax())
        mapping[int(cl)] = majority

    for cl, lbl in mapping.items():
        aligned[cluster_labels == cl] = lbl
    return aligned


def build_clustering_report(results: list[ClusteringResult]) -> pd.DataFrame:
    """Tableau récapitulatif comparable trié par ARI décroissant."""
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
    return df.sort_values("ari_vs_truth", ascending=False, na_position="last").reset_index(drop=True)


def assign_weak_labels(
    index_df: pd.DataFrame,
    aligned_labels: np.ndarray,
) -> pd.DataFrame:
    """Construit la table des pseudo-labels « faibles » sur le jeu non labellisé.

    On expose explicitement les colonnes ``image_id``, ``path``, ``weak_label_index``,
    ``weak_label_name``. Les images déjà labellisées sont *exclues* afin de
    respecter la consigne : « ne jamais mélanger faible et fort ».
    """
    if len(aligned_labels) != len(index_df):
        raise ValueError("aligned_labels et index_df doivent avoir la même longueur")

    df = index_df.copy()
    df["weak_label_index"] = aligned_labels
    name_map = {0: "normal", 1: "cancer", -1: "noise"}
    df["weak_label_name"] = df["weak_label_index"].map(lambda v: name_map.get(int(v), "unknown"))

    weak = df[df["split"] == "unlabeled"][
        ["image_id", "path", "weak_label_index", "weak_label_name"]
    ].copy()
    weak = weak[weak["weak_label_index"] != -1].reset_index(drop=True)
    return weak


def export_weak_labels(weak: pd.DataFrame, cfg: ClusteringConfig | None = None) -> str:
    """Sauvegarde les pseudo-labels au format CSV et renvoie le chemin."""
    cfg = cfg or ClusteringConfig()
    cfg.weak_labels_path.parent.mkdir(parents=True, exist_ok=True)
    weak.to_csv(cfg.weak_labels_path, index=False)
    return str(cfg.weak_labels_path)


__all__ = [
    "ClusteringResult",
    "align_cluster_labels",
    "assign_weak_labels",
    "build_clustering_report",
    "export_weak_labels",
    "fit_agglomerative",
    "fit_dbscan",
    "fit_gmm",
    "fit_kmeans",
    "reduce_pca",
    "standardise",
]
