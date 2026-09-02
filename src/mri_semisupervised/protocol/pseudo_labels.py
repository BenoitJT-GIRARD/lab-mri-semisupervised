"""Pseudo-labels fitted inside the training fold.

This module is the correction the audit asked for. Previously the pipeline did all three
of the following **once, over the whole dataset**, and reused the result in every fold:

* fit the clustering on features of all 1 506 images;
* choose which of five algorithms to use, by ARI against all 100 labels;
* map each cluster to a class, by majority vote over those same labels.

Every fold's test labels therefore helped decide the pseudo-labels used to pre-train the
model that was then evaluated on them. Here, all three happen inside the fold, and the
only labels they may read are the training fold's.

What the clustering is allowed to see: the unlabelled pool, plus the images of the
**training** fold. Not the test fold — not even its pixels. A transductive protocol could
defend showing it the test inputs; this one cannot, because the test fold is the scarce
resource the whole exercise claims to economise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from mri_semisupervised.config import ClusteringConfig
from mri_semisupervised.models.clustering import (
    fit_agglomerative,
    fit_dbscan,
    fit_gmm,
    fit_kmeans,
    reduce_pca,
    standardise,
)

NOISE = -1


@dataclass(frozen=True)
class PseudoLabelSet:
    """Pseudo-labels for one fold, and the record of how they were obtained."""

    method_name: str
    ari_on_train: float
    image_ids: np.ndarray
    labels: np.ndarray
    candidates: dict[str, float] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.image_ids)


def align_by_majority(
    cluster_labels: np.ndarray,
    truth: np.ndarray,
    allowed: np.ndarray,
) -> np.ndarray:
    """Map each cluster to a class, using only the entries ``allowed`` marks as readable.

    ``allowed`` is not a convenience: it is how the caller states, in code, which labels it
    has the right to look at. A boolean mask that must be passed explicitly is harder to
    forget than a comment saying "training labels only".
    """
    allowed = np.asarray(allowed, dtype=bool)
    if allowed.sum() == 0:
        raise ValueError("no label is allowed to be read: the alignment has nothing to go on")

    visible_truth = np.asarray(truth)[allowed].astype(int)
    visible_clusters = np.asarray(cluster_labels)[allowed]

    mapping: dict[int, int] = {}
    for cluster in np.unique(cluster_labels):
        if cluster == NOISE:
            mapping[NOISE] = NOISE
            continue
        mask = visible_clusters == cluster
        if not mask.any():
            # A cluster no visible label falls into cannot be named. It stays noise rather
            # than receiving an arbitrary class.
            mapping[int(cluster)] = NOISE
            continue
        mapping[int(cluster)] = int(np.bincount(visible_truth[mask]).argmax())

    aligned = np.asarray(cluster_labels).copy()
    for cluster, klass in mapping.items():
        aligned[np.asarray(cluster_labels) == cluster] = klass
    return aligned


def _candidates(features: np.ndarray, seed: int, cfg: ClusteringConfig):
    """The five algorithms, fitted without ever being told the truth.

    ``truth=None`` throughout: the ARI is computed afterwards, by the caller, on the labels
    it is allowed to read. The clustering itself is unsupervised, and stays so.
    """
    return [
        fit_kmeans(features, None, n_clusters=cfg.n_clusters, seed=seed),
        fit_agglomerative(features, None, n_clusters=cfg.n_clusters, linkage="ward"),
        fit_agglomerative(features, None, n_clusters=cfg.n_clusters, linkage="average"),
        fit_gmm(features, None, n_components=cfg.n_clusters, seed=seed),
        fit_dbscan(features, None, eps=8.0, min_samples=10),
    ]


def fit_pseudo_labels(
    features: np.ndarray,
    feature_ids: np.ndarray,
    unlabelled_ids: np.ndarray,
    train_ids: np.ndarray,
    train_labels: np.ndarray,
    seed: int = 42,
    cfg: ClusteringConfig | None = None,
) -> PseudoLabelSet:
    """Cluster, choose a method and align it — all inside the training fold.

    Parameters
    ----------
    features, feature_ids
        The embedding matrix and the image id of each row.
    unlabelled_ids
        The unlabelled pool, already deduplicated and stripped of the copies of labelled
        images.
    train_ids, train_labels
        The training fold. These are the **only** labels this function may read.
    """
    cfg = cfg or ClusteringConfig()

    id_to_row = {image_id: row for row, image_id in enumerate(feature_ids)}
    usable_ids = np.array(
        [i for i in (*unlabelled_ids, *train_ids) if i in id_to_row], dtype=object
    )
    rows = np.array([id_to_row[i] for i in usable_ids])
    if len(rows) == 0:
        raise ValueError("no feature row matches the requested image ids")

    subset, _ = standardise(features[rows])
    subset, _ = reduce_pca(subset, target_variance=cfg.pca_variance)

    train_id_set = set(train_ids.tolist())
    is_train = np.array([i in train_id_set for i in usable_ids])
    truth = np.full(len(usable_ids), -1, dtype=int)
    label_of = dict(zip(train_ids.tolist(), np.asarray(train_labels).tolist(), strict=True))
    truth[is_train] = [label_of[i] for i in usable_ids[is_train]]

    results = _candidates(subset, seed, cfg)
    scored = {
        result.name: float(adjusted_rand_score(truth[is_train], result.labels[is_train]))
        for result in results
    }
    best = max(results, key=lambda r: scored[r.name])

    aligned = align_by_majority(best.labels, truth, allowed=is_train)

    is_unlabelled = ~is_train
    keep = is_unlabelled & (aligned != NOISE)
    return PseudoLabelSet(
        method_name=best.name,
        ari_on_train=scored[best.name],
        image_ids=usable_ids[keep],
        labels=aligned[keep].astype(int),
        candidates=scored,
    )


def permute(pseudo: PseudoLabelSet, seed: int) -> PseudoLabelSet:
    """Shuffle the pseudo-labels while keeping their distribution.

    The negative control. Same images, same class proportions, same number of gradient
    steps — only the *correspondence* between an image and its pseudo-label is destroyed.
    If the semi-supervised arm cannot beat this, what helped was exposure to the images,
    not the information the clustering found.
    """
    rng = np.random.default_rng(seed)
    shuffled = pseudo.labels.copy()
    rng.shuffle(shuffled)
    return PseudoLabelSet(
        method_name=f"{pseudo.method_name}+permuted",
        ari_on_train=pseudo.ari_on_train,
        image_ids=pseudo.image_ids,
        labels=shuffled,
        candidates=pseudo.candidates,
    )


def to_frame(pseudo: PseudoLabelSet) -> pd.DataFrame:
    """Tabular view, for the run artefacts."""
    return pd.DataFrame({"image_id": pseudo.image_ids, "pseudo_label": pseudo.labels})


__all__ = [
    "NOISE",
    "PseudoLabelSet",
    "align_by_majority",
    "fit_pseudo_labels",
    "permute",
    "to_frame",
]
