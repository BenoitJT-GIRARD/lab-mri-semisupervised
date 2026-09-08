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

from dataclasses import dataclass, field, replace

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
    #: Margin to the nearest other cluster, in [0, 1], one per pseudo-label.
    confidence: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: Pool images the alignment could not name, dropped before this set was built. They
    #: are counted because otherwise a fold reports what survived and never what was lost,
    #: which is the more interesting half on a fold that clustered badly.
    n_noise: int = 0
    candidates: dict[str, float] = field(default_factory=dict)
    failed_candidates: dict[str, str] = field(default_factory=dict)

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


def cluster_confidence(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """How far inside its cluster each point sits, as a margin in ``[0, 1]``.

    ``1 - d(i, c) / d(i, c2)`` where ``c`` is the point own cluster centroid and ``c2`` the
    nearest other one. Zero for a point equidistant from two clusters, near one at an
    isolated centre, zero for noise, and zero when there is only one cluster to belong to.

    The centroid is empirical, computed from the points themselves, for every method alike.
    Two reasons rather than one: :class:`ClusteringResult` does not carry its estimator, so
    ``cluster_centers_`` is not available; and the chosen method varies from fold to fold,
    so a definition that changed with the method would not be comparable across the folds
    it has to be compared across.

    A point is excluded from its own centroid. Without that, a point drags the centroid it
    is measured against towards itself and scores as more central than it is — negligible
    on a cluster of five hundred points, large enough to invert the ranking on the small
    clusters DBSCAN produces.
    """
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels)
    confidence = np.zeros(len(labels), dtype=float)

    named = np.unique(labels[labels != NOISE])
    if len(named) < 2:
        return confidence

    sums = np.vstack([features[labels == c].sum(axis=0) for c in named])
    sizes = np.array([int((labels == c).sum()) for c in named])
    centroids = sums / sizes[:, None]
    distances = np.linalg.norm(features[:, None, :] - centroids[None, :, :], axis=2)

    for position, cluster in enumerate(named):
        mask = labels == cluster
        size = sizes[position]
        if not mask.any():
            continue
        if size > 1:
            own_centroid = (sums[position] - features[mask]) / (size - 1)
            own = np.linalg.norm(features[mask] - own_centroid, axis=1)
        else:
            own = distances[mask, position]
        others = np.delete(distances[mask], position, axis=1).min(axis=1)
        safe = np.where(others > 0, others, 1.0)
        confidence[mask] = np.clip(1.0 - own / safe, 0.0, 1.0)
    return confidence


def composition(pseudo: PseudoLabelSet) -> dict[str, float]:
    """What the clustering actually produced, in numbers the fold manifest can carry.

    The original protocol produced 928 cancer against 478 normal, for a truth that is
    50/50. The imbalance was noticed at the time and never acted on. The protocol has
    changed since; the question stayed unanswered because nothing recorded the answer.

    The share is taken over the labelled pseudo-labels alone. A fold where most of the pool
    landed in an unalignable cluster has a share that says nothing about the two classes,
    and dividing by the total would hide that behind a small number instead of showing it
    beside ``n_pseudo_noise``.
    """
    labels = np.asarray(pseudo.labels)
    negative = int((labels == 0).sum())
    positive = int((labels == 1).sum())
    total = negative + positive
    return {
        "n_pseudo_negative": negative,
        "n_pseudo_positive": positive,
        "n_pseudo_noise": int(pseudo.n_noise),
        "pseudo_positive_share": (positive / total) if total else float("nan"),
    }


def filter_by_confidence(pseudo: PseudoLabelSet, quantile: float) -> PseudoLabelSet:
    """Keep the pseudo-labels above the given quantile of the confidence distribution.

    ``quantile=0`` keeps everything, which is a real option and not a degenerate one: the
    arm must be able to choose not to filter, and to say so in its manifest.

    An empty pool is never returned. A quantile that would drop every point leaves the most
    confident one, because a pre-training set of size zero is not the same experiment as a
    pre-training set of size one — it would silently turn the arm into its own baseline.
    """
    if len(pseudo) == 0 or quantile <= 0.0:
        return pseudo

    threshold = float(np.quantile(pseudo.confidence, min(quantile, 1.0)))
    keep = pseudo.confidence >= threshold
    if not keep.any():
        keep = pseudo.confidence >= pseudo.confidence.max()

    return replace(
        pseudo,
        image_ids=pseudo.image_ids[keep],
        labels=pseudo.labels[keep],
        confidence=pseudo.confidence[keep],
    )


def _candidates(features: np.ndarray, seed: int, cfg: ClusteringConfig):
    """The five algorithms, fitted without ever being told the truth.

    ``truth=None`` throughout: the ARI is computed afterwards, by the caller, on the labels
    it is allowed to read. The clustering itself is unsupervised, and stays so.

    A candidate that cannot fit on a given fold — the GMM does this when the fold's
    covariance turns out singular — is dropped **for that fold**, and the failure is
    returned rather than swallowed. Silently substituting another method would change what
    the experiment compares without saying so.
    """
    factories = {
        "KMeans": lambda: fit_kmeans(features, None, n_clusters=cfg.n_clusters, seed=seed),
        "Agglomerative(ward)": lambda: fit_agglomerative(
            features, None, n_clusters=cfg.n_clusters, linkage="ward"
        ),
        "Agglomerative(average)": lambda: fit_agglomerative(
            features, None, n_clusters=cfg.n_clusters, linkage="average"
        ),
        "GMM": lambda: fit_gmm(features, None, n_components=cfg.n_clusters, seed=seed),
        "DBSCAN": lambda: fit_dbscan(features, None, eps=8.0, min_samples=10),
    }

    fitted, failures = [], {}
    for name, factory in factories.items():
        try:
            fitted.append(factory())
        except Exception as error:
            failures[name] = f"{type(error).__name__}: {error}"[:120]

    if not fitted:
        raise RuntimeError(f"every clustering candidate failed on this fold: {failures}")
    return fitted, failures


def fit_pseudo_labels(
    features: np.ndarray,
    feature_ids: np.ndarray,
    unlabelled_ids: np.ndarray,
    train_ids: np.ndarray,
    train_labels: np.ndarray,
    *,
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

    # The union, each image once. An image may belong to both sets — that is exactly the
    # defect the legacy protocol carries, and it must survive to be measured.
    seen: dict[object, None] = {}
    for image_id in (*unlabelled_ids, *train_ids):
        if image_id in id_to_row:
            seen.setdefault(image_id, None)
    usable_ids = np.array(list(seen), dtype=object)
    rows = np.array([id_to_row[i] for i in usable_ids])
    if len(rows) == 0:
        raise ValueError("no feature row matches the requested image ids")

    subset, _ = standardise(features[rows])
    subset, _ = reduce_pca(
        subset, target_variance=cfg.pca_variance, max_components=cfg.pca_max_components
    )

    train_id_set = set(train_ids.tolist())
    pool_id_set = set(np.asarray(unlabelled_ids).tolist())
    is_train = np.array([i in train_id_set for i in usable_ids])
    is_pool = np.array([i in pool_id_set for i in usable_ids])
    truth = np.full(len(usable_ids), -1, dtype=int)
    label_of = dict(zip(train_ids.tolist(), np.asarray(train_labels).tolist(), strict=True))
    truth[is_train] = [label_of[i] for i in usable_ids[is_train]]

    results, failures = _candidates(subset, seed, cfg)
    scored = {
        result.name: float(adjusted_rand_score(truth[is_train], result.labels[is_train]))
        for result in results
    }
    best = max(results, key=lambda r: scored[r.name])

    aligned = align_by_majority(best.labels, truth, allowed=is_train)

    # Pseudo-labels cover the pool the caller handed over. When that pool has been cleaned
    # of the copies of evaluation images, nothing labelled can appear here; when it has
    # not — the legacy protocol — the copies come through, and that is the leak.
    keep = is_pool & (aligned != NOISE)
    confidence = cluster_confidence(subset, best.labels)
    return PseudoLabelSet(
        method_name=best.name,
        ari_on_train=scored[best.name],
        image_ids=usable_ids[keep],
        labels=aligned[keep].astype(int),
        confidence=confidence[keep],
        n_noise=int((is_pool & (aligned == NOISE)).sum()),
        candidates=scored,
        failed_candidates=failures,
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
    return replace(pseudo, method_name=f"{pseudo.method_name}+permuted", labels=shuffled)


def to_frame(pseudo: PseudoLabelSet) -> pd.DataFrame:
    """Tabular view, for the run artefacts."""
    return pd.DataFrame({"image_id": pseudo.image_ids, "pseudo_label": pseudo.labels})


__all__ = [
    "NOISE",
    "PseudoLabelSet",
    "align_by_majority",
    "cluster_confidence",
    "composition",
    "filter_by_confidence",
    "fit_pseudo_labels",
    "permute",
    "to_frame",
]
