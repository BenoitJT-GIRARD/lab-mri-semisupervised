"""Unit tests of the per-fold pseudo-labels.

Two tests carry the weight, and they only mean something as a pair: one flips the labels
the mask hides and asserts nothing moves, the other shows those same labels *would* have
changed the answer if they had been read. A comment saying "training labels only" cannot be
checked; that pair can.
"""

from __future__ import annotations

import numpy as np
import pytest

from mri_semisupervised.protocol.pseudo_labels import (
    NOISE,
    align_by_majority,
    fit_pseudo_labels,
    permute,
)


def _two_blobs(n_per_class: int = 60, dim: int = 24, seed: int = 0):
    """Two separable blobs, so the clustering has something real to find.

    The separation is spread over every dimension on purpose: the pipeline standardises
    before clustering, and a signal carried by a single dimension would be scaled down to
    the level of the noise — which is a real property of the pipeline, not of this fixture.
    """
    rng = np.random.default_rng(seed)
    centre = np.full(dim, 1.5)
    a = rng.normal(-centre, 1.0, size=(n_per_class, dim))
    b = rng.normal(+centre, 1.0, size=(n_per_class, dim))
    features = np.vstack([a, b])
    truth = np.array([0] * n_per_class + [1] * n_per_class)
    ids = np.array([f"img{i:03d}" for i in range(len(features))], dtype=object)
    return features, ids, truth


def _fold(ids, truth, n_train: int = 40):
    """First `n_train` of each class are the training fold, the next ten are the test fold."""
    zeros = np.where(truth == 0)[0]
    ones = np.where(truth == 1)[0]
    train = np.concatenate([zeros[:n_train], ones[:n_train]])
    test = np.concatenate([zeros[n_train : n_train + 10], ones[n_train : n_train + 10]])
    unlabelled = np.concatenate([zeros[n_train + 10 :], ones[n_train + 10 :]])
    return ids[train], truth[train], ids[test], ids[unlabelled]


def test_the_pseudo_labels_cover_the_unlabelled_pool_only() -> None:
    features, ids, truth = _two_blobs()
    train_ids, train_labels, test_ids, unlabelled_ids = _fold(ids, truth)

    result = fit_pseudo_labels(features, ids, unlabelled_ids, train_ids, train_labels, seed=0)

    assert set(result.image_ids).issubset(set(unlabelled_ids))
    assert set(result.image_ids).isdisjoint(set(test_ids))
    assert set(result.image_ids).isdisjoint(set(train_ids))


def test_poisoning_the_labels_outside_the_mask_changes_nothing() -> None:
    """The alignment must be deaf to every label the mask does not open.

    The old code passed all 100 labels and voted on all of them. Here the labels of the
    held-out entries are flipped to their opposite: if the mask is respected, the mapping
    cannot move.
    """
    clusters = np.array([0, 0, 0, 1, 1, 1])
    truth = np.array([0, 0, 1, 1, 1, 0])
    allowed = np.array([True, True, False, True, True, False])

    honest = align_by_majority(clusters, truth, allowed=allowed)
    poisoned_truth = truth.copy()
    poisoned_truth[~allowed] = 1 - poisoned_truth[~allowed]
    poisoned = align_by_majority(clusters, poisoned_truth, allowed=allowed)

    assert np.array_equal(honest, poisoned)


def test_reading_the_forbidden_labels_would_have_changed_the_answer() -> None:
    """The counterpart, without which the test above proves nothing.

    A mask is only meaningful if the labels it hides could have changed the outcome. Here
    the cluster holds two visible `normal` and three hidden `cancer`: masked, the vote says
    `normal`; unmasked — the old behaviour — it says `cancer`.
    """
    clusters = np.array([0, 0, 0, 0, 0])
    truth = np.array([0, 0, 1, 1, 1])
    allowed = np.array([True, True, False, False, False])

    masked = align_by_majority(clusters, truth, allowed=allowed)
    leaky = align_by_majority(clusters, truth, allowed=np.ones(5, dtype=bool))

    assert set(masked.tolist()) == {0}
    assert set(leaky.tolist()) == {1}


def test_flipping_a_training_label_does_change_the_outcome() -> None:
    # The counterpart: the training labels are read, so they must matter. A test that only
    # proves inertia would pass on a function that ignores every input.
    features, ids, truth = _two_blobs()
    train_ids, train_labels, _, unlabelled_ids = _fold(ids, truth)

    honest = fit_pseudo_labels(features, ids, unlabelled_ids, train_ids, train_labels, seed=0)
    flipped = fit_pseudo_labels(features, ids, unlabelled_ids, train_ids, 1 - train_labels, seed=0)

    assert np.array_equal(honest.labels, 1 - flipped.labels), "the alignment must follow the labels"


def test_the_method_is_selected_on_the_training_labels() -> None:
    features, ids, truth = _two_blobs()
    train_ids, train_labels, _, unlabelled_ids = _fold(ids, truth)

    result = fit_pseudo_labels(features, ids, unlabelled_ids, train_ids, train_labels, seed=0)

    assert result.method_name in result.candidates
    assert result.ari_on_train == max(result.candidates.values())
    assert result.ari_on_train > 0.5, "two separated blobs should be found"


def test_align_refuses_to_read_labels_it_was_not_given() -> None:
    clusters = np.array([0, 0, 1, 1])
    truth = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError, match="no label is allowed"):
        align_by_majority(clusters, truth, allowed=np.zeros(4, dtype=bool))


def test_a_cluster_no_visible_label_falls_into_stays_noise() -> None:
    clusters = np.array([0, 0, 1, 1])
    truth = np.array([0, 0, -1, -1])
    allowed = np.array([True, True, False, False])

    aligned = align_by_majority(clusters, truth, allowed=allowed)

    assert aligned[0] == aligned[1] == 0
    assert aligned[2] == aligned[3] == NOISE


def test_the_permutation_keeps_the_distribution_and_destroys_the_pairing() -> None:
    features, ids, truth = _two_blobs()
    train_ids, train_labels, _, unlabelled_ids = _fold(ids, truth)
    pseudo = fit_pseudo_labels(features, ids, unlabelled_ids, train_ids, train_labels, seed=0)

    controlled = permute(pseudo, seed=1)

    assert np.array_equal(controlled.image_ids, pseudo.image_ids)
    assert sorted(controlled.labels) == sorted(pseudo.labels)
    assert not np.array_equal(controlled.labels, pseudo.labels)
