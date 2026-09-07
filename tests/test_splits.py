"""Unit tests of the folds, the inner split and the derived seeds."""

from __future__ import annotations

from collections import Counter

import numpy as np

from mri_semisupervised.protocol.splits import derive_seed, inner_split, outer_folds

LABELS = np.array([0] * 49 + [1] * 50)  # the real balance: 49 normal, 50 cancer


def test_every_outer_fold_is_stratified_and_disjoint() -> None:
    for spec in outer_folds(LABELS, n_splits=5, n_repeats=2, seed=42):
        assert set(spec.train_idx).isdisjoint(spec.test_idx)
        assert len(spec.train_idx) + len(spec.test_idx) == len(LABELS)
        counts = Counter(LABELS[spec.test_idx])
        assert abs(counts[0] - counts[1]) <= 1


def test_every_image_is_tested_exactly_once_per_repeat() -> None:
    specs = list(outer_folds(LABELS, n_splits=5, n_repeats=3, seed=42))
    for repeat in range(3):
        tested = np.concatenate([s.test_idx for s in specs if s.repeat == repeat])
        assert sorted(tested.tolist()) == list(range(len(LABELS)))


def test_repeats_differ_but_the_sequence_replays() -> None:
    first = [tuple(s.test_idx) for s in outer_folds(LABELS, 5, 3, seed=42)]
    again = [tuple(s.test_idx) for s in outer_folds(LABELS, 5, 3, seed=42)]
    assert first == again
    assert len(set(first)) == 15, "5 folds x 3 repeats must all differ"


def test_a_different_seed_gives_different_folds() -> None:
    a = [tuple(s.test_idx) for s in outer_folds(LABELS, 5, 1, seed=42)]
    b = [tuple(s.test_idx) for s in outer_folds(LABELS, 5, 1, seed=43)]
    assert a != b


def test_the_inner_split_partitions_the_training_fold_only() -> None:
    spec = next(iter(outer_folds(LABELS, 5, 1, seed=42)))
    inner_train, inner_val = inner_split(LABELS, spec.train_idx, 0.25, spec.seed)

    assert set(inner_train).isdisjoint(inner_val)
    assert set(inner_train) | set(inner_val) == set(spec.train_idx)
    assert set(inner_val).isdisjoint(spec.test_idx), "validation cannot come from the test fold"


def test_the_inner_validation_holds_both_classes() -> None:
    # A fold of twenty images can easily produce a single-class validation set by accident.
    for spec in outer_folds(LABELS, 5, 2, seed=42):
        _, inner_val = inner_split(LABELS, spec.train_idx, 0.25, spec.seed)
        assert set(LABELS[inner_val]) == {0, 1}


def test_derived_seeds_are_stable_and_distinct() -> None:
    assert derive_seed(42, "fold", 0, 1) == derive_seed(42, "fold", 0, 1)
    assert derive_seed(42, "fold", 0, 1) != derive_seed(42, "fold", 0, 2)
    assert derive_seed(42, "fold", 0, 1) != derive_seed(43, "fold", 0, 1)
    assert 0 <= derive_seed(42, "anything") < 2**32
