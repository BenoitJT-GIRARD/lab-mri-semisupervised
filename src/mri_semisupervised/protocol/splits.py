"""Folds, inner validation splits, and the seeds that make a run reproducible.

Two ideas carry the module.

**Repeated cross-validation.** With 99 evaluation images, one fold holds twenty. A recall
that moves by 0.05 has moved by one image. Repeating the whole cross-validation under
different splitting seeds separates the variance that comes from *where the split fell*
from the variance that comes from *how the network initialised* — and it is the only way to
say anything honest about an effect of that size.

**Derived seeds.** Every fold, every arm and every permutation draws its seed from the run
seed and its own coordinates. Nothing depends on the order in which things happen, so a
single fold can be re-run on its own and land on the same numbers.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


def derive_seed(base: int, *parts: object) -> int:
    """Derive a stable child seed from a base seed and a set of coordinates.

    Stable across processes and across Python versions: ``hash()`` is not, so the seed
    comes from a digest of the printed coordinates.
    """
    payload = "|".join([str(base), *(str(p) for p in parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


@dataclass(frozen=True)
class FoldSpec:
    """One outer fold: who trains, who is held out, and under which seed."""

    repeat: int
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    seed: int

    @property
    def name(self) -> str:
        return f"r{self.repeat}f{self.fold}"


def outer_folds(
    labels: np.ndarray,
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = 42,
) -> Iterator[FoldSpec]:
    """Yield the outer folds of a repeated stratified cross-validation.

    Each repeat re-shuffles under its own seed, so the folds of two repeats differ; the
    whole sequence is a function of ``seed`` alone, so it replays identically.
    """
    labels = np.asarray(labels)
    for repeat in range(n_repeats):
        repeat_seed = derive_seed(seed, "repeat", repeat)
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=repeat_seed)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(labels, labels)):
            yield FoldSpec(
                repeat=repeat,
                fold=fold,
                train_idx=train_idx,
                test_idx=test_idx,
                seed=derive_seed(seed, "fold", repeat, fold),
            )


def inner_split(
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_fraction: float = 0.25,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a training fold into an inner training set and an inner validation set.

    This is what the checkpoint and the decision threshold are chosen on. It is carved out
    of the training fold and never touches the test fold — which is the whole point.

    Stratified, so that a fold of twenty images does not end up with a validation set that
    holds a single class.
    """
    train_idx = np.asarray(train_idx)
    labels = np.asarray(labels)
    inner_train, inner_val = train_test_split(
        train_idx,
        test_size=val_fraction,
        stratify=labels[train_idx],
        random_state=seed,
        shuffle=True,
    )
    return np.sort(inner_train), np.sort(inner_val)


__all__ = ["FoldSpec", "derive_seed", "inner_split", "outer_folds"]
