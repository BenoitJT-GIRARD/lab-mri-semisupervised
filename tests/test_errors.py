"""Unit tests of the per-image error analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mri_semisupervised.protocol.errors import (
    per_image_errors,
    persistent_false_negatives,
    pool_overlap_rates,
)


def _manifest(pool_ids: list[str], labelled_ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_id": labelled_ids + pool_ids,
            "pool": ["labelled"] * len(labelled_ids) + ["unlabelled"] * len(pool_ids),
        }
    )


def _predictions(rows: list[tuple[str, str, str, int, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["fold", "arm", "image_id", "y_true", "y_score"])


def _per_fold(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["fold", "arm", "threshold"])


def test_an_image_wrong_on_every_fold_is_counted_as_such() -> None:
    predictions = _predictions(
        [
            ("f0", "a", "img", 1, 0.10),
            ("f1", "a", "img", 1, 0.20),
        ]
    )
    per_fold = _per_fold([("f0", "a", 0.5), ("f1", "a", 0.5)])

    out = per_image_errors(predictions, per_fold, _manifest([], ["img"]))

    assert out.loc[0, "n_folds"] == 2
    assert out.loc[0, "n_wrong"] == 2
    assert out.loc[0, "n_false_negative"] == 2
    assert out.loc[0, "wrong_share"] == 1.0


def test_each_fold_judges_with_its_own_threshold() -> None:
    """Averaging the thresholds would judge a fold by a model that was never trained."""
    predictions = _predictions(
        [
            ("f0", "a", "img", 1, 0.40),
            ("f1", "a", "img", 1, 0.40),
        ]
    )
    per_fold = _per_fold([("f0", "a", 0.30), ("f1", "a", 0.60)])

    out = per_image_errors(predictions, per_fold, _manifest([], ["img"]))

    assert out.loc[0, "n_wrong"] == 1, "correct under the lenient fold, wrong under the strict one"


def test_a_missing_threshold_raises_rather_than_dropping_the_row() -> None:
    predictions = _predictions([("f0", "a", "img", 1, 0.4), ("f9", "a", "img", 1, 0.4)])
    per_fold = _per_fold([("f0", "a", 0.5)])

    with pytest.raises(ValueError, match="no threshold"):
        per_image_errors(predictions, per_fold, _manifest([], ["img"]))


def test_pool_membership_comes_from_the_manifest() -> None:
    predictions = _predictions([("f0", "a", "dup", 1, 0.9), ("f0", "a", "solo", 0, 0.1)])
    per_fold = _per_fold([("f0", "a", 0.5)])

    out = per_image_errors(predictions, per_fold, _manifest(["dup"], ["dup", "solo"]))

    assert bool(out.loc[out["image_id"] == "dup", "also_in_pool"].iloc[0]) is True
    assert bool(out.loc[out["image_id"] == "solo", "also_in_pool"].iloc[0]) is False


def test_persistent_false_negatives_keep_only_positives_missed_often_enough() -> None:
    errors = pd.DataFrame(
        {
            "image_id": ["a", "b", "c"],
            "arm": ["x"] * 3,
            "n_folds": [5, 5, 5],
            "n_false_negative": [5, 2, 0],
            "n_wrong": [5, 2, 0],
            "mean_score": [0.05, 0.4, 0.9],
            "y_true": [1, 1, 0],
            "wrong_share": [1.0, 0.4, 0.0],
            "also_in_pool": [False, False, False],
        }
    )
    out = persistent_false_negatives(errors, min_share=0.8)

    assert list(out["image_id"]) == ["a"], "b is missed too rarely, c is not a positive"


def test_the_overlap_rates_split_the_two_populations() -> None:
    errors = pd.DataFrame(
        {
            "image_id": ["a", "b"],
            "arm": ["x", "x"],
            "n_folds": [5, 5],
            "n_false_negative": [5, 0],
            "n_wrong": [5, 0],
            "mean_score": [0.1, 0.9],
            "y_true": [1, 1],
            "wrong_share": [1.0, 0.0],
            "also_in_pool": [True, False],
        }
    )
    out = pool_overlap_rates(errors)

    assert len(out) == 2
    inside = out[out["also_in_pool"]].iloc[0]
    outside = out[~out["also_in_pool"]].iloc[0]
    assert inside["wrong_share"] == 1.0
    assert outside["wrong_share"] == 0.0


def test_the_rates_weight_images_by_how_often_they_were_tested() -> None:
    """An image tested twice must not weigh as much as one tested ten times."""
    errors = pd.DataFrame(
        {
            "image_id": ["a", "b"],
            "arm": ["x", "x"],
            "n_folds": [1, 9],
            "n_false_negative": [1, 0],
            "n_wrong": [1, 0],
            "mean_score": [0.1, 0.9],
            "y_true": [1, 1],
            "wrong_share": [1.0, 0.0],
            "also_in_pool": [False, False],
        }
    )
    out = pool_overlap_rates(errors)
    assert out.loc[0, "wrong_share"] == pytest.approx(0.1)


def test_the_arms_are_kept_apart() -> None:
    predictions = _predictions([("f0", "a", "img", 1, 0.9), ("f0", "b", "img", 1, 0.1)])
    per_fold = _per_fold([("f0", "a", 0.5), ("f0", "b", 0.5)])

    out = per_image_errors(predictions, per_fold, _manifest([], ["img"]))

    assert len(out) == 2
    assert set(out["arm"]) == {"a", "b"}
    assert out.set_index("arm").loc["a", "n_wrong"] == 0
    assert out.set_index("arm").loc["b", "n_wrong"] == 1


def test_the_mean_score_is_the_mean_over_the_folds_that_tested_it() -> None:
    predictions = _predictions([("f0", "a", "img", 1, 0.2), ("f1", "a", "img", 1, 0.8)])
    per_fold = _per_fold([("f0", "a", 0.5), ("f1", "a", 0.5)])

    out = per_image_errors(predictions, per_fold, _manifest([], ["img"]))
    assert out.loc[0, "mean_score"] == pytest.approx(0.5)
    assert np.isclose(out.loc[0, "wrong_share"], 0.5)
