"""Unit tests of the calibration measures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mri_semisupervised.protocol.calibration import (
    brier,
    expected_calibration_error,
    reliability,
    summarise_calibration,
)


def _calibrated(n: int = 20_000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Scores that mean exactly what they say: an outcome drawn at the stated rate."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.05, 0.95, n)
    y = (rng.uniform(size=n) < p).astype(int)
    return y, p


def test_a_perfectly_calibrated_score_has_near_zero_error() -> None:
    y, p = _calibrated()
    assert expected_calibration_error(y, p, n_bins=10) < 0.02


def test_pushing_the_scores_to_the_extremes_makes_them_worse() -> None:
    """Overconfidence is invisible to ROC AUC and is exactly what this measures."""
    y, p = _calibrated()
    overconfident = np.clip((p - 0.5) * 3 + 0.5, 0.0, 1.0)

    assert expected_calibration_error(y, overconfident, 10) > expected_calibration_error(y, p, 10)
    assert brier(y, overconfident) > brier(y, p)


def test_a_perfect_ranking_can_still_be_badly_calibrated() -> None:
    """The point of the module: the ranking is untouched, the scale is wrong."""
    from sklearn.metrics import roc_auc_score

    y = np.array([0] * 100 + [1] * 100)
    good = np.concatenate([np.full(100, 0.2), np.full(100, 0.8)])
    squashed = np.concatenate([np.full(100, 0.45), np.full(100, 0.55)])

    assert roc_auc_score(y, good) == roc_auc_score(y, squashed) == 1.0
    assert expected_calibration_error(y, squashed, 4) > expected_calibration_error(y, good, 4)


def test_brier_is_zero_on_a_confident_and_correct_prediction() -> None:
    assert brier(np.array([0, 1]), np.array([0.0, 1.0])) == 0.0
    assert brier(np.array([0, 1]), np.array([1.0, 0.0])) == 1.0


def test_the_bins_hold_equal_population() -> None:
    frame = reliability(np.array([0, 1] * 50), np.linspace(0, 1, 100), n_bins=10)
    assert len(frame) == 10
    assert frame["count"].nunique() == 1


def test_more_bins_than_points_does_not_produce_empty_bins() -> None:
    frame = reliability(np.array([0, 1, 1]), np.array([0.1, 0.6, 0.9]), n_bins=10)
    assert len(frame) == 3
    assert (frame["count"] > 0).all()


def test_an_empty_input_is_nan_rather_than_an_exception() -> None:
    empty = np.array([], dtype=float)
    assert np.isnan(brier(empty, empty))
    assert np.isnan(expected_calibration_error(empty, empty))
    assert reliability(empty, empty).empty


def test_the_summary_reports_one_row_per_arm_with_its_base_rate() -> None:
    rng = np.random.default_rng(1)
    frame = pd.DataFrame(
        {
            "arm": ["a"] * 200 + ["b"] * 200,
            "y_true": rng.integers(0, 2, 400),
            "y_score": rng.uniform(size=400),
        }
    )
    summary, curves = summarise_calibration(frame, n_bins=5)

    assert list(summary["arm"]) == ["a", "b"]
    assert (summary["n"] == 200).all()
    assert set(curves) == {"a", "b"}
    assert all(len(c) == 5 for c in curves.values())
    assert summary["base_rate"].between(0, 1).all()


def test_the_gap_column_is_signed_so_over_and_under_confidence_are_distinguishable() -> None:
    y = np.array([0] * 90 + [1] * 10)
    always_high = np.full(100, 0.9)

    frame = reliability(y, always_high, n_bins=2)
    assert (frame["gap"] < 0).all(), "claiming 0.9 on a 10% base rate is overconfidence"


@pytest.mark.parametrize("n_bins", [2, 5, 10, 20])
def test_the_error_stays_bounded_whatever_the_binning(n_bins: int) -> None:
    y, p = _calibrated(n=2000, seed=3)
    assert 0.0 <= expected_calibration_error(y, p, n_bins) <= 1.0
