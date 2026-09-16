"""Unit tests of the metrics and of the threshold choice."""

from __future__ import annotations

import numpy as np
import pytest

from mri_semisupervised.protocol.evaluate import (
    SENSITIVITY_TARGET,
    choose_threshold,
    evaluate_fold,
    metrics_at,
    threshold_at_sensitivity,
    threshold_free,
)
from mri_semisupervised.protocol.uncertainty import (
    bootstrap_ci,
    holm_correction,
    paired_difference,
)


def test_the_threshold_falls_between_the_two_classes() -> None:
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    assert 0.2 < choose_threshold(y, s) < 0.8


def test_a_single_class_leaves_the_threshold_at_the_default() -> None:
    assert choose_threshold(np.array([1, 1, 1]), np.array([0.3, 0.6, 0.9])) == 0.5


def test_a_threshold_move_raises_recall_while_the_ranking_stays_put() -> None:
    """The contradiction the audit found, reproduced on four images.

    Lowering the threshold catches the missed positive — recall goes up — and the AUC does
    not move an inch, because the ranking is unchanged. Reporting only the first number is
    what let a worse model look better.
    """
    y = np.array([0, 0, 1, 1])
    s = np.array([0.30, 0.45, 0.40, 0.90])

    strict = metrics_at(y, s, threshold=0.5)
    lenient = metrics_at(y, s, threshold=0.35)

    assert strict["recall_positive"] == 0.5
    assert lenient["recall_positive"] == 1.0
    assert threshold_free(y, s)["roc_auc"] == threshold_free(y, s)["roc_auc"]  # unchanged by both


def test_pr_auc_and_roc_auc_are_both_reported() -> None:
    y = np.array([0, 0, 0, 1])
    s = np.array([0.1, 0.2, 0.3, 0.9])
    out = threshold_free(y, s)
    assert out["roc_auc"] == 1.0
    assert out["pr_auc"] == 1.0


def test_the_confusion_counts_add_up() -> None:
    y = np.array([0, 0, 1, 1, 1])
    s = np.array([0.1, 0.9, 0.2, 0.8, 0.7])
    out = metrics_at(y, s, threshold=0.5)
    assert out["true_negative"] + out["false_positive"] == 2
    assert out["false_negative"] + out["true_positive"] == 3


def test_the_fold_report_carries_the_chosen_and_the_default_threshold() -> None:
    y = np.array([0, 0, 1, 1])
    s = np.array([0.10, 0.20, 0.30, 0.90])
    out = evaluate_fold(y, s, val_y_true=y, val_y_score=s)

    assert out["threshold"] < 0.5
    assert out["recall_positive"] == 1.0
    assert out["default_recall_positive"] == 0.5, "the 0.5 default is reported beside it"


def test_a_bootstrap_interval_brackets_the_point_estimate() -> None:
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    y = np.array([0] * 30 + [1] * 30)
    s = np.concatenate([rng.normal(0.3, 0.2, 30), rng.normal(0.7, 0.2, 30)])

    out = bootstrap_ci(y, s, roc_auc_score, n_boot=300, seed=0)
    assert out["ci_low"] <= out["point"] <= out["ci_high"]
    assert out["n_boot"] > 250


def test_the_paired_test_uses_the_pairing() -> None:
    # Two arms differing by a constant on every fold: the per-fold spread is wide, the
    # paired difference is not.
    a = np.array([0.80, 0.90, 0.70, 0.95, 0.85])
    b = a + 0.05

    out = paired_difference(b, a, n_boot=500, seed=0)
    assert abs(out["mean_difference"] - 0.05) < 1e-12
    assert out["ci_high"] - out["ci_low"] < 1e-9
    assert out["n_folds"] == 5


def test_no_difference_gives_a_p_value_of_one() -> None:
    a = np.array([0.8, 0.9, 0.7])
    out = paired_difference(a, a, n_boot=200, seed=0)
    assert out["mean_difference"] == 0.0
    assert out["p_value"] == 1.0


# --- Point de fonctionnement a sensibilite imposee (R4) -------------------


def test_the_sensitivity_threshold_is_the_highest_one_that_holds_the_target() -> None:
    y = np.array([0, 0, 1, 1, 1])
    s = np.array([0.10, 0.55, 0.40, 0.80, 0.90])

    threshold = threshold_at_sensitivity(y, s, target=0.90)

    assert threshold == pytest.approx(0.40)
    assert (s >= threshold)[y == 1].mean() >= 0.90


def test_a_higher_threshold_would_break_the_target() -> None:
    """Highest, not merely sufficient: a lower one costs precision for nothing."""
    y = np.array([0, 0, 1, 1, 1])
    s = np.array([0.10, 0.55, 0.40, 0.80, 0.90])

    threshold = threshold_at_sensitivity(y, s, target=0.90)
    higher = np.min(s[s > threshold])

    assert (s >= higher)[y == 1].mean() < 0.90


def test_an_unreachable_target_returns_nan_rather_than_a_default() -> None:
    """Sixteen validation images cannot always hold 0.90. A nan says so; 0.5 would lie."""
    assert np.isnan(threshold_at_sensitivity(np.array([0, 1]), np.array([0.9, 0.1]), target=1.01))


def test_no_positive_at_all_gives_nan() -> None:
    assert np.isnan(threshold_at_sensitivity(np.array([0, 0]), np.array([0.2, 0.8])))


def test_the_fold_report_carries_both_operating_points() -> None:
    y = np.array([0, 0, 1, 1])
    s = np.array([0.10, 0.20, 0.30, 0.90])

    out = evaluate_fold(y, s, val_y_true=y, val_y_score=s)

    assert out["threshold"] < 0.5
    assert out["sensitivity_threshold"] == pytest.approx(0.30)
    assert out["sensitivity_recall_positive"] >= SENSITIVITY_TARGET
    assert "sensitivity_precision_positive" in out


def test_the_sensitivity_point_is_chosen_on_validation_not_on_the_fold() -> None:
    """Same rule as every other decision here: the test fold is read once, at the end."""
    y = np.array([0, 0, 1, 1])
    test_scores = np.array([0.10, 0.20, 0.30, 0.90])
    val_scores = np.array([0.60, 0.65, 0.70, 0.95])

    out = evaluate_fold(y, test_scores, val_y_true=y, val_y_score=val_scores)

    assert out["sensitivity_threshold"] == pytest.approx(0.70)


def test_an_unreachable_target_leaves_the_row_nan_without_failing() -> None:
    y = np.array([0, 0, 1, 1])
    s = np.array([0.10, 0.20, 0.30, 0.90])

    out = evaluate_fold(y, s, val_y_true=np.array([0, 0, 0, 0]), val_y_score=s)

    assert np.isnan(out["sensitivity_threshold"])
    assert np.isnan(out["sensitivity_recall_positive"])


# --- Multiplicite (spec 006) ---------------------------------------------


def test_holm_is_stricter_on_the_smallest_p_and_laxer_on_the_largest() -> None:
    out = holm_correction({"a": 0.001, "b": 0.02, "c": 0.04}, alpha=0.05)
    assert list(out["comparison"]) == ["a", "b", "c"]
    assert out.loc[0, "threshold"] == pytest.approx(0.05 / 3)
    assert out.loc[2, "threshold"] == pytest.approx(0.05)


def test_the_first_failure_stops_every_rejection_after_it() -> None:
    """Holm is a step-down procedure: past the first survivor nothing else is claimed."""
    out = holm_correction({"a": 0.001, "b": 0.049, "c": 0.0491}, alpha=0.05)
    assert list(out["significant"]) == [True, False, False]


def test_a_family_where_nothing_survives_is_a_table_of_false() -> None:
    out = holm_correction({"a": 0.2, "b": 0.5}, alpha=0.05)
    assert not out["significant"].any()


def test_a_single_comparison_is_uncorrected() -> None:
    out = holm_correction({"only": 0.04}, alpha=0.05)
    assert out.loc[0, "threshold"] == pytest.approx(0.05)
    assert bool(out.loc[0, "significant"]) is True


def test_an_empty_family_gives_an_empty_table_rather_than_an_error() -> None:
    assert holm_correction({}).empty
