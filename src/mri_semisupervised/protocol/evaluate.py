"""Metrics, and the separation the audit showed was missing.

The published comparison had the semi-supervised arm raising recall on the positive class
from 0.900 to 0.960 **while its ROC AUC fell** from 0.982 to 0.954. A model that ranks
better cannot do that. What had moved was the implicit decision threshold at 0.5, not the
discrimination.

So every result here comes in two parts:

* **threshold-free** — ROC AUC and PR-AUC, which say how well the scores rank;
* **at a threshold** — and the threshold is *chosen on the inner validation*, never on the
  test fold, with the 0.5 default reported alongside so the gap between the two is visible.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

DEFAULT_THRESHOLD = 0.5


def choose_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Pick the threshold that maximises F1 on the positive class.

    Evaluated on every midpoint between consecutive distinct scores, which is where the
    decision can actually change. Ties go to the lower threshold: on a screening problem,
    the cheaper mistake is the false positive.
    """
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    if len(set(y_true.tolist())) < 2:
        return DEFAULT_THRESHOLD

    unique = np.unique(y_score)
    if len(unique) == 1:
        return DEFAULT_THRESHOLD
    candidates = (unique[:-1] + unique[1:]) / 2.0

    best_threshold, best_f1 = DEFAULT_THRESHOLD, -1.0
    for threshold in candidates:
        score = f1_score(y_true, (y_score >= threshold).astype(int), zero_division=0)
        if score > best_f1:
            best_threshold, best_f1 = float(threshold), float(score)
    return best_threshold


SENSITIVITY_TARGET = 0.90


def threshold_at_sensitivity(
    y_true: np.ndarray, y_score: np.ndarray, target: float = SENSITIVITY_TARGET
) -> float:
    """The highest threshold that still catches ``target`` of the positives.

    ``choose_threshold`` maximises F1 on the positive class and breaks ties towards the
    false positive. That is defensible in general and it contradicts the frame this
    repository claims: on a screening problem the two errors do not cost the same, and F1
    treats them as if they did. It was pointed out more than once, and independently, that
    ``recall_positive`` was computed here and never governed a decision.

    So this is published beside the F1 threshold, not instead of it. The first serves the
    comparison between arms, where symmetry is what you want; the second serves the medical
    frame. Reading them together shows what the frame costs.

    Returns ``nan`` when the target cannot be met — on sixteen validation images that
    happens, and an honest nan beats a silent fallback to 0.5.
    """
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    positives = np.sort(y_score[y_true == 1])[::-1]
    if len(positives) == 0:
        return float("nan")

    needed = int(np.ceil(target * len(positives)))
    if needed <= 0 or needed > len(positives):
        return float("nan")
    return float(positives[needed - 1])


def threshold_free(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """The metrics that do not depend on where the line is drawn."""
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    if len(set(y_true.tolist())) < 2:
        return {"roc_auc": float("nan"), "pr_auc": float("nan")}
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        # PR-AUC answers the question a screening problem actually asks: of the cases the
        # model flags, how many are real, across every operating point.
        "pr_auc": float(average_precision_score(y_true, y_score)),
    }


def metrics_at(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float]:
    """Accuracy, F1 and per-class precision/recall at a given threshold."""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_score) >= threshold).astype(int)

    precision, recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1], zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_positive": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision_positive": float(precision[1]),
        "recall_positive": float(recall[1]),
        "precision_negative": float(precision[0]),
        "recall_negative": float(recall[0]),
        "true_negative": int(matrix[0, 0]),
        "false_positive": int(matrix[0, 1]),
        "false_negative": int(matrix[1, 0]),
        "true_positive": int(matrix[1, 1]),
    }


def evaluate_fold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    val_y_true: np.ndarray,
    val_y_score: np.ndarray,
) -> dict[str, float]:
    """Every number for one fold: threshold-free, at the chosen threshold, and at 0.5."""
    threshold = choose_threshold(val_y_true, val_y_score)
    result = threshold_free(y_true, y_score)
    result.update(metrics_at(y_true, y_score, threshold))
    default = metrics_at(y_true, y_score, DEFAULT_THRESHOLD)
    result.update({f"default_{k}": v for k, v in default.items() if k != "threshold"})

    # The sensitivity-first operating point, chosen on the validation split like every
    # other decision here, and reported on the test fold.
    sensitivity = threshold_at_sensitivity(val_y_true, val_y_score)
    if np.isnan(sensitivity):
        at_sensitivity = dict.fromkeys(
            ("recall_positive", "precision_positive", "f1_positive"), float("nan")
        )
    else:
        at_sensitivity = metrics_at(y_true, y_score, sensitivity)
    result["sensitivity_threshold"] = sensitivity
    result.update(
        {
            f"sensitivity_{k}": at_sensitivity[k]
            for k in ("recall_positive", "precision_positive", "f1_positive")
        }
    )
    return result


__all__ = [
    "DEFAULT_THRESHOLD",
    "SENSITIVITY_TARGET",
    "choose_threshold",
    "evaluate_fold",
    "metrics_at",
    "threshold_at_sensitivity",
    "threshold_free",
]
