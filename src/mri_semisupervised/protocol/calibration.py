"""How much a probability from this model is worth.

The repository publishes ROC AUC, PR-AUC and a threshold chosen on validation. None of
them says anything about the **scale**: a score of 0.90 is not a claim that the risk is
90%. On a screening problem that is exactly what a clinical reader would look at, and it
is what makes a threshold transportable or not.

Everything here reads the out-of-fold predictions that are already on disk. Nothing is
retrained, and nothing is recalibrated: a network fine-tuned on twenty images per fold has
little chance of being calibrated, and measuring that and saying so *is* the result.
Recalibrating on this little data would add a layer of estimation noisier than the thing
it corrects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["brier", "expected_calibration_error", "reliability", "summarise_calibration"]


def brier(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Mean squared error between the probability and the outcome.

    Zero for a confident and correct prediction, 0.25 for a coin flip, 1.0 for a confident
    and wrong one. Unlike ROC AUC it punishes a badly scaled score even when the ranking is
    perfect, which is precisely the gap being measured.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean((y_score - y_true) ** 2))


def reliability(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Observed frequency against predicted probability, in equal-population bins.

    Equal population, not equal width. On fifteen hundred predictions clustered near 0 and
    1 — which is what a confident network produces — equal-width bins leave the middle
    nearly empty, and the calibration error becomes an average over noise. Equal-population
    bins put the same weight behind every point of the curve.

    Returns one row per bin: ``count``, ``mean_score``, ``observed`` and the signed ``gap``.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    if len(y_true) == 0:
        return pd.DataFrame(columns=["bin", "count", "mean_score", "observed", "gap"])

    order = np.argsort(y_score, kind="stable")
    chunks = np.array_split(order, min(n_bins, len(order)))

    rows = []
    for index, chunk in enumerate(chunks):
        if len(chunk) == 0:
            continue
        predicted = float(y_score[chunk].mean())
        observed = float(y_true[chunk].mean())
        rows.append(
            {
                "bin": index,
                "count": len(chunk),
                "mean_score": predicted,
                "observed": observed,
                "gap": observed - predicted,
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> float:
    """Average distance between predicted probability and observed frequency.

    Weighted by bin population, which with equal-population bins means a plain mean. Read
    it as: *by how much, on average, is this model's stated probability off.*
    """
    frame = reliability(y_true, y_score, n_bins)
    if frame.empty:
        return float("nan")
    weights = frame["count"].to_numpy(dtype=float)
    return float(np.average(np.abs(frame["gap"].to_numpy()), weights=weights))


def summarise_calibration(
    predictions: pd.DataFrame, n_bins: int = 10
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Per-arm Brier and ECE, plus the reliability curve of each arm.

    ``predictions`` is the run's ``predictions.parquet``: one row per image, per fold, per
    arm. The scores are pooled across folds, which is what makes the bins large enough to
    mean anything — twenty images per fold would not.
    """
    summary, curves = [], {}
    for arm, group in predictions.groupby("arm"):
        y_true = group["y_true"].to_numpy()
        y_score = group["y_score"].to_numpy()
        curves[str(arm)] = reliability(y_true, y_score, n_bins)
        summary.append(
            {
                "arm": arm,
                "n": len(group),
                "brier": brier(y_true, y_score),
                "ece": expected_calibration_error(y_true, y_score, n_bins),
                "mean_score": float(y_score.mean()),
                "base_rate": float(y_true.mean()),
            }
        )
    return pd.DataFrame(summary).sort_values("arm").reset_index(drop=True), curves
