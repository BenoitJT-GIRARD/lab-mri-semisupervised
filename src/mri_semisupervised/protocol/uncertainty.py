"""Intervals and paired comparisons.

With ninety-nine evaluation images and folds of twenty, a recall that moves by 0.05 has
moved by one image. Publishing a point estimate for a difference that size is not a
simplification, it is a claim the data cannot support. So the uncertainty is not an
appendix here: it is the result.

Two tools, and they answer different questions.

**Bootstrap on the pooled out-of-fold predictions** — how precisely do we know this arm's
performance? Resampling images, not folds, because the images are the unit of observation.

**Paired difference across shared folds** — is arm A better than arm B? The arms are
trained on exactly the same folds, so the comparison is paired. Treating them as two
independent samples would throw away the pairing and widen the interval for nothing.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Percentile bootstrap interval for a metric, resampling the images."""
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    rng = np.random.default_rng(seed)
    n = len(y_true)

    point = float(metric(y_true, y_score))
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(set(y_true[idx].tolist())) < 2:
            continue  # a resample with one class says nothing about a ranking metric
        draws.append(float(metric(y_true[idx], y_score[idx])))

    if not draws:
        return {"point": point, "ci_low": float("nan"), "ci_high": float("nan"), "n_boot": 0}

    low, high = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": point, "ci_low": float(low), "ci_high": float(high), "n_boot": len(draws)}


def paired_difference(
    per_fold_a: np.ndarray,
    per_fold_b: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Bootstrap the mean of the per-fold differences, and read a two-sided p-value off it.

    ``per_fold_a`` and ``per_fold_b`` must be aligned: entry *i* of each comes from the same
    fold, of the same repeat, with the same split. That alignment is the whole point.

    The p-value is the proportion of resampled means that fall on the other side of zero,
    doubled. It is a bootstrap p-value, not a t-test: with twenty-five folds and a bounded
    metric, normality is an assumption worth not making.
    """
    a, b = np.asarray(per_fold_a, dtype=float), np.asarray(per_fold_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"the arms must share their folds: {a.shape} against {b.shape}")

    differences = a - b
    finite = differences[np.isfinite(differences)]
    if len(finite) == 0:
        return {
            "mean_difference": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_value": float("nan"),
            "n_folds": 0,
        }

    rng = np.random.default_rng(seed)
    means = np.array(
        [finite[rng.integers(0, len(finite), size=len(finite))].mean() for _ in range(n_boot)]
    )
    observed = float(finite.mean())
    low, high = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    tail = float(min((means <= 0).mean(), (means >= 0).mean()))

    return {
        "mean_difference": observed,
        "ci_low": float(low),
        "ci_high": float(high),
        "p_value": float(min(1.0, 2 * tail)),
        "n_folds": len(finite),
    }


__all__ = ["bootstrap_ci", "paired_difference"]
