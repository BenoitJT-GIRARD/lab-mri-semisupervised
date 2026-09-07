"""Which images the model gets wrong, and whether they have anything in common.

The fold reports give confusion counts. They do not say *which* images are missed, nor
whether it is the same ones every time. On ninety-nine images a recurring error is
identifiable by eye, and one question in particular is worth answering here: thirty-one of
the evaluation images also exist in the unlabelled pool. If the false negatives concentrate
on those, that is a fact about the dataset rather than about the model, and it belongs in
the README.

Every threshold used here is the one **its own fold** chose. Averaging the thresholds and
applying the average would judge each fold by a model that was never trained.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["per_image_errors", "persistent_false_negatives", "pool_overlap_rates"]


def per_image_errors(
    predictions: pd.DataFrame,
    per_fold: pd.DataFrame,
    manifest: pd.DataFrame,
    threshold_column: str = "threshold",
) -> pd.DataFrame:
    """Count, per image and per arm, how often it is misclassified.

    Returns one row per ``(image_id, arm)``: how many folds tested it, how many it got
    wrong, its mean score, its true label, and whether a copy of it also sits in the
    unlabelled pool.
    """
    thresholds = per_fold[["fold", "arm", threshold_column]].rename(
        columns={threshold_column: "fold_threshold"}
    )
    joined = predictions.merge(thresholds, on=["fold", "arm"], how="left", validate="many_to_one")
    if joined["fold_threshold"].isna().any():
        missing = joined.loc[joined["fold_threshold"].isna(), ["fold", "arm"]].drop_duplicates()
        raise ValueError(f"no threshold for {len(missing)} (fold, arm) pair(s)")

    joined["predicted"] = (joined["y_score"] >= joined["fold_threshold"]).astype(int)
    joined["wrong"] = (joined["predicted"] != joined["y_true"]).astype(int)
    joined["false_negative"] = ((joined["y_true"] == 1) & (joined["predicted"] == 0)).astype(int)

    in_pool = set(manifest.loc[manifest["pool"] == "unlabelled", "image_id"])

    grouped = (
        joined.groupby(["image_id", "arm"])
        .agg(
            n_folds=("wrong", "size"),
            n_wrong=("wrong", "sum"),
            n_false_negative=("false_negative", "sum"),
            mean_score=("y_score", "mean"),
            y_true=("y_true", "first"),
        )
        .reset_index()
    )
    grouped["wrong_share"] = grouped["n_wrong"] / grouped["n_folds"]
    grouped["also_in_pool"] = grouped["image_id"].isin(in_pool)
    return grouped.sort_values(["arm", "wrong_share"], ascending=[True, False]).reset_index(
        drop=True
    )


def persistent_false_negatives(errors: pd.DataFrame, min_share: float = 0.8) -> pd.DataFrame:
    """Cancer images missed on at least ``min_share`` of the folds that tested them.

    A single miss is noise on twenty images. A miss on four folds out of five is a property
    of the image, and those are the ones worth looking at.
    """
    positives = errors[errors["y_true"] == 1]
    share = positives["n_false_negative"] / positives["n_folds"]
    return positives[share >= min_share].sort_values("mean_score").reset_index(drop=True)


def pool_overlap_rates(errors: pd.DataFrame) -> pd.DataFrame:
    """Error rate split by whether the image also lives in the unlabelled pool.

    The question this module exists to answer. A gap here says the duplicates are harder or
    easier than the rest of the evaluation set, which is a statement about the dataset that
    no aggregate metric would ever surface.
    """
    rows = []
    for (arm, in_pool), group in errors.groupby(["arm", "also_in_pool"]):
        rows.append(
            {
                "arm": arm,
                "also_in_pool": bool(in_pool),
                "n_images": len(group),
                "wrong_share": float(np.average(group["wrong_share"], weights=group["n_folds"])),
                "false_negative_share": float(
                    group["n_false_negative"].sum() / max(group["n_folds"].sum(), 1)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["arm", "also_in_pool"]).reset_index(drop=True)
