"""Draw the figures the README publishes, from the artefacts of a run.

Every figure here reads `reports/experiments/<mode>/`, so a picture in the README and a
number in the README come from the same run. Regenerating them is one command, and it is
the only way they get regenerated: none is edited by hand.

Usage:
    uv run python scripts/build_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib.pyplot as plt
import pandas as pd

from mri_semisupervised.config import (
    EXPERIMENTS_DIR,
    FIGURES_DIR,
    MANIFEST_PATH,
    ensure_dirs,
)
from mri_semisupervised.protocol.calibration import summarise_calibration
from mri_semisupervised.protocol.errors import per_image_errors, pool_overlap_rates
from mri_semisupervised.protocol.uncertainty import paired_difference
from mri_semisupervised.viz.plots import plot_roc_compare

ARM_LABEL = {
    "supervised": "supervised",
    "semi_supervised": "semi-supervised",
    "semi_supervised_confident": "semi-supervised, filtered",
    "permuted_control": "permuted control",
}
ARM_COLOUR = {
    "supervised": "#4c78a8",
    "semi_supervised": "#f58518",
    "semi_supervised_confident": "#e45756",
    "permuted_control": "#9c9c9c",
}
BUDGETS = (10, 20, 40)


def _load(mode: str) -> tuple[pd.DataFrame, pd.DataFrame, dict] | None:
    directory = EXPERIMENTS_DIR / mode
    if not (directory / "per_fold.parquet").exists():
        print(f"[warn] no {mode} run under {directory}")
        return None
    return (
        pd.read_parquet(directory / "per_fold.parquet"),
        pd.read_parquet(directory / "predictions.parquet"),
        json.loads((directory / "manifest.json").read_text(encoding="utf-8")),
    )


def figure_roc(predictions: pd.DataFrame, path: Path) -> None:
    """Pooled out-of-fold ROC for the three arms, on one axis."""
    curves = {
        ARM_LABEL.get(arm, arm): (group["y_true"].to_numpy(), group["y_score"].to_numpy())
        for arm, group in predictions.groupby("arm")
    }
    figure = plot_roc_compare(curves, title="Pooled out-of-fold ROC, 5 folds x 5 repeats")
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"[ok] {path.name}")


def figure_arms(per_fold: pd.DataFrame, path: Path) -> None:
    """Per-arm means with the spread across folds, on the metrics that matter."""
    metrics = ["roc_auc", "pr_auc", "recall_positive", "f1_macro"]
    titles = ["ROC AUC", "PR-AUC", "recall (cancer)", "F1 macro"]
    arms = [a for a in ARM_LABEL if a in set(per_fold["arm"])]

    figure, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.6))
    for axis, metric, title in zip(axes, metrics, titles, strict=True):
        means = [per_fold.loc[per_fold["arm"] == a, metric].mean() for a in arms]
        errors = [per_fold.loc[per_fold["arm"] == a, metric].std() for a in arms]
        axis.bar(
            range(len(arms)),
            means,
            yerr=errors,
            capsize=5,
            color=[ARM_COLOUR[a] for a in arms],
        )
        axis.set_xticks(range(len(arms)))
        axis.set_xticklabels([ARM_LABEL[a] for a in arms], rotation=20, ha="right")
        axis.set_title(title)
        axis.set_ylim(0.5, 1.0)
        axis.grid(axis="y", alpha=0.3)
    figure.suptitle("Three arms, same folds, same budget — bars are one standard deviation")
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"[ok] {path.name}")


def figure_paired(per_fold: pd.DataFrame, path: Path) -> None:
    """The paired differences, with their intervals. Zero is the line that matters."""
    pivot = per_fold.pivot(index="fold", columns="arm")
    pairs = [
        ("semi_supervised", "supervised"),
        ("semi_supervised", "permuted_control"),
        ("permuted_control", "supervised"),
    ]
    metrics = ["roc_auc", "pr_auc", "recall_positive"]

    rows = []
    for metric in metrics:
        for a, b in pairs:
            if (metric, a) not in pivot or (metric, b) not in pivot:
                continue
            out = paired_difference(pivot[(metric, a)].to_numpy(), pivot[(metric, b)].to_numpy())
            rows.append({"label": f"{ARM_LABEL[a]}\nminus {ARM_LABEL[b]}", "metric": metric, **out})
    frame = pd.DataFrame(rows)

    figure, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.4), sharey=True)
    for axis, metric in zip(axes, metrics, strict=True):
        subset = frame[frame["metric"] == metric].reset_index(drop=True)
        positions = range(len(subset))
        axis.errorbar(
            subset["mean_difference"],
            positions,
            xerr=[
                subset["mean_difference"] - subset["ci_low"],
                subset["ci_high"] - subset["mean_difference"],
            ],
            fmt="o",
            color="#333333",
            capsize=4,
        )
        axis.axvline(0.0, color="#c0392b", linestyle="--", linewidth=1)
        axis.set_yticks(list(positions))
        axis.set_yticklabels(subset["label"], fontsize=8)
        axis.set_title(metric)
        axis.grid(axis="x", alpha=0.3)
    figure.suptitle("Paired differences across shared folds, 95% bootstrap intervals")
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"[ok] {path.name}")


def figure_leak_price(corrected: pd.DataFrame, legacy: pd.DataFrame, path: Path) -> None:
    """What the three leaks were worth, arm by arm."""
    metrics = ["roc_auc", "recall_positive"]
    arms = [a for a in ARM_LABEL if a in set(corrected["arm"]) & set(legacy["arm"])]

    figure, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 3.6))
    width = 0.35
    for axis, metric in zip(axes, metrics, strict=True):
        legacy_means = [legacy.loc[legacy["arm"] == a, metric].mean() for a in arms]
        corrected_means = [corrected.loc[corrected["arm"] == a, metric].mean() for a in arms]
        positions = range(len(arms))
        axis.bar(
            [p - width / 2 for p in positions],
            legacy_means,
            width,
            label="with the three leaks",
            color="#c0392b",
            alpha=0.8,
        )
        axis.bar(
            [p + width / 2 for p in positions],
            corrected_means,
            width,
            label="leaks closed",
            color="#4c78a8",
        )
        axis.set_xticks(list(positions))
        axis.set_xticklabels([ARM_LABEL[a] for a in arms], rotation=20, ha="right")
        axis.set_title(metric)
        axis.set_ylim(0.5, 1.0)
        axis.grid(axis="y", alpha=0.3)
    axes[0].legend(loc="lower left", fontsize=8)
    figure.suptitle("What the three leaks were worth — everything else held identical")
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"[ok] {path.name}")


def figure_calibration(predictions: pd.DataFrame, path: Path) -> None:
    """Reliability curves: what a score of 0.4 is actually worth.

    ROC AUC cannot see this. A model whose ranking is perfect and whose scale is squashed
    scores 1.0 and still tells a clinician the wrong number.
    """
    summary, curves = summarise_calibration(predictions, n_bins=10)
    figure, axis = plt.subplots(figsize=(6.5, 6))
    axis.plot([0, 1], [0, 1], color="#bbbbbb", linestyle="--", linewidth=1, label="perfect")

    for arm, curve in curves.items():
        row = summary[summary["arm"] == arm].iloc[0]
        axis.plot(
            curve["mean_score"],
            curve["observed"],
            marker="o",
            markersize=4,
            color=ARM_COLOUR.get(arm, "#333333"),
            label=f"{ARM_LABEL.get(arm, arm)} — Brier {row['brier']:.3f}, ECE {row['ece']:.3f}",
        )

    axis.set_xlabel("mean predicted probability (equal-population bins)")
    axis.set_ylabel("observed cancer rate")
    axis.set_title("Calibration, pooled out of fold")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.legend(fontsize=8, loc="upper left")
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"[ok] {path.name}")


def figure_label_efficiency(path: Path) -> pd.DataFrame | None:
    """Two panels; the second is the one that answers the question.

    The top panel shows the task getting easier as labels are added, which is expected and
    says nothing about semi-supervision. Only the paired difference against the permuted
    control does, and it gets its own axis so it cannot be read off the first by eye.
    """
    points = []
    for budget in BUDGETS:
        loaded = _load(f"budget-{budget}")
        if loaded is None:
            continue
        points.append((budget, loaded[0]))
    full = _load("corrected")
    if full is not None:
        points.append((int(full[2]["protocol"].get("label_budget") or 59), full[0]))
    if len(points) < 2:
        print("[warn] not enough budgets to draw the curve")
        return None

    rows = []
    for budget, per_fold in sorted(points):
        pivot = per_fold.pivot(index="fold", columns="arm", values="roc_auc")
        for arm in pivot.columns:
            rows.append(
                {
                    "budget": budget,
                    "arm": arm,
                    "mean": float(pivot[arm].mean()),
                    "sd": float(pivot[arm].std()),
                }
            )
        if {"semi_supervised", "permuted_control"} <= set(pivot.columns):
            out = paired_difference(
                pivot["semi_supervised"].to_numpy(), pivot["permuted_control"].to_numpy()
            )
            rows.append(
                {
                    "budget": budget,
                    "arm": "difference",
                    "mean": out["mean_difference"],
                    "ci_low": out["ci_low"],
                    "ci_high": out["ci_high"],
                    "p_value": out["p_value"],
                }
            )
    frame = pd.DataFrame(rows)

    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(7, 7.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    for arm in ("supervised", "semi_supervised", "permuted_control"):
        part = frame[frame["arm"] == arm].sort_values("budget")
        if part.empty:
            continue
        top.errorbar(
            part["budget"],
            part["mean"],
            yerr=part["sd"],
            marker="o",
            capsize=3,
            color=ARM_COLOUR.get(arm, "#333333"),
            label=ARM_LABEL.get(arm, arm),
        )
    top.set_ylabel("ROC AUC")
    top.set_title("Label efficiency: the task, and then the question")
    top.legend(fontsize=9)

    diff = frame[frame["arm"] == "difference"].sort_values("budget")
    bottom.axhline(0.0, color="#bbbbbb", linestyle="--", linewidth=1)
    bottom.errorbar(
        diff["budget"],
        diff["mean"],
        yerr=[diff["mean"] - diff["ci_low"], diff["ci_high"] - diff["mean"]],
        marker="o",
        capsize=3,
        color="#f58518",
    )
    bottom.set_xlabel("training labels per fold")
    bottom.set_ylabel("semi-supervised minus control")
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"[ok] {path.name}")
    return frame


def figure_hardest_images(errors: pd.DataFrame, manifest: pd.DataFrame, path: Path) -> None:
    """The twelve images the supervised arm misses most, with what it said about them.

    An aggregate rate says a model is wrong 7% of the time. It does not say the misses are
    the same handful of scans every fold, which on ninety-nine images is checkable by eye.
    """
    from PIL import Image

    worst = (
        errors[errors["arm"] == "supervised"]
        .sort_values(["wrong_share", "n_wrong"], ascending=False)
        .head(12)
    )
    paths = dict(zip(manifest["image_id"], manifest["path"], strict=True))

    figure, axes = plt.subplots(3, 4, figsize=(11, 9))
    for axis, row in zip(axes.ravel(), worst.itertuples(), strict=False):
        source = paths.get(row.image_id)
        if source and Path(source).exists():
            with Image.open(source) as image:
                axis.imshow(image.convert("L"), cmap="gray")
        truth = "cancer" if row.y_true == 1 else "normal"
        pool = ", also in pool" if row.also_in_pool else ""
        axis.set_title(
            f"{truth}, score {row.mean_score:.2f}\nmissed {row.n_wrong}/{row.n_folds}{pool}",
            fontsize=8,
        )
        axis.axis("off")
    for axis in axes.ravel()[len(worst) :]:
        axis.axis("off")

    figure.suptitle("What the supervised arm gets wrong, and how confidently", fontsize=11)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"[ok] {path.name}")


def main() -> None:
    ensure_dirs()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    loaded = _load("corrected")
    if loaded is None:
        print("[warn] run scripts/run_experiment.py first")
        return
    per_fold, predictions, meta = loaded
    print(f"[info] corrected run {meta['run_id']}, fingerprint {meta['dataset_fingerprint']}")

    figure_roc(predictions, FIGURES_DIR / "roc_arms.png")
    figure_arms(per_fold, FIGURES_DIR / "arms_comparison.png")
    figure_paired(per_fold, FIGURES_DIR / "paired_differences.png")

    figure_calibration(predictions, FIGURES_DIR / "calibration.png")

    legacy_loaded = _load("legacy")
    if legacy_loaded is not None:
        figure_leak_price(per_fold, legacy_loaded[0], FIGURES_DIR / "leak_price.png")

    manifest_path = Path(MANIFEST_PATH)
    if manifest_path.exists():
        manifest = pd.read_parquet(manifest_path)
        errors = per_image_errors(predictions, per_fold, manifest)
        target = EXPERIMENTS_DIR / "corrected" / "per_image_errors.parquet"
        errors.to_parquet(target, index=False)
        print(f"[ok] {target.name}")
        print(pool_overlap_rates(errors).round(3).to_string(index=False))
        figure_hardest_images(errors, manifest, FIGURES_DIR / "hardest_images.png")

    curve = figure_label_efficiency(FIGURES_DIR / "label_efficiency.png")
    if curve is not None:
        target = EXPERIMENTS_DIR / "label_efficiency.parquet"
        curve.to_parquet(target, index=False)
        print(f"[ok] {target.name}")


if __name__ == "__main__":
    main()
