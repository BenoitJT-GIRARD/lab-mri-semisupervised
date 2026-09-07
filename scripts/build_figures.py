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

from mri_semisupervised.config import EXPERIMENTS_DIR, FIGURES_DIR, ensure_dirs
from mri_semisupervised.protocol.uncertainty import paired_difference
from mri_semisupervised.viz.plots import plot_roc_compare

ARM_LABEL = {
    "supervised": "supervised",
    "semi_supervised": "semi-supervised",
    "permuted_control": "permuted control",
}
ARM_COLOUR = {
    "supervised": "#4c78a8",
    "semi_supervised": "#f58518",
    "permuted_control": "#9c9c9c",
}


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
        axis.bar([p - width / 2 for p in positions], legacy_means, width,
                 label="original protocol", color="#c0392b", alpha=0.8)
        axis.bar([p + width / 2 for p in positions], corrected_means, width,
                 label="leak-free protocol", color="#4c78a8")
        axis.set_xticks(list(positions))
        axis.set_xticklabels([ARM_LABEL[a] for a in arms], rotation=20, ha="right")
        axis.set_title(metric)
        axis.set_ylim(0.5, 1.0)
        axis.grid(axis="y", alpha=0.3)
    axes[0].legend(loc="lower left", fontsize=8)
    figure.suptitle("What the three leaks were worth")
    figure.tight_layout()
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

    legacy_loaded = _load("legacy")
    if legacy_loaded is not None:
        figure_leak_price(per_fold, legacy_loaded[0], FIGURES_DIR / "leak_price.png")


if __name__ == "__main__":
    main()
