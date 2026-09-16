"""Draw the figures the README publishes, from the artefacts of a run.

Every figure here reads `reports/experiments/<mode>/`, so a picture in the README and a
number in the README come from the same run. Regenerating them is one command, and it is
the only way they get regenerated: none is edited by hand.

Colours come from `mri_semisupervised.figure_style`, by role and by name: an arm keeps its
colour across every figure, and a control is grey wherever it appears. The writer is the same
module, and it refuses a figure whose axes say nothing or whose error bars are unnamed.

Usage:
    uv run python scripts/build_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib.pyplot as plt
import pandas as pd

from mri_semisupervised.config import (
    EXPERIMENTS_DIR,
    FIGURES_DIR,
    MANIFEST_PATH,
    ensure_dirs,
)
from mri_semisupervised.figure_style import (
    apply_style,
    close,
    reference_line,
    save_figure,
    series_colours,
)
from mri_semisupervised.protocol.calibration import summarise_calibration
from mri_semisupervised.protocol.errors import per_image_errors, pool_overlap_rates
from mri_semisupervised.protocol.uncertainty import paired_difference
from mri_semisupervised.viz.plots import plot_roc_compare

SOURCE = "scripts/build_figures.py"

#: Every arm, in reading order, with the name a figure shows. A control is named here too:
#: it is drawn as often as the arm it answers.
ARM_LABEL = {
    "supervised": "supervised",
    "semi_supervised": "semi-supervised",
    "semi_supervised_confident": "semi-supervised, filtered",
    "semi_supervised_joint": "joint training",
    "self_training": "self-training",
    "permuted_control": "permuted control",
    "joint_permuted_control": "joint permuted control",
    "self_training_control": "self-training control",
}

#: The arms that measure nothing on purpose. Grey, by contract, on every figure.
CONTROL_ARMS = ("permuted_control", "joint_permuted_control", "self_training_control")

#: One spelling per metric, used for a panel title and for an axis label alike. Publishing
#: the DataFrame column name instead — `roc_auc`, `recall_positive` — asks the reader to
#: know the schema.
METRIC_LABEL = {
    "roc_auc": "ROC AUC",
    "pr_auc": "PR AUC",
    "recall_positive": "recall, cancer class",
    "f1_macro": "F1, macro",
}

BUDGETS = (10, 20, 40)

#: Each mechanism arm and the control it must be read against. The joint arm carries a
#: second treatment — its pseudo batches update BatchNorm on the pool — so comparing it to
#: the plain supervised baseline would conflate that with the pseudo-label loss.
MECHANISMS = {
    "semi_supervised_joint": "joint_permuted_control",
    "self_training": "self_training_control",
}
#: Where the mechanism arms ran: the lowest budget, where there is room to see an effect,
#: and the full budget, which is the one the headline uses.
MECHANISM_RUNS = (("budget-10-stageb", 10), ("corrected-stageb", 59))

ARM_COLOUR = series_colours(list(ARM_LABEL), control=CONTROL_ARMS)


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


def _folds(meta: dict) -> int:
    protocol = meta.get("protocol") or {}
    return int(protocol.get("n_splits", 0)) * int(protocol.get("n_repeats", 0))


def _population(meta: dict) -> dict[str, int]:
    """What a figure of this run rests on: the images scored, and the folds they came from."""
    return {"evaluation images": int(meta.get("evaluation_images", 0)), "folds": _folds(meta)}


def _present(frame: pd.DataFrame) -> list[str]:
    """The arms of `frame`, in the reading order of ARM_LABEL."""
    seen = set(frame["arm"])
    return [arm for arm in ARM_LABEL if arm in seen]


# --- The figures ------------------------------------------------------------


def figure_roc(predictions: pd.DataFrame, meta: dict, path: Path) -> None:
    """Pooled out-of-fold ROC.

    This is **not** the number the table publishes. The table averages the ROC AUC of each
    fold; this pools every out-of-fold score into one curve, which mixes folds calibrated
    differently and lands a point or two lower. Both are honest; they answer different
    questions, and the legend says which one is on screen.
    """
    curves = {
        ARM_LABEL.get(arm, arm): (group["y_true"].to_numpy(), group["y_score"].to_numpy())
        for arm, group in predictions.groupby("arm")
    }
    colours = {ARM_LABEL.get(arm, arm): colour for arm, colour in ARM_COLOUR.items()}
    figure = plot_roc_compare(
        curves,
        title=f"Pooled out-of-fold ROC, {_folds(meta)} folds",
        colours=colours,
    )
    save_figure(figure, path, n=_population(meta), source=SOURCE)
    close(figure)
    print(f"[ok] {path.name}")


def figure_arms(per_fold: pd.DataFrame, meta: dict, path: Path) -> None:
    """Per-arm means with the spread across folds, on the metrics that matter."""
    metrics = ["roc_auc", "pr_auc", "recall_positive", "f1_macro"]
    arms = _present(per_fold)

    figure, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.8))
    for axis, metric in zip(axes, metrics, strict=True):
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
        axis.set_xlabel("arm")
        axis.set_ylabel(METRIC_LABEL[metric])
        axis.set_title(METRIC_LABEL[metric])
        axis.set_ylim(0.5, 1.0)
    figure.suptitle(f"{len(arms)} arms, same folds, same label budget")
    figure.tight_layout()
    save_figure(
        figure,
        path,
        n=_population(meta),
        dispersion="±1 standard deviation across folds",
        source=SOURCE,
    )
    close(figure)
    print(f"[ok] {path.name}")


def _difference_panels(
    frame: pd.DataFrame, metrics: list[str], y_label: str, n_bootstrap: int
) -> plt.Figure:
    """One panel per metric, all on the same x scale, zero drawn in the reference colour.

    The shared scale is the point: three panels with three scales invite the eye to compare
    lengths that are not comparable.
    """
    span = max(
        abs(float(frame["ci_low"].min())),
        abs(float(frame["ci_high"].max())),
    )
    figure, axes = plt.subplots(1, len(metrics), figsize=(4.4 * len(metrics), 3.6), sharey=True)
    axes = [axes] if len(metrics) == 1 else list(axes)
    for axis, metric in zip(axes, metrics, strict=True):
        subset = frame[frame["metric"] == metric].reset_index(drop=True)
        positions = range(len(subset))
        # One call per row: a single errorbar call paints every row in one colour, and the
        # control ended up in the colour of the treatment above it.
        for position, row in zip(positions, subset.itertuples(), strict=True):
            axis.errorbar(
                [row.mean_difference],
                [position],
                xerr=[[row.mean_difference - row.ci_low], [row.ci_high - row.mean_difference]],
                fmt="o",
                color=row.colour,
                capsize=4,
            )
        reference_line(axis, x=0.0)
        axis.set_yticks(list(positions))
        axis.set_yticklabels(subset["label"], fontsize=8)
        axis.set_xlim(-1.15 * span, 1.15 * span)
        axis.set_xlabel(f"difference in {METRIC_LABEL[metric]}")
        axis.set_ylabel(y_label)
        axis.set_title(METRIC_LABEL[metric])
    figure.suptitle(f"Paired differences across shared folds, {n_bootstrap} bootstrap resamples")
    figure.tight_layout()
    return figure


def figure_paired(per_fold: pd.DataFrame, meta: dict, path: Path) -> None:
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
            rows.append(
                {
                    "label": f"{ARM_LABEL[a]}\nminus {ARM_LABEL[b]}",
                    "metric": metric,
                    "colour": ARM_COLOUR[a],
                    **out,
                }
            )
    frame = pd.DataFrame(rows)
    figure = _difference_panels(
        frame,
        metrics,
        y_label="pair compared",
        n_bootstrap=int((meta.get("protocol") or {}).get("n_bootstrap", 0)),
    )
    save_figure(
        figure,
        path,
        n=_population(meta),
        dispersion="95 % bootstrap interval on the paired difference",
        source=SOURCE,
    )
    close(figure)
    print(f"[ok] {path.name}")


def figure_leak_price(
    corrected: pd.DataFrame, legacy: pd.DataFrame, meta: dict, path: Path
) -> None:
    """What the three leaks were worth, as the paired difference they are.

    Drawn as two bars per arm, the leak was a hairline on a 0.5-to-1.0 axis: the figure said
    « nothing happened » while the text said the opposite. The quantity the text talks about
    is the difference, so the difference is what is drawn, against a zero line.
    """
    metrics = ["roc_auc", "recall_positive"]
    arms = [a for a in _present(corrected) if a in set(legacy["arm"])]

    rows = []
    for metric in metrics:
        for arm in arms:
            # Joined on the fold, never zipped by position: the two runs share their splits,
            # and a missing fold on one side would silently shift every pair by one.
            pair = legacy.loc[legacy["arm"] == arm, ["fold", metric]].merge(
                corrected.loc[corrected["arm"] == arm, ["fold", metric]],
                on="fold",
                suffixes=("_legacy", "_corrected"),
            )
            if pair.empty:
                continue
            out = paired_difference(
                pair[f"{metric}_legacy"].to_numpy(), pair[f"{metric}_corrected"].to_numpy()
            )
            rows.append(
                {
                    "label": ARM_LABEL[arm],
                    "metric": metric,
                    "colour": ARM_COLOUR[arm],
                    **out,
                }
            )
    frame = pd.DataFrame(rows)
    figure = _difference_panels(
        frame,
        metrics,
        y_label="arm",
        n_bootstrap=int((meta.get("protocol") or {}).get("n_bootstrap", 0)),
    )
    figure.suptitle("What the three leaks were worth: leaking minus corrected, same folds")
    save_figure(
        figure,
        path,
        n=_population(meta),
        dispersion="95 % bootstrap interval on the paired difference",
        source=SOURCE,
    )
    close(figure)
    print(f"[ok] {path.name}")


def figure_calibration(predictions: pd.DataFrame, meta: dict, path: Path) -> None:
    """Reliability curves: what a score of 0.4 is actually worth.

    ROC AUC cannot see this. A model whose ranking is perfect and whose scale is squashed
    scores 1.0 and still tells a clinician the wrong number.
    """
    summary, curves = summarise_calibration(predictions, n_bins=10)
    figure, axis = plt.subplots(figsize=(6.5, 6))
    reference_line(axis, diagonal=True, label="perfectly calibrated")

    for arm, curve in curves.items():
        row = summary[summary["arm"] == arm].iloc[0]
        axis.plot(
            curve["mean_score"],
            curve["observed"],
            marker="o",
            markersize=4,
            color=ARM_COLOUR.get(arm),
            label=f"{ARM_LABEL.get(arm, arm)} — Brier {row['brier']:.3f}, ECE {row['ece']:.3f}",
        )

    axis.set_xlabel("mean predicted probability, equal-population bins")
    axis.set_ylabel("observed cancer rate")
    axis.set_title("Calibration, pooled out of fold")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.legend(fontsize=8, loc="upper left")
    save_figure(figure, path, n=_population(meta), source=SOURCE)
    close(figure)
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
    drawn = ("supervised", "semi_supervised", "permuted_control")
    # A small horizontal offset per arm: at ten and twenty labels the three error bars sat
    # on the same x and covered each other entirely.
    offsets = {arm: (index - 1) * 0.9 for index, arm in enumerate(drawn)}
    for arm in drawn:
        part = frame[frame["arm"] == arm].sort_values("budget")
        if part.empty:
            continue
        top.errorbar(
            part["budget"] + offsets[arm],
            part["mean"],
            yerr=part["sd"],
            marker="o",
            capsize=3,
            color=ARM_COLOUR.get(arm),
            label=ARM_LABEL.get(arm, arm),
        )
    top.set_xlabel("training labels per fold")
    top.set_ylabel(METRIC_LABEL["roc_auc"])
    top.set_title("The task gets easier with labels, which answers nothing on its own")
    top.legend(fontsize=9)

    diff = frame[frame["arm"] == "difference"].sort_values("budget")
    reference_line(bottom, y=0.0)
    bottom.errorbar(
        diff["budget"],
        diff["mean"],
        yerr=[diff["mean"] - diff["ci_low"], diff["ci_high"] - diff["mean"]],
        marker="o",
        capsize=3,
        color=ARM_COLOUR["semi_supervised"],
    )
    bottom.set_xlabel("training labels per fold")
    bottom.set_ylabel("semi-supervised minus\npermuted control")
    bottom.set_title("And the answer: every interval spans zero")
    figure.tight_layout()
    save_figure(
        figure,
        path,
        n={"evaluation images": 99, "folds": 25, "budgets": len(points)},
        dispersion="top: ±1 SD across folds; bottom: 95 % bootstrap interval",
        source=SOURCE,
    )
    close(figure)
    print(f"[ok] {path.name}")
    return frame


def figure_mechanisms(path: Path) -> pd.DataFrame | None:
    """Each mechanism against its own control, at the budgets where it ran.

    Two hypotheses about *why* the sequential pseudo-labels do nothing, each with a control
    that isolates it. Joint training tests whether the pre-training is simply forgotten;
    self-training tests whether the labels came from the wrong source. Neither is read
    against the plain supervised baseline — the joint arm updates BatchNorm on the pool, so
    that comparison would measure two treatments at once.
    """
    rows = []
    for name, budget in MECHANISM_RUNS:
        loaded = _load(name)
        if loaded is None:
            continue
        pivot = loaded[0].pivot(index="fold", columns="arm", values="roc_auc")
        for arm, control in MECHANISMS.items():
            if not {arm, control} <= set(pivot.columns):
                continue
            out = paired_difference(pivot[arm].to_numpy(), pivot[control].to_numpy())
            rows.append(
                {
                    "budget": budget,
                    "arm": arm,
                    "control": control,
                    "arm_mean": float(pivot[arm].mean()),
                    "control_mean": float(pivot[control].mean()),
                    "difference": out["mean_difference"],
                    "ci_low": out["ci_low"],
                    "ci_high": out["ci_high"],
                    "p_value": out["p_value"],
                }
            )
    if not rows:
        print("[warn] no mechanism run found")
        return None
    frame = pd.DataFrame(rows)

    figure, axis = plt.subplots(figsize=(7, 4.5))
    reference_line(axis, x=0.0)
    labels = []
    for position, row in enumerate(frame.itertuples()):
        axis.errorbar(
            row.difference,
            position,
            xerr=[[row.difference - row.ci_low], [row.ci_high - row.difference]],
            fmt="o",
            capsize=4,
            color=ARM_COLOUR.get(row.arm),
        )
        labels.append(f"{ARM_LABEL.get(row.arm, row.arm)} — {row.budget} labels")
    axis.set_yticks(range(len(frame)))
    axis.set_yticklabels(labels, fontsize=8)
    axis.set_xlabel(f"difference in {METRIC_LABEL['roc_auc']}, arm minus its own control")
    axis.set_ylabel("mechanism tested")
    axis.set_title("Two hypotheses about why the pseudo-labels do nothing")
    axis.invert_yaxis()
    figure.tight_layout()
    save_figure(
        figure,
        path,
        n={"evaluation images": 99, "folds": 25, "comparisons": len(frame)},
        dispersion="95 % bootstrap interval on the paired difference",
        source=SOURCE,
    )
    close(figure)
    print(f"[ok] {path.name}")
    return frame


def main() -> None:
    ensure_dirs()
    apply_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    loaded = _load("corrected")
    if loaded is None:
        print("[warn] run scripts/run_experiment.py first")
        return
    per_fold, predictions, meta = loaded
    print(f"[info] corrected run {meta['run_id']}, fingerprint {meta['dataset_fingerprint']}")

    figure_roc(predictions, meta, FIGURES_DIR / "roc_arms.png")
    figure_arms(per_fold, meta, FIGURES_DIR / "arms_comparison.png")
    figure_paired(per_fold, meta, FIGURES_DIR / "paired_differences.png")

    figure_calibration(predictions, meta, FIGURES_DIR / "calibration.png")

    legacy_loaded = _load("legacy")
    if legacy_loaded is not None:
        figure_leak_price(per_fold, legacy_loaded[0], meta, FIGURES_DIR / "leak_price.png")

    manifest_path = Path(MANIFEST_PATH)
    if manifest_path.exists():
        manifest = pd.read_parquet(manifest_path)
        errors = per_image_errors(predictions, per_fold, manifest)
        target = EXPERIMENTS_DIR / "corrected" / "per_image_errors.parquet"
        errors.to_parquet(target, index=False)
        print(f"[ok] {target.name}")
        print(pool_overlap_rates(errors).round(3).to_string(index=False))

    curve = figure_label_efficiency(FIGURES_DIR / "label_efficiency.png")
    if curve is not None:
        target = EXPERIMENTS_DIR / "label_efficiency.parquet"
        curve.to_parquet(target, index=False)
        print(f"[ok] {target.name}")

    mechanisms = figure_mechanisms(FIGURES_DIR / "mechanisms.png")
    if mechanisms is not None:
        target = EXPERIMENTS_DIR / "mechanisms.parquet"
        mechanisms.to_parquet(target, index=False)
        print(f"[ok] {target.name}")
        print(mechanisms.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
