"""The markdown a reader opens after a run, rendered from the run's own artefacts.

It lives in the package and not in the script that writes it, because it is a pure function
of what a run produced: `scripts/rebuild_summaries.py` re-renders every past run with it, and
the system tier checks what it renders.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from mri_semisupervised.config import EXPERIMENTS_DIR
from mri_semisupervised.protocol.experiment import ExperimentResult
from mri_semisupervised.protocol.uncertainty import bootstrap_ci, paired_difference


def load_result(directory: Path) -> ExperimentResult:
    """Rebuild the result of a past run from the files it left behind."""
    meta = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    folds_meta = directory / "folds.parquet"
    return ExperimentResult(
        run_id=meta["run_id"],
        mode=meta["mode"],
        per_fold=pd.read_parquet(directory / "per_fold.parquet"),
        predictions=pd.read_parquet(directory / "predictions.parquet"),
        folds_meta=pd.read_parquet(folds_meta) if folds_meta.exists() else pd.DataFrame(),
        directory=directory,
    )


def runs() -> list[Path]:
    """Every run directory under `reports/experiments/` that carries a result."""
    if not EXPERIMENTS_DIR.exists():
        return []
    return sorted(d for d in EXPERIMENTS_DIR.iterdir() if (d / "per_fold.parquet").exists())


HEADLINE = ("roc_auc", "pr_auc", "recall_positive", "f1_macro", "accuracy")


def summarise(result, n_bootstrap: int = 2000) -> str:
    """Build the markdown summary that sits next to the raw artefacts.

    The header carries what makes the numbers quotable: which protocol produced them, over
    which dataset, on how many images and how many folds. The manifest beside it held all
    four; the page a reader actually opens held none.
    """
    manifest = json.loads((result.directory / "manifest.json").read_text(encoding="utf-8"))
    protocol = manifest.get("protocol") or {}
    folds = int(protocol.get("n_splits", 0)) * int(protocol.get("n_repeats", 0))
    lines = [
        f"# {result.run_id}",
        "",
        "Written by `scripts/rebuild_summaries.py` from the artefacts of this run.",
        "",
        f"Protocol: **{result.mode}**",
        "",
        f"- dataset fingerprint: `{manifest.get('dataset_fingerprint')}`",
        f"- evaluation images: {manifest.get('evaluation_images')}",
        f"- folds: {folds} ({protocol.get('n_splits')} splits x {protocol.get('n_repeats')} repeats)",
        f"- unlabelled pool: {manifest.get('unlabelled_pool')}",
        "",
    ]

    means = result.per_fold.groupby("arm")[list(HEADLINE)].agg(["mean", "std"])
    lines += [
        "## Per-arm means across folds",
        "",
        "| arm | " + " | ".join(HEADLINE) + " |",
        "|---|" + "---|" * len(HEADLINE),
    ]
    for arm in means.index:
        cells = [
            f"{means.loc[arm, (m, 'mean')]:.3f} +/- {means.loc[arm, (m, 'std')]:.3f}"
            for m in HEADLINE
        ]
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Pooled out-of-fold, with bootstrap intervals",
        "",
        "| arm | ROC AUC | PR-AUC |",
        "|---|---|---|",
    ]
    for arm, group in result.predictions.groupby("arm"):
        roc = bootstrap_ci(
            group["y_true"].to_numpy(),
            group["y_score"].to_numpy(),
            roc_auc_score,
            n_boot=n_bootstrap,
        )
        pr = bootstrap_ci(
            group["y_true"].to_numpy(),
            group["y_score"].to_numpy(),
            average_precision_score,
            n_boot=n_bootstrap,
        )
        lines.append(
            f"| {arm} | {roc['point']:.3f} [{roc['ci_low']:.3f}, {roc['ci_high']:.3f}] "
            f"| {pr['point']:.3f} [{pr['ci_low']:.3f}, {pr['ci_high']:.3f}] |"
        )

    lines += [
        "",
        "## Paired differences across the shared folds",
        "",
        "| comparison | metric | difference | 95% CI | p |",
        "|---|---|---|---|---|",
    ]
    arms = list(result.per_fold["arm"].unique())
    pivot = result.per_fold.pivot(index="fold", columns="arm")
    for metric in ("roc_auc", "recall_positive"):
        for a, b in [(x, y) for i, x in enumerate(arms) for y in arms[i + 1 :]]:
            out = paired_difference(
                pivot[(metric, a)].to_numpy(),
                pivot[(metric, b)].to_numpy(),
                n_boot=n_bootstrap,
            )
            lines.append(
                f"| {a} vs {b} | {metric} | {out['mean_difference']:+.3f} "
                f"| [{out['ci_low']:+.3f}, {out['ci_high']:+.3f}] | {out['p_value']:.3f} |"
            )

    if "pseudo_method" in result.folds_meta:
        chosen = result.folds_meta["pseudo_method"].value_counts()
        lines += ["", "## Clustering method chosen, fold by fold", ""]
        for name, count in chosen.items():
            lines.append(f"- `{name}`: {count} fold(s)")
        ari = result.folds_meta["pseudo_ari_on_train"]
        lines.append(f"\nARI on the training labels: {ari.mean():.3f} ± {ari.std():.3f}")

    return "\n".join(lines) + "\n"
