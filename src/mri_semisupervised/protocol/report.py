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


def _ran_on(result: ExperimentResult) -> str:
    """The date of the run, read from its identifier.

    The date of the *rendering* would change every time the page is rebuilt, and a page that
    changes without its numbers changing is a page nobody can compare.
    """
    stamp = result.run_id.rsplit("-", 2)
    if len(stamp) == 3 and len(stamp[1]) == 8 and stamp[1].isdigit():
        return f"{stamp[1][:4]}-{stamp[1][4:6]}-{stamp[1][6:]}"
    return "an undated run"


def runs() -> list[Path]:
    """Every run directory under `reports/experiments/` that carries a result."""
    if not EXPERIMENTS_DIR.exists():
        return []
    return sorted(d for d in EXPERIMENTS_DIR.iterdir() if (d / "per_fold.parquet").exists())


HEADLINE = ("roc_auc", "pr_auc", "recall_positive", "f1_macro", "accuracy")

#: The run every other run is a variant of: same folds, same seeds, no preprocessing change
#: and the full label budget. A number that compares two runs has to live somewhere a reader
#: can open, and the summary of the variant is where it belongs.
REFERENCE = "corrected"


#: The two runs whose arms the README publishes side by side: the four arms of the corrected
#: protocol, and the four the stage-B run adds. Every arm appears in exactly one of them.
PUBLISHED_RUNS = ("corrected", "corrected-stageb")


def published_arms() -> pd.DataFrame:
    """The eight arms of the headline table, each with the run it was measured in.

    The table crosses two runs, so no single run's `summary.md` carries it, and a README
    table with no artefact behind it is a table nobody can check. This writes the artefact.
    """
    rows = []
    for name in PUBLISHED_RUNS:
        directory = EXPERIMENTS_DIR / name
        if not (directory / "per_fold.parquet").exists():
            continue
        result = load_result(directory)
        meta = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        for arm, values in result.per_fold.groupby("arm")["roc_auc"]:
            rows.append(
                {
                    "arm": arm,
                    "run": name,
                    "mean_roc_auc": round(float(values.mean()), 6),
                    "sd_across_folds": round(float(values.std()), 6),
                    "folds": int(values.count()),
                    "evaluation_images": int(meta.get("evaluation_images", 0)),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    # An arm measured in both runs is published from the one that introduced it.
    frame = frame.drop_duplicates(subset="arm", keep="first")
    return frame.sort_values("mean_roc_auc", ascending=False).reset_index(drop=True)


def _reference_per_fold(result) -> pd.DataFrame | None:
    """The per-fold table of the reference run, when this run is a variant of it."""
    if result.mode == REFERENCE:
        return None
    path = result.directory.parent / REFERENCE / "per_fold.parquet"
    return pd.read_parquet(path) if path.exists() else None


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
        f"Written by `scripts/rebuild_summaries.py` from the artefacts of the run of {_ran_on(result)}.",
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

    reference = _reference_per_fold(result)
    if reference is not None:
        here = result.per_fold.pivot(index="fold", columns="arm", values="roc_auc")
        there = reference.pivot(index="fold", columns="arm", values="roc_auc")
        shared_folds = here.index.intersection(there.index)
        shared_arms = [arm for arm in here.columns if arm in there.columns]
        if len(shared_folds) and shared_arms:
            lines += [
                "",
                f"## Against the `{REFERENCE}` run, paired over the "
                f"{len(shared_folds)} shared folds",
                "",
                f"| arm | this run | {REFERENCE} | difference | 95% CI | p |",
                "|---|---|---|---|---|---|",
            ]
            for arm in shared_arms:
                a = here.loc[shared_folds, arm].to_numpy()
                b = there.loc[shared_folds, arm].to_numpy()
                out = paired_difference(a, b, n_boot=n_bootstrap)
                lines.append(
                    f"| {arm} | {a.mean():.3f} | {b.mean():.3f} "
                    f"| {out['mean_difference']:+.3f} "
                    f"| [{out['ci_low']:+.3f}, {out['ci_high']:+.3f}] "
                    f"| {out['p_value']:.3f} |"
                )

    if "pseudo_method" in result.folds_meta:
        chosen = result.folds_meta["pseudo_method"].value_counts()
        lines += ["", "## Clustering method chosen, fold by fold", ""]
        for name, count in chosen.items():
            lines.append(f"- `{name}`: {count} fold(s)")
        ari = result.folds_meta["pseudo_ari_on_train"]
        lines.append(f"\nARI on the training labels: {ari.mean():.3f} ± {ari.std():.3f}")

    return "\n".join(lines) + "\n"
