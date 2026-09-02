"""Run the comparison and summarise it.

Usage:
    uv run python scripts/run_experiment.py                       # the corrected protocol
    uv run python scripts/run_experiment.py --mode legacy         # reproduce the old one
    uv run python scripts/run_experiment.py --repeats 1 --folds 5 # a quick pass
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Windows consoles default to cp1252, which cannot print the report.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sklearn.metrics import average_precision_score, roc_auc_score

from mri_semisupervised.config import ProtocolConfig, TrainingConfig, ensure_dirs
from mri_semisupervised.protocol.experiment import CORRECTED, LEGACY, run_experiment
from mri_semisupervised.protocol.uncertainty import bootstrap_ci, paired_difference

HEADLINE = ("roc_auc", "pr_auc", "recall_positive", "f1_macro", "accuracy")


def summarise(result) -> str:
    """Build the markdown summary that sits next to the raw artefacts."""
    lines = [f"# {result.run_id}", "", f"Protocol: **{result.mode}**", ""]

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
        roc = bootstrap_ci(group["y_true"].to_numpy(), group["y_score"].to_numpy(), roc_auc_score)
        pr = bootstrap_ci(
            group["y_true"].to_numpy(), group["y_score"].to_numpy(), average_precision_score
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
            out = paired_difference(pivot[(metric, a)].to_numpy(), pivot[(metric, b)].to_numpy())
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=[CORRECTED, LEGACY], default=CORRECTED)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--epochs-strong", type=int, default=None)
    args = parser.parse_args()

    ensure_dirs()
    protocol = ProtocolConfig()
    if args.repeats:
        protocol = ProtocolConfig(**{**protocol.__dict__, "n_repeats": args.repeats})
    if args.folds:
        protocol = ProtocolConfig(**{**protocol.__dict__, "n_splits": args.folds})

    training = TrainingConfig()
    if args.epochs_strong:
        training = TrainingConfig(**{**training.__dict__, "epochs_strong": args.epochs_strong})

    print(f"[info] mode={args.mode} folds={protocol.n_splits} repeats={protocol.n_repeats}")
    result = run_experiment(mode=args.mode, protocol=protocol, training=training)

    summary = summarise(result)
    (result.directory / "summary.md").write_text(summary, encoding="utf-8")
    print()
    print(summary)
    print(f"[ok] artefacts -> {result.directory}")


if __name__ == "__main__":
    main()
