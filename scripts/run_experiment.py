"""Run the comparison and summarise it.

Usage:
    uv run python scripts/run_experiment.py                       # the corrected protocol
    uv run python scripts/run_experiment.py --mode legacy         # reproduce the old one
    uv run python scripts/run_experiment.py --mode equalized      # with equalisation
    uv run python scripts/run_experiment.py --repeats 1 --folds 5 # a quick pass
    uv run python scripts/run_experiment.py --label-budget 10 --arms supervised,semi_supervised
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Windows consoles default to cp1252, which cannot print the report.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from mri_semisupervised.config import ProtocolConfig, TrainingConfig, ensure_dirs
from mri_semisupervised.protocol.arms import ARMS
from mri_semisupervised.protocol.experiment import (
    CORRECTED,
    EQUALIZED,
    LEGACY,
    run_experiment,
)
from mri_semisupervised.protocol.report import summarise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=[CORRECTED, EQUALIZED, LEGACY], default=CORRECTED)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--epochs-strong", type=int, default=None)
    parser.add_argument(
        "--label-budget",
        type=int,
        default=None,
        help="cap the training labels per fold; writes to reports/experiments/budget-N/",
    )
    parser.add_argument(
        "--arms",
        default=None,
        help="comma-separated subset of the arms, for a cheaper sweep",
    )
    parser.add_argument(
        "--out-name",
        default=None,
        help="output directory name under reports/experiments/, if not the default",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        type=Path,
        help="write the run somewhere other than reports/experiments/ (a trial run, a test)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="resize images to this side before the network; the published runs use 224",
    )
    args = parser.parse_args()

    ensure_dirs()
    protocol = ProtocolConfig()
    if args.repeats:
        protocol = ProtocolConfig(**{**protocol.__dict__, "n_repeats": args.repeats})
    if args.folds:
        protocol = ProtocolConfig(**{**protocol.__dict__, "n_splits": args.folds})
    if args.label_budget:
        protocol = ProtocolConfig(**{**protocol.__dict__, "label_budget": args.label_budget})
    if args.arms:
        chosen = tuple(a.strip() for a in args.arms.split(",") if a.strip())
        unknown = set(chosen) - set(ARMS)
        if unknown:
            parser.error(f"unknown arm(s): {sorted(unknown)}")
        protocol = ProtocolConfig(**{**protocol.__dict__, "arms": chosen})

    training = TrainingConfig(equalize=args.mode == EQUALIZED)
    if args.epochs_strong:
        training = TrainingConfig(**{**training.__dict__, "epochs_strong": args.epochs_strong})
    if args.image_size:
        training = TrainingConfig(**{**training.__dict__, "image_size": args.image_size})

    print(
        f"[info] mode={args.mode} folds={protocol.n_splits} "
        f"repeats={protocol.n_repeats} budget={protocol.label_budget} "
        f"arms={len(protocol.arms)}"
    )
    result = run_experiment(
        mode=args.mode,
        protocol=protocol,
        training=training,
        run_name=args.out_name,
        output_root=args.output_root,
    )

    summary = summarise(result, n_bootstrap=protocol.n_bootstrap)
    (result.directory / "summary.md").write_text(summary, encoding="utf-8")
    print()
    print(summary)
    print(f"[ok] artefacts -> {result.directory}")


if __name__ == "__main__":
    main()
