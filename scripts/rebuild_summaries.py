"""Re-render `summary.md` for every run under reports/experiments/, and the arm table beside them.

A run takes over an hour on a GPU, and the summary beside it is a pure function of the three
files it wrote. When the renderer changes — as it did the day the dataset fingerprint entered
the header — every past run gets the new page without anyone training anything again.

    uv run python scripts/rebuild_summaries.py
"""

from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from mri_semisupervised.config import EXPERIMENTS_DIR
from mri_semisupervised.config import PROJECT_ROOT as ROOT
from mri_semisupervised.protocol.report import load_result, published_arms, runs, summarise


def main() -> int:
    directories = runs()
    if not directories:
        print("[warn] no run under reports/experiments/")
        return 1
    for directory in directories:
        result = load_result(directory)
        target = directory / "summary.md"
        target.write_text(summarise(result), encoding="utf-8", newline="")
        print(f"[ok] {target.relative_to(ROOT)}")
    arms = published_arms()
    if not arms.empty:
        target = EXPERIMENTS_DIR / "arms.csv"
        arms.to_csv(target, index=False, lineterminator="\n")
        print(f"[ok] {target.relative_to(ROOT)}")

    print(f"{len(directories)} summary file(s) rebuilt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
