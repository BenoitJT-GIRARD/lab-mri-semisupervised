"""Execute the notebooks of this project, in order, and commit the state of that execution.

Outputs that no single execution produced make a notebook unreadable: the counters say 3, 7,
12, the reader believes they are looking at one run, and the cells they are reading never saw
each other. This script runs each notebook from a clean kernel, so the execution counters
come out as 1..N and every output is from that one pass.

    uv sync --group notebook
    uv run python scripts/run_notebooks.py
    uv run python scripts/run_notebooks.py --check    # fail if a notebook is stale

A notebook writes nothing outside itself. What is published under ``reports/`` comes from
the scripts, because a figure a notebook redrew is a figure no manifest can trace back to
a command.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from mri_semisupervised.config import PROJECT_ROOT as ROOT_DIR

NOTEBOOKS_DIR = ROOT_DIR / "notebooks"

#: How long one cell may take. A notebook that needs more is doing work that belongs in a
#: script, where it can be tested.
CELL_TIMEOUT = 300


def notebooks() -> list[Path]:
    return sorted(NOTEBOOKS_DIR.glob("*.ipynb"))


def execute(path: Path) -> None:
    """Run one notebook from a clean kernel, in place, with the project root as cwd."""
    import nbformat
    from nbclient import NotebookClient

    document = nbformat.read(path, as_version=4)
    client = NotebookClient(
        document,
        timeout=CELL_TIMEOUT,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT_DIR)}},
    )
    client.execute()
    # `newline=""` so a notebook executed on Windows is byte-identical to one executed on
    # Linux. Without it every line of a re-executed notebook shows up as changed.
    with path.open("w", encoding="utf-8", newline="") as handle:
        nbformat.write(document, handle)


def counters_are_a_run(path: Path) -> list[str]:
    """What is wrong with the published state of a notebook, or nothing."""
    import nbformat

    document = nbformat.read(path, as_version=4)
    code = [
        cell for cell in document.cells if cell.cell_type == "code" and "".join(cell.source).strip()
    ]
    if not code:
        return []
    problems = []
    if not any(cell.get("outputs") for cell in code):
        problems.append(f"{path.name}: {len(code)} code cell(s) and no output")
    counters = [cell.get("execution_count") for cell in code]
    if counters != list(range(1, len(counters) + 1)):
        problems.append(f"{path.name}: execution counters {counters[:10]} are not one run")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", help="a single notebook, by file name")
    parser.add_argument(
        "--check", action="store_true", help="run nothing; report the notebooks that are stale"
    )
    arguments = parser.parse_args(argv)

    wanted = [n for n in notebooks() if not arguments.only or n.name == arguments.only]
    if not wanted:
        print("no notebook selected", file=sys.stderr)
        return 1

    if arguments.check:
        problems = [line for path in wanted for line in counters_are_a_run(path)]
        for line in problems:
            print(f"  {line}")
        return 1 if problems else 0

    for path in wanted:
        # A failed execution must not leave half a run committed: the notebook is replaced
        # only once the whole thing has run.
        with tempfile.TemporaryDirectory() as scratch:
            backup = Path(scratch) / path.name
            shutil.copy2(path, backup)
            try:
                execute(path)
            except Exception:
                shutil.copy2(backup, path)
                raise
        print(f"{path.relative_to(ROOT_DIR)} — {len(counters_are_a_run(path)) == 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
