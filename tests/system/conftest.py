"""The system tier: the product started exactly as the README says to start it.

Started the way ``## Running it`` says, questioned through its public surface, judged on what
a user would see. A test that imports the package and calls a function is not a system test,
whatever directory it sits in — it must cross a process boundary: a subprocess, a container,
an HTTP call to a listening port, a CLI runner.

Every file here carries its tier where a reader sees it:

    pytestmark = pytest.mark.system

The collection hook below refuses a file that does not, and the skips of this tier name the
command that starts what is missing — ``skip_unless(port_is_open(8000), command="uv run
uvicorn ...")`` from the root conftest. A system test that skips on every checkout proves
nothing and inflates the published count.
"""

from __future__ import annotations

from pathlib import Path

import pytest

TIER = "system"

#: This directory. `pytest_collection_modifyitems` is handed EVERY collected item, not only
#: the ones below the conftest that defines it, so the hook filters by path: without this the
#: system tier reports the integration tier's files as unmarked. The marker is read with
#: `get_closest_marker`, never from `item.keywords` — the keywords carry the names of the
#: parent nodes, so the directory called `integration` makes every file in it look marked.
HERE = Path(__file__).resolve().parent


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    unmarked = sorted(
        {
            str(item.path.relative_to(HERE))
            for item in items
            if getattr(item, "path", None) is not None
            and HERE in item.path.parents
            and item.get_closest_marker(TIER) is None
        }
    )
    if unmarked:
        raise pytest.UsageError(
            f"{len(unmarked)} file(s) under tests/{TIER}/ without "
            f"`pytestmark = pytest.mark.{TIER}`: " + ", ".join(unmarked)
        )
