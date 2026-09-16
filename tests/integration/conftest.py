"""The integration tier: two real components of this repository wired together.

Either two real components of the repository face to face, or one real component against a
real external dependency — a database in a container, an index on disk, an engine binary, a
real checkpoint, the real FastAPI or Streamlit application object. A double is admitted only
for a paid or remote service, and the test's docstring says which one and why.

Every file here carries its tier where a reader sees it, at the top of the module:

    pytestmark = pytest.mark.integration

The collection hook below refuses a file that does not. Without it the rule is checked once
a day by the audit; with it, the suite refuses to run the moment a file arrives unmarked, and
``-m "not integration"`` keeps meaning what it says.

Fixtures shared by this tier — a session against a container, a temporary index, a loaded
model — belong in this file, so that starting the dependency is written once and skipped
once, with the command that starts it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

TIER = "integration"

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
