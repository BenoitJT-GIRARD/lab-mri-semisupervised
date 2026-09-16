"""The plotting helpers return a figure and write nothing.

`viz/` had no test at all, which is how a French axis label survived a commit named "finish
the translation" and how one figure ended up assigning colours by draw order — putting an arm
and its own control in each other's colour from one figure to the next.
"""

from __future__ import annotations

import numpy as np
import pytest

from mri_semisupervised.figure_style import PALETTE
from mri_semisupervised.viz import plots


@pytest.fixture()
def two_curves() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    truth = np.array([0, 0, 1, 1, 0, 1])
    return {
        "supervised": (truth, np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])),
        "permuted control": (truth, np.array([0.5, 0.6, 0.4, 0.5, 0.5, 0.5])),
    }


def test_the_roc_axes_are_labelled_in_english(two_curves) -> None:
    figure = plots.plot_roc_compare(two_curves)
    axis = figure.axes[0]

    assert axis.get_xlabel() == "False positive rate"
    assert axis.get_ylabel() == "True positive rate"
    assert "chance" in [line.get_label() for line in axis.lines]


def test_each_curve_takes_the_colour_it_is_given(two_curves) -> None:
    """Matplotlib assigns by draw order; an arm keeps its colour across figures only if told."""
    colours = {"supervised": PALETTE["primary"], "permuted control": PALETTE["control"]}

    figure = plots.plot_roc_compare(two_curves, colours=colours)

    drawn = {
        line.get_label().split(" (")[0]: line.get_color()
        for line in figure.axes[0].lines
        if line.get_label() != "chance"
    }
    assert drawn == colours


def test_a_curve_without_a_colour_is_still_drawn(two_curves) -> None:
    figure = plots.plot_roc_compare(two_curves, colours={"supervised": PALETTE["primary"]})

    assert len(figure.axes[0].lines) == 3  # two curves and the chance diagonal


def test_the_projection_refuses_a_method_it_does_not_know() -> None:
    features = np.random.default_rng(0).normal(size=(30, 4))

    with pytest.raises(ValueError, match="unknown 2D projection"):
        plots.project_2d(features, method="pca")


def test_the_projection_returns_two_dimensions_per_row() -> None:
    features = np.random.default_rng(0).normal(size=(40, 6))

    coordinates = plots.project_2d(features, method="tsne", seed=0)

    assert coordinates.shape == (40, 2)


def test_an_unlabelled_point_is_named_and_not_dropped() -> None:
    assert plots._normalise_label(None) == "unlabeled"
    assert plots._normalise_label(float("nan")) == "unlabeled"
    assert plots._normalise_label("cancer") == "cancer"
