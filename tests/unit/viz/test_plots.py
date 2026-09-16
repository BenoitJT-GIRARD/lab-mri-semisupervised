"""The plotting helpers return a figure and write nothing.

`viz/` had no test at all, which is how a French axis label survived a commit named "finish
the translation" and how one figure ended up assigning colours by draw order — putting an arm
and its own control in each other's colour from one figure to the next.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from mri_semisupervised.figure_style import PALETTE
from mri_semisupervised.viz import plots

#: The style sheet puts every title on the left, so that is where matplotlib stores it.
TITLE_SIDE = plt.rcParams["axes.titlelocation"]


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


@pytest.fixture()
def four_scans(tmp_path: Path) -> list[Path]:
    """Four small greyscale PNGs, drawn so that their statistics differ."""
    rng = np.random.default_rng(3)
    written = []
    for index in range(4):
        pixels = (rng.normal(60 + 40 * index, 20, size=(24, 24)).clip(0, 255)).astype("uint8")
        path = tmp_path / f"scan_{index}.png"
        Image.fromarray(pixels, mode="L").save(path)
        written.append(path)
    return written


def test_the_grid_holds_one_cell_per_image_and_hides_every_axis(four_scans) -> None:
    """A grid of scans is shown, not measured: a tick mark on it would mean nothing."""
    figure = plots.plot_image_grid(four_scans, cols=3)

    assert len(figure.axes) == 6  # two rows of three, the last two empty
    assert not any(axis.axison for axis in figure.axes)
    assert [axis.get_title(loc=TITLE_SIDE) for axis in figure.axes[:4]] == [
        p.name for p in four_scans
    ]


def test_the_grid_takes_the_titles_it_is_given(four_scans) -> None:
    figure = plots.plot_image_grid(four_scans[:2], titles=["cancer", "normal"], cols=2)

    assert [axis.get_title(loc=TITLE_SIDE) for axis in figure.axes] == ["cancer", "normal"]


def test_the_pixel_histograms_name_what_they_count() -> None:
    stats = pd.DataFrame({"mean": np.linspace(0.1, 0.9, 40), "std": np.linspace(0.02, 0.3, 40)})

    figure = plots.plot_pixel_stats(stats)

    assert [axis.get_xlabel() for axis in figure.axes] == [
        "mean intensity (0-1)",
        "standard deviation",
    ]


def test_the_equalisation_panel_is_written_in_english(four_scans) -> None:
    """A French title survived a commit named "finish the translation"; this is the guard."""
    figure = plots.plot_equalization(four_scans[0])
    titles = [axis.get_title(loc=TITLE_SIDE) for axis in figure.axes]

    assert len(titles) == 4
    assert all(title.isascii() for title in titles)
    assert "as the archive ships it" in titles


def test_the_scatter_gives_the_unlabelled_pool_the_neutral_colour() -> None:
    """The pool is background: a series colour would make it read as a third class."""
    coords = np.column_stack([np.linspace(0, 1, 6), np.linspace(1, 0, 6)])

    figure = plots.plot_2d_scatter(
        coords, ["cancer", "normal", None, "", float("nan"), "cancer"], "t-SNE"
    )
    axis = figure.axes[0]

    assert axis.get_xlabel() == "dim 1"
    assert axis.get_ylabel() == "dim 2"
    assert axis.get_title(loc=TITLE_SIDE) == "t-SNE"
    assert {text.get_text() for text in axis.get_legend().get_texts()} == {
        "cancer",
        "normal",
        "unlabeled",
    }


def test_the_scatter_survives_labels_it_has_no_colour_for() -> None:
    """A clustering returns 0, 1, 2, …; an unknown label must not take the figure down."""
    coords = np.column_stack([np.arange(4.0), np.arange(4.0)])

    figure = plots.plot_2d_scatter(coords, ["c3", "c4", "c5", "c6"], "clusters")

    assert len(figure.axes[0].collections) == 1


def test_umap_is_refused_when_the_extra_is_not_installed(monkeypatch) -> None:
    """`umap-learn` is optional; the published projection is t-SNE. The refusal says which."""
    monkeypatch.setattr(plots, "HAS_UMAP", False)
    features = np.random.default_rng(0).normal(size=(20, 4))

    with pytest.raises(RuntimeError, match="umap-learn is not installed"):
        plots.project_2d(features, method="umap")
