"""Plotting helpers: image grids, 2D projections, ROC curves, confusion matrices."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageOps
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, roc_curve

try:
    import umap

    HAS_UMAP = True
except ImportError:  # pragma: no cover
    HAS_UMAP = False

sns.set_theme(context="notebook", style="whitegrid")


def plot_image_grid(
    paths: Iterable[str | Path],
    titles: Iterable[str] | None = None,
    *,
    cols: int = 5,
    figsize_per_cell: tuple[float, float] = (2.4, 2.4),
    cmap: str = "gray",
    save_path: Path | None = None,
) -> plt.Figure:
    """Draw a grid of images and return the figure."""
    paths_list = [Path(p) for p in paths]
    titles_list = list(titles) if titles is not None else [p.name for p in paths_list]
    n = len(paths_list)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * figsize_per_cell[0], rows * figsize_per_cell[1]),
        squeeze=False,
    )
    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        ax.axis("off")
        if i < n:
            with Image.open(paths_list[i]) as img:
                ax.imshow(np.asarray(img.convert("L")), cmap=cmap)
            ax.set_title(titles_list[i], fontsize=9)
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig


def plot_pixel_stats(
    stats: pd.DataFrame,
    save_path: Path | None = None,
) -> plt.Figure:
    """Histograms of the per-image pixel statistics."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(stats["mean"], bins=30, ax=axes[0], color="#3182bd")
    axes[0].set_title("Distribution of the mean luminance")
    axes[0].set_xlabel("mean intensity (0-1)")
    sns.histplot(stats["std"], bins=30, ax=axes[1], color="#fd8d3c")
    axes[1].set_title("Distribution of the per-image pixel standard deviation")
    axes[1].set_xlabel("standard deviation")
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig


def plot_equalization(
    image_path: str | Path,
    save_path: Path | None = None,
) -> plt.Figure:
    """Show one MRI before and after histogram equalisation.

    The two images side by side, then the two intensity histograms, so the stretch is
    visible rather than asserted.
    """
    with Image.open(image_path) as img:
        original = np.asarray(img.convert("L"))
    equalized = np.asarray(ImageOps.equalize(Image.fromarray(original)))

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    axes[0][0].imshow(original, cmap="gray")
    axes[0][0].set_title("IRM d'origine")
    axes[0][0].axis("off")
    axes[0][1].imshow(equalized, cmap="gray")
    axes[0][1].set_title("after equalisation")
    axes[0][1].axis("off")
    axes[1][0].hist(original.ravel(), bins=256, range=(0, 255), color="#3182bd")
    axes[1][0].set_title("Histogramme d'origine")
    axes[1][0].set_xlabel("intensity")
    axes[1][1].hist(equalized.ravel(), bins=256, range=(0, 255), color="#fd8d3c")
    axes[1][1].set_title("equalised histogram")
    axes[1][1].set_xlabel("intensity")
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig


def project_2d(
    features: np.ndarray,
    method: str = "tsne",
    *,
    seed: int = 42,
    perplexity: float = 30.0,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> np.ndarray:
    """Project ``features`` down to two dimensions, with t-SNE or UMAP."""
    method = method.lower()
    if method == "tsne":
        model = TSNE(
            n_components=2,
            random_state=seed,
            perplexity=min(perplexity, max(5, len(features) // 4)),
            init="pca",
            learning_rate="auto",
        )
        return model.fit_transform(features)
    if method == "umap":
        if not HAS_UMAP:
            raise RuntimeError("umap-learn is not installed")
        model = umap.UMAP(
            n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=seed
        )
        return model.fit_transform(features)
    raise ValueError(f"unknown 2D projection: {method!r}; expected 'tsne' or 'umap'")


def _normalise_label(value: object) -> str:
    if value is None:
        return "unlabeled"
    try:
        if isinstance(value, float) and pd.isna(value):
            return "unlabeled"
    except TypeError:
        pass
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return "unlabeled"
    return text


def plot_2d_scatter(
    coords: np.ndarray,
    labels: Iterable[str | int | None],
    title: str,
    save_path: Path | None = None,
) -> plt.Figure:
    """A 2D scatter coloured by label.

    Tolerates ``None``, ``NaN`` and empty strings: they all become "unlabeled" rather than
    crashing the plot or, worse, forming a silent category of their own.
    """
    cleaned = [_normalise_label(v) for v in labels]
    df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "label": cleaned})

    reference = {
        "unlabeled": "#bdbdbd",
        "normal": "#2ca25f",
        "cancer": "#de2d26",
        "noise": "#444444",
        "0": "#1f77b4",
        "1": "#d62728",
    }
    palette = {k: reference[k] for k in df["label"].unique() if k in reference}
    if not palette:
        palette = None

    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.scatterplot(
        data=df,
        x="x",
        y="y",
        hue="label",
        palette=palette,
        s=18,
        alpha=0.75,
        ax=ax,
        edgecolor="none",
    )
    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig


def plot_roc_compare(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    title: str = "Courbes ROC — comparaison",
    save_path: Path | None = None,
) -> plt.Figure:
    """Overlay several binary ROC curves on one axis.

    ``curves`` maps an arm name to ``(y_true, y_score)``, where ``y_score`` is the
    probability of the positive class. The AUC goes in the legend, because a curve without
    its number invites the reader to guess.
    """
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    for name, (y_true, y_score) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        ax.plot(fpr, tpr, linewidth=1.8, label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="hasard")
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs")
    ax.set_title(title)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig


__all__ = [
    "plot_2d_scatter",
    "plot_equalization",
    "plot_image_grid",
    "plot_pixel_stats",
    "plot_roc_compare",
    "project_2d",
]
