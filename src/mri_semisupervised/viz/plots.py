"""Helpers de visualisation : grilles d'images, projections 2D, ROC, CM."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.metrics import RocCurveDisplay, roc_curve

try:
    import umap

    HAS_UMAP = True
except ImportError:  # pragma: no cover
    HAS_UMAP = False

sns.set_theme(context="notebook", style="whitegrid")


def plot_image_grid(
    paths: Iterable[str | Path],
    titles: Iterable[str] | None = None,
    cols: int = 5,
    figsize_per_cell: tuple[float, float] = (2.4, 2.4),
    cmap: str = "gray",
    save_path: Path | None = None,
) -> plt.Figure:
    """Affiche une grille d'images. Renvoie la figure."""
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
    """Histogrammes des statistiques de pixels (mean / std)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(stats["mean"], bins=30, ax=axes[0], color="#3182bd")
    axes[0].set_title("Distribution de la luminance moyenne")
    axes[0].set_xlabel("intensité moyenne (0-1)")
    sns.histplot(stats["std"], bins=30, ax=axes[1], color="#fd8d3c")
    axes[1].set_title("Distribution de l'écart-type des pixels")
    axes[1].set_xlabel("écart-type")
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig


def project_2d(
    features: np.ndarray,
    method: str = "tsne",
    seed: int = 42,
    perplexity: float = 30.0,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> np.ndarray:
    """Projette ``features`` en 2D via t-SNE ou UMAP."""
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
            raise RuntimeError("umap-learn n'est pas installé")
        model = umap.UMAP(
            n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=seed
        )
        return model.fit_transform(features)
    raise ValueError(f"Méthode 2D inconnue : {method}")


def plot_2d_scatter(
    coords: np.ndarray,
    labels: Iterable[str | int | None],
    title: str,
    save_path: Path | None = None,
) -> plt.Figure:
    """Nuage 2D coloré par label (catégoriel)."""
    df = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "label": ["unlabeled" if v is None else str(v) for v in labels],
        }
    )
    fig, ax = plt.subplots(figsize=(7.5, 6))
    palette = {"unlabeled": "#bdbdbd", "normal": "#2ca25f", "cancer": "#de2d26", "0": "#1f77b4", "1": "#d62728"}
    sns.scatterplot(
        data=df,
        x="x",
        y="y",
        hue="label",
        palette={k: palette.get(k, None) for k in df["label"].unique() if palette.get(k) is not None}
        or None,
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


def plot_confusion_matrix(
    cm: list[list[int]] | np.ndarray,
    class_names: list[str],
    title: str = "Matrice de confusion",
    save_path: Path | None = None,
) -> plt.Figure:
    """Heatmap d'une matrice de confusion."""
    cm_arr = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    sns.heatmap(
        cm_arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar=False,
    )
    ax.set_xlabel("prédiction")
    ax.set_ylabel("vérité")
    ax.set_title(title)
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig


def plot_roc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    title: str = "Courbe ROC",
    save_path: Path | None = None,
) -> plt.Figure:
    """Courbe ROC (binaire)."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 4.4))
    RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_title(title)
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig


def plot_metrics_bar(
    runs: dict[str, dict[str, float]],
    metric_name: str = "f1_macro",
    title: str | None = None,
    save_path: Path | None = None,
) -> plt.Figure:
    """Diagramme à barres pour comparer plusieurs runs sur une même métrique."""
    df = pd.DataFrame(
        [{"run": name, metric_name: vals.get(metric_name, np.nan)} for name, vals in runs.items()]
    )
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    sns.barplot(data=df, x="run", y=metric_name, ax=ax, palette="muted")
    for p in ax.patches:
        h = p.get_height()
        if not np.isnan(h):
            ax.text(p.get_x() + p.get_width() / 2, h + 0.01, f"{h:.3f}", ha="center", fontsize=9)
    ax.set_title(title or f"Comparaison {metric_name}")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("")
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig


def plot_history(history_df: pd.DataFrame, save_path: Path | None = None) -> plt.Figure:
    """Courbes loss / accuracy par phase d'entraînement."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    sns.lineplot(data=history_df, x="step", y="train_loss", hue="phase", ax=axes[0], marker="o")
    axes[0].set_title("Loss par phase")
    axes[0].set_xlabel("étape")
    sns.lineplot(data=history_df, x="step", y="train_acc", hue="phase", ax=axes[1], marker="o")
    axes[1].set_title("Accuracy par phase")
    axes[1].set_xlabel("étape")
    axes[1].set_ylim(0, 1.05)
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig


__all__ = [
    "plot_2d_scatter",
    "plot_confusion_matrix",
    "plot_history",
    "plot_image_grid",
    "plot_metrics_bar",
    "plot_pixel_stats",
    "plot_roc",
    "project_2d",
]
