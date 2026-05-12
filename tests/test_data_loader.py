"""Tests pour ``curelyticsia.data.loader``."""

from __future__ import annotations

from pathlib import Path

import pytest

from curelyticsia.data.loader import (
    compute_pixel_stats,
    detect_outlier_ids,
    discover_images,
    records_to_dataframe,
    summarise_dataset,
)


def test_discover_images_inventory(synthetic_dataset: Path) -> None:
    records, corrupted = discover_images(synthetic_dataset)

    # 4 + 4 + 6 = 14 images valides ; 1 corrompue.
    assert len(records) == 14
    assert len(corrupted) == 1
    assert corrupted[0].name == "broken.jpg"

    df = records_to_dataframe(records)
    assert df["split"].value_counts()["labeled"] == 8
    assert df["split"].value_counts()["unlabeled"] == 6
    assert set(df["label_name"].dropna().unique()) == {"cancer", "normal"}


def test_records_dataframe_schema(synthetic_dataset: Path) -> None:
    records, _ = discover_images(synthetic_dataset)
    df = records_to_dataframe(records)
    expected_cols = {
        "image_id",
        "path",
        "split",
        "label_name",
        "label_index",
        "width",
        "height",
        "mode",
        "file_size_bytes",
    }
    assert expected_cols.issubset(df.columns)
    assert df["width"].iloc[0] == 32
    assert df["height"].iloc[0] == 32


def test_summarise_dataset(synthetic_dataset: Path) -> None:
    records, _ = discover_images(synthetic_dataset)
    summary = summarise_dataset(records)
    assert summary["total"] == 14
    assert summary["by_split"]["labeled"] == 8
    assert summary["by_split"]["unlabeled"] == 6
    assert summary["by_label"]["cancer"] == 4
    assert summary["by_label"]["normal"] == 4


def test_pixel_stats_and_outliers(synthetic_dataset: Path) -> None:
    records, _ = discover_images(synthetic_dataset)
    stats = compute_pixel_stats(records)
    assert len(stats) == 14
    assert stats["mean"].between(0.0, 1.0).all()

    # Avec un seuil très bas, la moitié des images deviennent "outliers" car
    # les classes ont des intensités contrastées.
    aggressive = detect_outlier_ids(stats, z_threshold=0.5)
    assert len(aggressive) > 0

    # Avec un seuil très haut, aucune image ne doit être déclarée outlier.
    soft = detect_outlier_ids(stats, z_threshold=10.0)
    assert soft == []


@pytest.mark.parametrize("sample_size", [5, 10, None])
def test_pixel_stats_sampling(synthetic_dataset: Path, sample_size: int | None) -> None:
    records, _ = discover_images(synthetic_dataset)
    stats = compute_pixel_stats(records, sample_size=sample_size)
    expected = sample_size if sample_size is not None else len(records)
    assert len(stats) == min(expected, len(records))
