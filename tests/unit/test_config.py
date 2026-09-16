"""Unit tests of the configuration."""

from __future__ import annotations

from mri_semisupervised import config


def test_class_index_consistent() -> None:
    assert config.CLASSES == ("normal", "cancer")
    assert config.CLASS_TO_INDEX == {"normal": 0, "cancer": 1}


def test_set_global_seeds_idempotent() -> None:
    import random

    config.set_global_seeds(123)
    a = random.random()
    config.set_global_seeds(123)
    b = random.random()
    assert a == b


def test_ensure_dirs(tmp_path, monkeypatch) -> None:
    # Paths are monkeypatched so the test never writes into the project tree.
    interim = tmp_path / "interim"
    processed = tmp_path / "processed"
    figures = tmp_path / "figures"
    monkeypatch.setattr(config, "INTERIM_DIR", interim)
    monkeypatch.setattr(config, "PROCESSED_DIR", processed)
    monkeypatch.setattr(config, "FIGURES_DIR", figures)
    config.ensure_dirs()
    assert interim.exists()
    assert processed.exists()
    assert figures.exists()
