"""Tests pour ``curelyticsia.models.semi_supervised``."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

if importlib.util.find_spec("torch") is None:  # pragma: no cover
    pytest.skip("torch indisponible", allow_module_level=True)

import torch
from torch.utils.data import DataLoader, TensorDataset

from curelyticsia.config import set_global_seeds
from curelyticsia.models.semi_supervised import build_classifier, evaluate


def test_build_classifier_output_shape() -> None:
    set_global_seeds(0)
    model = build_classifier("resnet18", num_classes=2)
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 224, 224))
    assert out.shape == (1, 2)


def test_build_classifier_unsupported_arch_raises() -> None:
    with pytest.raises(ValueError):
        build_classifier("bogus", num_classes=2)


def _toy_loader(num_samples: int = 6) -> DataLoader:
    rng = np.random.default_rng(0)
    x = torch.tensor(rng.standard_normal((num_samples, 3, 224, 224)).astype(np.float32))
    y = torch.tensor([i % 2 for i in range(num_samples)], dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=2)


def test_evaluate_returns_complete_report() -> None:
    set_global_seeds(0)
    model = build_classifier("resnet18", num_classes=2)
    report = evaluate(
        model, _toy_loader(), class_names=["normal", "cancer"], dev=torch.device("cpu")
    )
    assert 0.0 <= report.accuracy <= 1.0
    assert 0.0 <= report.f1_macro <= 1.0
    assert set(report.f1_per_class) == {"normal", "cancer"}
    assert len(report.confusion_matrix) == 2
    assert report.roc_auc is None or 0.0 <= report.roc_auc <= 1.0
