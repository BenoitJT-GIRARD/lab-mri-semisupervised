"""Unit tests of the shared classifier."""

from __future__ import annotations

import pytest
import torch

from mri_semisupervised.models.semi_supervised import build_classifier


def test_the_head_is_replaced_and_the_output_has_the_right_width() -> None:
    model = build_classifier("resnet18", num_classes=2)
    model.eval()
    with torch.inference_mode():
        logits = model(torch.zeros(2, 3, 224, 224))
    assert logits.shape == (2, 2)


def test_the_backbone_stays_trainable() -> None:
    # Freezing it would put the same bottleneck under all three arms and hide the
    # comparison behind it.
    model = build_classifier("resnet18")
    assert all(p.requires_grad for p in model.parameters())


def test_resnet50_is_supported_too() -> None:
    model = build_classifier("resnet50", num_classes=2)
    assert model.fc.out_features == 2


def test_an_unknown_architecture_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="unsupported architecture"):
        build_classifier("efficientnet_b0")
