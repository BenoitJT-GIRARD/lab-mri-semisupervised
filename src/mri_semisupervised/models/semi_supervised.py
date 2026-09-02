"""The classifier the three arms share.

This module used to hold the training loops and the cross-validation as well. They now
live in :mod:`mri_semisupervised.protocol`, because what they do is protocol — which fold
sees which image, which labels may be read, where the checkpoint is chosen — and not
modelling. Keeping them here alongside the architecture is what let a leak look like an
ordinary line.

What is left is the one thing that really is a model: a pre-trained ResNet with a fresh
two-class head.
"""

from __future__ import annotations

from torch import nn
from torchvision import models

SUPPORTED = ("resnet18", "resnet50")


def build_classifier(architecture: str = "resnet18", num_classes: int = 2) -> nn.Module:
    """A pre-trained ResNet whose classification head is replaced.

    The backbone stays trainable: on a hundred images the head alone has nothing like
    enough capacity to adapt ImageNet features to greyscale MRI, and freezing it would
    hide the comparison behind a bottleneck common to every arm.
    """
    arch = architecture.lower()
    if arch == "resnet18":
        net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif arch == "resnet50":
        net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    else:
        raise ValueError(f"unsupported architecture: {architecture}; expected one of {SUPPORTED}")

    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net


__all__ = ["SUPPORTED", "build_classifier"]
