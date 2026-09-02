"""Preprocessing for an ImageNet backbone.

Two pipelines, and the difference between them matters:

* **evaluation** is deterministic — resize, centre crop, normalise. Anything random here
  would make a measurement depend on a draw.
* **training** adds light augmentations chosen to be plausible for an MRI.

ImageNet CNNs expect RGB and brain MRI is greyscale, so the channel is replicated
explicitly rather than left to whatever the loader happens to do.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

from mri_semisupervised.config import IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE


def equalize_image(image: Image.Image) -> Image.Image:
    """Histogram equalisation on a greyscale MRI.

    Intensities are stretched back over the full 0-255 range, which lifts the contrast and
    brings out structure on a dull scan. Used for visual exploration: the feature
    extraction keeps the ImageNet normalisation instead, because that is what the backbone
    was trained under.
    """
    gray = image.convert("L")
    return ImageOps.equalize(gray)


def build_eval_transform(image_size: int = INPUT_SIZE) -> Callable[[Image.Image], object]:
    """The deterministic pipeline, for feature extraction and for evaluation."""
    return transforms.Compose(
        [
            transforms.Lambda(lambda im: im.convert("RGB")),
            transforms.Resize(int(image_size * 1.14)),  # ≈ 256 quand image_size=224
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_train_transform(image_size: int = INPUT_SIZE) -> Callable[[Image.Image], object]:
    """The training pipeline, with augmentations that respect the anatomy.

    No vertical flip — a brain MRI has a top and a bottom — and rotation is capped at ten
    degrees. An augmentation that changes what the image means is not augmentation, it is
    corruption with extra steps.
    """
    return transforms.Compose(
        [
            transforms.Lambda(lambda im: im.convert("RGB")),
            transforms.Resize(int(image_size * 1.14)),
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class ImagePathsDataset(Dataset):
    """A minimal PyTorch dataset reading images from a list of paths.

    ``labels`` is optional: with it, ``__getitem__`` returns ``(tensor, label)``; without
    it, ``(tensor, index)``, so the caller can put the outputs back in order.
    """

    def __init__(
        self,
        paths: list[str | Path],
        transform: Callable[[Image.Image], object],
        labels: list[int] | None = None,
    ) -> None:
        self.paths = [Path(p) for p in paths]
        self.transform = transform
        self.labels = labels
        if labels is not None and len(labels) != len(self.paths):
            raise ValueError("labels and paths must have the same length")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[object, int]:
        path = self.paths[index]
        with Image.open(path) as img:
            tensor = self.transform(img)
        target = self.labels[index] if self.labels is not None else index
        return tensor, target


__all__ = [
    "ImagePathsDataset",
    "build_eval_transform",
    "build_train_transform",
    "equalize_image",
]
