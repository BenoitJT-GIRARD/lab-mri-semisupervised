"""Pipeline de prétraitement adapté aux backbones ImageNet.

Les transformations sont conçues pour être :

* déterministes en mode évaluation (resize + center-crop + normalisation),
* enrichies en mode entraînement par des augmentations légères et plausibles
  pour des IRM (flip horizontal, rotation modérée).

Les CNN ImageNet attendent du RGB ; les IRM cérébrales sont en niveaux de gris :
on convertit explicitement en RGB en répliquant le canal.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

from curelyticsia.config import IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE


def equalize_image(image: Image.Image) -> Image.Image:
    """Égalisation d'histogramme sur une IRM (niveaux de gris).

    On repasse l'image en niveaux de gris puis on égalise son histogramme :
    les intensités sont ré-étalées sur toute la plage 0-255, ce qui rehausse
    le contraste et fait mieux ressortir les structures sur des IRM ternes.
    On l'utilise surtout pour l'exploration visuelle (voir notebook 1).
    """
    gray = image.convert("L")
    return ImageOps.equalize(gray)


def build_eval_transform(image_size: int = INPUT_SIZE) -> Callable[[Image.Image], object]:
    """Pipeline déterministe pour l'extraction de features et l'évaluation."""
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
    """Pipeline d'entraînement avec augmentations IRM-safe.

    On évite les augmentations qui modifient la sémantique médicale :
    pas de flip vertical (l'IRM cérébrale a un haut/bas), rotations limitées.
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
    """Dataset PyTorch minimaliste lisant des images depuis une liste de chemins.

    ``labels`` est optionnel : si fourni, ``__getitem__`` renvoie ``(tensor, label)``,
    sinon ``(tensor, index)`` afin de préserver l'ordre côté appelant.
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
            raise ValueError("labels et paths doivent avoir la même longueur")

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
