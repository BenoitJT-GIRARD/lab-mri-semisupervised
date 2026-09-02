"""The three arms, sharing folds, architecture and budget.

| arm | pre-training | fine-tuning | what it isolates |
|---|---|---|---|
| `supervised` | — | inner train | the reference |
| `semi_supervised` | the fold's pseudo-labels | inner train | the supposed contribution |
| `permuted_control` | the same images, labels shuffled | inner train | budget and exposure |

The third arm is what makes the experiment conclusive. It sees exactly the same images,
takes exactly the same number of gradient steps, and starts from the same weights; only the
correspondence between an image and its pseudo-label is destroyed. If `semi_supervised`
cannot beat it, then what helped was exposure to the images and not the information the
clustering found — and that has to be said.

Checkpoints and stopping are decided on the **inner validation split**, carved out of the
training fold. The previous code kept the epoch with the best *training* accuracy, which on
eighty images selects the most overfit one.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from mri_semisupervised.config import POSITIVE_INDEX, TrainingConfig, device
from mri_semisupervised.data.preprocess import (
    ImagePathsDataset,
    build_eval_transform,
    build_train_transform,
)
from mri_semisupervised.models.semi_supervised import build_classifier
from mri_semisupervised.protocol.pseudo_labels import PseudoLabelSet, permute

SUPERVISED = "supervised"
SEMI_SUPERVISED = "semi_supervised"
PERMUTED_CONTROL = "permuted_control"
ARMS = (SUPERVISED, SEMI_SUPERVISED, PERMUTED_CONTROL)


@dataclass
class ArmResult:
    """What one arm produced on one fold."""

    arm: str
    fold: str
    y_true: np.ndarray
    y_score: np.ndarray
    val_y_true: np.ndarray
    val_y_score: np.ndarray
    best_epoch: int
    pretrain_steps: int
    finetune_steps: int
    pretrain_image_ids: tuple[str, ...] = ()
    history: list[dict] = field(default_factory=list)

    @property
    def steps(self) -> int:
        return self.pretrain_steps + self.finetune_steps


def _loader(
    paths: list[Path],
    labels: list[int],
    cfg: TrainingConfig,
    *,
    train: bool,
    seed: int,
) -> DataLoader:
    transform = (
        build_train_transform(cfg.image_size) if train else build_eval_transform(cfg.image_size)
    )
    dataset = ImagePathsDataset(paths=paths, transform=transform, labels=labels)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=train,
        num_workers=0,
        pin_memory=False,
        generator=generator if train else None,
    )


def _train_epoch(model, loader, optimiser, criterion, dev) -> tuple[float, int]:
    model.train()
    total, steps = 0.0, 0
    for batch, targets in loader:
        batch, targets = batch.to(dev), targets.to(dev)
        optimiser.zero_grad(set_to_none=True)
        loss = criterion(model(batch), targets)
        loss.backward()
        optimiser.step()
        total += float(loss.item()) * batch.size(0)
        steps += 1
    return total / max(len(loader.dataset), 1), steps


@torch.inference_mode()
def _predict(model, loader, dev) -> tuple[np.ndarray, np.ndarray]:
    """Return the true labels and the probability of the positive class."""
    model.eval()
    truths, scores = [], []
    for batch, targets in loader:
        probabilities = torch.softmax(model(batch.to(dev)), dim=1)
        truths.extend(int(t) for t in targets.cpu().tolist())
        scores.extend(float(p) for p in probabilities[:, POSITIVE_INDEX].cpu().tolist())
    return np.asarray(truths), np.asarray(scores)


def _validation_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Rank the epochs on the inner validation.

    ROC AUC rather than accuracy: on sixteen validation images, accuracy moves in steps of
    0.0625 and ties constantly, which turns model selection into a coin toss.
    """
    from sklearn.metrics import roc_auc_score

    if len(set(y_true.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _pretrain(
    model: nn.Module,
    pseudo: PseudoLabelSet,
    paths_by_id: dict[str, Path],
    cfg: TrainingConfig,
    dev: torch.device,
    seed: int,
) -> tuple[int, list[dict]]:
    """Pre-train on pseudo-labels; return the number of steps and the history."""
    paths = [paths_by_id[i] for i in pseudo.image_ids]
    loader = _loader(paths, list(pseudo.labels), cfg, train=True, seed=seed)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    steps, history = 0, []
    for epoch in range(1, cfg.epochs_weak + 1):
        loss, epoch_steps = _train_epoch(model, loader, optimiser, criterion, dev)
        steps += epoch_steps
        history.append({"phase": "pretrain", "epoch": epoch, "train_loss": loss})
    return steps, history


def run_arm(
    arm: str,
    *,
    fold_name: str,
    paths_by_id: dict[str, Path],
    inner_train: list[tuple[str, int]],
    inner_val: list[tuple[str, int]],
    test: list[tuple[str, int]],
    pseudo: PseudoLabelSet | None,
    cfg: TrainingConfig,
    seed: int,
) -> ArmResult:
    """Train one arm on one fold and return its predictions.

    ``inner_train``, ``inner_val`` and ``test`` are lists of ``(image_id, label)``. The test
    entries are used once, at the very end, to produce scores — never to choose anything.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    if arm != SUPERVISED and pseudo is None:
        raise ValueError(f"arm {arm} needs pseudo-labels")

    torch.manual_seed(seed)
    dev = torch.device(device())
    model = build_classifier(cfg.architecture, cfg.num_classes).to(dev)

    pretrain_steps, history, pretrain_ids = 0, [], ()
    if arm in (SEMI_SUPERVISED, PERMUTED_CONTROL):
        used = pseudo if arm == SEMI_SUPERVISED else permute(pseudo, seed)
        pretrain_ids = tuple(used.image_ids.tolist())
        pretrain_steps, history = _pretrain(model, used, paths_by_id, cfg, dev, seed)

    train_loader = _loader(
        [paths_by_id[i] for i, _ in inner_train],
        [y for _, y in inner_train],
        cfg,
        train=True,
        seed=seed,
    )
    val_loader = _loader(
        [paths_by_id[i] for i, _ in inner_val],
        [y for _, y in inner_val],
        cfg,
        train=False,
        seed=seed,
    )
    test_loader = _loader(
        [paths_by_id[i] for i, _ in test],
        [y for _, y in test],
        cfg,
        train=False,
        seed=seed,
    )

    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate * (cfg.finetune_lr_factor if pretrain_steps else 1.0),
        weight_decay=cfg.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    best_score, best_epoch, best_state, waited = -np.inf, 0, None, 0
    finetune_steps = 0
    for epoch in range(1, cfg.epochs_strong + 1):
        loss, epoch_steps = _train_epoch(model, train_loader, optimiser, criterion, dev)
        finetune_steps += epoch_steps
        val_true, val_score = _predict(model, val_loader, dev)
        score = _validation_score(val_true, val_score)
        history.append({"phase": "finetune", "epoch": epoch, "train_loss": loss, "val_auc": score})

        if np.isnan(score) or score <= best_score:
            waited += 1
            if waited >= cfg.early_stopping_patience:
                break
            continue

        best_score, best_epoch, waited = score, epoch, 0
        best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    val_true, val_score = _predict(model, val_loader, dev)
    y_true, y_score = _predict(model, test_loader, dev)

    return ArmResult(
        arm=arm,
        fold=fold_name,
        y_true=y_true,
        y_score=y_score,
        val_y_true=val_true,
        val_y_score=val_score,
        best_epoch=best_epoch,
        pretrain_steps=pretrain_steps,
        finetune_steps=finetune_steps,
        pretrain_image_ids=pretrain_ids,
        history=history,
    )


__all__ = ["ARMS", "PERMUTED_CONTROL", "SEMI_SUPERVISED", "SUPERVISED", "ArmResult", "run_arm"]
