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
import math
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
from mri_semisupervised.protocol.pseudo_labels import (
    PseudoLabelSet,
    filter_by_confidence,
    permute,
)

SUPERVISED = "supervised"
SEMI_SUPERVISED = "semi_supervised"
SEMI_SUPERVISED_CONFIDENT = "semi_supervised_confident"
PERMUTED_CONTROL = "permuted_control"
#: Pseudo-labels carried through fine-tuning, which the phased arms leave behind. The
#: sequential arms pre-train for 246 steps and then converge in two to four epochs, so
#: whatever the pre-training taught has every opportunity to be forgotten. These two put
#: the pseudo-label loss beside the supervised one at every step, and the second is the
#: control that tells information apart from extra gradient work.
SEMI_SUPERVISED_JOINT = "semi_supervised_joint"
JOINT_PERMUTED_CONTROL = "joint_permuted_control"
#: Pseudo-labels taken from the decision function being optimised, and not from a
#: k-means on ImageNet embeddings. The clustering reaches an ARI of 0.46 against the
#: training labels, so its clusters and the classes only half agree; a first supervised
#: pass has at least been asked the right question.
SELF_TRAINING = "self_training"
SELF_TRAINING_CONTROL = "self_training_control"

#: Every implemented arm. A test asserts each one is either run by default or listed as
#: opt-in, so an arm cannot be written and then quietly never run.
ARMS = (
    SUPERVISED,
    SEMI_SUPERVISED,
    SEMI_SUPERVISED_CONFIDENT,
    PERMUTED_CONTROL,
    SEMI_SUPERVISED_JOINT,
    JOINT_PERMUTED_CONTROL,
    SELF_TRAINING,
    SELF_TRAINING_CONTROL,
)
#: Arms that need the fold's pseudo-labels.
PRETRAINING_ARMS = (
    SEMI_SUPERVISED,
    SEMI_SUPERVISED_CONFIDENT,
    PERMUTED_CONTROL,
    SEMI_SUPERVISED_JOINT,
    JOINT_PERMUTED_CONTROL,
)
#: Arms that build their own pseudo-labels from a first supervised pass. They need the
#: pool's image ids, which they take from ``pseudo``, and none of its labels.
SELF_TRAINING_ARMS = (SELF_TRAINING, SELF_TRAINING_CONTROL)
#: Arms that train on both losses at once, in one phase.
JOINT_ARMS = (SEMI_SUPERVISED_JOINT, JOINT_PERMUTED_CONTROL)
#: Opt-in via ``--arms``: they answer a separate question and cost a run of their own.
OPTIONAL_ARMS = JOINT_ARMS + SELF_TRAINING_ARMS


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
    #: Quantile of the confidence distribution the arm kept, when it filters. ``None``
    #: for every other arm; ``0.0`` when the filtering arm chose not to filter.
    confidence_quantile: float | None = None
    #: Pseudo-labels actually pre-trained on, after filtering.
    n_pseudo_used: int = 0
    #: How hard the pseudo-label loss pulled. ``None`` for every arm that trains in phases.
    pseudo_weight: float | None = None
    #: Pool images whose own prediction was confident enough to be reused, for the
    #: self-training arms only.
    n_self_labelled: int | None = None
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
    builder = build_train_transform if train else build_eval_transform
    transform = builder(cfg.image_size, equalize=cfg.equalize)
    dataset = ImagePathsDataset(paths=paths, transform=transform, labels=labels)
    generator = torch.Generator()
    generator.manual_seed(seed)
    # A training batch of one cannot pass through BatchNorm: the layer has no variance to
    # normalise with, and torch raises. It happens whenever the fold size leaves a remainder
    # of one, so the last batch is dropped in that single case and kept in every other.
    lonely_tail = train and len(dataset) % cfg.batch_size == 1
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=train,
        num_workers=0,
        pin_memory=False,
        generator=generator if train else None,
        drop_last=lonely_tail,
    )


def _train_epoch(model, loader, optimiser, criterion, dev, *, max_steps=None) -> tuple[float, int]:
    model.train()
    total, steps = 0.0, 0
    for batch, targets in loader:
        if max_steps is not None and steps >= max_steps:
            break
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

    ROC AUC, and not accuracy: on sixteen validation images, accuracy moves in steps of
    0.0625 and ties constantly, which turns model selection into a coin toss.
    """
    from sklearn.metrics import roc_auc_score

    if len(set(y_true.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def budget_steps(n_items: int, cfg: TrainingConfig) -> int:
    """The number of pre-training steps the unfiltered pool would take.

    Expressed in steps, not epochs, and that is the whole point. Filtering shrinks the pool,
    so an equal number of *epochs* would be an unequal number of gradient updates — and a
    budget confound between the arms is exactly the defect the leak-free protocol removed
    once already.
    """
    per_epoch = math.ceil(n_items / cfg.batch_size) if n_items else 0
    return per_epoch * cfg.epochs_weak


def _pretrain(
    model: nn.Module,
    pseudo: PseudoLabelSet,
    paths_by_id: dict[str, Path],
    *,
    cfg: TrainingConfig,
    dev: torch.device,
    seed: int,
    target_steps: int | None = None,
) -> tuple[int, list[dict]]:
    """Pre-train on pseudo-labels; return the number of steps and the history.

    With ``target_steps`` the loop runs until that many gradient updates have happened,
    passing over a reduced pool as many times as it takes and stopping mid-epoch on the
    last one. Without it, the usual ``epochs_weak`` passes.
    """
    paths = [paths_by_id[i] for i in pseudo.image_ids]
    loader = _loader(paths, list(pseudo.labels), cfg, train=True, seed=seed)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    steps, history, epoch = 0, [], 0
    while True:
        epoch += 1
        if target_steps is None:
            if epoch > cfg.epochs_weak:
                break
            remaining = None
        else:
            remaining = target_steps - steps
            if remaining <= 0:
                break
        loss, epoch_steps = _train_epoch(
            model, loader, optimiser, criterion, dev, max_steps=remaining
        )
        if epoch_steps == 0:
            break
        steps += epoch_steps
        history.append({"phase": "pretrain", "epoch": epoch, "train_loss": loss})
    return steps, history


def _train_epoch_joint(
    model,
    labelled_loader,
    pseudo_loader,
    optimiser,
    criterion,
    *,
    dev,
    weight: float,
) -> tuple[float, int]:
    """One epoch of ``supervised loss + weight * pseudo-label loss``, both at every step.

    One pseudo batch per labelled batch, cycling the pseudo loader, so the two losses carry
    the same number of examples per step and ``weight`` alone decides their relative pull.
    The epoch is counted in labelled batches: the arm sees the labels exactly as often as
    the supervised baseline does, and the pseudo-labels are the only thing added.

    One consequence has to be said out loud. The pseudo batches go through the network in
    train mode, so BatchNorm updates its running statistics on the unlabelled pool even
    when ``weight`` is zero. That is unsupervised adaptation to the target distribution,
    arguably a form of semi-supervision in itself, and it means this arm carries two
    treatments at once. It is why the joint arm is only ever compared to the joint
    control, which pushes exactly the same images through exactly the same BatchNorm: the
    difference between them is the label information and nothing else.
    """
    model.train()
    total, steps = 0.0, 0
    pseudo_iterator = iter(pseudo_loader)
    for batch, targets in labelled_loader:
        try:
            weak_batch, weak_targets = next(pseudo_iterator)
        except StopIteration:
            pseudo_iterator = iter(pseudo_loader)
            weak_batch, weak_targets = next(pseudo_iterator)

        batch, targets = batch.to(dev), targets.to(dev)
        weak_batch, weak_targets = weak_batch.to(dev), weak_targets.to(dev)

        optimiser.zero_grad(set_to_none=True)
        loss = criterion(model(batch), targets) + weight * criterion(
            model(weak_batch), weak_targets
        )
        loss.backward()
        optimiser.step()
        total += float(loss.item()) * batch.size(0)
        steps += 1
    return total / max(len(labelled_loader.dataset), 1), steps


def _finetune(
    model,
    train_loader,
    val_loader,
    cfg: TrainingConfig,
    dev: torch.device,
    *,
    history: list[dict],
    warm_start: bool = False,
    pseudo_loader=None,
    pseudo_weight: float | None = None,
    phase: str = "finetune",
) -> tuple[int, int]:
    """Train on the labels, keeping the epoch that scores best on the inner validation.

    Returns the number of steps taken and the epoch kept. The checkpoint criterion is the
    validation ROC AUC, never the training accuracy: on sixteen validation images accuracy
    moves in steps of 0.0625 and ties constantly, which turns selection into a coin toss.

    ``pseudo_loader`` switches the epoch to the joint form. ``warm_start`` lowers the
    learning rate for a model that has already been pre-trained.
    """
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate * (cfg.finetune_lr_factor if warm_start else 1.0),
        weight_decay=cfg.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    best_score, best_epoch, best_state, waited = -np.inf, 0, None, 0
    steps = 0
    for epoch in range(1, cfg.epochs_strong + 1):
        if pseudo_loader is not None:
            loss, epoch_steps = _train_epoch_joint(
                model,
                train_loader,
                pseudo_loader,
                optimiser,
                criterion,
                dev=dev,
                weight=pseudo_weight,
            )
        else:
            loss, epoch_steps = _train_epoch(model, train_loader, optimiser, criterion, dev)
        steps += epoch_steps
        val_true, val_score = _predict(model, val_loader, dev)
        score = _validation_score(val_true, val_score)
        history.append({"phase": phase, "epoch": epoch, "train_loss": loss, "val_auc": score})

        if np.isnan(score) or score <= best_score:
            waited += 1
            if waited >= cfg.early_stopping_patience:
                break
            continue

        best_score, best_epoch, waited = score, epoch, 0
        best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    return steps, best_epoch


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
    if arm in (*PRETRAINING_ARMS, *SELF_TRAINING_ARMS) and pseudo is None:
        raise ValueError(f"arm {arm} needs pseudo-labels")

    dev = torch.device(device())

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

    def fresh_model() -> nn.Module:
        # Same seed for every arm and every candidate: the starting weights are shared, so
        # a difference between arms cannot come from where they started.
        torch.manual_seed(seed)
        return build_classifier(cfg.architecture, cfg.num_classes).to(dev)

    model = fresh_model()
    pretrain_steps, history, pretrain_ids = 0, [], ()
    quantile, n_pseudo_used, n_self_labelled = None, 0, None
    pseudo_loader, pseudo_weight = None, None

    if arm in JOINT_ARMS:
        # No pre-training phase at all. The pseudo-labels ride alongside the supervised
        # loss for the whole of training, which is the difference being tested.
        used = pseudo if arm == SEMI_SUPERVISED_JOINT else permute(pseudo, seed)
        pretrain_ids = tuple(used.image_ids.tolist())
        n_pseudo_used = len(used)
        pseudo_weight = cfg.pseudo_loss_weight
        pseudo_loader = _loader(
            [paths_by_id[i] for i in used.image_ids],
            list(used.labels),
            cfg,
            train=True,
            seed=seed,
        )

    if arm in SELF_TRAINING_ARMS:
        # A first supervised pass, then its own confident predictions on the pool as
        # pseudo-labels, then a pre-training on those. The labels come from the decision
        # function actually being optimised, where the other arms read a k-means on
        # dominant structure turned out to be acquisition contrast.
        scout = fresh_model()
        _finetune(scout, train_loader, val_loader, cfg, dev, history=history, phase="scout")
        pool_paths = [paths_by_id[i] for i in pseudo.image_ids]
        pool_loader = _loader(pool_paths, [0] * len(pool_paths), cfg, train=False, seed=seed)
        _, scores = _predict(scout, pool_loader, dev)

        keep = (scores >= cfg.self_training_threshold) | (
            scores <= 1.0 - cfg.self_training_threshold
        )
        if not keep.any():
            keep = np.zeros(len(scores), dtype=bool)
            keep[np.argsort(np.abs(scores - 0.5))[-cfg.batch_size :]] = True

        self_labels = (scores[keep] >= 0.5).astype(int)
        used = PseudoLabelSet(
            method_name="self-training",
            ari_on_train=float("nan"),
            image_ids=pseudo.image_ids[keep],
            labels=self_labels,
            confidence=np.abs(scores[keep] - 0.5) * 2.0,
            n_noise=int((~keep).sum()),
        )
        if arm == SELF_TRAINING_CONTROL:
            used = permute(used, seed)

        pretrain_ids = tuple(used.image_ids.tolist())
        n_pseudo_used = len(used)
        n_self_labelled = len(used)
        model = fresh_model()
        pretrain_steps, pretrain_history = _pretrain(
            model, used, paths_by_id, cfg=cfg, dev=dev, seed=seed
        )
        history.extend(pretrain_history)

    elif arm in (SEMI_SUPERVISED, PERMUTED_CONTROL):
        used = pseudo if arm == SEMI_SUPERVISED else permute(pseudo, seed)
        pretrain_ids = tuple(used.image_ids.tolist())
        n_pseudo_used = len(used)
        pretrain_steps, history = _pretrain(model, used, paths_by_id, cfg=cfg, dev=dev, seed=seed)

    elif arm == SEMI_SUPERVISED_CONFIDENT:
        # The answer to "your pseudo-labels were simply too noisy". Each quantile gets the
        # same step budget as the unfiltered arm, and the one that scores best on the
        # inner validation is kept. The test fold takes no part in this choice.
        budget = budget_steps(len(pseudo), cfg)
        best_score, best_state = -np.inf, None
        for candidate in cfg.confidence_quantiles:
            subset = filter_by_confidence(pseudo, candidate)
            trial = fresh_model()
            steps, trial_history = _pretrain(
                trial, subset, paths_by_id, cfg=cfg, dev=dev, seed=seed, target_steps=budget
            )
            val_true, val_score = _predict(trial, val_loader, dev)
            score = _validation_score(val_true, val_score)
            trial_history.append(
                {
                    "phase": "confidence_search",
                    "quantile": candidate,
                    "n_pseudo": len(subset),
                    "pretrain_steps": steps,
                    "val_auc": score,
                }
            )
            if not np.isnan(score) and score > best_score:
                best_score = score
                best_state = copy.deepcopy(trial.state_dict())
                quantile, n_pseudo_used = float(candidate), len(subset)
                pretrain_steps, history = steps, trial_history
                pretrain_ids = tuple(subset.image_ids.tolist())
            elif best_state is None:
                # Every candidate so far scored nan — a single-class validation split. Keep
                # the first one, so that no untrained model is ever returned.
                best_state = copy.deepcopy(trial.state_dict())
                quantile, n_pseudo_used = float(candidate), len(subset)
                pretrain_steps, history = steps, trial_history
                pretrain_ids = tuple(subset.image_ids.tolist())
        model.load_state_dict(best_state)
    test_loader = _loader(
        [paths_by_id[i] for i, _ in test],
        [y for _, y in test],
        cfg,
        train=False,
        seed=seed,
    )

    finetune_steps, best_epoch = _finetune(
        model,
        train_loader,
        val_loader,
        cfg,
        dev,
        history=history,
        warm_start=bool(pretrain_steps),
        pseudo_loader=pseudo_loader,
        pseudo_weight=pseudo_weight,
    )

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
        confidence_quantile=quantile,
        n_pseudo_used=n_pseudo_used,
        pseudo_weight=pseudo_weight,
        n_self_labelled=n_self_labelled,
        history=history,
    )


__all__ = [
    "ARMS",
    "JOINT_ARMS",
    "JOINT_PERMUTED_CONTROL",
    "OPTIONAL_ARMS",
    "PERMUTED_CONTROL",
    "PRETRAINING_ARMS",
    "SELF_TRAINING",
    "SELF_TRAINING_ARMS",
    "SELF_TRAINING_CONTROL",
    "SEMI_SUPERVISED",
    "SEMI_SUPERVISED_CONFIDENT",
    "SEMI_SUPERVISED_JOINT",
    "SUPERVISED",
    "ArmResult",
    "budget_steps",
    "run_arm",
]
