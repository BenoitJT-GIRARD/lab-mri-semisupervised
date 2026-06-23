"""Modèle CNN et boucles d'entraînement supervisé / semi-supervisé.

Stratégies comparées :

* **Supervised baseline** : ResNet18 pré-entraîné ImageNet, ré-entraîné
  uniquement sur les 80 images du train fortement labellisé.
* **Semi-supervisée** : même architecture, pré-entraînée sur les ~1 400
  pseudo-labels « faibles » issus du clustering, puis fine-tunée sur les
  80 images fortement labellisées.

Les deux modèles sont évalués sur le **même** jeu de test fortement
labellisé (20 images) afin que la comparaison soit honnête.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from curelyticsia.config import TrainingConfig, device
from curelyticsia.data.preprocess import (
    ImagePathsDataset,
    build_eval_transform,
    build_train_transform,
)


@dataclass
class EpochLog:
    """Journal d'une époque."""

    epoch: int
    phase: str  # "weak" / "strong" / "supervised"
    train_loss: float
    train_acc: float


@dataclass
class EvalReport:
    """Synthèse d'une évaluation sur jeu de test."""

    accuracy: float
    f1_macro: float
    f1_per_class: dict[str, float]
    precision_per_class: dict[str, float]
    recall_per_class: dict[str, float]
    roc_auc: float | None
    confusion_matrix: list[list[int]]
    # On garde aussi la vérité terrain et la probabilité de la classe "cancer"
    # pour pouvoir retracer la courbe ROC dans le notebook.
    y_true: list[int] = field(default_factory=list)
    y_score: list[float] = field(default_factory=list)
    history: list[EpochLog] = field(default_factory=list)


def build_classifier(architecture: str = "resnet18", num_classes: int = 2) -> nn.Module:
    """ResNet pré-entraîné avec une nouvelle tête de classification."""
    arch = architecture.lower()
    if arch == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        net = models.resnet18(weights=weights)
        in_features = net.fc.in_features
        net.fc = nn.Linear(in_features, num_classes)
    elif arch == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        net = models.resnet50(weights=weights)
        in_features = net.fc.in_features
        net.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Architecture non supportée : {architecture}")
    return net


def _set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Active/désactive l'apprentissage des couches convolutionnelles."""
    for name, param in model.named_parameters():
        if name.startswith("fc."):
            param.requires_grad_(True)
        else:
            param.requires_grad_(trainable)


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    criterion: nn.Module,
    dev: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    for batch, targets in loader:
        batch = batch.to(dev)
        targets = targets.to(dev)
        optimiser.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = criterion(logits, targets)
        loss.backward()
        optimiser.step()
        total_loss += float(loss.item()) * batch.size(0)
        total_correct += int((logits.argmax(dim=1) == targets).sum().item())
        total_seen += batch.size(0)
    return total_loss / max(total_seen, 1), total_correct / max(total_seen, 1)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    class_names: list[str],
    dev: torch.device | None = None,
) -> EvalReport:
    """Évalue ``model`` sur ``loader`` et renvoie un rapport complet."""
    if dev is None:
        dev = torch.device(device())
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[float] = []
    for batch, targets in loader:
        batch = batch.to(dev)
        logits = model(batch)
        proba = torch.softmax(logits, dim=1)
        preds = proba.argmax(dim=1)
        y_true.extend(int(t) for t in targets.cpu().tolist())
        y_pred.extend(int(p) for p in preds.cpu().tolist())
        if proba.shape[1] == 2:
            y_prob.extend(float(p) for p in proba[:, 1].cpu().tolist())

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    f1m = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        average=None,
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=list(range(len(class_names)))).tolist()
    auc: float | None
    if y_prob and len(set(y_true_arr.tolist())) == 2:
        try:
            auc = float(roc_auc_score(y_true_arr, np.asarray(y_prob)))
        except ValueError:
            auc = None
    else:
        auc = None
    return EvalReport(
        accuracy=acc,
        f1_macro=f1m,
        f1_per_class={class_names[i]: float(f1[i]) for i in range(len(class_names))},
        precision_per_class={class_names[i]: float(prec[i]) for i in range(len(class_names))},
        recall_per_class={class_names[i]: float(rec[i]) for i in range(len(class_names))},
        roc_auc=auc,
        confusion_matrix=cm,
        y_true=[int(v) for v in y_true],
        y_score=[float(p) for p in y_prob],
    )


def make_loader(
    paths: list[str | Path],
    labels: list[int],
    batch_size: int,
    train: bool,
    image_size: int,
    seed: int = 42,
) -> DataLoader:
    transform = build_train_transform(image_size) if train else build_eval_transform(image_size)
    ds = ImagePathsDataset(paths=paths, transform=transform, labels=labels)
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        num_workers=0,
        pin_memory=False,
        generator=g if train else None,
    )


def train_supervised(
    train_paths: list[str | Path],
    train_labels: list[int],
    test_paths: list[str | Path],
    test_labels: list[int],
    class_names: list[str],
    cfg: TrainingConfig | None = None,
) -> tuple[nn.Module, EvalReport]:
    """Baseline 100 % supervisée sur le jeu fortement labellisé uniquement."""
    cfg = cfg or TrainingConfig()
    dev = torch.device(device())
    model = build_classifier(cfg.architecture, cfg.num_classes).to(dev)
    _set_backbone_trainable(model, trainable=True)

    train_loader = make_loader(
        train_paths, train_labels, cfg.batch_size, train=True, image_size=cfg.image_size
    )
    test_loader = make_loader(
        test_paths, test_labels, cfg.batch_size, train=False, image_size=cfg.image_size
    )

    optimiser = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    history: list[EpochLog] = []
    best_state: dict | None = None
    best_acc = -1.0
    for epoch in range(1, cfg.epochs_strong + 1):
        loss, acc = _train_one_epoch(model, train_loader, optimiser, criterion, dev)
        history.append(EpochLog(epoch=epoch, phase="supervised", train_loss=loss, train_acc=acc))
        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    report = evaluate(model, test_loader, class_names=class_names, dev=dev)
    report.history = history
    return model, report


def train_semi_supervised(
    weak_paths: list[str | Path],
    weak_labels: list[int],
    strong_train_paths: list[str | Path],
    strong_train_labels: list[int],
    strong_test_paths: list[str | Path],
    strong_test_labels: list[int],
    class_names: list[str],
    cfg: TrainingConfig | None = None,
) -> tuple[nn.Module, EvalReport]:
    """Pré-entraîne sur les pseudo-labels faibles puis fine-tune sur les forts."""
    cfg = cfg or TrainingConfig()
    dev = torch.device(device())
    model = build_classifier(cfg.architecture, cfg.num_classes).to(dev)

    # Phase 1 : pré-entraînement sur labels faibles, head + backbone entraînables.
    _set_backbone_trainable(model, trainable=True)
    weak_loader = make_loader(
        weak_paths, weak_labels, cfg.batch_size, train=True, image_size=cfg.image_size
    )
    optimiser = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    history: list[EpochLog] = []
    for epoch in range(1, cfg.epochs_weak + 1):
        loss, acc = _train_one_epoch(model, weak_loader, optimiser, criterion, dev)
        history.append(EpochLog(epoch=epoch, phase="weak", train_loss=loss, train_acc=acc))

    # Phase 2 : fine-tuning sur jeu fortement labellisé.
    strong_train_loader = make_loader(
        strong_train_paths,
        strong_train_labels,
        cfg.batch_size,
        train=True,
        image_size=cfg.image_size,
    )
    strong_test_loader = make_loader(
        strong_test_paths,
        strong_test_labels,
        cfg.batch_size,
        train=False,
        image_size=cfg.image_size,
    )

    optimiser = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate * 0.5,  # lr réduit pour le fine-tuning
        weight_decay=cfg.weight_decay,
    )

    best_state: dict | None = None
    best_acc = -1.0
    for epoch in range(1, cfg.epochs_strong + 1):
        loss, acc = _train_one_epoch(model, strong_train_loader, optimiser, criterion, dev)
        history.append(EpochLog(epoch=epoch, phase="strong", train_loss=loss, train_acc=acc))
        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    report = evaluate(model, strong_test_loader, class_names=class_names, dev=dev)
    report.history = history
    return model, report


def cross_validate(
    strong_paths: list[str | Path],
    strong_labels: list[int],
    weak_paths: list[str | Path] | None,
    weak_labels: list[int] | None,
    class_names: list[str],
    cfg: TrainingConfig | None = None,
    seed: int = 42,
) -> dict[str, list[EvalReport]]:
    """Évaluation 5-fold stratifiée des deux stratégies sur les labels forts.

    À chaque fold, le test set est composé d'IRM fortement labellisées **jamais
    vues** ; le train set est utilisé tel quel pour la baseline supervisée et,
    augmenté des pseudo-labels faibles, pour la stratégie semi-supervisée.

    Returns
    -------
    dict
        ``{"supervised": [EvalReport, ...], "semi_supervised": [...]}``
    """
    from sklearn.model_selection import StratifiedKFold

    cfg = cfg or TrainingConfig()
    skf = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=seed)
    paths_arr = np.asarray(strong_paths, dtype=object)
    labels_arr = np.asarray(strong_labels, dtype=int)

    reports_sup: list[EvalReport] = []
    reports_semi: list[EvalReport] = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(paths_arr, labels_arr), start=1):
        s_train_paths = paths_arr[train_idx].tolist()
        s_train_labels = labels_arr[train_idx].tolist()
        s_test_paths = paths_arr[test_idx].tolist()
        s_test_labels = labels_arr[test_idx].tolist()

        _, sup_report = train_supervised(
            train_paths=s_train_paths,
            train_labels=s_train_labels,
            test_paths=s_test_paths,
            test_labels=s_test_labels,
            class_names=class_names,
            cfg=cfg,
        )
        reports_sup.append(sup_report)

        if weak_paths is not None and weak_labels is not None:
            _, semi_report = train_semi_supervised(
                weak_paths=list(weak_paths),
                weak_labels=list(weak_labels),
                strong_train_paths=s_train_paths,
                strong_train_labels=s_train_labels,
                strong_test_paths=s_test_paths,
                strong_test_labels=s_test_labels,
                class_names=class_names,
                cfg=cfg,
            )
            reports_semi.append(semi_report)

        print(
            f"  fold {fold}/{cfg.cv_folds} | sup acc={sup_report.accuracy:.3f} f1={sup_report.f1_macro:.3f}"
            + (
                f" | semi acc={reports_semi[-1].accuracy:.3f} f1={reports_semi[-1].f1_macro:.3f}"
                if reports_semi
                else ""
            ),
            flush=True,
        )

    return {"supervised": reports_sup, "semi_supervised": reports_semi}


def aggregate_reports(reports: list[EvalReport]) -> dict[str, dict[str, float]]:
    """Renvoie ``{metric: {"mean": ..., "std": ..., "values": [...]}}``."""
    if not reports:
        return {}
    metrics: dict[str, list[float]] = {
        "accuracy": [r.accuracy for r in reports],
        "f1_macro": [r.f1_macro for r in reports],
        "recall_cancer": [r.recall_per_class.get("cancer", float("nan")) for r in reports],
        "precision_cancer": [r.precision_per_class.get("cancer", float("nan")) for r in reports],
        "f1_cancer": [r.f1_per_class.get("cancer", float("nan")) for r in reports],
    }
    if all(r.roc_auc is not None for r in reports):
        metrics["roc_auc"] = [r.roc_auc for r in reports]  # type: ignore[misc]
    return {
        name: {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=0)),
            "values": [float(v) for v in values],
        }
        for name, values in metrics.items()
    }


__all__ = [
    "EpochLog",
    "EvalReport",
    "TrainingConfig",
    "aggregate_reports",
    "build_classifier",
    "cross_validate",
    "evaluate",
    "make_loader",
    "train_semi_supervised",
    "train_supervised",
]
