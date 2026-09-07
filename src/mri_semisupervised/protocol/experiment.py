"""Run the whole comparison, and leave behind enough to check it.

Two modes, and the second one exists so that the before/after is **measured** rather than
quoted from an old JSON file:

* ``corrected`` — the protocol this package now defends. Duplicates removed by content,
  pseudo-labels fitted inside each training fold, three arms, checkpoint and threshold
  chosen on an inner validation split.
* ``legacy`` — **the three leaks, and only those**: duplicates left in place, the
  clustering method and the cluster-to-class alignment decided once over every label
  including the test folds. Everything else — the inner validation split, the checkpoint,
  the chosen threshold — stays as the corrected protocol does it.

  That is deliberate, and it is narrower than "reproduce the original". It isolates what
  the *leaks* were worth from what the *training changes* were worth. Running the original
  end to end would confound the two, and the question here is the leaks.

Every run writes a manifest naming its data, its seeds, its versions and its git revision.
A number that cannot be traced back to one of those does not belong in the README.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess  # nosec B404 — one call, on a constant argv; see _git_revision
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mri_semisupervised.config import (
    EXPERIMENTS_DIR,
    MANIFEST_PATH,
    FeatureConfig,
    ProtocolConfig,
    TrainingConfig,
    set_global_seeds,
)
from mri_semisupervised.data.manifest import LABELLED, UNLABELLED, dataset_fingerprint
from mri_semisupervised.protocol.arms import SUPERVISED, run_arm
from mri_semisupervised.protocol.evaluate import evaluate_fold
from mri_semisupervised.protocol.pseudo_labels import PseudoLabelSet, fit_pseudo_labels
from mri_semisupervised.protocol.splits import derive_seed, inner_split, outer_folds

CORRECTED = "corrected"
LEGACY = "legacy"


@dataclass(frozen=True)
class ExperimentResult:
    """What a run produced, and where it was written."""

    run_id: str
    mode: str
    per_fold: pd.DataFrame
    predictions: pd.DataFrame
    folds_meta: pd.DataFrame
    directory: Path


def _git_revision() -> str:
    """The commit the run was produced from, recorded in the manifest.

    `git` is resolved to an absolute path first. Calling it as a bare name would let
    whatever comes first on PATH answer instead — the argv is a constant, so this is the
    only part of the call that an environment could subvert.
    """
    git = shutil.which("git")
    if git is None:
        return "unknown"
    try:
        return subprocess.check_output(  # nosec B603 — absolute path, constant argv
            [git, "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def _versions() -> dict[str, str]:
    import sklearn
    import torch

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "cuda": torch.version.cuda or "none",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def _load_inputs(mode: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, Path]]:
    """Manifest, features and the id → path map, filtered according to the mode."""
    manifest = pd.read_parquet(MANIFEST_PATH)
    cached = pd.read_parquet(FeatureConfig().cache_path)
    feature_cols = [c for c in cached.columns if c.startswith("f_")]

    # One feature row per distinct image: the cache holds one row per *file*.
    cached = cached.drop_duplicates(subset="image_id").reset_index(drop=True)
    features = cached[feature_cols].to_numpy(dtype=np.float32)
    feature_ids = cached["image_id"].to_numpy(dtype=object)

    paths_by_id: dict[str, Path] = {}
    for image_id, path in zip(manifest["image_id"], manifest["path"], strict=True):
        paths_by_id.setdefault(image_id, Path(path))

    return manifest, features, feature_ids, paths_by_id


def _evaluation_set(manifest: pd.DataFrame, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """The images the arms are scored on."""
    labelled = manifest[manifest["pool"] == LABELLED]
    if mode == CORRECTED:
        labelled = labelled[labelled["kept_for_training"]]
    return (
        labelled["image_id"].to_numpy(dtype=object),
        labelled["label_index"].to_numpy(dtype=int),
    )


def _unlabelled_pool(manifest: pd.DataFrame, mode: str) -> np.ndarray:
    """The pool the pre-training draws on.

    In ``legacy`` the pool is taken as the folder gives it — copies of evaluation images
    included, which is the defect being priced.
    """
    pool = manifest[manifest["pool"] == UNLABELLED]
    if mode == CORRECTED:
        pool = pool[pool["kept_for_training"]]
    return pool["image_id"].drop_duplicates().to_numpy(dtype=object)


def run_experiment(
    mode: str = CORRECTED,
    protocol: ProtocolConfig | None = None,
    training: TrainingConfig | None = None,
    output_root: Path | None = None,
    progress: bool = True,
) -> ExperimentResult:
    """Run every fold of every arm, and write the artefacts."""
    if mode not in (CORRECTED, LEGACY):
        raise ValueError(f"unknown mode: {mode}")

    protocol = protocol or ProtocolConfig()
    training = training or TrainingConfig()
    output_root = output_root or EXPERIMENTS_DIR

    started = time.time()
    set_global_seeds(protocol.seed)

    manifest, features, feature_ids, paths_by_id = _load_inputs(mode)
    eval_ids, eval_labels = _evaluation_set(manifest, mode)
    pool_ids = _unlabelled_pool(manifest, mode)

    # The legacy protocol decides the clustering method and the alignment once, over every
    # label it can see — including the ones every fold is about to be tested on.
    global_pseudo: PseudoLabelSet | None = None
    if mode == LEGACY:
        global_pseudo = fit_pseudo_labels(
            features, feature_ids, pool_ids, eval_ids, eval_labels, seed=protocol.seed
        )

    rows, predictions, folds_meta = [], [], []
    specs = list(outer_folds(eval_labels, protocol.n_splits, protocol.n_repeats, protocol.seed))

    for index, spec in enumerate(specs, start=1):
        train_ids, train_labels = eval_ids[spec.train_idx], eval_labels[spec.train_idx]
        test_ids, test_labels = eval_ids[spec.test_idx], eval_labels[spec.test_idx]

        if mode == CORRECTED:
            pseudo = fit_pseudo_labels(
                features, feature_ids, pool_ids, train_ids, train_labels, seed=spec.seed
            )
        else:
            pseudo = global_pseudo

        inner_train_idx, inner_val_idx = inner_split(
            eval_labels, spec.train_idx, training.inner_val_fraction, spec.seed
        )
        inner_train = list(
            zip(eval_ids[inner_train_idx], eval_labels[inner_train_idx], strict=True)
        )
        inner_val = list(zip(eval_ids[inner_val_idx], eval_labels[inner_val_idx], strict=True))
        test = list(zip(test_ids, test_labels, strict=True))

        folds_meta.append(
            {
                "fold": spec.name,
                "repeat": spec.repeat,
                "split": spec.fold,
                "n_train": len(spec.train_idx),
                "n_test": len(spec.test_idx),
                "n_inner_val": len(inner_val_idx),
                "pseudo_method": pseudo.method_name if pseudo else None,
                "pseudo_ari_on_train": pseudo.ari_on_train if pseudo else None,
                "n_pseudo": len(pseudo) if pseudo else 0,
                "failed_candidates": (";".join(sorted(pseudo.failed_candidates)) if pseudo else ""),
            }
        )

        for arm in protocol.arms:
            result = run_arm(
                arm,
                fold_name=spec.name,
                paths_by_id=paths_by_id,
                inner_train=inner_train,
                inner_val=inner_val,
                test=test,
                pseudo=None if arm == SUPERVISED else pseudo,
                cfg=training,
                seed=derive_seed(spec.seed, "arm", arm),
            )
            metrics = evaluate_fold(
                result.y_true, result.y_score, result.val_y_true, result.val_y_score
            )
            rows.append(
                {
                    "fold": spec.name,
                    "repeat": spec.repeat,
                    "split": spec.fold,
                    "arm": arm,
                    "best_epoch": result.best_epoch,
                    "pretrain_steps": result.pretrain_steps,
                    "finetune_steps": result.finetune_steps,
                    **metrics,
                }
            )
            predictions.append(
                pd.DataFrame(
                    {
                        "fold": spec.name,
                        "arm": arm,
                        "image_id": [i for i, _ in test],
                        "y_true": result.y_true,
                        "y_score": result.y_score,
                    }
                )
            )

        if progress:
            print(f"  [{index:>3}/{len(specs)}] {spec.name} done", flush=True)

    per_fold = pd.DataFrame(rows)
    all_predictions = pd.concat(predictions, ignore_index=True)
    folds_frame = pd.DataFrame(folds_meta)

    run_id = f"{mode}-{time.strftime('%Y%m%d-%H%M%S')}"
    # One canonical directory per mode rather than one per run: the repository publishes a
    # current result, not an archive of attempts. The run id and the timestamp live in the
    # manifest, so a published figure is still traceable to the run that produced it.
    directory = Path(output_root) / mode
    directory.mkdir(parents=True, exist_ok=True)
    per_fold.to_parquet(directory / "per_fold.parquet", index=False)
    all_predictions.to_parquet(directory / "predictions.parquet", index=False)
    folds_frame.to_parquet(directory / "folds.parquet", index=False)

    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "mode": mode,
                "dataset_fingerprint": dataset_fingerprint(manifest),
                "evaluation_images": len(eval_ids),
                "unlabelled_pool": len(pool_ids),
                "protocol": asdict(protocol),
                "training": {k: str(v) for k, v in asdict(training).items()},
                "git_revision": _git_revision(),
                "versions": _versions(),
                "duration_seconds": round(time.time() - started, 1),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return ExperimentResult(run_id, mode, per_fold, all_predictions, folds_frame, directory)


__all__ = ["CORRECTED", "LEGACY", "ExperimentResult", "run_experiment"]
