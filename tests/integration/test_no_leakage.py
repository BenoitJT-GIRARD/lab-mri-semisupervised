"""Assert that the protocol cannot leak.

The audit found three leaks that a reader could not have seen from the code, because each
of them looked like an ordinary line. These tests state the invariants those lines broke,
so that breaking them again fails a build rather than producing a better-looking number.

They watch the orchestration rather than the training: a spy replaces `run_arm` and records
what every fold was handed. That runs in milliseconds and covers exactly the wiring where a
leak would live.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mri_semisupervised.config import ProtocolConfig, TrainingConfig
from mri_semisupervised.protocol import experiment as experiment_module
from mri_semisupervised.protocol.arms import ArmResult
from mri_semisupervised.protocol.experiment import CORRECTED, LEGACY, run_experiment

#: Integration tier: these run the protocol's own machinery over a synthetic dataset,
#: which is what makes them able to catch a leak the unit tests cannot see.
pytestmark = pytest.mark.integration

N_LABELLED = 40
N_UNLABELLED = 60
N_LEAKED = 8  # labelled images whose copy also sits in the unlabelled pool


@pytest.fixture()
def wired(tmp_path: Path, monkeypatch):
    """A synthetic dataset, its manifest and its features, wired into the experiment.

    The unlabelled pool deliberately contains copies of eight labelled images: the defect
    the real dataset carries, reproduced so the rules can be seen doing their job.
    """
    rng = np.random.default_rng(0)

    labelled_ids = [f"lab{i:03d}" for i in range(N_LABELLED)]
    labels = [0] * (N_LABELLED // 2) + [1] * (N_LABELLED // 2)
    own_unlabelled = [f"unl{i:03d}" for i in range(N_UNLABELLED)]
    leaked = labelled_ids[:N_LEAKED]

    rows = []
    for image_id, label in zip(labelled_ids, labels, strict=True):
        rows.append(
            {
                "image_id": image_id,
                "path": str(tmp_path / f"{image_id}.png"),
                "pool": "labelled",
                "label": "cancer" if label else "normal",
                "label_index": label,
                "kept_for_training": True,
                "exclusion_reason": None,
            }
        )
    for image_id in own_unlabelled:
        rows.append(
            {
                "image_id": image_id,
                "path": str(tmp_path / f"{image_id}.png"),
                "pool": "unlabelled",
                "label": None,
                "label_index": -1,
                "kept_for_training": True,
                "exclusion_reason": None,
            }
        )
    for image_id in leaked:
        rows.append(
            {
                "image_id": image_id,
                "path": str(tmp_path / f"{image_id}_copy.png"),
                "pool": "unlabelled",
                "label": None,
                "label_index": -1,
                "kept_for_training": False,
                "exclusion_reason": "copy_of_labelled",
            }
        )

    manifest = pd.DataFrame(rows)
    manifest_path = tmp_path / "manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)

    every_id = labelled_ids + own_unlabelled
    features = rng.normal(size=(len(every_id), 12)).astype(np.float32)
    features[[every_id.index(i) for i in labelled_ids if labels[labelled_ids.index(i)] == 1]] += 3.0
    cache = pd.concat(
        [
            pd.DataFrame({"image_id": every_id}),
            pd.DataFrame(features, columns=[f"f_{i:04d}" for i in range(12)]),
        ],
        axis=1,
    )
    cache_path = tmp_path / "features.parquet"
    cache.to_parquet(cache_path, index=False)

    features_config = SimpleNamespace(cache_path=cache_path)

    monkeypatch.setattr(experiment_module, "MANIFEST_PATH", manifest_path)
    # The double has to carry the real interface, not just be callable. These tests
    # exercise the corrected and legacy modes, which read the same cache, so one
    # config answers every variant.
    monkeypatch.setattr(
        experiment_module,
        "FeatureConfig",
        SimpleNamespace(for_variant=lambda **_: features_config),
    )

    seen: list[dict] = []

    def spy(arm, *, fold_name, paths_by_id, inner_train, inner_val, test, pseudo, cfg, seed):
        seen.append(
            {
                "arm": arm,
                "fold": fold_name,
                "inner_train": [i for i, _ in inner_train],
                "inner_val": [i for i, _ in inner_val],
                "test": [i for i, _ in test],
                "pretrain": list(pseudo.image_ids) if pseudo is not None else [],
            }
        )
        n = len(test)
        return ArmResult(
            arm=arm,
            fold=fold_name,
            y_true=np.array([y for _, y in test]),
            y_score=np.linspace(0.1, 0.9, n),
            val_y_true=np.array([y for _, y in inner_val]),
            val_y_score=np.linspace(0.1, 0.9, len(inner_val)),
            best_epoch=1,
            pretrain_steps=len(pseudo.image_ids) if pseudo is not None else 0,
            finetune_steps=1,
        )

    monkeypatch.setattr(experiment_module, "run_arm", spy)
    return seen, set(labelled_ids), set(leaked)


def _run(mode: str, output: Path):
    return run_experiment(
        mode=mode,
        protocol=ProtocolConfig(n_splits=4, n_repeats=2, n_bootstrap=10),
        training=TrainingConfig(),
        output_root=output,
        progress=False,
    )


def test_no_test_image_is_ever_in_a_pretraining_set(wired, tmp_path: Path) -> None:
    seen, _, _ = wired
    _run(CORRECTED, tmp_path / "out")

    for call in seen:
        assert set(call["test"]).isdisjoint(call["pretrain"]), (
            f"{call['arm']} on {call['fold']} pre-trained on its own test images"
        )


def test_no_training_image_is_in_a_pretraining_set_either(wired, tmp_path: Path) -> None:
    # The pseudo-labels cover the unlabelled pool alone; a labelled image appearing there
    # would mean an image carried both a true label and a pseudo-label in the same fold.
    seen, labelled_ids, _ = wired
    _run(CORRECTED, tmp_path / "out")

    for call in seen:
        assert set(call["pretrain"]).isdisjoint(labelled_ids)


def test_the_copies_of_evaluation_images_never_reach_the_pretraining(wired, tmp_path: Path):
    seen, _, leaked = wired
    _run(CORRECTED, tmp_path / "out")

    for call in seen:
        assert set(call["pretrain"]).isdisjoint(leaked)


def test_the_legacy_mode_does_leak(wired, tmp_path: Path) -> None:
    """Without this, the tests above could pass on a protocol that never had a problem.

    The legacy mode reproduces the original: the leaked copies stay in the pool, so a third
    of every test fold is pre-trained on. It must fail the invariant the corrected mode
    holds — otherwise the correction is measuring nothing.
    """
    seen, _, leaked = wired
    _run(LEGACY, tmp_path / "out")

    pretrained_on_leaked = [c for c in seen if set(c["pretrain"]) & leaked]
    assert pretrained_on_leaked, "the legacy mode is supposed to carry the defect"


def test_the_inner_validation_never_comes_from_the_test_fold(wired, tmp_path: Path) -> None:
    seen, _, _ = wired
    _run(CORRECTED, tmp_path / "out")

    for call in seen:
        assert set(call["inner_val"]).isdisjoint(call["test"])
        assert set(call["inner_train"]).isdisjoint(call["test"])
        assert set(call["inner_train"]).isdisjoint(call["inner_val"])


def test_every_arm_of_a_fold_shares_exactly_the_same_split(wired, tmp_path: Path) -> None:
    seen, _, _ = wired
    _run(CORRECTED, tmp_path / "out")

    by_fold: dict[str, list[dict]] = {}
    for call in seen:
        by_fold.setdefault(call["fold"], []).append(call)

    for fold, calls in by_fold.items():
        tests = {tuple(c["test"]) for c in calls}
        trains = {tuple(c["inner_train"]) for c in calls}
        assert len(tests) == 1, f"the arms of {fold} were scored on different images"
        assert len(trains) == 1, f"the arms of {fold} were trained on different images"


def test_the_pretraining_set_is_identical_for_the_semi_arm_and_its_control(
    wired, tmp_path: Path
) -> None:
    # Same images, same count: the control differs only in which label goes with which
    # image. If the sets differed, the comparison would confound information with exposure.
    seen, _, _ = wired
    _run(CORRECTED, tmp_path / "out")

    by_fold: dict[str, dict[str, list]] = {}
    for call in seen:
        by_fold.setdefault(call["fold"], {})[call["arm"]] = call["pretrain"]

    for fold, arms in by_fold.items():
        assert arms["semi_supervised"] == arms["permuted_control"], f"fold {fold}"
        assert arms["supervised"] == [], "the baseline pre-trains on nothing"
        # The filtered arm keeps a subset of the same images, never other ones.
        assert set(arms["semi_supervised_confident"]) <= set(arms["semi_supervised"]), fold


def test_every_evaluation_image_is_tested_once_per_repeat(wired, tmp_path: Path) -> None:
    seen, labelled_ids, _ = wired
    _run(CORRECTED, tmp_path / "out")

    per_repeat: dict[str, list[str]] = {}
    for call in seen:
        if call["arm"] != "supervised":
            continue
        per_repeat.setdefault(call["fold"][:2], []).extend(call["test"])

    for repeat, tested in per_repeat.items():
        assert sorted(tested) == sorted(labelled_ids), f"repeat {repeat} did not cover the set"


def test_no_implemented_arm_can_be_written_and_then_never_run() -> None:
    """ARMS is the registry of what exists; ProtocolConfig.arms is what runs by default.

    Caught in the act on 2026-09-03: a fourth arm was added to ARMS and not to the config,
    so the experiment ran three arms while every test passed. An arm may legitimately be
    opt-in — the joint arms answer a separate question and cost a run of their own — but it
    has to say so by being listed, not by being forgotten.
    """
    from mri_semisupervised.config import ProtocolConfig
    from mri_semisupervised.protocol.arms import ARMS, OPTIONAL_ARMS

    default = set(ProtocolConfig().arms)
    assert default <= set(ARMS), "the config names an arm that does not exist"
    assert set(ARMS) - default == set(OPTIONAL_ARMS), (
        "every implemented arm must be run by default or declared opt-in"
    )
