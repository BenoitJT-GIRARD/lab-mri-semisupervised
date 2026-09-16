"""End-to-end checks of one arm on one fold, on a handful of tiny images.

The leakage suite watches the orchestration with a spy, so ``run_arm`` itself is never
executed there. These tests run it for real — small enough to stay in the suite, big enough
to catch a fourth arm that raises on its first fold after forty minutes of GPU time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mri_semisupervised.config import TrainingConfig
from mri_semisupervised.protocol.arms import (
    JOINT_PERMUTED_CONTROL,
    PERMUTED_CONTROL,
    SELF_TRAINING,
    SELF_TRAINING_CONTROL,
    SEMI_SUPERVISED,
    SEMI_SUPERVISED_CONFIDENT,
    SEMI_SUPERVISED_JOINT,
    SUPERVISED,
    budget_steps,
    run_arm,
)
from mri_semisupervised.protocol.pseudo_labels import PseudoLabelSet

TINY = TrainingConfig(
    architecture="resnet18",
    batch_size=4,
    epochs_weak=1,
    epochs_strong=1,
    early_stopping_patience=1,
    confidence_quantiles=(0.0, 0.5),
)


@pytest.fixture()
def fold(synthetic_dataset: Path):
    """One fold built from the synthetic tree: eight labelled images, six in the pool."""
    from mri_semisupervised.data.loader import discover_images

    records, _ = discover_images(synthetic_dataset)
    paths_by_id = {r.image_id: r.path for r in records}

    labelled = [r for r in records if r.split == "labeled"]
    pool = [r for r in records if r.split != "labeled"]

    inner_train = [(r.image_id, r.label_index) for r in labelled[:2] + labelled[-2:]]
    inner_val = [(r.image_id, r.label_index) for r in labelled[2:4] + labelled[-4:-2]]
    test = inner_val

    rng = np.random.default_rng(0)
    pseudo = PseudoLabelSet(
        method_name="KMeans",
        ari_on_train=0.4,
        image_ids=np.array([r.image_id for r in pool], dtype=object),
        labels=np.array([i % 2 for i in range(len(pool))]),
        confidence=rng.uniform(0.1, 0.9, len(pool)),
        n_noise=1,
    )
    return paths_by_id, inner_train, inner_val, test, pseudo


def _run(arm, fold, **kwargs):
    paths_by_id, inner_train, inner_val, test, pseudo = fold
    return run_arm(
        arm,
        fold_name="f0",
        paths_by_id=paths_by_id,
        inner_train=inner_train,
        inner_val=inner_val,
        test=test,
        pseudo=None if arm == SUPERVISED else pseudo,
        cfg=TINY,
        seed=0,
        **kwargs,
    )


def test_the_budget_counts_the_steps_the_unfiltered_pool_would_take() -> None:
    assert budget_steps(10, TrainingConfig(batch_size=4, epochs_weak=3)) == 9
    assert budget_steps(8, TrainingConfig(batch_size=4, epochs_weak=2)) == 4
    assert budget_steps(0, TrainingConfig()) == 0


def test_the_baseline_pretrains_on_nothing(fold) -> None:
    result = _run(SUPERVISED, fold)
    assert result.pretrain_steps == 0
    assert result.pretrain_image_ids == ()
    assert result.confidence_quantile is None


def test_the_confident_arm_reports_the_quantile_it_chose(fold) -> None:
    result = _run(SEMI_SUPERVISED_CONFIDENT, fold)
    assert result.confidence_quantile in TINY.confidence_quantiles
    assert 0 < result.n_pseudo_used <= len(fold[4])


def test_the_confident_arm_spends_the_same_budget_as_the_unfiltered_one(fold) -> None:
    """Filtering shrinks the pool. Equal epochs would mean unequal gradient updates, and a
    budget confound between arms is the defect the leak-free protocol removed once."""
    plain = _run(SEMI_SUPERVISED, fold)
    confident = _run(SEMI_SUPERVISED_CONFIDENT, fold)

    assert confident.pretrain_steps == plain.pretrain_steps


def test_the_control_pretrains_on_the_same_images_as_the_semi_arm(fold) -> None:
    plain = _run(SEMI_SUPERVISED, fold)
    control = _run(PERMUTED_CONTROL, fold)

    assert set(plain.pretrain_image_ids) == set(control.pretrain_image_ids)


def test_every_arm_scores_the_test_images_and_only_those(fold) -> None:
    _, _, _, test, _ = fold
    for arm in (SUPERVISED, SEMI_SUPERVISED, SEMI_SUPERVISED_CONFIDENT, PERMUTED_CONTROL):
        result = _run(arm, fold)
        assert len(result.y_score) == len(test)
        assert ((result.y_score >= 0.0) & (result.y_score <= 1.0)).all()


def test_an_unknown_arm_is_refused_by_name(fold) -> None:
    with pytest.raises(ValueError, match="unknown arm"):
        _run("semi_supervised_confident_v2", fold)


def test_the_joint_arm_never_pretrains(fold) -> None:
    """Its whole point is that the pseudo-labels are not left behind by the fine-tuning."""
    result = _run(SEMI_SUPERVISED_JOINT, fold)

    assert result.pretrain_steps == 0
    assert result.n_pseudo_used > 0
    assert result.pseudo_weight == TINY.pseudo_loss_weight


def test_the_joint_arm_sees_the_labels_as_often_as_the_baseline(fold) -> None:
    """The epoch is counted in labelled batches; the pseudo-labels are the only addition."""
    plain = _run(SUPERVISED, fold)
    joint = _run(SEMI_SUPERVISED_JOINT, fold)

    assert joint.finetune_steps == plain.finetune_steps


def test_the_joint_control_uses_the_same_images_with_the_pairing_destroyed(fold) -> None:
    joint = _run(SEMI_SUPERVISED_JOINT, fold)
    control = _run(JOINT_PERMUTED_CONTROL, fold)

    assert set(joint.pretrain_image_ids) == set(control.pretrain_image_ids)
    assert control.n_pseudo_used == joint.n_pseudo_used


def test_a_zero_weight_still_moves_the_predictions_because_of_batchnorm(fold) -> None:
    """The joint arm carries a second treatment, and it must be named rather than hidden.

    With the pseudo loss weighted to zero the gradient is identical — the training loss is
    the same to the last digit — and the predictions still differ. The pseudo batches go
    through the network in train mode, so BatchNorm updates its running statistics on the
    unlabelled pool. That is unsupervised adaptation to the target distribution, and it is
    arguably a form of semi-supervision in its own right.

    It is also why the joint arm may only be judged against the joint control, which passes
    exactly the same images through the same BatchNorm. Compared against the plain
    supervised baseline, the two treatments would be indistinguishable.
    """
    from dataclasses import replace

    paths_by_id, inner_train, inner_val, test, pseudo = fold
    zeroed = replace(TINY, pseudo_loss_weight=0.0)
    joint = run_arm(
        SEMI_SUPERVISED_JOINT,
        fold_name="f0",
        paths_by_id=paths_by_id,
        inner_train=inner_train,
        inner_val=inner_val,
        test=test,
        pseudo=pseudo,
        cfg=zeroed,
        seed=0,
    )
    plain = _run(SUPERVISED, fold)

    assert joint.history[0]["train_loss"] == pytest.approx(plain.history[0]["train_loss"]), (
        "a zero weight must leave the loss untouched"
    )
    assert not np.allclose(joint.y_score, plain.y_score, atol=1e-6), (
        "the batchnorm statistics did absorb the pool, and the arm must be read knowing it"
    )


def test_self_training_builds_its_labels_from_its_own_first_pass(fold) -> None:
    """The pseudo-labels come from the decision function, not from a clustering."""
    result = _run(SELF_TRAINING, fold)

    assert result.n_self_labelled is not None
    assert result.n_self_labelled == result.n_pseudo_used > 0
    assert result.pretrain_steps > 0
    assert any(entry["phase"] == "scout" for entry in result.history)


def test_self_training_ignores_the_clustering_labels_it_is_handed(fold) -> None:
    """Only the pool's identities are taken from `pseudo`; its labels are never read."""
    from dataclasses import replace

    paths_by_id, inner_train, inner_val, test, pseudo = fold
    flipped = replace(pseudo, labels=1 - pseudo.labels)

    a = _run(SELF_TRAINING, fold)
    b = run_arm(
        SELF_TRAINING,
        fold_name="f0",
        paths_by_id=paths_by_id,
        inner_train=inner_train,
        inner_val=inner_val,
        test=test,
        pseudo=flipped,
        cfg=TINY,
        seed=0,
    )
    np.testing.assert_allclose(a.y_score, b.y_score)


def test_the_self_training_control_keeps_the_same_images(fold) -> None:
    a = _run(SELF_TRAINING, fold)
    b = _run(SELF_TRAINING_CONTROL, fold)

    assert set(a.pretrain_image_ids) == set(b.pretrain_image_ids)


def test_a_pool_where_nothing_is_confident_still_yields_a_pretraining_set(fold) -> None:
    """A first pass that is sure of nothing must not silently turn the arm into a baseline."""
    from dataclasses import replace

    paths_by_id, inner_train, inner_val, test, pseudo = fold
    impossible = replace(TINY, self_training_threshold=1.01)
    result = run_arm(
        SELF_TRAINING,
        fold_name="f0",
        paths_by_id=paths_by_id,
        inner_train=inner_train,
        inner_val=inner_val,
        test=test,
        pseudo=pseudo,
        cfg=impossible,
        seed=0,
    )
    assert result.n_self_labelled > 0
