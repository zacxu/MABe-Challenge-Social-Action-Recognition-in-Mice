from __future__ import annotations

import torch

from mabe.phase9_config import load_phase9_config
from mabe.phase10_config import load_phase10_config
from mabe.phase10_pipeline import _experiment_phase9_config
from mabe.temporal_model import combined_temporal_loss
from mabe.temporal_training import build_temporal_model


def test_phase10_config_has_one_control_and_bounded_families() -> None:
    config = load_phase10_config("configs/advanced_temporal.yaml")

    assert config.experiments[0].family == "control"
    assert len(config.experiments) == 17
    assert config.screen.failures_before_family_stop == 2
    assert config.screen.maximum_full_oof_candidates == 1


def test_experiment_overrides_do_not_mutate_phase9_control() -> None:
    phase10 = load_phase10_config("configs/advanced_temporal.yaml")
    base = load_phase9_config("configs/temporal.yaml")
    experiment = next(value for value in phase10.experiments if value.name == "keypoints_7")

    variant = _experiment_phase9_config(base, phase10, experiment)

    assert len(base.data.landmarks) == 5
    assert len(variant.data.landmarks) == 7
    assert variant.training.steps_per_epoch == phase10.screen.steps_per_epoch
    assert variant.temporal_version.endswith("keypoints_7")


def test_dynamic_keypoint_and_conv_attention_model_shapes() -> None:
    phase10 = load_phase10_config("configs/advanced_temporal.yaml")
    base = load_phase9_config("configs/temporal.yaml")
    experiment = next(
        value for value in phase10.experiments if value.name == "conv_attention_lite"
    )
    variant = _experiment_phase9_config(base, phase10, experiment)
    model = build_temporal_model(
        config=variant,
        lab_count=15,
        solo_actions=11,
        pair_actions=26,
    )

    solo = model(
        "solo",
        torch.randn(2, 31, 23),
        torch.tensor([0, 3]),
        torch.randn(2, 3),
    )
    pair = model(
        "pair",
        torch.randn(2, 31, 54),
        torch.tensor([0, 3]),
        torch.randn(2, 3),
    )

    assert solo.shape == (2, 31, 11)
    assert pair.shape == (2, 31, 26)
    assert any("attention" in name for name, _ in model.named_parameters())


def test_soft_f1_component_rewards_correct_logits_and_respects_mask() -> None:
    targets = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    mask = torch.ones_like(targets)
    correct = torch.tensor([[[8.0, -8.0], [-8.0, 8.0]]])
    wrong = -correct

    correct_loss = combined_temporal_loss(
        correct,
        targets,
        mask,
        positive_weight=1.0,
        soft_f1_weight=0.1,
    )
    wrong_loss = combined_temporal_loss(
        wrong,
        targets,
        mask,
        positive_weight=1.0,
        soft_f1_weight=0.1,
    )

    assert correct_loss < wrong_loss
