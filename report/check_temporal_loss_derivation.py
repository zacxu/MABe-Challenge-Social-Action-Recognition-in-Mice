"""Check the report's loss equations against the implemented temporal objective.

This is a documentation audit on synthetic, float64 tensors. It does not train a
model, read a dataset, or change model/configuration/experimental-result files.
Run from any directory with the project's environment, for example:
    .venv\\Scripts\\python.exe report\\check_temporal_loss_derivation.py
"""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
import sys

import torch
from torch import Tensor
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mabe.temporal_model import (  # noqa: E402
    combined_temporal_loss,
    masked_bce_with_logits,
    masked_macro_soft_f1_loss,
)

ATOL = 1e-12
RTOL = 1e-10
EPSILON = 1e-6


@dataclass(frozen=True)
class SyntheticCase:
    name: str
    logits: Tensor
    targets: Tensor
    loss_mask: Tensor
    sample_weights: Tensor | None = None
    gamma: float | None = None
    auxiliary_weight: float | None = None


def mathematical_components(
    logits: Tensor,
    targets: Tensor,
    loss_mask: Tensor,
    *,
    positive_weight: float,
    gamma: float,
    auxiliary_weight: float,
    sample_weights: Tensor | None,
) -> dict[str, Tensor]:
    """Evaluate the report's equations without using the model's loss helpers."""
    probability = logits.sigmoid()
    supervision_weight = loss_mask
    if sample_weights is not None:
        supervision_weight = loss_mask * sample_weights[:, None, None]

    weighted_bce = -(
        positive_weight * targets * torch.log(probability)
        + (1 - targets) * torch.log1p(-probability)
    )
    probability_true = probability * targets + (1 - probability) * (1 - targets)
    focal_factor = (1 - probability_true).pow(gamma)
    normalizer = supervision_weight.sum().clamp_min(1.0)
    focal_loss = (supervision_weight * focal_factor * weighted_bce).sum() / normalizer

    true_positive = (supervision_weight * probability * targets).sum(dim=(0, 1))
    false_positive = (supervision_weight * probability * (1 - targets)).sum(dim=(0, 1))
    false_negative = (supervision_weight * (1 - probability) * targets).sum(dim=(0, 1))
    active_actions = supervision_weight.sum(dim=(0, 1)) > 0
    per_action_soft_f1 = (2 * true_positive + EPSILON) / (
        2 * true_positive + false_positive + false_negative + EPSILON
    )
    if bool(active_actions.any()):
        soft_f1_loss = 1 - per_action_soft_f1[active_actions].mean()
    else:
        # The empty active-action mean is not defined; the implementation
        # explicitly supplies a differentiable zero in this case.
        soft_f1_loss = logits.sum() * 0
    combined_loss = focal_loss + auxiliary_weight * soft_f1_loss

    return {
        "focal_loss": focal_loss,
        "soft_f1_loss": soft_f1_loss,
        "combined_loss": combined_loss,
        "normalizer": normalizer,
        "supervision_weight_sum": supervision_weight.sum(),
        "active_action_count": active_actions.sum(),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "per_action_soft_f1": per_action_soft_f1,
        "weighted_bce": weighted_bce,
        "probability_true": probability_true,
        "focal_factor": focal_factor,
    }


def source_record(relative_path: str, symbols: list[str]) -> dict[str, object]:
    path = PROJECT_ROOT / relative_path
    return {
        "path": relative_path,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "symbols": symbols,
    }


def synthetic_cases() -> list[SyntheticCase]:
    logits = torch.tensor(
        [[[-2.0, 0.4], [1.3, -0.7], [0.1, 2.2]],
         [[0.8, -1.6], [-0.5, 0.2], [2.6, -2.8]]],
        dtype=torch.float64,
    )
    binary_targets = torch.tensor(
        [[[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
         [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]],
        dtype=torch.float64,
    )
    full_mask = torch.ones_like(logits)
    mixed_targets = 0.65 * binary_targets + 0.35 * binary_targets.flip(0)
    three_action_logits = torch.cat([logits, logits[..., :1]], dim=2)
    three_action_targets = torch.cat([binary_targets, binary_targets[..., :1]], dim=2)
    partial_mask = torch.tensor(
        [[[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
         [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]],
        dtype=torch.float64,
    )
    worked_probability = torch.tensor([[[0.8], [0.6]]], dtype=torch.float64)
    worked_logits = torch.logit(worked_probability)
    worked_targets = torch.tensor([[[1.0], [0.0]]], dtype=torch.float64)

    return [
        SyntheticCase("binary_targets", logits, binary_targets, full_mask),
        SyntheticCase("fractional_mixup_targets", logits, mixed_targets, full_mask),
        SyntheticCase(
            "partial_masks_and_inactive_action",
            three_action_logits,
            three_action_targets,
            partial_mask,
            torch.tensor([1.3, 0.7], dtype=torch.float64),
        ),
        SyntheticCase(
            "unequal_per_sample_weights",
            logits,
            binary_targets,
            full_mask,
            torch.tensor([0.25, 2.3], dtype=torch.float64),
        ),
        SyntheticCase("zero_supervision", logits, binary_targets, torch.zeros_like(logits)),
        SyntheticCase(
            "subunit_mask_denominator_clamp",
            worked_logits,
            worked_targets,
            torch.full_like(worked_logits, 0.1),
            torch.tensor([0.5], dtype=torch.float64),
        ),
        SyntheticCase("lambda_zero", logits, mixed_targets, full_mask, auxiliary_weight=0.0),
        SyntheticCase("gamma_zero", logits, binary_targets, full_mask, gamma=0.0),
        SyntheticCase("valid_action_without_positive_targets", logits, torch.zeros_like(logits), full_mask),
        SyntheticCase("worked_two_frame_example", worked_logits, worked_targets, torch.ones_like(worked_logits)),
    ]


def check_case(case: SyntheticCase, training: dict[str, object]) -> dict[str, object]:
    positive_weight = float(training["positive_class_weight"])
    gamma = float(training["focal_gamma"]) if case.gamma is None else case.gamma
    auxiliary_weight = float(training["soft_f1_weight"]) if case.auxiliary_weight is None else case.auxiliary_weight
    actual_logits = case.logits.clone().requires_grad_(True)
    mathematical_logits = case.logits.clone().requires_grad_(True)
    expected = mathematical_components(
        mathematical_logits,
        case.targets,
        case.loss_mask,
        positive_weight=positive_weight,
        gamma=gamma,
        auxiliary_weight=auxiliary_weight,
        sample_weights=case.sample_weights,
    )
    actual_components = {
        "focal_loss": masked_bce_with_logits(
            actual_logits, case.targets, case.loss_mask,
            positive_weight=positive_weight, focal_gamma=gamma,
            sample_weights=case.sample_weights,
        ),
        "soft_f1_loss": masked_macro_soft_f1_loss(
            actual_logits, case.targets, case.loss_mask,
            sample_weights=case.sample_weights, epsilon=EPSILON,
        ),
        "combined_loss": combined_temporal_loss(
            actual_logits, case.targets, case.loss_mask,
            positive_weight=positive_weight, soft_f1_weight=auxiliary_weight,
            focal_gamma=gamma, sample_weights=case.sample_weights,
        ),
    }
    errors: dict[str, float] = {}
    for component_name, actual in actual_components.items():
        torch.testing.assert_close(actual, expected[component_name], rtol=RTOL, atol=ATOL)
        errors[component_name] = float((actual - expected[component_name]).detach().abs())

    actual_gradient = torch.autograd.grad(actual_components["combined_loss"], actual_logits)[0]
    expected_gradient = torch.autograd.grad(expected["combined_loss"], mathematical_logits)[0]
    if not bool(torch.isfinite(actual_gradient).all() and torch.isfinite(expected_gradient).all()):
        raise AssertionError(f"Non-finite gradient in {case.name}")
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=RTOL, atol=ATOL)
    errors["gradient_maximum_absolute_error"] = float((actual_gradient - expected_gradient).abs().max())
    record: dict[str, object] = {
        "name": case.name,
        "passed": True,
        "shape": list(case.logits.shape),
        "positive_weight": positive_weight,
        "focal_gamma": gamma,
        "soft_f1_weight": auxiliary_weight,
        "supervision_weight_sum": float(expected["supervision_weight_sum"]),
        "focal_denominator": float(expected["normalizer"]),
        "active_action_count": int(expected["active_action_count"]),
        "components": {name: float(value.detach()) for name, value in actual_components.items()},
        "errors": errors,
        "gradients_finite": True,
    }
    if case.name == "zero_supervision":
        if float(actual_components["combined_loss"].detach()) != 0.0 or bool(actual_gradient.any()):
            raise AssertionError("Zero supervision must yield zero loss and gradient")
        record["zero_loss_and_zero_gradient"] = True
    if case.name == "subunit_mask_denominator_clamp":
        if float(expected["normalizer"]) != 1.0:
            raise AssertionError("A supervision-weight sum below one must be clamped to one")
        record["denominator_clamped_to_one"] = True
    if case.name == "worked_two_frame_example":
        record["worked_example"] = {
            "probabilities": case.logits.sigmoid().flatten().tolist(),
            "targets": case.targets.flatten().tolist(),
            "supervision_weights": case.loss_mask.flatten().tolist(),
            "weighted_bce_per_frame": expected["weighted_bce"].detach().flatten().tolist(),
            "probability_true_per_frame": expected["probability_true"].detach().flatten().tolist(),
            "focal_factor_per_frame": expected["focal_factor"].detach().flatten().tolist(),
            "true_positive": float(expected["true_positive"][0].detach()),
            "false_positive": float(expected["false_positive"][0].detach()),
            "false_negative": float(expected["false_negative"][0].detach()),
            "soft_f1": float(expected["per_action_soft_f1"][0].detach()),
            "scaled_soft_f1_loss": float((auxiliary_weight * expected["soft_f1_loss"]).detach()),
        }
    return record


def main() -> None:
    config_path = PROJECT_ROOT / "configs/temporal-performance.yaml"
    configuration = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    training = configuration["training"]
    # Fixed test parameters are intentional: a future configuration change
    # should prompt an update of both the derivation and this documentation audit.
    required_values = {
        "positive_class_weight": 1.5,
        "focal_gamma": 2.0,
        "soft_f1_weight": 0.20,
        "lab_loss_weight_power": 0.5,
    }
    for name, required in required_values.items():
        if float(training[name]) != required:
            raise AssertionError(f"Configuration drift for {name}: {training[name]} != {required}")
    default_epsilon = inspect.signature(masked_macro_soft_f1_loss).parameters["epsilon"].default
    if float(default_epsilon) != EPSILON:
        raise AssertionError(f"Soft-F1 epsilon drift: {default_epsilon} != {EPSILON}")

    results = [check_case(case, training) for case in synthetic_cases()]
    report = {
        "audit_type": "documentation-only synthetic numerical verification",
        "scope": "Loss equations only; no training, inference, dataset access, or empirical performance validation.",
        "all_passed": all(result["passed"] for result in results),
        "case_count": len(results),
        "tensor_dtype": "float64",
        "device": "cpu",
        "torch_version": torch.__version__,
        "tolerances": {"absolute": ATOL, "relative": RTOL},
        "parameters_from_configuration": {
            **required_values,
        },
        "implementation_constant": {"soft_f1_epsilon": EPSILON, "minimum_focal_denominator": 1.0},
        "formula_summary": {
            "p": "sigmoid(logit)",
            "q": "loss_mask * laboratory_sample_weight; sample weight defaults to 1 when absent",
            "weighted_bce": "-(rho*y*log(p) + (1-y)*log(1-p))",
            "pt": "p*y + (1-p)*(1-y)",
            "focal": "sum(q*(1-pt)^gamma*weighted_bce) / max(sum(q), 1)",
            "soft_counts": "TP=sum(q*p*y); FP=sum(q*p*(1-y)); FN=sum(q*(1-p)*y), over batch and time per action",
            "soft_f1": "(2*TP+epsilon)/(2*TP+FP+FN+epsilon)",
            "active_actions": "actions with sum(q)>0; active does not mean a positive target is present",
            "auxiliary": "1-mean(soft_f1 over active actions); zero if none are active",
            "combined": "focal + lambda*auxiliary",
        },
        "sources": [
            source_record("src/mabe/temporal_model.py", [
                "masked_bce_with_logits", "masked_macro_soft_f1_loss", "combined_temporal_loss",
            ]),
            source_record("configs/temporal-performance.yaml", ["training"]),
            source_record("src/mabe/temporal_training.py", [
                "_lab_loss_weights", "_sample_weights", "combined_temporal_loss call sites",
            ]),
        ],
        "cases": results,
    }
    output_path = PROJECT_ROOT / "report/temporal_loss_derivation_validation.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved {output_path}")


if __name__ == "__main__":
    main()
