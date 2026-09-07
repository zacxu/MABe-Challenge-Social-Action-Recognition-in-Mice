"""Validated Phase 8 experiment and nested-decoder configuration."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class FeatureCandidate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    solo: tuple[str, ...]
    pair: tuple[str, ...]

    @field_validator("solo", "pair")
    @classmethod
    def unique_groups(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or len(value) != len(set(value)):
            raise ValueError("feature candidate groups must be non-empty and unique")
        return value


class HyperparameterCandidate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    n_estimators: int | None = Field(default=None, gt=0)
    learning_rate: float | None = Field(default=None, gt=0, le=1)
    max_depth: int | None = Field(default=None, gt=0)
    min_child_weight: float | None = Field(default=None, ge=0)
    gamma: float | None = Field(default=None, ge=0)
    subsample: float | None = Field(default=None, gt=0, le=1)
    colsample_bytree: float | None = Field(default=None, gt=0, le=1)
    reg_alpha: float | None = Field(default=None, ge=0)
    reg_lambda: float | None = Field(default=None, ge=0)
    positive_class_weight: Literal["auto", "sqrt_auto"] | float | None = None
    maximum_training_rows_per_route: int | None = Field(default=None, gt=1)
    negative_to_positive_ratio: float | None = Field(default=None, gt=0)

    def model_updates(self) -> dict[str, object]:
        excluded = {"maximum_training_rows_per_route", "negative_to_positive_ratio"}
        return {
            key: value
            for key, value in self.model_dump(exclude_none=True).items()
            if key not in excluded
        }


class ScreenConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    held_out_folds: tuple[int, ...]
    representative_routes_per_lab: int = Field(gt=0)
    maximum_training_rows_per_route: int = Field(gt=1)
    maximum_validation_rows_per_route: int = Field(gt=1)
    threshold_grid: tuple[float, ...]
    feature_candidates: dict[str, FeatureCandidate]
    hyperparameter_candidates: dict[str, HyperparameterCandidate]

    @field_validator("held_out_folds")
    @classmethod
    def valid_folds(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value or tuple(sorted(set(value))) != value:
            raise ValueError("screen folds must be sorted and unique")
        return value

    @field_validator("threshold_grid")
    @classmethod
    def valid_thresholds(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if tuple(sorted(set(value))) != value or any(item <= 0 or item >= 1 for item in value):
            raise ValueError("screen thresholds must be sorted, unique, and inside (0, 1)")
        return value


class HardNegativeConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool
    top_rows_per_lab_action_fold: int = Field(gt=0)
    minimum_probability: float = Field(ge=0, le=1)
    repeat_count: int = Field(gt=0)


class ImportanceConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    permutation_models_per_route: int = Field(gt=0)
    maximum_validation_rows: int = Field(gt=1)
    seed: int = Field(ge=0)


class RoutingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    blend_weights: tuple[float, ...]
    minimum_component_improvement: float = Field(ge=0)
    minimum_winning_folds: int = Field(gt=0)

    @field_validator("blend_weights")
    @classmethod
    def valid_weights(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if tuple(sorted(set(value))) != value or any(item < 0 or item > 1 for item in value):
            raise ValueError("blend weights must be sorted, unique, and in [0, 1]")
        return value


class Phase8DecoderConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    additional_smoothing_seconds: tuple[float, ...]
    threshold_grid: tuple[float, ...]
    hysteresis_delta: tuple[float, ...]
    gap_fill_seconds: tuple[float, ...]
    minimum_run_seconds: tuple[float, ...]
    minimum_lab_action_positive_frames: int = Field(gt=0)
    minimum_lab_action_positive_videos: int = Field(gt=0)
    minimum_action_positive_frames: int = Field(gt=0)

    @field_validator(
        "additional_smoothing_seconds",
        "hysteresis_delta",
        "gap_fill_seconds",
        "minimum_run_seconds",
    )
    @classmethod
    def nonnegative_grid(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if not value or tuple(sorted(set(value))) != value or any(item < 0 for item in value):
            raise ValueError("decoder grids must be non-empty, sorted, unique, and non-negative")
        return value

    @field_validator("threshold_grid")
    @classmethod
    def decoder_thresholds(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if tuple(sorted(set(value))) != value or any(item <= 0 or item >= 1 for item in value):
            raise ValueError("decoder thresholds must be sorted, unique, and inside (0, 1)")
        return value


class PromotionConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    minimum_exact_fbeta_gain: float = Field(ge=0)
    seed: int = Field(ge=0)
    retrain_selected_model: bool = False
    promoted_model_version: str | None = None
    minimum_exact_model_gain: float = Field(default=0.0, ge=0)


class Phase8Config(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(ge=1)
    phase8_version: str = Field(min_length=1)
    baseline_model_version: str = Field(min_length=1)
    screen: ScreenConfig
    hard_negative: HardNegativeConfig
    importance: ImportanceConfig
    routing: RoutingConfig
    decoder: Phase8DecoderConfig
    promotion: PromotionConfig


def load_phase8_config(path: str | Path = Path("configs/phase8.yaml")) -> Phase8Config:
    source = Path(path)
    loaded: object = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Phase 8 configuration must be a mapping: {source}")
    return Phase8Config.model_validate(cast(dict[str, Any], loaded))


def phase8_config_fingerprint(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
