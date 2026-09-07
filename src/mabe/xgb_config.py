"""Validated, editable Phase 7 XGBoost baseline configuration."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class XGBoostParameters(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    objective: Literal["binary:logistic"]
    eval_metric: Literal["logloss", "aucpr"]
    tree_method: Literal["hist"]
    device: Literal["auto", "cpu", "cuda"]
    n_estimators: int = Field(gt=0)
    early_stopping_rounds: int = Field(gt=0)
    learning_rate: float = Field(gt=0, le=1)
    max_depth: int = Field(gt=0)
    min_child_weight: float = Field(ge=0)
    gamma: float = Field(ge=0)
    subsample: float = Field(gt=0, le=1)
    colsample_bytree: float = Field(gt=0, le=1)
    reg_alpha: float = Field(ge=0)
    reg_lambda: float = Field(ge=0)
    max_bin: int = Field(gt=1)
    n_jobs: int
    positive_class_weight: Literal["auto", "sqrt_auto"] | float

    @field_validator("positive_class_weight")
    @classmethod
    def validate_positive_weight(
        cls, value: Literal["auto", "sqrt_auto"] | float
    ) -> Literal["auto", "sqrt_auto"] | float:
        if isinstance(value, str):
            return value
        if value <= 0:
            raise ValueError(
                "positive class weight must be 'auto', 'sqrt_auto', or positive"
            )
        return value


class SamplerConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_positive_rows_per_lab_action: int = Field(gt=0)
    negative_to_positive_ratio: float = Field(gt=0)
    minimum_negative_rows_per_lab_action: int = Field(ge=0)
    max_rows_per_lab_action: int = Field(gt=1)
    near_boundary_seconds: float = Field(ge=0)
    background_blocks_per_video_triplet: int = Field(ge=0)
    background_block_frames: int = Field(gt=0)


class SupportConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    minimum_positive_rows: int = Field(gt=0)
    minimum_positive_videos: int = Field(gt=0)
    pooled_minimum_positive_rows: int = Field(gt=0)
    allow_pooled_without_direct_positives: bool = True
    auxiliary_labs_by_action: dict[str, tuple[str, ...]]


class ModelFeatureConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    solo_groups: tuple[str, ...]
    pair_groups: tuple[str, ...]
    maximum_rows_per_chunk: int = Field(gt=0)

    @field_validator("solo_groups", "pair_groups")
    @classmethod
    def validate_groups(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or len(value) != len(set(value)):
            raise ValueError("feature groups must be non-empty and unique")
        return value


class DecoderConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    smoothing_seconds: float = Field(ge=0)
    threshold_grid: tuple[float, ...]
    fallback_threshold: float = Field(ge=0, le=1)

    @field_validator("threshold_grid")
    @classmethod
    def validate_thresholds(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if not value or tuple(sorted(set(value))) != value:
            raise ValueError("threshold grid must be sorted and unique")
        if any(item <= 0 or item >= 1 for item in value):
            raise ValueError("threshold grid entries must be in (0, 1)")
        return value


class ExecutionConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    folds: tuple[int, ...]
    include_auxiliary_training: bool
    seed: int = Field(ge=0, le=2**32 - 1)

    @field_validator("folds")
    @classmethod
    def validate_folds(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value or tuple(sorted(set(value))) != value or any(item < 0 for item in value):
            raise ValueError("folds must be sorted, unique, and non-negative")
        return value


class XGBoostBaselineConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(ge=1)
    model_version: str = Field(min_length=1)
    xgboost: XGBoostParameters
    sampler: SamplerConfig
    support: SupportConfig
    features: ModelFeatureConfig
    decoder: DecoderConfig
    execution: ExecutionConfig


def load_xgboost_config(path: str | Path = Path("configs/xgboost.yaml")) -> XGBoostBaselineConfig:
    source = Path(path)
    loaded: object = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"XGBoost configuration must be a mapping: {source}")
    return XGBoostBaselineConfig.model_validate(cast(dict[str, Any], loaded))


def xgboost_config_fingerprint(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
