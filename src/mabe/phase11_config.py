"""Strict configuration for Phase 11 leakage-safe ensembling."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class BaseOOFStreamConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    manifest: str = Field(min_length=1)
    deployable: bool
    eligible_for_optimized_blend: bool


class EnsembleOptimizationConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    folds: tuple[int, ...]
    temporal_weight_grid: tuple[float, ...]
    threshold_grid: tuple[float, ...]
    minimum_action_positive_frames: int = Field(ge=0)
    minimum_lab_positive_frames: int = Field(ge=0)
    minimum_lab_action_positive_frames: int = Field(ge=0)
    shrinkage_positive_frames: int = Field(gt=0)
    minimum_exact_fbeta_gain: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_grids(self) -> EnsembleOptimizationConfig:
        if not self.folds or len(set(self.folds)) != len(self.folds):
            raise ValueError("folds must be non-empty and unique")
        if not self.temporal_weight_grid or any(value < 0 or value > 1 for value in self.temporal_weight_grid):
            raise ValueError("temporal weights must be in [0, 1]")
        if 0.0 not in self.temporal_weight_grid:
            raise ValueError("temporal weight grid must include the strongest-member fallback 0.0")
        if tuple(sorted(set(self.temporal_weight_grid))) != self.temporal_weight_grid:
            raise ValueError("temporal weights must be unique and increasing")
        if not self.threshold_grid or any(value <= 0 or value >= 1 for value in self.threshold_grid):
            raise ValueError("thresholds must be in (0, 1)")
        return self


class EnsembleDecoderConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    phase8_config: str = Field(min_length=1)
    baseline_smoothing_seconds: float = Field(ge=0)
    additional_smoothing_seconds: tuple[float, ...]
    hysteresis_delta: tuple[float, ...]
    global_threshold_references: tuple[float, ...]
    gap_fill_seconds: tuple[float, ...]
    minimum_run_seconds: tuple[float, ...]

    @model_validator(mode="after")
    def validate_search(self) -> EnsembleDecoderConfig:
        if not self.additional_smoothing_seconds or any(value < 0 for value in self.additional_smoothing_seconds):
            raise ValueError("decoder smoothing candidates must be non-negative")
        if not self.hysteresis_delta or any(value < 0 for value in self.hysteresis_delta):
            raise ValueError("decoder hysteresis candidates must be non-negative")
        if not self.global_threshold_references or any(
            value <= 0 or value >= 1 for value in self.global_threshold_references
        ):
            raise ValueError("global decoder thresholds must be in (0, 1)")
        if not self.gap_fill_seconds or any(value < 0 for value in self.gap_fill_seconds):
            raise ValueError("gap candidates must be non-negative")
        if not self.minimum_run_seconds or any(value < 0 for value in self.minimum_run_seconds):
            raise ValueError("minimum-run candidates must be non-negative")
        return self


class EnsembleRuntimeConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    hidden_video_count: int = Field(gt=0)
    kaggle_limit_seconds: float = Field(gt=0)
    required_headroom_fraction: float = Field(ge=0, lt=1)
    phase8_gbdt_seconds_per_video: float = Field(ge=0)
    temporal_phase9_seconds_per_video: float = Field(ge=0)
    temporal_phase10_seconds_per_video: float = Field(ge=0)


class StackerConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled_on_blend_plateau: bool
    always_evaluate: bool = False
    require_base_geometric_features: bool
    maximum_projected_fraction_of_budget: float = Field(gt=0, le=1)
    maximum_positive_rows_per_action: int = Field(default=25000, gt=0)
    maximum_negative_rows_per_action: int = Field(default=250000, gt=0)
    negative_hash_keep_per_thousand: int = Field(default=10, gt=0, le=1000)
    n_estimators: int = Field(default=300, gt=0)
    learning_rate: float = Field(default=0.05, gt=0, le=1)
    max_depth: int = Field(default=3, gt=0)
    min_child_weight: float = Field(default=10.0, ge=0)
    subsample: float = Field(default=0.9, gt=0, le=1)
    colsample_bytree: float = Field(default=1.0, gt=0, le=1)
    reg_lambda: float = Field(default=5.0, ge=0)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    minimum_exact_fbeta_gain: float = Field(default=0.0005, ge=0)


class Phase11Config(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(ge=1)
    phase11_version: str = Field(min_length=1)
    base_streams: tuple[BaseOOFStreamConfig, ...]
    optimization: EnsembleOptimizationConfig
    decoder: EnsembleDecoderConfig
    runtime: EnsembleRuntimeConfig
    stacker: StackerConfig

    @model_validator(mode="after")
    def validate_streams(self) -> Phase11Config:
        names = [stream.name for stream in self.base_streams]
        if len(names) < 2 or len(names) != len(set(names)):
            raise ValueError("base stream names must be unique and contain at least two systems")
        eligible = [stream for stream in self.base_streams if stream.eligible_for_optimized_blend]
        if len(eligible) != 2:
            raise ValueError("Phase 11 convex optimization requires exactly two eligible streams")
        if any(not stream.deployable for stream in eligible):
            raise ValueError("optimized streams must have full-test inference routes")
        return self


def load_phase11_config(path: str | Path = Path("configs/ensemble.yaml")) -> Phase11Config:
    source = Path(path)
    loaded: object = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Phase 11 configuration must be a mapping: {source}")
    return Phase11Config.model_validate(cast(dict[str, Any], loaded))


def phase11_config_fingerprint(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
