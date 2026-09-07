"""Strict configuration and lineage hashing for Phase 9."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TemporalDataConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    target_fps: float = Field(gt=0)
    landmarks: tuple[str, ...]
    window_frames: int = Field(gt=15)
    inference_stride_frames: int = Field(gt=0)
    inference_edge_frames: int = Field(ge=0)
    positive_fraction: float = Field(ge=0, le=1)
    background_fraction: float = Field(ge=0, le=1)
    hard_negative_fraction: float = Field(ge=0, le=1)
    maximum_positive_centers_per_action_entity: int = Field(gt=0)
    include_auxiliary_training: bool = False

    @model_validator(mode="after")
    def valid_window_and_mixture(self) -> TemporalDataConfig:
        if self.inference_stride_frames > self.window_frames:
            raise ValueError("inference stride cannot exceed window length")
        if self.inference_edge_frames * 2 >= self.window_frames:
            raise ValueError("inference edges must leave a non-empty center")
        total = self.positive_fraction + self.background_fraction + self.hard_negative_fraction
        if abs(total - 1.0) > 1e-9:
            raise ValueError("window sampling fractions must sum to one")
        canonical = (
            "nose",
            "ear_left",
            "ear_right",
            "body_center",
            "tail_base",
            "lateral_left",
            "lateral_right",
        )
        if not 4 <= len(self.landmarks) <= 7:
            raise ValueError("temporal models require between four and seven landmarks")
        if tuple(value for value in canonical if value in self.landmarks) != self.landmarks:
            raise ValueError(
                "temporal landmarks must be unique and follow canonical order: "
                + ", ".join(canonical)
            )
        if self.landmarks[:3] != ("nose", "ear_left", "ear_right"):
            raise ValueError("nose, ear_left, and ear_right are required for augmentation")
        return self


class TemporalArchitectureConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    hidden_channels: int = Field(gt=3)
    stem_kernel_sizes: tuple[int, ...]
    tcn_dilations: tuple[int, ...]
    tcn_kernel_size: int = Field(gt=1)
    dropout: float = Field(ge=0, lt=1)
    use_bigru: bool
    gru_hidden_size: int = Field(gt=0)
    lab_embedding_dim: int = Field(gt=0)
    metadata_hidden_dim: int = Field(gt=0)
    family: Literal["tcn", "conv_attention"] = "tcn"
    attention_heads: int = Field(default=2, gt=0)
    attention_layers: int = Field(default=1, ge=0)
    attention_ff_multiplier: int = Field(default=2, gt=0)
    use_lab_adapter: bool = False
    lab_adapter_dim: int = Field(default=4, gt=0)

    @field_validator("stem_kernel_sizes", "tcn_dilations")
    @classmethod
    def positive_unique_grid(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value or len(value) != len(set(value)) or any(item <= 0 for item in value):
            raise ValueError("architecture grids must be non-empty, positive, and unique")
        return value

    @model_validator(mode="after")
    def odd_kernels(self) -> TemporalArchitectureConfig:
        if any(kernel % 2 == 0 for kernel in self.stem_kernel_sizes):
            raise ValueError("stem kernels must be odd for length-preserving padding")
        if self.tcn_kernel_size % 2 == 0:
            raise ValueError("TCN kernel must be odd")
        if self.family == "conv_attention":
            if self.attention_layers == 0:
                raise ValueError("conv-attention models require at least one attention layer")
            if self.hidden_channels % self.attention_heads:
                raise ValueError("hidden channels must be divisible by attention heads")
        return self


class TemporalTrainingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    folds: tuple[int, ...]
    gate_fold: int
    epochs: int = Field(gt=0)
    steps_per_epoch: int = Field(gt=0)
    validation_windows_per_route: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    warmup_steps: int = Field(ge=0)
    early_stopping_patience: int = Field(gt=0)
    positive_class_weight: float = Field(gt=0)
    lab_loss_weight_power: float = Field(ge=0, le=1)
    gradient_clip_norm: float = Field(gt=0)
    ema_decay: float = Field(gt=0, lt=1)
    mixed_precision: bool
    minimum_gate_average_precision: float = Field(ge=0, le=1)
    seed: int = Field(ge=0)
    soft_f1_weight: float = Field(default=0.0, ge=0, le=0.5)
    focal_gamma: float = Field(default=0.0, ge=0, le=5)
    early_stopping_metric: Literal["validation_loss", "average_precision"] = (
        "validation_loss"
    )

    @model_validator(mode="after")
    def valid_folds(self) -> TemporalTrainingConfig:
        if not self.folds or len(self.folds) != len(set(self.folds)):
            raise ValueError("training folds must be non-empty and unique")
        if self.gate_fold not in self.folds:
            raise ValueError("gate fold must be one of the training folds")
        return self


class TemporalAugmentationConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    coordinate_noise_std: float = Field(ge=0)
    time_stretch_min: float = Field(gt=0)
    time_stretch_max: float = Field(gt=0)
    keypoint_dropout_probability: float = Field(ge=0, le=1)
    left_right_swap_probability: float = Field(ge=0, le=1)
    mouse_permutation_probability: float = Field(ge=0, le=1)
    geometric_transform_probability: float = Field(ge=0, le=1)
    maximum_rotation_degrees: float = Field(ge=0, le=360)
    reflection_probability: float = Field(ge=0, le=1)
    scale_probability: float = Field(default=0.0, ge=0, le=1)
    scale_min: float = Field(default=1.0, gt=0)
    scale_max: float = Field(default=1.0, gt=0)
    frame_shift_probability: float = Field(default=0.0, ge=0, le=1)
    maximum_frame_shift: int = Field(default=0, ge=0)
    mixup_probability: float = Field(default=0.0, ge=0, le=1)
    mixup_alpha: float = Field(default=0.4, gt=0)
    cutmix_probability: float = Field(default=0.0, ge=0, le=1)
    cutmix_min_fraction: float = Field(default=0.1, gt=0, lt=1)
    cutmix_max_fraction: float = Field(default=0.5, gt=0, lt=1)

    @model_validator(mode="after")
    def valid_stretch(self) -> TemporalAugmentationConfig:
        if self.time_stretch_min > self.time_stretch_max:
            raise ValueError("minimum time stretch cannot exceed maximum")
        if self.scale_min > self.scale_max:
            raise ValueError("minimum spatial scale cannot exceed maximum")
        if self.cutmix_min_fraction > self.cutmix_max_fraction:
            raise ValueError("minimum CutMix fraction cannot exceed maximum")
        return self


class TemporalExecutionConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    device: str
    cache_workers: int = Field(ge=1)
    num_workers: int = Field(ge=0)
    inference_batch_size: int = Field(gt=0)
    oof_part_rows: int = Field(gt=0)
    reflection_tta: bool = False


class TemporalDecoderConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    threshold_grid: tuple[float, ...]
    fallback_threshold: float = Field(gt=0, lt=1)
    smoothing_seconds: float = Field(ge=0)

    @field_validator("threshold_grid")
    @classmethod
    def valid_thresholds(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if not value or tuple(sorted(set(value))) != value:
            raise ValueError("threshold grid must be non-empty, sorted, and unique")
        if any(item <= 0 or item >= 1 for item in value):
            raise ValueError("thresholds must be inside (0, 1)")
        return value


class Phase9Config(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(ge=1)
    temporal_version: str = Field(min_length=1)
    data: TemporalDataConfig
    architecture: TemporalArchitectureConfig
    training: TemporalTrainingConfig
    augmentation: TemporalAugmentationConfig
    execution: TemporalExecutionConfig
    decoder: TemporalDecoderConfig


def load_phase9_config(path: str | Path = Path("configs/temporal.yaml")) -> Phase9Config:
    source = Path(path)
    loaded: object = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Phase 9 configuration must be a mapping: {source}")
    return Phase9Config.model_validate(cast(dict[str, Any], loaded))


def phase9_config_fingerprint(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
