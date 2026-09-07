"""Strict experiment configuration for bounded Phase 10 temporal screens."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Phase10ScreenConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    held_out_fold: int = Field(ge=0)
    epochs: int = Field(gt=0)
    steps_per_epoch: int = Field(gt=0)
    validation_windows_per_route: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    warmup_steps: int = Field(ge=0)
    early_stopping_patience: int = Field(gt=0)
    minimum_average_precision_gain: float = Field(ge=0)
    minimum_exact_fbeta_gain_over_phase9: float = Field(ge=0)
    failures_before_family_stop: int = Field(gt=0)
    maximum_full_oof_candidates: int = Field(ge=0)


class Phase10ExperimentConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    family: str = Field(min_length=1)
    data_overrides: dict[str, Any] = Field(default_factory=dict)
    architecture_overrides: dict[str, Any] = Field(default_factory=dict)
    training_overrides: dict[str, Any] = Field(default_factory=dict)
    augmentation_overrides: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def safe_name(self) -> Phase10ExperimentConfig:
        if re.fullmatch(r"[a-z0-9_]+", self.name) is None:
            raise ValueError("experiment names may contain lowercase letters, digits, and underscores")
        return self


class Phase10Config(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(ge=1)
    phase10_version: str = Field(min_length=1)
    base_temporal_config: str = Field(min_length=1)
    screen: Phase10ScreenConfig
    experiments: tuple[Phase10ExperimentConfig, ...]

    @model_validator(mode="after")
    def valid_experiments(self) -> Phase10Config:
        names = [value.name for value in self.experiments]
        if not names or len(names) != len(set(names)):
            raise ValueError("Phase 10 experiment names must be non-empty and unique")
        controls = [value for value in self.experiments if value.family == "control"]
        if len(controls) != 1 or self.experiments[0].family != "control":
            raise ValueError("exactly one control experiment must appear first")
        return self


def load_phase10_config(
    path: str | Path = Path("configs/advanced_temporal.yaml"),
) -> Phase10Config:
    source = Path(path)
    loaded: object = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Phase 10 configuration must be a mapping: {source}")
    return Phase10Config.model_validate(cast(dict[str, Any], loaded))


def phase10_config_fingerprint(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
