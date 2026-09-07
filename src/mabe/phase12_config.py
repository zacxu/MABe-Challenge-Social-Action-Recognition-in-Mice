"""Strict configuration for offline Phase 12 inference."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class InferenceAssetConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    xgboost_registry: str
    blend_weights: str
    decoder_parameters: str
    probability_decoder_selection: str
    deployment_weights: str
    deployment_decoder_parameters: str
    category_registry: str
    vocabulary: str
    temporal_lab_registry: str
    temporal_checkpoint_root: str
    pose_config: str
    feature_config: str
    xgboost_config: str
    temporal_config: str
    stacker_registry: str | None = None
    final_model_registry: str | None = None


class InferenceEnsembleConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    folds: tuple[int, ...]
    test_fold_aggregation: Literal["mean"]
    crossfit_parameter_aggregation: Literal["median"]
    weight_strategy: Literal["action"]
    probability_model: Literal["auto", "blend", "stacker"] = "blend"

    @model_validator(mode="after")
    def unique_folds(self) -> InferenceEnsembleConfig:
        if not self.folds or len(set(self.folds)) != len(self.folds):
            raise ValueError("inference folds must be non-empty and unique")
        return self


class InferenceRuntimeConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    hidden_video_count: int = Field(gt=0)
    kaggle_limit_seconds: int = Field(gt=0)
    required_headroom_fraction: float = Field(ge=0, lt=1)
    resume: bool
    retain_probabilities: bool


class InferenceSpecialCasesConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    adaptable_snail_25fps_policy: Literal["none", "scale_30_over_25", "drop"] = (
        "none"
    )


class Phase12Config(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(ge=1)
    inference_version: str = Field(min_length=1)
    assets: InferenceAssetConfig
    ensemble: InferenceEnsembleConfig
    runtime: InferenceRuntimeConfig
    special_cases: InferenceSpecialCasesConfig = Field(
        default_factory=InferenceSpecialCasesConfig
    )
    config_path: Path = Field(exclude=True)

    @property
    def project_root(self) -> Path:
        return self.config_path.resolve().parent.parent

    def asset_path(self, value: str) -> Path:
        path = Path(value)
        return path.resolve() if path.is_absolute() else (self.project_root / path).resolve()


def load_phase12_config(path: str | Path = Path("configs/inference.yaml")) -> Phase12Config:
    source = Path(path).resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Phase 12 configuration must be a mapping: {source}")
    payload["config_path"] = source
    return Phase12Config.model_validate(cast(dict[str, object], payload))


def phase12_config_fingerprint(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
