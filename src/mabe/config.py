"""Immutable, portable project configuration."""

from __future__ import annotations

import copy
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

DEFAULT_CONFIG_PATH = Path("configs/base.yaml")
PATH_ENVIRONMENT_OVERRIDES = {
    "dataset_root": "MABE_DATASET_ROOT",
    "artifact_root": "MABE_ARTIFACT_ROOT",
    "cache_root": "MABE_CACHE_ROOT",
    "model_root": "MABE_MODEL_ROOT",
}


class PathConfig(BaseModel):
    """Resolved project paths."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_root: Path
    artifact_root: Path
    cache_root: Path
    model_root: Path

    @property
    def train_csv(self) -> Path:
        return self.dataset_root / "train.csv"

    @property
    def test_csv(self) -> Path:
        return self.dataset_root / "test.csv"


class LoggingConfig(BaseModel):
    """Logging settings shared by commands and scripts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    level: str = "INFO"

    @field_validator("level")
    @classmethod
    def normalize_level(cls, value: str) -> str:
        normalized = value.upper()
        if normalized not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"unsupported logging level: {value}")
        return normalized


class RunConfig(BaseModel):
    """The single immutable configuration object for a pipeline run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(ge=1)
    run_id: str = Field(min_length=1)
    phase: str = Field(min_length=1)
    seed: int = Field(ge=0, le=2**32 - 1)
    logging: LoggingConfig
    paths: PathConfig
    config_file: Path = Field(exclude=True)

    def require_train_csv(self) -> Path:
        """Return train.csv or fail with a path-specific error."""
        if not self.paths.dataset_root.is_dir():
            raise FileNotFoundError(f"dataset root does not exist: {self.paths.dataset_root}")
        if not self.paths.train_csv.is_file():
            raise FileNotFoundError(f"train.csv does not exist: {self.paths.train_csv}")
        return self.paths.train_csv


def _read_yaml(config_file: Path) -> dict[str, Any]:
    if not config_file.is_file():
        raise FileNotFoundError(f"configuration file does not exist: {config_file}")
    loaded: object = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"configuration must be a YAML mapping: {config_file}")
    return cast(dict[str, Any], loaded)


def _resolve_path(value: object, *, base_directory: Path, key: str) -> str:
    if not isinstance(value, (str, Path)):
        raise TypeError(f"paths.{key} must be a path string")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = base_directory / candidate
    return str(candidate.resolve())


def load_config(
    config_path: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> RunConfig:
    """Load, override, resolve, and validate one frozen run configuration.

    ``MABE_CONFIG`` selects the YAML file. The remaining supported overrides are
    ``MABE_DATASET_ROOT``, ``MABE_ARTIFACT_ROOT``, ``MABE_CACHE_ROOT``,
    ``MABE_MODEL_ROOT``, ``MABE_RUN_ID``, ``MABE_PHASE``, ``MABE_SEED``, and
    ``MABE_LOG_LEVEL``.
    """
    environment = os.environ if environ is None else environ
    selected_path = config_path or environment.get("MABE_CONFIG") or DEFAULT_CONFIG_PATH
    config_file = Path(selected_path).expanduser().resolve()
    payload = copy.deepcopy(_read_yaml(config_file))

    paths = payload.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("configuration requires a 'paths' mapping")
    for key, environment_name in PATH_ENVIRONMENT_OVERRIDES.items():
        if environment_name in environment:
            paths[key] = environment[environment_name]
        if key not in paths:
            raise ValueError(f"configuration requires paths.{key}")
        paths[key] = _resolve_path(paths[key], base_directory=config_file.parent, key=key)

    scalar_overrides = {
        "MABE_RUN_ID": (payload, "run_id"),
        "MABE_PHASE": (payload, "phase"),
        "MABE_SEED": (payload, "seed"),
    }
    for environment_name, (destination, key) in scalar_overrides.items():
        if environment_name in environment:
            destination[key] = environment[environment_name]

    logging_payload = payload.get("logging")
    if not isinstance(logging_payload, dict):
        raise ValueError("configuration requires a 'logging' mapping")
    if "MABE_LOG_LEVEL" in environment:
        logging_payload["level"] = environment["MABE_LOG_LEVEL"]

    payload["config_file"] = config_file
    return RunConfig.model_validate(payload)
