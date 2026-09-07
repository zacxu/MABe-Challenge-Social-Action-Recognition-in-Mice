from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from mabe.cli import app
from mabe.config import load_config

ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs" / "base.yaml"


def test_base_config_is_frozen_and_locates_train_csv() -> None:
    config = load_config(BASE_CONFIG, environ={})

    assert config.paths.dataset_root == (ROOT / "MABe-mouse-behavior-detection").resolve()
    assert config.require_train_csv() == (ROOT / "MABe-mouse-behavior-detection" / "train.csv").resolve()
    assert config.seed == 2025

    with pytest.raises(ValidationError):
        config.seed = 1  # type: ignore[misc]
    with pytest.raises(ValidationError):
        config.paths.dataset_root = ROOT  # type: ignore[misc]


def test_environment_overrides_are_validated_and_resolved(tmp_path: Path) -> None:
    dataset_root = tmp_path / "input"
    dataset_root.mkdir()
    (dataset_root / "train.csv").write_text("video_id\n", encoding="utf-8")
    environment = {
        "MABE_DATASET_ROOT": str(dataset_root),
        "MABE_ARTIFACT_ROOT": str(tmp_path / "working" / "artifacts"),
        "MABE_CACHE_ROOT": str(tmp_path / "working" / "cache"),
        "MABE_MODEL_ROOT": str(tmp_path / "working" / "models"),
        "MABE_RUN_ID": "kaggle-test",
        "MABE_PHASE": "inference",
        "MABE_SEED": "17",
        "MABE_LOG_LEVEL": "warning",
    }

    config = load_config(BASE_CONFIG, environ=environment)

    assert config.paths.dataset_root == dataset_root.resolve()
    assert config.paths.artifact_root == (tmp_path / "working" / "artifacts").resolve()
    assert config.run_id == "kaggle-test"
    assert config.phase == "inference"
    assert config.seed == 17
    assert config.logging.level == "WARNING"
    assert config.require_train_csv().is_file()
    assert not config.paths.artifact_root.exists(), "loading configuration must not create directories"


def test_missing_required_path_fails_loudly(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(
        "schema_version: 1\nrun_id: x\nphase: test\nseed: 1\nlogging:\n  level: INFO\npaths: {}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="paths.dataset_root"):
        load_config(invalid, environ={})


def test_smoke_cli_prints_required_foundation_fields_without_writing_data() -> None:
    train_csv = ROOT / "MABe-mouse-behavior-detection" / "train.csv"
    before = train_csv.stat()

    result = CliRunner().invoke(
        app,
        ["smoke", "--config", str(BASE_CONFIG), "--skip-import-check"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["dataset_root"] == str((ROOT / "MABe-mouse-behavior-detection").resolve())
    assert payload["artifact_root"] == str((ROOT / "artifacts").resolve())
    assert payload["seed"] == 2025
    assert "numpy" in payload["dependencies"]["dependencies"]
    assert train_csv.stat() == before
