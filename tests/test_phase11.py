from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from mabe.phase11_config import load_phase11_config
from mabe.phase11_pipeline import LoadedStream, audit_oof_alignment, learn_cross_fitted_weights
from mabe.probability_stacker import build_probability_stacker
from mabe.schemas import OOF_COLUMNS


def _oof(probability: list[float], *, target: list[int] | None = None) -> pl.DataFrame:
    rows = len(probability)
    return pl.DataFrame(
        {
            "lab_id": ["lab"] * rows,
            "video_id": [1] * rows,
            "video_frame": list(range(rows)),
            "agent_id": ["1"] * rows,
            "target_id": ["self"] * rows,
            "action": ["rear"] * rows,
            "fold": [0] * rows,
            "loss_mask": [1] * rows,
            "target": target or [0] * rows,
            "probability": probability,
            "model_id": ["model"] * rows,
            "feature_version": ["feature"] * rows,
        }
    ).select(OOF_COLUMNS)


def _stream(tmp_path: Path, name: str, frame: pl.DataFrame) -> LoadedStream:
    path = tmp_path / name / "fold_0" / "lab" / "1" / "part-00000.parquet"
    path.parent.mkdir(parents=True)
    frame.write_parquet(path)
    manifest_path = tmp_path / name / "manifest.json"
    payload = {"parts": [path.as_posix()], "row_count": frame.height}
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return LoadedStream(name, manifest_path, payload, {"fold_0/lab/1/part-00000.parquet": path}, True, True)


def test_alignment_accepts_exact_keys_and_different_probabilities(tmp_path: Path) -> None:
    left = _stream(tmp_path, "left", _oof([0.1, 0.2]))
    right = _stream(tmp_path, "right", _oof([0.8, 0.9]))
    audit = audit_oof_alignment((left, right), tmp_path / "audit.parquet")
    assert audit.height == 1
    assert audit["row_count"].sum() == 2


def test_alignment_rejects_target_mismatch(tmp_path: Path) -> None:
    left = _stream(tmp_path, "left", _oof([0.1, 0.2], target=[0, 1]))
    right = _stream(tmp_path, "right", _oof([0.8, 0.9], target=[1, 1]))
    with pytest.raises(ValueError, match="key/mask/target mismatch"):
        audit_oof_alignment((left, right), tmp_path / "audit.parquet")


def test_cross_fitted_weights_are_nonnegative_and_ignore_held_fold() -> None:
    config = load_phase11_config()
    rows = []
    for fold in config.optimization.folds:
        for weight in config.optimization.temporal_weight_grid:
            for threshold in config.optimization.threshold_grid:
                # All non-held calibration folds consistently favor the GBDT fallback.
                predicted = 10 if weight == 0 else 100
                rows.append(
                    {
                        "lab_id": "lab",
                        "action": "rear",
                        "fold": fold,
                        "temporal_weight": weight,
                        "threshold": threshold,
                        "positive_frames": 10,
                        "predicted_frames": predicted,
                        "true_positive_frames": 10,
                    }
                )
    registry = learn_cross_fitted_weights(pl.DataFrame(rows), config)
    assert not registry["used_held_out_labels"].any()
    assert (registry["gbdt_weight"] >= 0).all()
    assert (registry["temporal_phase9_weight"] >= 0).all()
    assert ((registry["gbdt_weight"] + registry["temporal_phase9_weight"] - 1).abs() < 1e-12).all()
    assert (registry.filter(pl.col("strategy") == "global")["temporal_phase9_weight"] == 0).all()


def test_cross_fitted_weights_emit_fallback_for_held_fold_only_component() -> None:
    config = load_phase11_config()
    rows = []
    for fold in config.optimization.folds:
        actions = ["common", "held_only"] if fold == 0 else ["common"]
        for action in actions:
            for weight in config.optimization.temporal_weight_grid:
                for threshold in config.optimization.threshold_grid:
                    rows.append(
                        {
                            "lab_id": "lab",
                            "action": action,
                            "fold": fold,
                            "temporal_weight": weight,
                            "threshold": threshold,
                            "positive_frames": 10,
                            "predicted_frames": 10,
                            "true_positive_frames": 10,
                        }
                    )
    registry = learn_cross_fitted_weights(pl.DataFrame(rows), config)
    route = registry.filter(
        (pl.col("strategy") == "lab_action")
        & (pl.col("held_out_fold") == 0)
        & (pl.col("lab_id") == "lab")
        & (pl.col("action") == "held_only")
    )
    assert route.height == 1
    assert route["calibration_positive_frames"][0] == 0
    assert route["shrinkage_alpha"][0] == 0


def test_phase11_configuration_declares_equal_reference_and_two_eligible_streams() -> None:
    config = load_phase11_config()
    assert len(config.base_streams) == 3
    assert sum(stream.eligible_for_optimized_blend for stream in config.base_streams) == 2
    assert 0.0 in config.optimization.temporal_weight_grid


def test_probability_stacker_builds_crossfit_and_deployment_models(
    tmp_path: Path,
) -> None:
    gbdt_parts: dict[str, Path] = {}
    temporal_parts: dict[str, Path] = {}
    for fold in (0, 1):
        target = [index % 2 for index in range(20)]
        gbdt = _oof(
            [0.8 if value else 0.2 for value in target], target=target
        ).with_columns(
            pl.lit(fold, dtype=pl.Int8).alias("fold"),
            pl.lit(fold + 1, dtype=pl.Int64).alias("video_id"),
        )
        temporal = gbdt.with_columns(
            pl.Series(
                "probability",
                [0.7 if value else 0.3 for value in target],
                dtype=pl.Float32,
            )
        )
        suffix = f"fold_{fold}/lab/{fold + 1}/part-00000.parquet"
        gbdt_path = tmp_path / "gbdt" / suffix
        temporal_path = tmp_path / "temporal" / suffix
        gbdt_path.parent.mkdir(parents=True)
        temporal_path.parent.mkdir(parents=True)
        gbdt.write_parquet(gbdt_path)
        temporal.write_parquet(temporal_path)
        gbdt_parts[suffix] = gbdt_path
        temporal_parts[suffix] = temporal_path
    stacker = load_phase11_config(
        "configs/ensemble-performance.yaml"
    ).stacker.model_copy(
        update={
            "device": "cpu",
            "n_estimators": 5,
            "negative_hash_keep_per_thousand": 1000,
            "maximum_positive_rows_per_action": 4,
            "maximum_negative_rows_per_action": 6,
        }
    )
    artifacts = build_probability_stacker(
        gbdt_parts=gbdt_parts,
        temporal_parts=temporal_parts,
        folds=(0, 1),
        output_root=tmp_path / "stacker",
        oof_root=tmp_path / "oof",
        configuration_sha256="abc",
        source_names=("gbdt", "temporal"),
        config=stacker,
    )

    registry = pl.read_parquet(artifacts.registry_path)
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    assert set(registry["held_out_fold"]) == {-1, 0, 1}
    assert not registry["used_held_out_labels"].any()
    assert artifacts.model_count == 3
    assert manifest["row_count"] == 40
    sample = pl.read_parquet(tmp_path / "stacker" / "training_sample.parquet")
    counts = sample.group_by("fold", "target").len().sort("fold", "target")
    assert counts["len"].to_list() == [3, 2, 3, 2]
