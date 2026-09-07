"""Leakage-safe OOF alignment, blending, selection, and freezing for Phase 11."""

from __future__ import annotations

import json
import math
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import polars as pl

from mabe.config import RunConfig
from mabe.diagnostics import metric_breakdown
from mabe.fingerprints import sha256_file
from mabe.labels import build_validation_solution
from mabe.phase8_config import load_phase8_config
from mabe.phase8_decoder import nested_interval_postprocessing
from mabe.phase8_probability_decoder import screen_probability_decoder
from mabe.phase9_pipeline import _decode_oof, _nested_thresholds, _threshold_counts
from mabe.phase11_config import Phase11Config, load_phase11_config, phase11_config_fingerprint
from mabe.probability_stacker import build_probability_stacker
from mabe.schemas import OOF_COLUMNS
from mabe.submission import validate_submission

KEY_COLUMNS = (
    "lab_id",
    "video_id",
    "video_frame",
    "agent_id",
    "target_id",
    "action",
    "fold",
)
ALIGNMENT_COLUMNS = (*KEY_COLUMNS, "loss_mask", "target")


@dataclass(frozen=True, slots=True)
class LoadedStream:
    name: str
    manifest_path: Path
    manifest: dict[str, Any]
    parts: dict[str, Path]
    deployable: bool
    eligible_for_optimized_blend: bool


@dataclass(frozen=True, slots=True)
class CandidateResult:
    name: str
    manifest_path: Path
    submission_path: Path
    decoder_parameters_path: Path
    by_lab_path: Path
    exact_fbeta: float
    projected_seconds_200_videos: float


@dataclass(frozen=True, slots=True)
class Phase11Outputs:
    report_path: Path
    alignment_audit_path: Path
    blend_weight_registry_path: Path
    final_oof_manifest_path: Path
    final_submission_path: Path
    final_registry_path: Path
    overall_fbeta: float


def _project_path(config_path: Path, raw: str) -> Path:
    value = Path(raw)
    return value.resolve() if value.is_absolute() else (config_path.resolve().parent.parent / value).resolve()


def _suffix(path: Path) -> str:
    return "/".join(path.parts[-4:])


def _load_streams(config: Phase11Config, config_path: Path) -> tuple[LoadedStream, ...]:
    streams: list[LoadedStream] = []
    for item in config.base_streams:
        manifest_path = _project_path(config_path, item.manifest)
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing Phase 11 OOF manifest: {manifest_path}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw_parts = payload.get("parts")
        if not isinstance(raw_parts, list) or not raw_parts:
            raise ValueError(f"OOF manifest has no parts: {manifest_path}")
        parts = {_suffix(Path(str(path))): Path(str(path)) for path in raw_parts}
        if len(parts) != len(raw_parts):
            raise ValueError(f"OOF manifest contains duplicate part routes: {manifest_path}")
        missing = [path for path in parts.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"OOF manifest references {len(missing)} missing parts; first={missing[0]}")
        streams.append(
            LoadedStream(
                name=item.name,
                manifest_path=manifest_path,
                manifest=payload,
                parts=parts,
                deployable=item.deployable,
                eligible_for_optimized_blend=item.eligible_for_optimized_blend,
            )
        )
    return tuple(streams)


def audit_oof_alignment(streams: tuple[LoadedStream, ...], output_path: Path) -> pl.DataFrame:
    """Prove exact row alignment and fail before any probability arithmetic."""
    reference = streams[0]
    reference_suffixes = set(reference.parts)
    for stream in streams[1:]:
        missing = sorted(reference_suffixes - set(stream.parts))
        unexpected = sorted(set(stream.parts) - reference_suffixes)
        if missing or unexpected:
            raise ValueError(
                f"OOF route mismatch for {stream.name}: missing={missing[:3]}, unexpected={unexpected[:3]}"
            )
    records: list[dict[str, object]] = []
    previous_by_video: dict[tuple[str, str, str], tuple[int, str]] = {}
    for index, suffix in enumerate(sorted(reference_suffixes), start=1):
        reference_path = reference.parts[suffix]
        expected = pl.read_parquet(reference_path, columns=list(ALIGNMENT_COLUMNS))
        if expected.is_empty():
            raise ValueError(f"empty OOF part: {reference_path}")
        if expected.select(KEY_COLUMNS).is_duplicated().any():
            raise ValueError(f"duplicate OOF keys within part: {reference_path}")
        fold, lab, video, _ = suffix.split("/")
        first_frame = cast(int, expected["video_frame"].min())
        last_frame = cast(int, expected["video_frame"].max())
        video_key = (fold, lab, video)
        previous = previous_by_video.get(video_key)
        if previous is not None and first_frame <= previous[0]:
            raise ValueError(f"overlapping part frame ranges for {video_key}: {previous[1]} then {suffix}")
        previous_by_video[video_key] = (last_frame, suffix)
        row: dict[str, object] = {
            "part_suffix": suffix,
            "row_count": expected.height,
            "first_frame": first_frame,
            "last_frame": last_frame,
            "within_part_duplicate_keys": 0,
            f"{reference.name}_aligned": True,
        }
        expected_probability = pl.read_parquet(reference_path, columns=["probability"])["probability"]
        expected_values = expected_probability.to_numpy()
        if not np.isfinite(expected_values).all() or ((expected_values < 0) | (expected_values > 1)).any():
            raise ValueError(f"invalid probability in {reference_path}")
        for stream in streams[1:]:
            other_path = stream.parts[suffix]
            other = pl.read_parquet(other_path, columns=list(ALIGNMENT_COLUMNS))
            aligned = expected.equals(other)
            row[f"{stream.name}_aligned"] = aligned
            if not aligned:
                raise ValueError(f"OOF key/mask/target mismatch: {reference.name} vs {stream.name}: {suffix}")
            values = pl.read_parquet(other_path, columns=["probability"])["probability"].to_numpy()
            if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                raise ValueError(f"invalid probability in {other_path}")
        records.append(row)
        if index == 1 or index % 100 == 0 or index == len(reference_suffixes):
            print(f"Phase 11 alignment: {index}/{len(reference_suffixes)} parts", flush=True)
    audit = pl.DataFrame(records).sort("part_suffix")
    expected_rows = int(reference.manifest.get("row_count", audit["row_count"].sum()))
    if int(audit["row_count"].sum()) != expected_rows:
        raise ValueError("aligned OOF row count differs from the reference manifest")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit.write_parquet(output_path, compression="zstd", statistics=True)
    return audit


def _weight_statistics(
    reference: LoadedStream,
    temporal: LoadedStream,
    *,
    weights: tuple[float, ...],
    thresholds: tuple[float, ...],
    output_path: Path,
) -> pl.DataFrame:
    cached_meta = Path(f"{output_path}.meta.json")
    if output_path.is_file() and cached_meta.is_file():
        return pl.read_parquet(output_path)
    records: list[dict[str, object]] = []
    suffixes = sorted(reference.parts)
    for index, suffix in enumerate(suffixes, start=1):
        frame = pl.read_parquet(
            reference.parts[suffix], columns=["lab_id", "action", "fold", "loss_mask", "target", "probability"]
        )
        neural = pl.read_parquet(temporal.parts[suffix], columns=["probability"])
        base_values = frame["probability"].to_numpy().astype(np.float32, copy=False)
        neural_values = neural["probability"].to_numpy().astype(np.float32, copy=False)
        targets = frame["target"].to_numpy().astype(bool, copy=False)
        valid = frame["loss_mask"].to_numpy().astype(bool, copy=False)
        actions = frame["action"].to_numpy()
        lab = str(frame["lab_id"][0])
        fold = int(frame["fold"][0])
        for action in np.unique(actions):
            selected = valid & (actions == action)
            y = targets[selected]
            positive = int(y.sum())
            base_action = base_values[selected]
            delta_action = neural_values[selected] - base_action
            for weight in weights:
                probability = base_action + np.float32(weight) * delta_action
                for threshold in thresholds:
                    predicted = probability >= threshold
                    records.append(
                        {
                            "lab_id": lab,
                            "action": str(action),
                            "fold": fold,
                            "temporal_weight": weight,
                            "threshold": threshold,
                            "positive_frames": positive,
                            "predicted_frames": int(predicted.sum()),
                            "true_positive_frames": int(np.logical_and(predicted, y).sum()),
                        }
                    )
        if index == 1 or index % 100 == 0 or index == len(suffixes):
            print(f"Phase 11 weight statistics: {index}/{len(suffixes)} parts", flush=True)
    statistics = (
        pl.DataFrame(records)
        .group_by("lab_id", "action", "fold", "temporal_weight", "threshold")
        .agg(
            pl.col("positive_frames").sum(),
            pl.col("predicted_frames").sum(),
            pl.col("true_positive_frames").sum(),
        )
        .sort("lab_id", "action", "fold", "temporal_weight", "threshold")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    statistics.write_parquet(output_path, compression="zstd", statistics=True)
    cached_meta.write_text(
        json.dumps(
            {
                "row_count": statistics.height,
                "reference_manifest_sha256": sha256_file(reference.manifest_path),
                "temporal_manifest_sha256": sha256_file(temporal.manifest_path),
                "weights": list(weights),
                "thresholds": list(thresholds),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return statistics


def _scored_components(statistics: pl.DataFrame, held_fold: int) -> pl.DataFrame:
    return (
        statistics.filter(pl.col("fold") != held_fold)
        .group_by("lab_id", "action", "temporal_weight", "threshold")
        .agg(
            pl.col("positive_frames").sum(),
            pl.col("predicted_frames").sum(),
            pl.col("true_positive_frames").sum(),
        )
        .with_columns(
            pl.when(pl.col("positive_frames") + pl.col("predicted_frames") > 0)
            .then(2 * pl.col("true_positive_frames") / (pl.col("positive_frames") + pl.col("predicted_frames")))
            .otherwise(0.0)
            .alias("frame_f1")
        )
        .sort(
            ["lab_id", "action", "temporal_weight", "frame_f1", "threshold"],
            descending=[False, False, False, True, False],
        )
        .unique(subset=["lab_id", "action", "temporal_weight"], keep="first", maintain_order=True)
    )


def _best_weight(frame: pl.DataFrame) -> tuple[float, float]:
    if frame.is_empty():
        return 0.0, 0.0
    winner = frame.sort(["score", "temporal_weight"], descending=[True, False]).row(0, named=True)
    return float(winner["temporal_weight"]), float(winner["score"])


def _scope_scores(components: pl.DataFrame, scope: str) -> pl.DataFrame:
    if scope == "global":
        return (
            components.group_by("temporal_weight", "lab_id")
            .agg(pl.col("frame_f1").mean().alias("lab_score"))
            .group_by("temporal_weight")
            .agg(pl.col("lab_score").mean().alias("score"))
        )
    group = [scope, "temporal_weight"]
    return components.group_by(group).agg(pl.col("frame_f1").mean().alias("score"))


def _shrink(raw: float, parent: float, support: int, shrinkage: int) -> tuple[float, float]:
    alpha = support / (support + shrinkage)
    return alpha * raw + (1.0 - alpha) * parent, alpha


def _positive_support(components: pl.DataFrame) -> int:
    """Count positives once; component rows repeat support for every tested weight."""
    first_weight = cast(float, components["temporal_weight"].min())
    return int(components.filter(pl.col("temporal_weight") == first_weight)["positive_frames"].sum())


def learn_cross_fitted_weights(statistics: pl.DataFrame, config: Phase11Config) -> pl.DataFrame:
    """Fit non-negative convex weights only on folds other than the reported fold."""
    settings = config.optimization
    records: list[dict[str, object]] = []
    requested_actions = sorted(str(value) for value in statistics["action"].unique())
    requested_labs = sorted(str(value) for value in statistics["lab_id"].unique())
    requested_components = sorted(
        (str(row["lab_id"]), str(row["action"]))
        for row in statistics.select("lab_id", "action").unique().iter_rows(named=True)
    )
    for held_fold in settings.folds:
        components = _scored_components(statistics, held_fold)
        global_weight, global_score = _best_weight(_scope_scores(components, "global"))
        records.append(
            {
                "strategy": "global",
                "held_out_fold": held_fold,
                "lab_id": None,
                "action": None,
                "gbdt_weight": 1.0 - global_weight,
                "temporal_phase9_weight": global_weight,
                "raw_temporal_weight": global_weight,
                "parent_temporal_weight": global_weight,
                "shrinkage_alpha": 1.0,
                "calibration_positive_frames": _positive_support(components),
                "surrogate_calibration_score": global_score,
                "calibration_folds": json.dumps([fold for fold in settings.folds if fold != held_fold]),
                "used_held_out_labels": False,
            }
        )
        action_weights: dict[str, float] = {}
        for action in requested_actions:
            selected = components.filter(pl.col("action") == action)
            support = _positive_support(selected) if selected.height else 0
            raw, score = (
                _best_weight(_scope_scores(selected, "action").drop("action"))
                if selected.height
                else (global_weight, global_score)
            )
            if support < settings.minimum_action_positive_frames:
                final, alpha = global_weight, 0.0
            else:
                final, alpha = _shrink(raw, global_weight, support, settings.shrinkage_positive_frames)
            action_weights[action] = final
            records.append(
                {
                    "strategy": "action",
                    "held_out_fold": held_fold,
                    "lab_id": None,
                    "action": action,
                    "gbdt_weight": 1.0 - final,
                    "temporal_phase9_weight": final,
                    "raw_temporal_weight": raw,
                    "parent_temporal_weight": global_weight,
                    "shrinkage_alpha": alpha,
                    "calibration_positive_frames": support,
                    "surrogate_calibration_score": score,
                    "calibration_folds": json.dumps([fold for fold in settings.folds if fold != held_fold]),
                    "used_held_out_labels": False,
                }
            )
        lab_weights: dict[str, float] = {}
        for lab in requested_labs:
            selected = components.filter(pl.col("lab_id") == lab)
            support = _positive_support(selected) if selected.height else 0
            raw, score = (
                _best_weight(_scope_scores(selected, "lab_id").drop("lab_id"))
                if selected.height
                else (global_weight, global_score)
            )
            if support < settings.minimum_lab_positive_frames:
                final, alpha = global_weight, 0.0
            else:
                final, alpha = _shrink(raw, global_weight, support, settings.shrinkage_positive_frames)
            lab_weights[lab] = final
            records.append(
                {
                    "strategy": "lab",
                    "held_out_fold": held_fold,
                    "lab_id": lab,
                    "action": None,
                    "gbdt_weight": 1.0 - final,
                    "temporal_phase9_weight": final,
                    "raw_temporal_weight": raw,
                    "parent_temporal_weight": global_weight,
                    "shrinkage_alpha": alpha,
                    "calibration_positive_frames": support,
                    "surrogate_calibration_score": score,
                    "calibration_folds": json.dumps([fold for fold in settings.folds if fold != held_fold]),
                    "used_held_out_labels": False,
                }
            )
        for lab, action in requested_components:
            selected = components.filter((pl.col("lab_id") == lab) & (pl.col("action") == action))
            parent = 0.5 * (lab_weights[lab] + action_weights[action])
            support = _positive_support(selected) if selected.height else 0
            raw, score = (
                _best_weight(selected.select("temporal_weight", pl.col("frame_f1").alias("score")))
                if selected.height
                else (parent, 0.0)
            )
            if support < settings.minimum_lab_action_positive_frames:
                final, alpha = parent, 0.0
            else:
                final, alpha = _shrink(raw, parent, support, settings.shrinkage_positive_frames)
            records.append(
                {
                    "strategy": "lab_action",
                    "held_out_fold": held_fold,
                    "lab_id": lab,
                    "action": action,
                    "gbdt_weight": 1.0 - final,
                    "temporal_phase9_weight": final,
                    "raw_temporal_weight": raw,
                    "parent_temporal_weight": parent,
                    "shrinkage_alpha": alpha,
                    "calibration_positive_frames": support,
                    "surrogate_calibration_score": score,
                    "calibration_folds": json.dumps([fold for fold in settings.folds if fold != held_fold]),
                    "used_held_out_labels": False,
                }
            )
    registry = pl.DataFrame(records).sort("strategy", "held_out_fold", "lab_id", "action")
    sums = registry["gbdt_weight"] + registry["temporal_phase9_weight"]
    if registry.filter(
        (pl.col("gbdt_weight") < 0)
        | (pl.col("temporal_phase9_weight") < 0)
        | ((sums - 1.0).abs() > 1e-12)
        | pl.col("used_held_out_labels")
    ).height:
        raise AssertionError("invalid or leaky Phase 11 blend weight registry")
    return registry


def _weights_for_part(
    frame: pl.DataFrame,
    registry: pl.DataFrame,
    strategy: str,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    fold = int(frame["fold"][0])
    lab = str(frame["lab_id"][0])
    selected = registry.filter((pl.col("strategy") == strategy) & (pl.col("held_out_fold") == fold))
    if strategy == "global":
        return np.full(frame.height, float(selected["temporal_phase9_weight"][0]), dtype=np.float32)
    if strategy == "lab":
        row = selected.filter(pl.col("lab_id") == lab)
        if row.height != 1:
            raise ValueError(f"missing unique lab blend weight for {(fold, lab)}")
        return np.full(frame.height, float(row["temporal_phase9_weight"][0]), dtype=np.float32)
    actions = frame["action"].to_numpy()
    values = np.empty(frame.height, dtype=np.float32)
    for action in np.unique(actions):
        condition = pl.col("action") == str(action)
        if strategy == "lab_action":
            condition &= pl.col("lab_id") == lab
        row = selected.filter(condition)
        if row.height != 1:
            raise ValueError(f"missing unique {strategy} blend weight for {(fold, lab, action)}")
        values[actions == action] = float(row["temporal_phase9_weight"][0])
    return values


def _write_oof_manifest(
    *,
    candidate: str,
    paths: list[Path],
    row_count: int,
    output_root: Path,
    configuration_sha256: str,
    lineage: dict[str, object],
) -> Path:
    manifest = {
        "schema_version": 1,
        "model_version": candidate,
        "configuration_sha256": configuration_sha256,
        "row_count": row_count,
        "columns": list(OOF_COLUMNS),
        "parts": [path.as_posix() for path in paths],
        "held_out_labels_used_for_training_or_early_stopping": False,
        "probability_lineage": lineage,
    }
    path = output_root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def build_blend_oof(
    *,
    candidate: str,
    streams: tuple[LoadedStream, ...],
    output_root: Path,
    configuration_sha256: str,
    constants: dict[str, float] | None = None,
    registry: pl.DataFrame | None = None,
    strategy: str | None = None,
) -> Path:
    """Blend aligned probabilities while preserving the immutable OOF index."""
    manifest_path = output_root / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        cached_paths = [Path(str(path)) for path in existing.get("parts", [])]
        if (
            existing.get("configuration_sha256") == configuration_sha256
            and cached_paths
            and all(path.is_file() for path in cached_paths)
        ):
            return manifest_path
    if (constants is None) == (registry is None or strategy is None):
        raise ValueError("provide either constant weights or a registry strategy")
    lookup = {stream.name: stream for stream in streams}
    reference = streams[0]
    eligible = [stream for stream in streams if stream.eligible_for_optimized_blend]
    if registry is not None and len(eligible) != 2:
        raise ValueError("cross-fitted registry blending requires two eligible streams")
    paths: list[Path] = []
    row_count = 0
    suffixes = sorted(reference.parts)
    for index, suffix in enumerate(suffixes, start=1):
        base = pl.read_parquet(reference.parts[suffix])
        if constants is not None:
            if set(constants) - set(lookup):
                raise KeyError(f"unknown constant blend streams: {sorted(set(constants) - set(lookup))}")
            if any(value < 0 for value in constants.values()) or not math.isclose(
                sum(constants.values()), 1.0, abs_tol=1e-12
            ):
                raise ValueError("constant blend weights must be non-negative and sum to one")
            probability = np.zeros(base.height, dtype=np.float32)
            for name, weight in constants.items():
                values = (
                    pl.read_parquet(lookup[name].parts[suffix], columns=["probability"])["probability"]
                    .to_numpy()
                    .astype(np.float32, copy=False)
                )
                probability += np.float32(weight) * values
            lineage: dict[str, object] = {"type": "equal_reference", "weights": constants}
        else:
            if registry is None:
                raise AssertionError("registry is required for a learned blend")
            temporal_weights = _weights_for_part(base, registry, str(strategy))
            base_values = base["probability"].to_numpy().astype(np.float32, copy=False)
            temporal_values = (
                pl.read_parquet(eligible[1].parts[suffix], columns=["probability"])["probability"]
                .to_numpy()
                .astype(np.float32, copy=False)
            )
            probability = base_values + temporal_weights * (temporal_values - base_values)
            lineage = {
                "type": "cross_fitted_convex_blend",
                "strategy": strategy,
                "base_stream": eligible[0].name,
                "temporal_stream": eligible[1].name,
                "used_held_out_labels": False,
            }
        output_path = output_root / suffix
        output_path.parent.mkdir(parents=True, exist_ok=True)
        base.with_columns(
            pl.Series("probability", probability, dtype=pl.Float32),
            pl.lit(candidate).alias("model_id"),
            pl.lit(candidate).alias("feature_version"),
        ).select(OOF_COLUMNS).write_parquet(output_path, compression="zstd", statistics=True)
        paths.append(output_path)
        row_count += base.height
        if index == 1 or index % 100 == 0 or index == len(suffixes):
            print(f"Phase 11 {candidate} OOF: {index}/{len(suffixes)} parts", flush=True)
    return _write_oof_manifest(
        candidate=candidate,
        paths=paths,
        row_count=row_count,
        output_root=output_root,
        configuration_sha256=configuration_sha256,
        lineage=lineage,
    )


def _validation_context(run_config: RunConfig) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pd.DataFrame]:
    videos_path = run_config.paths.artifact_root / "manifests" / "videos.parquet"
    triplets_path = run_config.paths.artifact_root / "manifests" / "triplets.parquet"
    folds_path = run_config.paths.artifact_root / "folds" / "folds.parquet"
    videos = pl.read_parquet(videos_path)
    triplets = pl.read_parquet(triplets_path)
    folds = pl.read_parquet(folds_path)
    keys = [
        (str(row["lab_id"]), int(row["video_id"]))
        for row in folds.filter(pl.col("is_primary_validation")).iter_rows(named=True)
    ]
    solution = build_validation_solution(
        videos_path=videos_path,
        dataset_root=run_config.paths.dataset_root,
        validation_videos=keys,
    )
    return videos, triplets, folds, solution


def _projected_candidate_runtime(manifest: dict[str, Any], config: Phase11Config) -> float:
    lineage = manifest.get("probability_lineage", {})
    weights = lineage.get("weights", {}) if isinstance(lineage, dict) else {}
    lineage_type = lineage.get("type") if isinstance(lineage, dict) else None
    routed_sources = lineage.get("source_row_counts", {}) if isinstance(lineage, dict) else {}
    uses_temporal10 = isinstance(weights, dict) and float(weights.get("temporal_phase10_reference", 0.0)) > 0
    uses_temporal9 = (
        lineage_type == "cross_fitted_convex_blend"
        or lineage_type == "cross_fitted_probability_xgboost_stacker"
        or (isinstance(weights, dict) and float(weights.get("temporal_phase9", 0.0)) > 0)
        or (
            lineage_type == "cross_fitted_per_lab_probability_router"
            and isinstance(routed_sources, dict)
            and any(name != "gbdt" and int(rows) > 0 for name, rows in routed_sources.items())
        )
    )
    per_video = config.runtime.phase8_gbdt_seconds_per_video
    if uses_temporal9:
        per_video += config.runtime.temporal_phase9_seconds_per_video
    if uses_temporal10:
        per_video += config.runtime.temporal_phase10_seconds_per_video
    return per_video * config.runtime.hidden_video_count


def evaluate_candidate(
    *,
    name: str,
    manifest_path: Path,
    output_root: Path,
    run_config: RunConfig,
    config: Phase11Config,
    phase8_config_path: Path,
    source_fingerprint: str,
    baseline_root: Path,
    interval_global_only: bool = True,
) -> CandidateResult:
    """Run nested threshold, smoothing/hysteresis, and interval decoding for one blend."""
    report_path = output_root / "candidate_report.json"
    if report_path.is_file():
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return CandidateResult(
            name=name,
            manifest_path=manifest_path,
            submission_path=output_root / "held_out_submission.csv",
            decoder_parameters_path=output_root / "decoder_parameters.parquet",
            by_lab_path=output_root / "cv_by_lab.csv",
            exact_fbeta=float(payload["exact_fbeta"]),
            projected_seconds_200_videos=_projected_candidate_runtime(manifest, config),
        )
    started = time.perf_counter()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = [Path(str(path)) for path in manifest["parts"]]
    videos, triplets, folds, solution = _validation_context(run_config)
    primary = folds.filter(pl.col("is_primary_validation")).select("lab_id", "video_id", "fold")
    counts = _threshold_counts(paths, config.optimization.threshold_grid)
    requested = (
        pl.read_parquet(baseline_root / "final_model_registry.parquet")
        .select("lab_id", "action")
        .unique()
    )
    thresholds = _nested_thresholds(
        counts,
        requested,
        folds=config.optimization.folds,
        threshold_grid=config.optimization.threshold_grid,
        fallback=0.5,
    )
    thresholds.write_parquet(output_root / "initial_thresholds.parquet", compression="zstd", statistics=True)
    initial = _decode_oof(paths, primary, thresholds)
    decoder_config = load_phase8_config(phase8_config_path).decoder.model_copy(
        update={
            "additional_smoothing_seconds": config.decoder.additional_smoothing_seconds,
            "hysteresis_delta": config.decoder.hysteresis_delta,
            "threshold_grid": config.decoder.global_threshold_references,
            "gap_fill_seconds": config.decoder.gap_fill_seconds,
            "minimum_run_seconds": config.decoder.minimum_run_seconds,
        }
    )
    interval_config = (
        decoder_config.model_copy(
            update={
                "minimum_action_positive_frames": 2**60,
                "minimum_lab_action_positive_frames": 2**60,
                "minimum_lab_action_positive_videos": 2**30,
            }
        )
        if interval_global_only
        else decoder_config
    )
    probability_screen = screen_probability_decoder(
        solution=solution,
        oof_paths=tuple(path.as_posix() for path in paths),
        videos=videos,
        folds=folds,
        baseline_parameters=thresholds,
        config=decoder_config,
    )
    probability_screen.search_summary.write_parquet(
        output_root / "probability_decoder_search.parquet", compression="zstd", statistics=True
    )
    probability_screen.nested_selection.write_parquet(
        output_root / "probability_decoder_selection.parquet", compression="zstd", statistics=True
    )
    probability_screen.submission.to_csv(output_root / "probability_decoder_submission.csv", index=False)
    postprocessed = nested_interval_postprocessing(
        solution=solution,
        baseline_submission=probability_screen.submission,
        videos=videos,
        folds=folds,
        baseline_parameters=probability_screen.route_parameters,
        config=interval_config,
        baseline_smoothing_seconds=config.decoder.baseline_smoothing_seconds,
    )
    validated = validate_submission(
        postprocessed.submission,
        videos=videos,
        triplets=triplets,
        allowed_splits=("train",),
    )
    exact = metric_breakdown(solution, validated.to_pandas(), beta=1.0)
    submission_path = output_root / "held_out_submission.csv"
    validated.write_csv(submission_path)
    decoder_parameters_path = output_root / "decoder_parameters.parquet"
    postprocessed.parameters.write_parquet(decoder_parameters_path, compression="zstd", statistics=True)
    postprocessed.search_summary.write_parquet(
        output_root / "interval_decoder_search.parquet", compression="zstd", statistics=True
    )
    postprocessed.by_fold.to_csv(output_root / "cv_by_fold.csv", index=False)
    by_lab_path = output_root / "cv_by_lab.csv"
    exact.by_lab.to_csv(by_lab_path, index=False)
    exact.by_lab_action.to_csv(output_root / "cv_by_lab_action.csv", index=False)
    # Runtime cost is the sum of unique required base inference routes; blend arithmetic is negligible.
    projected = _projected_candidate_runtime(manifest, config)
    available = config.runtime.kaggle_limit_seconds * (1 - config.runtime.required_headroom_fraction)
    report = {
        "schema_version": 1,
        "candidate": name,
        "configuration_sha256": manifest["configuration_sha256"],
        "source_data_fingerprint": source_fingerprint,
        "oof_manifest": manifest_path.as_posix(),
        "oof_rows": int(manifest["row_count"]),
        "initial_submission_rows": len(initial),
        "final_submission_rows": validated.height,
        "nested_probability_decoder_fbeta": probability_screen.overall_fbeta,
        "exact_fbeta": exact.overall,
        "decoder_used_held_out_labels": bool(postprocessed.parameters["used_held_out_labels"].any()),
        "interval_selection_scope": "global" if interval_global_only else "global_action_lab_action",
        "blend_weights_used_held_out_labels": False,
        "projected_seconds_200_videos": projected,
        "runtime_budget_with_headroom_seconds": available,
        "runtime_feasible": projected <= available,
        "evaluation_seconds": time.perf_counter() - started,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return CandidateResult(
        name=name,
        manifest_path=manifest_path,
        submission_path=submission_path,
        decoder_parameters_path=decoder_parameters_path,
        by_lab_path=by_lab_path,
        exact_fbeta=exact.overall,
        projected_seconds_200_videos=projected,
    )


def _baseline_result(
    root: Path, *, name: str, projected_seconds: float
) -> CandidateResult:
    report = json.loads((root / "phase8_report.json").read_text(encoding="utf-8"))
    return CandidateResult(
        name=name,
        manifest_path=root / "final_oof_manifest.json",
        submission_path=root / "held_out_submission.csv",
        decoder_parameters_path=root / "decoder_parameters.parquet",
        by_lab_path=root / "cv_by_lab.csv",
        exact_fbeta=float(report["final_overall_fbeta"]),
        projected_seconds_200_videos=projected_seconds,
    )


def select_models_by_lab_cross_fitted(
    *,
    candidates: list[CandidateResult],
    run_config: RunConfig,
    folds: tuple[int, ...],
    output_path: Path,
) -> pl.DataFrame:
    """Select a source for each held fold/lab using only other-fold exact OOF."""
    if output_path.is_file():
        return pl.read_parquet(output_path)
    _, _, fold_frame, solution = _validation_context(run_config)
    membership = fold_frame.filter(pl.col("is_primary_validation")).select("lab_id", "video_id", "fold")
    submissions = {candidate.name: pd.read_csv(candidate.submission_path) for candidate in candidates}
    records: list[dict[str, object]] = []
    labs = sorted(str(value) for value in membership["lab_id"].unique())
    complexity = {candidate.name: index for index, candidate in enumerate(candidates)}
    for held_fold in folds:
        calibration = membership.filter(pl.col("fold") != held_fold)
        for lab in labs:
            video_ids = set(int(value) for value in calibration.filter(pl.col("lab_id") == lab)["video_id"].to_list())
            if not video_ids:
                raise ValueError(f"no calibration videos for per-lab selector: {(held_fold, lab)}")
            lab_solution = solution[solution["video_id"].isin(video_ids)].reset_index(drop=True)
            scored: list[tuple[float, int, str]] = []
            for candidate in candidates:
                submission = submissions[candidate.name]
                selected = submission[submission["video_id"].isin(video_ids)].reset_index(drop=True)
                score = 0.0 if selected.empty else metric_breakdown(lab_solution, selected).overall
                scored.append((score, -complexity[candidate.name], candidate.name))
            score, _, winner = max(scored)
            records.append(
                {
                    "held_out_fold": held_fold,
                    "lab_id": lab,
                    "selected_candidate": winner,
                    "calibration_exact_fbeta": score,
                    "calibration_folds": json.dumps([fold for fold in folds if fold != held_fold]),
                    "used_held_out_labels": False,
                    "candidate_scores_json": json.dumps(
                        {name: value for value, _, name in sorted(scored, key=lambda item: item[2])},
                        sort_keys=True,
                    ),
                }
            )
    registry = pl.DataFrame(records).sort("held_out_fold", "lab_id")
    if registry.filter(pl.col("used_held_out_labels")).height:
        raise AssertionError("held-out labels entered per-lab ensemble selection")
    registry.write_parquet(output_path, compression="zstd", statistics=True)
    return registry


def build_selected_oof(
    *,
    candidate: str,
    selector: pl.DataFrame,
    sources: list[CandidateResult],
    reference: LoadedStream,
    output_root: Path,
    configuration_sha256: str,
) -> Path:
    manifest_path = output_root / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        paths = [Path(str(path)) for path in existing.get("parts", [])]
        if existing.get("configuration_sha256") == configuration_sha256 and all(path.is_file() for path in paths):
            return manifest_path
    source_lookups: dict[str, dict[str, Path]] = {}
    for source in sources:
        if source.name == reference.name:
            source_lookups[source.name] = reference.parts
        else:
            payload = json.loads(source.manifest_path.read_text(encoding="utf-8"))
            source_lookups[source.name] = {_suffix(Path(str(path))): Path(str(path)) for path in payload["parts"]}
    output_paths: list[Path] = []
    row_count = 0
    choices: defaultdict[str, int] = defaultdict(int)
    suffixes = sorted(reference.parts)
    for index, suffix in enumerate(suffixes, start=1):
        fold_text, lab, _, _ = suffix.split("/")
        fold = int(fold_text.removeprefix("fold_"))
        row = selector.filter((pl.col("held_out_fold") == fold) & (pl.col("lab_id") == lab))
        if row.height != 1:
            raise ValueError(f"missing unique per-lab selector route for {(fold, lab)}")
        selected_name = str(row["selected_candidate"][0])
        source_path = source_lookups[selected_name][suffix]
        frame = (
            pl.read_parquet(source_path)
            .with_columns(
                pl.lit(candidate).alias("model_id"),
                pl.lit(candidate).alias("feature_version"),
            )
            .select(OOF_COLUMNS)
        )
        output_path = output_root / suffix
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(output_path, compression="zstd", statistics=True)
        output_paths.append(output_path)
        row_count += frame.height
        choices[selected_name] += frame.height
        if index == 1 or index % 100 == 0 or index == len(suffixes):
            print(f"Phase 11 {candidate} OOF: {index}/{len(suffixes)} parts", flush=True)
    return _write_oof_manifest(
        candidate=candidate,
        paths=output_paths,
        row_count=row_count,
        output_root=output_root,
        configuration_sha256=configuration_sha256,
        lineage={
            "type": "cross_fitted_per_lab_probability_router",
            "selection_registry": "per_lab_selector.parquet",
            "source_row_counts": dict(choices),
            "used_held_out_labels": False,
        },
    )


def _artifact_record(path: Path) -> dict[str, object]:
    return {"path": path.as_posix(), "sha256": sha256_file(path)}


def run_phase11_ensemble(
    *,
    run_config: RunConfig,
    phase11_config_path: str | Path,
    source_fingerprint: str,
) -> Phase11Outputs:
    started = time.perf_counter()
    config_path = Path(phase11_config_path)
    config = load_phase11_config(config_path)
    configuration_sha = phase11_config_fingerprint(config_path)
    output_root = run_config.paths.artifact_root / "phase11" / config.phase11_version
    output_root.mkdir(parents=True, exist_ok=True)
    streams = _load_streams(config, config_path)
    alignment_path = output_root / "oof_alignment_audit.parquet"
    if alignment_path.is_file():
        alignment = pl.read_parquet(alignment_path)
    else:
        alignment = audit_oof_alignment(streams, alignment_path)
    eligible = [stream for stream in streams if stream.eligible_for_optimized_blend]
    baseline_root = eligible[0].manifest_path.parent
    statistics_path = output_root / "blend_weight_statistics.parquet"
    statistics = _weight_statistics(
        eligible[0],
        eligible[1],
        weights=config.optimization.temporal_weight_grid,
        thresholds=config.optimization.threshold_grid,
        output_path=statistics_path,
    )
    weights = learn_cross_fitted_weights(statistics, config)
    weights_path = output_root / "blend_weight_registry.parquet"
    weights.write_parquet(weights_path, compression="zstd", statistics=True)
    oof_root = run_config.paths.artifact_root / "oof"
    definitions: list[tuple[str, dict[str, float] | None, str | None]] = [
        ("phase11-equal-gbdt-phase9", {eligible[0].name: 0.5, eligible[1].name: 0.5}, None),
        ("phase11-crossfit-global", None, "global"),
        ("phase11-crossfit-action", None, "action"),
        ("phase11-crossfit-lab", None, "lab"),
        ("phase11-crossfit-lab-action", None, "lab_action"),
    ]
    phase8_config_path = _project_path(config_path, config.decoder.phase8_config)
    results: list[CandidateResult] = []
    for candidate, constants, strategy in definitions:
        candidate_oof_root = oof_root / candidate / configuration_sha[:12]
        manifest_path = build_blend_oof(
            candidate=candidate,
            streams=streams,
            output_root=candidate_oof_root,
            configuration_sha256=configuration_sha,
            constants=constants,
            registry=weights if strategy is not None else None,
            strategy=strategy,
        )
        print(f"Phase 11 exact evaluation: {candidate}", flush=True)
        results.append(
            evaluate_candidate(
                name=candidate,
                manifest_path=manifest_path,
                output_root=output_root / "candidates" / candidate,
                run_config=run_config,
                config=config,
                phase8_config_path=phase8_config_path,
                source_fingerprint=source_fingerprint,
                baseline_root=baseline_root,
            )
        )
    baseline = _baseline_result(
        baseline_root,
        name=eligible[0].name,
        projected_seconds=(
            config.runtime.phase8_gbdt_seconds_per_video
            * config.runtime.hidden_video_count
        ),
    )
    preliminary = max(
        [baseline, *results],
        key=lambda item: (item.exact_fbeta, -item.projected_seconds_200_videos),
    )
    preliminary_gain = preliminary.exact_fbeta - baseline.exact_fbeta
    stacker_requested = config.stacker.always_evaluate or (
        config.stacker.enabled_on_blend_plateau
        and preliminary_gain < config.optimization.minimum_exact_fbeta_gain
    )
    stacker_registry_path: Path | None = None
    stacker_result: CandidateResult | None = None
    if stacker_requested:
        stacker_artifacts = build_probability_stacker(
            gbdt_parts=eligible[0].parts,
            temporal_parts=eligible[1].parts,
            folds=config.optimization.folds,
            output_root=output_root / "stacker",
            oof_root=oof_root / "phase11-crossfit-stacker" / configuration_sha[:12],
            configuration_sha256=configuration_sha,
            source_names=(eligible[0].name, eligible[1].name),
            config=config.stacker,
        )
        stacker_registry_path = stacker_artifacts.registry_path
        stacker_result = evaluate_candidate(
            name="phase11-crossfit-stacker",
            manifest_path=stacker_artifacts.manifest_path,
            output_root=output_root / "candidates" / "phase11-crossfit-stacker",
            run_config=run_config,
            config=config,
            phase8_config_path=phase8_config_path,
            source_fingerprint=source_fingerprint,
            baseline_root=baseline_root,
        )
        results.append(stacker_result)
    selector_sources = [baseline, *results]
    selector_path = output_root / "per_lab_selector.parquet"
    selector = select_models_by_lab_cross_fitted(
        candidates=selector_sources,
        run_config=run_config,
        folds=config.optimization.folds,
        output_path=selector_path,
    )
    selected_name = "phase11-crossfit-per-lab-selector"
    selected_manifest = build_selected_oof(
        candidate=selected_name,
        selector=selector,
        sources=selector_sources,
        reference=streams[0],
        output_root=oof_root / selected_name / configuration_sha[:12],
        configuration_sha256=configuration_sha,
    )
    print(f"Phase 11 exact evaluation: {selected_name}", flush=True)
    selected_result = evaluate_candidate(
        name=selected_name,
        manifest_path=selected_manifest,
        output_root=output_root / "candidates" / selected_name,
        run_config=run_config,
        config=config,
        phase8_config_path=phase8_config_path,
        source_fingerprint=source_fingerprint,
        baseline_root=baseline_root,
    )
    results.append(selected_result)
    all_results = [baseline, *results]
    feasible_limit = config.runtime.kaggle_limit_seconds * (1 - config.runtime.required_headroom_fraction)
    deployable_names = {
        baseline.name,
        "phase11-crossfit-action",
        "phase11-crossfit-stacker",
    }

    def clears_promotion_gate(result: CandidateResult) -> bool:
        if result.name == baseline.name:
            return True
        required_gain = (
            config.stacker.minimum_exact_fbeta_gain
            if result.name == "phase11-crossfit-stacker"
            else config.optimization.minimum_exact_fbeta_gain
        )
        return result.exact_fbeta - baseline.exact_fbeta >= required_gain

    def clears_runtime_gate(result: CandidateResult) -> bool:
        candidate_limit = feasible_limit
        if result.name == "phase11-crossfit-stacker":
            candidate_limit *= config.stacker.maximum_projected_fraction_of_budget
        return result.projected_seconds_200_videos <= candidate_limit

    feasible = [
        result
        for result in all_results
        if result.name in deployable_names
        and clears_promotion_gate(result)
        and clears_runtime_gate(result)
    ]
    strongest = max(feasible, key=lambda item: (item.exact_fbeta, -item.projected_seconds_200_videos))
    improvement = strongest.exact_fbeta - baseline.exact_fbeta
    acceptance_gate_passed = strongest.name != baseline.name and clears_promotion_gate(
        strongest
    )
    stacker_promoted = (
        stacker_result is not None
        and strongest.name == stacker_result.name
        and stacker_result.exact_fbeta - baseline.exact_fbeta
        >= config.stacker.minimum_exact_fbeta_gain
    )
    stacker_decision = {
        "triggered": stacker_requested,
        "evaluated": stacker_result is not None,
        "promoted": stacker_promoted,
        "status": (
            "promoted"
            if stacker_promoted
            else "evaluated_not_promoted"
            if stacker_result is not None
            else "not_triggered_blending_improved"
        ),
        "require_base_geometric_features": config.stacker.require_base_geometric_features,
        "registry": None
        if stacker_registry_path is None
        else stacker_registry_path.as_posix(),
        "exact_fbeta": None
        if stacker_result is None
        else stacker_result.exact_fbeta,
        "reason": (
            "The cross-fitted stacker cleared both exact-F-beta and runtime gates."
            if stacker_promoted
            else "The stacker was evaluated but the stronger deployable candidate was retained."
            if stacker_result is not None
            else "A convex blend improved before the optional plateau-triggered stacker gate."
        ),
    }
    stacker_path = output_root / "stacker_decision.json"
    stacker_path.write_text(json.dumps(stacker_decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    selected_decoder_path = output_root / "selected_decoder_parameters.parquet"
    shutil.copy2(strongest.decoder_parameters_path, selected_decoder_path)
    selected_probability_source = (
        strongest.submission_path.parent / "probability_decoder_selection.parquet"
    )
    if not selected_probability_source.is_file():
        raise FileNotFoundError(
            "selected candidate is missing probability decoder lineage: "
            f"{selected_probability_source}"
        )
    selected_probability_path = (
        output_root / "selected_probability_decoder_selection.parquet"
    )
    shutil.copy2(selected_probability_source, selected_probability_path)
    selected_weights = weights.filter(pl.col("strategy") == "action")
    if strongest.name == baseline.name:
        selected_weights = selected_weights.with_columns(
            pl.lit(1.0).alias("gbdt_weight"),
            pl.lit(0.0).alias("temporal_phase9_weight"),
            pl.lit(0.0).alias("raw_temporal_weight"),
            pl.lit(0.0).alias("parent_temporal_weight"),
        )
    selected_weights_path = output_root / "selected_blend_weight_registry.parquet"
    selected_weights.write_parquet(
        selected_weights_path, compression="zstd", statistics=True
    )
    final_registry = {
        "schema_version": 1,
        "phase11_version": config.phase11_version,
        "configuration_sha256": configuration_sha,
        "frozen_at": datetime.now(UTC).isoformat(),
        "source_data_fingerprint": source_fingerprint,
        "selected_model_id": strongest.name,
        "selected_exact_oof_fbeta": strongest.exact_fbeta,
        "strongest_member_model_id": baseline.name,
        "strongest_member_exact_oof_fbeta": baseline.exact_fbeta,
        "exact_fbeta_gain": improvement,
        "acceptance_gate_passed": acceptance_gate_passed,
        "oof_manifest": _artifact_record(strongest.manifest_path),
        "decoder_parameters": _artifact_record(strongest.decoder_parameters_path),
        "held_out_submission": _artifact_record(strongest.submission_path),
        "blend_weight_registry": _artifact_record(weights_path),
        "selected_blend_weight_registry": _artifact_record(selected_weights_path),
        "selected_probability_model": (
            "stacker" if strongest.name == "phase11-crossfit-stacker" else "blend"
        ),
        "selected_stacker_registry": (
            None
            if strongest.name != "phase11-crossfit-stacker"
            or stacker_registry_path is None
            else _artifact_record(stacker_registry_path)
        ),
        "selected_probability_decoder_selection": _artifact_record(
            selected_probability_path
        ),
        "per_lab_selector": _artifact_record(selector_path),
        "base_streams": [
            {
                "name": stream.name,
                "manifest": _artifact_record(stream.manifest_path),
                "deployable": stream.deployable,
                "eligible_for_optimized_blend": stream.eligible_for_optimized_blend,
            }
            for stream in streams
        ],
        "normalization": {
            "pose_config": _artifact_record(config_path.parent / "pose.yaml"),
            "fold_statistics_root": (run_config.paths.artifact_root / "pose" / "normalization").as_posix(),
        },
        "fallback": {
            "model_id": baseline.name,
            "oof_manifest": baseline.manifest_path.as_posix(),
            "decoder_parameters": baseline.decoder_parameters_path.as_posix(),
        },
        "runtime": {
            "projected_seconds_200_videos": strongest.projected_seconds_200_videos,
            "budget_with_headroom_seconds": feasible_limit,
            "feasible": strongest.projected_seconds_200_videos <= feasible_limit,
        },
        "stacker_decision": _artifact_record(stacker_path),
    }
    registry_path = output_root / "final_model_registry.json"
    registry_path.write_text(json.dumps(final_registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    leaderboard = pl.DataFrame(
        [
            {
                "candidate": result.name,
                "exact_fbeta": result.exact_fbeta,
                "delta_vs_gbdt": result.exact_fbeta - baseline.exact_fbeta,
                "projected_seconds_200_videos": result.projected_seconds_200_videos,
                "runtime_feasible": clears_runtime_gate(result),
                "promotion_gate_passed": clears_promotion_gate(result),
                "selected": result.name == strongest.name,
            }
            for result in all_results
        ]
    ).sort("exact_fbeta", descending=True)
    leaderboard_path = output_root / "candidate_leaderboard.csv"
    leaderboard.write_csv(leaderboard_path)
    report = {
        "schema_version": 1,
        "phase11_version": config.phase11_version,
        "configuration_sha256": configuration_sha,
        "source_data_fingerprint": source_fingerprint,
        "alignment": {
            "parts": alignment.height,
            "rows": int(alignment["row_count"].sum()),
            "failures": 0,
            "missing_routes": 0,
            "duplicated_keys": 0,
            "mask_or_target_mismatches": 0,
        },
        "weight_optimization": {
            "method": "cross_fitted_nonnegative_convex_grid_search_with_hierarchical_shrinkage",
            "scopes": ["global", "action", "lab", "lab_action"],
            "held_out_labels_used": False,
            "registry_rows": weights.height,
        },
        "candidate_count": len(all_results),
        "excluded_reference_streams": [
            {
                "name": stream.name,
                "reason": "not eligible for optimized blending because its prior phase retention gate failed",
            }
            for stream in streams
            if not stream.eligible_for_optimized_blend
        ],
        "candidate_scores": {result.name: result.exact_fbeta for result in all_results},
        "strongest_member_fbeta": baseline.exact_fbeta,
        "selected_model_id": strongest.name,
        "final_exact_fbeta": strongest.exact_fbeta,
        "exact_fbeta_gain": improvement,
        "acceptance_gate_passed": acceptance_gate_passed,
        "all_candidate_decoders_nested": True,
        "all_probability_operations_pre_decode": True,
        "stacker": stacker_decision,
        "runtime_feasible": strongest.projected_seconds_200_videos <= feasible_limit,
        "runtime_seconds": time.perf_counter() - started,
        "deliverables": {
            "final_oof_manifest": strongest.manifest_path.as_posix(),
            "final_submission": strongest.submission_path.as_posix(),
            "final_registry": registry_path.as_posix(),
            "candidate_leaderboard": leaderboard_path.as_posix(),
        },
    }
    report_path = output_root / "phase11_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return Phase11Outputs(
        report_path=report_path,
        alignment_audit_path=alignment_path,
        blend_weight_registry_path=weights_path,
        final_oof_manifest_path=strongest.manifest_path,
        final_submission_path=strongest.submission_path,
        final_registry_path=registry_path,
        overall_fbeta=strongest.exact_fbeta,
    )
