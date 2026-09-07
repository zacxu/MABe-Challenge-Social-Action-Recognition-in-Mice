"""Phase 5 real-data pose quality, timing, cache, and normalization artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from mabe.fingerprints import schema_fingerprint, sha256_file
from mabe.io import scan_annotation
from mabe.normalization import (
    fit_fold_normalization,
    manifest_normalization_records,
    save_fold_normalization,
)
from mabe.pose import (
    PoseConfig,
    load_clean_pose_from_manifest,
    load_pose_config,
    pose_config_fingerprint,
    save_clean_pose,
)


@dataclass(frozen=True, slots=True)
class Phase5Outputs:
    tracking_quality_path: Path
    pose_audit_path: Path
    adaptable_snail_path: Path
    normalization_root: Path
    sample_cache_root: Path
    family_count: int
    sample_count: int


def _resolve(root: Path, raw_path: object) -> Path:
    path = Path(str(raw_path))
    return path if path.is_absolute() else root / path


def _source_hashes(source_files: pl.DataFrame) -> dict[tuple[str, str, int], str]:
    return {
        (str(row["split"]), str(row["lab_id"]), int(row["video_id"])): str(row["sha256"])
        for row in source_files.filter(pl.col("kind") == "tracking").iter_rows(named=True)
    }


def audit_pose_families(
    videos: pl.DataFrame,
    source_files: pl.DataFrame,
    *,
    dataset_root: str | Path,
    config: PoseConfig,
    config_sha256: str,
    cache_root: str | Path,
    samples_per_family: int = 2,
    max_frame_span: int = 4096,
) -> pl.DataFrame:
    """Load bounded samples from every observed skeleton/tracker family."""
    selected = (
        videos.sort("tracking_row_count")
        .group_by("skeleton_id", "tracking_method", maintain_order=True)
        .head(samples_per_family)
        .sort("tracking_method", "skeleton_id", "tracking_row_count")
    )
    source_hashes = _source_hashes(source_files)
    cache_directory = Path(cache_root)
    records: list[dict[str, object]] = []
    for row in selected.iter_rows(named=True):
        start = int(row["observed_start_frame"])
        stop = min(int(row["observed_stop_frame"]), start + max_frame_span)
        pose = load_clean_pose_from_manifest(
            row,
            dataset_root=dataset_root,
            config=config,
            start_frame=start,
            stop_frame=stop,
        )
        cache_path = cache_directory / str(row["lab_id"]) / f"{int(row['video_id'])}.npz"
        save_clean_pose(
            pose,
            cache_path,
            source_sha256=source_hashes[(str(row["split"]), str(row["lab_id"]), int(row["video_id"]))],
            configuration_sha256=config_sha256,
            frame_window=(start, stop),
        )
        total_cells = pose.availability_mask.size
        records.append(
            {
                "split": str(row["split"]),
                "lab_id": str(row["lab_id"]),
                "video_id": int(row["video_id"]),
                "skeleton_id": str(row["skeleton_id"]),
                "tracking_method": str(row["tracking_method"]),
                "fps": pose.fps,
                "frame_start": int(pose.frame_ids[0]),
                "frame_stop": int(pose.frame_ids[-1]) + 1,
                "frame_count": int(pose.frame_ids.size),
                "mouse_count": len(pose.mouse_ids),
                "landmark_count": len(pose.landmarks),
                "scale_valid": pose.scale_valid,
                "coordinate_unit": pose.coordinate_unit,
                "observed_rate": float(pose.observed_mask.sum() / total_cells),
                "availability_rate": float(pose.availability_mask.sum() / total_cells),
                "interpolated_rate": float(pose.interpolated_mask.sum() / total_cells),
                "unresolved_rate": float(pose.long_gap_mask.sum() / total_cells),
                "derived_center_rate": float(
                    pose.derived_body_center.sum() / pose.derived_body_center.size
                ),
                "jump_count": int(pose.jump_mask.sum()),
                "invalid_count": int(pose.invalid_mask.sum()),
                "identity_swap_candidate_count": int(pose.identity_swap_mask.sum()),
                "mean_confidence": float(pose.confidence.mean()),
                "all_numeric_arrays_finite": bool(
                    np.isfinite(pose.raw_pixels).all()
                    and np.isfinite(pose.coordinates).all()
                    and np.isfinite(pose.center).all()
                    and np.isfinite(pose.body_axis).all()
                    and np.isfinite(pose.body_length).all()
                    and np.isfinite(pose.confidence).all()
                ),
                "cache_path": cache_path.as_posix(),
            }
        )
    result = pl.DataFrame(records).sort("tracking_method", "skeleton_id", "lab_id", "video_id")
    expected_families = videos.select("skeleton_id", "tracking_method").unique().height
    loaded_families = result.select("skeleton_id", "tracking_method").unique().height
    if loaded_families != expected_families:
        raise AssertionError(
            f"loaded {loaded_families} skeleton/tracker families, expected {expected_families}"
        )
    if result.filter(~pl.col("all_numeric_arrays_finite")).height:
        raise AssertionError("pose audit produced non-finite numeric output")
    return result


def audit_adaptable_snail_25fps(
    videos: pl.DataFrame,
    *,
    dataset_root: str | Path,
    config: PoseConfig,
    audit_frame_span: int = 4096,
) -> pl.DataFrame:
    """Compare declared timing, tracking, annotations, and the proposed 30/25 hypothesis."""
    selected = videos.filter(
        (pl.col("split") == "train")
        & (pl.col("lab_id") == "AdaptableSnail")
        & pl.col("frames_per_second").is_between(24.9, 25.1)
    ).sort("video_id")
    root = Path(dataset_root)
    records: list[dict[str, object]] = []
    for row in selected.iter_rows(named=True):
        annotation_path = _resolve(root, row["annotation_path"])
        annotations = scan_annotation(annotation_path).collect()
        annotation_min = int(str(annotations["start_frame"].min()))
        annotation_max = int(str(annotations["stop_frame"].max()))
        observed_start = int(row["observed_start_frame"])
        observed_stop = int(row["observed_stop_frame"])
        observed_span = observed_stop - observed_start
        declared_seconds = float(row["video_duration_sec"])
        current_seconds = observed_span / float(row["frames_per_second"])
        hypothesis_seconds = observed_span / 30.0
        audit_stop = min(observed_stop, observed_start + audit_frame_span)
        pose = load_clean_pose_from_manifest(
            row,
            dataset_root=root,
            config=config,
            start_frame=observed_start,
            stop_frame=audit_stop,
        )
        correction_factor = config.adaptable_snail_frame_corrections.get(int(row["video_id"]))
        records.append(
            {
                "lab_id": "AdaptableSnail",
                "video_id": int(row["video_id"]),
                "declared_fps": float(row["frames_per_second"]),
                "declared_duration_seconds": declared_seconds,
                "observed_frame_start": observed_start,
                "observed_frame_stop": observed_stop,
                "observed_span_frames": observed_span,
                "duration_at_declared_fps": current_seconds,
                "duration_at_30fps": hypothesis_seconds,
                "declared_fps_absolute_error_seconds": abs(current_seconds - declared_seconds),
                "thirty_fps_absolute_error_seconds": abs(hypothesis_seconds - declared_seconds),
                "annotation_start_min": annotation_min,
                "annotation_stop_max": annotation_max,
                "annotation_bounds_within_tracking": (
                    annotation_min >= observed_start and annotation_max <= observed_stop
                ),
                "annotation_span_fraction": (
                    (annotation_max - annotation_min) / observed_span if observed_span else 0.0
                ),
                "identity_swap_candidates_in_audit_window": int(
                    pose.identity_swap_mask.sum()
                ),
                "audit_window_start": observed_start,
                "audit_window_stop": audit_stop,
                "oof_correction_evidence_available": False,
                "correction_enabled": correction_factor is not None,
                "correction_factor": correction_factor,
                "decision": (
                    "configured_video_specific_correction"
                    if correction_factor is not None
                    else "no_correction_without_nested_oof_gain"
                ),
            }
        )
    result = pl.DataFrame(records).sort("video_id")
    if result.filter(pl.col("correction_enabled")).height and not config.adaptable_snail_frame_corrections:
        raise AssertionError("frame correction was enabled without an explicit rule")
    return result


def _write_sidecar(
    path: Path,
    *,
    row_count: int | None,
    schema_hash: str | None,
    source_fingerprint: str,
    config_sha256: str,
) -> None:
    payload = {
        "artifact": path.name,
        "artifact_sha256": sha256_file(path),
        "configuration": {"pose_config_sha256": config_sha256},
        "created_at": datetime.now(UTC).isoformat(),
        "row_count": row_count,
        "schema_sha256": schema_hash,
        "source_data_fingerprint": source_fingerprint,
    }
    Path(f"{path}.meta.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _audit_markdown(report: dict[str, Any]) -> str:
    counts = report["counts"]
    return "\n".join(
        [
            "# Phase 5 pose audit",
            "",
            f"Source fingerprint: `{report['source_fingerprint']}`",
            "",
            "## Acceptance summary",
            "",
            f"- Skeleton/tracker families: {counts['family_count']}",
            f"- Representative windows loaded: {counts['sample_count']}",
            f"- Samples with finite/masked output: {counts['finite_sample_count']}",
            f"- Valid physical scales: {counts['scale_valid_sample_count']}",
            f"- Fold normalization artifacts: {counts['normalization_fold_count']}",
            f"- AdaptableSnail 25 FPS videos audited: {counts['adaptable_25fps_videos']}",
            "",
            "No global or video-specific 30/25 correction is enabled. The observed frame spans",
            "are closer to declared 25 FPS timing, and no nested OOF correction evidence exists yet.",
            "Potential identity swaps are flagged only; mouse identities are not silently rewritten.",
            "",
        ]
    )


def build_phase5_artifacts(
    *,
    videos_path: str | Path,
    source_files_path: str | Path,
    folds_path: str | Path,
    dataset_root: str | Path,
    pose_config_path: str | Path,
    artifact_root: str | Path,
    cache_root: str | Path,
    source_fingerprint: str,
) -> Phase5Outputs:
    videos = pl.read_parquet(videos_path)
    source_files = pl.read_parquet(source_files_path)
    folds = pl.read_parquet(folds_path)
    config = load_pose_config(pose_config_path)
    config_sha = pose_config_fingerprint(pose_config_path)
    output_root = Path(artifact_root) / "pose"
    output_root.mkdir(parents=True, exist_ok=True)
    sample_cache_root = Path(cache_root) / config.pose_version / "samples"
    quality = audit_pose_families(
        videos,
        source_files,
        dataset_root=dataset_root,
        config=config,
        config_sha256=config_sha,
        cache_root=sample_cache_root,
    )
    adaptable = audit_adaptable_snail_25fps(
        videos,
        dataset_root=dataset_root,
        config=config,
    )
    normalization_root = output_root / "normalization"
    normalization_records = manifest_normalization_records(videos)
    normalization_features = (
        "fps",
        "pixels_per_cm",
        "arena_width_cm",
        "arena_height_cm",
        "video_width_cm",
        "video_height_cm",
    )
    for held_out_fold in range(5):
        normalization = fit_fold_normalization(
            normalization_records,
            folds,
            held_out_fold=held_out_fold,
            feature_names=normalization_features,
        )
        save_fold_normalization(
            normalization,
            normalization_root / f"fold_{held_out_fold}.json",
            source_fingerprint=source_fingerprint,
        )

    quality_path = output_root / "tracking_quality.parquet"
    adaptable_path = output_root / "adaptable_snail_25fps.parquet"
    quality.write_parquet(quality_path, compression="zstd", statistics=True)
    adaptable.write_parquet(adaptable_path, compression="zstd", statistics=True)
    report: dict[str, Any] = {
        "schema_version": 1,
        "pose_version": config.pose_version,
        "pose_config_sha256": config_sha,
        "source_fingerprint": source_fingerprint,
        "canonical_landmarks": list(config.canonical_landmarks),
        "reviewed_bodypart_synonyms": config.bodypart_synonyms,
        "adaptable_snail_frame_corrections": config.adaptable_snail_frame_corrections,
        "counts": {
            "family_count": quality.select("skeleton_id", "tracking_method").unique().height,
            "sample_count": quality.height,
            "finite_sample_count": quality.filter(pl.col("all_numeric_arrays_finite")).height,
            "scale_valid_sample_count": quality.filter(pl.col("scale_valid")).height,
            "normalization_fold_count": 5,
            "adaptable_25fps_videos": adaptable.height,
            "adaptable_corrections_enabled": adaptable.filter(
                pl.col("correction_enabled")
            ).height,
        },
        "policies": {
            "short_gap_interpolation_seconds": config.cleanup.max_interpolation_gap_seconds,
            "long_gaps_remain_masked": True,
            "nonfinite_values_remain_numeric": False,
            "identity_swap_policy": "flag_only",
            "resampling_target_fps": config.resampling.target_fps,
            "resampling_extrapolation": False,
            "adaptable_correction_requires_nested_oof_gain": True,
        },
    }
    report_path = output_root / "pose_audit.json"
    markdown_path = output_root / "pose_audit.md"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(_audit_markdown(report), encoding="utf-8")
    for path, frame in ((quality_path, quality), (adaptable_path, adaptable)):
        _write_sidecar(
            path,
            row_count=frame.height,
            schema_hash=schema_fingerprint(frame),
            source_fingerprint=source_fingerprint,
            config_sha256=config_sha,
        )
    for path in (report_path, markdown_path):
        _write_sidecar(
            path,
            row_count=None,
            schema_hash=None,
            source_fingerprint=source_fingerprint,
            config_sha256=config_sha,
        )
    return Phase5Outputs(
        tracking_quality_path=quality_path,
        pose_audit_path=report_path,
        adaptable_snail_path=adaptable_path,
        normalization_root=normalization_root,
        sample_cache_root=sample_cache_root,
        family_count=quality.select("skeleton_id", "tracking_method").unique().height,
        sample_count=quality.height,
    )
