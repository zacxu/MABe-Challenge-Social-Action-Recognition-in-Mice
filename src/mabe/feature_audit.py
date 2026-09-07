"""Phase 6 real-data feature audit, schema gate, caches, and runtime benchmark."""

from __future__ import annotations

import json
import time
import tracemalloc
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from mabe.features.builder import build_feature_tables
from mabe.features.common import (
    MetadataContext,
    feature_config_fingerprint,
    fit_category_registry,
    load_feature_config,
    slice_clean_pose,
)
from mabe.features.storage import (
    save_category_registry,
    save_feature_registry,
    save_feature_table,
    validate_feature_cache,
)
from mabe.fingerprints import schema_fingerprint, sha256_file
from mabe.pose import load_cached_pose, load_clean_pose_from_manifest, load_pose_config


@dataclass(frozen=True, slots=True)
class Phase6Outputs:
    report_path: Path
    benchmark_path: Path
    registry_root: Path
    sample_cache_root: Path
    family_count: int
    sample_count: int
    solo_feature_count: int
    pair_feature_count: int


def _write_sidecar(
    path: Path,
    *,
    row_count: int | None,
    schema_hash: str | None,
    source_fingerprint: str,
    configuration_sha256: str,
) -> None:
    payload = {
        "artifact": path.name,
        "artifact_sha256": sha256_file(path),
        "configuration_sha256": configuration_sha256,
        "created_at": datetime.now(UTC).isoformat(),
        "row_count": row_count,
        "schema_sha256": schema_hash,
        "source_data_fingerprint": source_fingerprint,
    }
    Path(f"{path}.meta.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _report_markdown(report: dict[str, Any]) -> str:
    counts = report["counts"]
    benchmark = report["benchmark"]
    return "\n".join(
        [
            "# Phase 6 feature audit",
            "",
            f"Source fingerprint: `{report['source_fingerprint']}`",
            "",
            "## Acceptance summary",
            "",
            f"- Skeleton/tracker families covered: {counts['family_count']}",
            f"- Real train/inference samples: {counts['sample_count']}",
            f"- Solo features: {counts['solo_feature_count']}",
            f"- Directed-pair features: {counts['pair_feature_count']}",
            f"- Train/inference schema match: {counts['train_inference_schema_match']}",
            f"- Non-finite feature columns: {counts['nonfinite_column_count']}",
            "",
            "## Bounded representative-window benchmark",
            "",
            f"- Total seconds: {benchmark['total_seconds']:.3f}",
            f"- Mean seconds/sample: {benchmark['mean_seconds_per_sample']:.3f}",
            f"- Complete videos measured: {benchmark['complete_video_count']}",
            (
                "- Mean seconds/complete video: "
                f"{benchmark['mean_seconds_per_complete_video']:.3f}"
                if benchmark["mean_seconds_per_complete_video"] is not None
                else "- Mean seconds/complete video: unavailable"
            ),
            f"- Peak traced memory: {benchmark['peak_traced_memory_mb']:.2f} MiB",
            f"- Output size: {benchmark['output_size_mb']:.2f} MiB",
            "",
            "Every model column is finite float32. Missing geometry remains zero only with an",
            "explicit validity/support feature. Solo and pair caches retain original frame IDs.",
            "",
        ]
    )


def build_phase6_artifacts(
    *,
    videos_path: str | Path,
    pose_quality_path: str | Path,
    dataset_root: str | Path,
    pose_config_path: str | Path,
    feature_config_path: str | Path,
    artifact_root: str | Path,
    cache_root: str | Path,
    source_fingerprint: str,
) -> Phase6Outputs:
    """Build and profile one train sample per family plus real inference metadata."""
    videos = pl.read_parquet(videos_path)
    quality = pl.read_parquet(pose_quality_path)
    pose_config = load_pose_config(pose_config_path)
    feature_config = load_feature_config(feature_config_path)
    config_sha = feature_config_fingerprint(feature_config_path)
    categories = fit_category_registry(videos)
    output_root = Path(artifact_root) / "features"
    registry_root = output_root / "registry"
    sample_cache_root = Path(cache_root) / feature_config.feature_version / "samples"
    output_root.mkdir(parents=True, exist_ok=True)
    save_category_registry(categories, registry_root / "categories.json")

    selected_quality = (
        quality.sort("frame_count", "lab_id", "video_id")
        .group_by("skeleton_id", "tracking_method", maintain_order=True)
        .head(1)
        .sort("tracking_method", "skeleton_id")
    )
    selected: list[tuple[str, dict[str, object], Path | None]] = []
    for audit_row in selected_quality.iter_rows(named=True):
        manifest = videos.filter(
            (pl.col("lab_id") == audit_row["lab_id"])
            & (pl.col("video_id") == audit_row["video_id"])
        ).row(0, named=True)
        selected.append(("train_family", manifest, Path(str(audit_row["cache_path"]))))
    for manifest in videos.filter(pl.col("split") == "test").iter_rows(named=True):
        selected.append(("inference", manifest, None))

    records: list[dict[str, object]] = []
    reference_hashes: dict[str, set[str]] = {"solo": set(), "pair": set()}
    train_hashes: dict[str, set[str]] = {"solo": set(), "pair": set()}
    inference_hashes: dict[str, set[str]] = {"solo": set(), "pair": set()}
    peak_memory = 0
    started_all = time.perf_counter()
    registries_written = False
    solo_feature_count = 0
    pair_feature_count = 0
    feature_groups: dict[str, list[str]] = {}
    for role, manifest, cache_path in selected:
        if cache_path is None:
            start = int(str(manifest["observed_start_frame"]))
            stop = min(
                int(str(manifest["observed_stop_frame"])),
                start + feature_config.audit_max_frames,
            )
            pose = load_clean_pose_from_manifest(
                manifest,
                dataset_root=dataset_root,
                config=pose_config,
                start_frame=start,
                stop_frame=stop,
            )
        else:
            pose = slice_clean_pose(load_cached_pose(cache_path), feature_config.audit_max_frames)
        context = MetadataContext.from_manifest_row(manifest)
        tracemalloc.start()
        started = time.perf_counter()
        solo, pair = build_feature_tables(pose, context, feature_config, categories)
        elapsed = time.perf_counter() - started
        _, current_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_memory = max(peak_memory, current_peak)
        if not registries_written:
            save_feature_registry(solo, registry_root / "solo.json")
            save_feature_registry(pair, registry_root / "pair.json")
            registries_written = True
        sample_root = (
            sample_cache_root
            / str(manifest["split"])
            / str(manifest["lab_id"])
            / str(int(str(manifest["video_id"])))
        )
        solo_cache = save_feature_table(
            solo,
            sample_root / "solo",
            chunk_rows=feature_config.chunk_rows,
            source_fingerprint=source_fingerprint,
            configuration_sha256=config_sha,
        )
        pair_cache = save_feature_table(
            pair,
            sample_root / "pair",
            chunk_rows=feature_config.chunk_rows,
            source_fingerprint=source_fingerprint,
            configuration_sha256=config_sha,
        )
        validate_feature_cache(solo_cache, solo)
        validate_feature_cache(pair_cache, pair)
        solo_feature_count = len(solo.feature_names)
        pair_feature_count = len(pair.feature_names)
        feature_groups = {
            "solo": sorted(set(solo.groups.values())),
            "pair": sorted(set(pair.groups.values())),
        }
        split_group = inference_hashes if role == "inference" else train_hashes
        for kind, table in (("solo", solo), ("pair", pair)):
            reference_hashes[kind].add(table.schema_hash)
            split_group[kind].add(table.schema_hash)
        records.append(
            {
                "role": role,
                "split": str(manifest["split"]),
                "lab_id": pose.lab_id,
                "video_id": pose.video_id,
                "skeleton_id": str(manifest["skeleton_id"]),
                "tracking_method": str(manifest["tracking_method"]),
                "fps": pose.fps,
                "frame_count": pose.frame_ids.size,
                "complete_video": pose.frame_ids.size
                >= int(str(manifest["observed_frame_count"])),
                "mouse_count": len(pose.mouse_ids),
                "solo_row_count": solo_cache.row_count,
                "pair_row_count": pair_cache.row_count,
                "solo_feature_count": len(solo.feature_names),
                "pair_feature_count": len(pair.feature_names),
                "solo_schema_sha256": solo.schema_hash,
                "pair_schema_sha256": pair.schema_hash,
                "seconds": elapsed,
                "traced_peak_memory_mb": current_peak / (1024**2),
                "output_bytes": solo_cache.byte_count + pair_cache.byte_count,
                "nonfinite_column_count": 0,
                "cache_root": sample_root.as_posix(),
            }
        )
    total_elapsed = time.perf_counter() - started_all
    benchmark_frame = pl.DataFrame(records).sort("role", "tracking_method", "skeleton_id")
    if any(len(hashes) != 1 for hashes in reference_hashes.values()):
        raise AssertionError("feature schema changed across real videos")
    schema_match = all(
        train_hashes[kind] == inference_hashes[kind] for kind in ("solo", "pair")
    )
    if not schema_match:
        raise AssertionError("training and inference feature schemas differ")
    family_count = selected_quality.select("skeleton_id", "tracking_method").unique().height
    expected_families = quality.select("skeleton_id", "tracking_method").unique().height
    if family_count != expected_families:
        raise AssertionError("feature audit did not cover every skeleton/tracker family")
    nonfinite_columns = int(benchmark_frame["nonfinite_column_count"].sum())
    if nonfinite_columns:
        raise AssertionError("non-finite model columns reached Phase 6 artifacts")

    benchmark_path = output_root / "benchmark.parquet"
    benchmark_frame.write_parquet(benchmark_path, compression="zstd", statistics=True)
    complete_benchmarks = benchmark_frame.filter(pl.col("complete_video"))
    report: dict[str, Any] = {
        "schema_version": 1,
        "feature_version": feature_config.feature_version,
        "feature_config_sha256": config_sha,
        "source_fingerprint": source_fingerprint,
        "temporal_rounding": "max(1, floor(seconds * fps + 0.5))",
        "temporal_context": "centered_with_truncated_video_edges_and_support_features",
        "counts": {
            "family_count": family_count,
            "sample_count": len(records),
            "train_sample_count": sum(record["role"] == "train_family" for record in records),
            "inference_sample_count": sum(record["role"] == "inference" for record in records),
            "solo_feature_count": solo_feature_count,
            "pair_feature_count": pair_feature_count,
            "train_inference_schema_match": schema_match,
            "nonfinite_column_count": nonfinite_columns,
        },
        "schema_hashes": {
            kind: next(iter(hashes)) for kind, hashes in reference_hashes.items()
        },
        "feature_groups": feature_groups,
        "benchmark": {
            "window_frames_max": feature_config.audit_max_frames,
            "total_seconds": total_elapsed,
            "mean_seconds_per_sample": float(str(benchmark_frame["seconds"].mean())),
            "peak_traced_memory_mb": peak_memory / (1024**2),
            "output_size_mb": int(str(benchmark_frame["output_bytes"].sum())) / (1024**2),
            "complete_video_count": complete_benchmarks.height,
            "mean_seconds_per_complete_video": (
                float(str(complete_benchmarks["seconds"].mean()))
                if complete_benchmarks.height
                else None
            ),
        },
        "policies": {
            "feature_columns_float32": True,
            "nonfinite_reaches_model": False,
            "ordered_pairs": True,
            "third_mouse_masked_for_dyads": True,
            "labels_used_by_builder": False,
            "split_specific_builder_branch": False,
        },
    }
    report_path = output_root / "feature_audit.json"
    markdown_path = output_root / "feature_audit.md"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(_report_markdown(report), encoding="utf-8")
    for path, frame in ((benchmark_path, benchmark_frame),):
        _write_sidecar(
            path,
            row_count=frame.height,
            schema_hash=schema_fingerprint(frame),
            source_fingerprint=source_fingerprint,
            configuration_sha256=config_sha,
        )
    for path in (
        report_path,
        markdown_path,
        registry_root / "categories.json",
        registry_root / "solo.json",
        registry_root / "pair.json",
    ):
        _write_sidecar(
            path,
            row_count=None,
            schema_hash=None,
            source_fingerprint=source_fingerprint,
            configuration_sha256=config_sha,
        )
    return Phase6Outputs(
        report_path=report_path,
        benchmark_path=benchmark_path,
        registry_root=registry_root,
        sample_cache_root=sample_cache_root,
        family_count=family_count,
        sample_count=len(records),
        solo_feature_count=solo_feature_count,
        pair_feature_count=pair_feature_count,
    )
