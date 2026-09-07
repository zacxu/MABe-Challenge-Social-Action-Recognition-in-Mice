"""Deterministic, read-only inventory and anomaly audit for the MABe dataset."""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from mabe.config import RunConfig
from mabe.fingerprints import inventory_fingerprint, schema_fingerprint, sha256_file
from mabe.io import scan_annotation_files, scan_metadata
from mabe.masks import BehaviorMaskError, ParsedBehaviorMask, parse_behaviors_labeled
from mabe.schemas import (
    MOUSE_METADATA_FIELDS,
    MOUSE_SLOTS,
    TRIPLET_MANIFEST_COLUMNS,
    VIDEO_MANIFEST_COLUMNS,
)

EXPECTED_FACTS = {
    "train_metadata_rows": 8_789,
    "train_tracking_files": 8_789,
    "annotation_files": 847,
    "unlabeled_mabe22_rows": 7_926,
    "canonical_action_count": 37,
    "self_action_count": 11,
    "pair_action_count": 26,
}
DEFAULT_EXTREME_JUMP_PIXELS_PER_FRAME = 100.0
OVERLAP_SCHEMA = pl.Schema(
    {
        "lab_id": pl.String,
        "video_id": pl.Int64,
        "agent_id": pl.String,
        "target_id": pl.String,
        "left_row_id": pl.Int64,
        "right_row_id": pl.Int64,
        "left_action": pl.String,
        "right_action": pl.String,
        "same_action": pl.Boolean,
        "overlap_start_frame": pl.Int64,
        "overlap_stop_frame": pl.Int64,
        "overlap_frame_count": pl.Int64,
    }
)


@dataclass(frozen=True, slots=True)
class AuditOutputs:
    videos_path: Path
    triplets_path: Path
    source_files_path: Path
    report_json_path: Path
    report_markdown_path: Path
    source_fingerprint: str
    facts: dict[str, int]


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _path_key(path: str | Path) -> str:
    return str(Path(path).resolve()).casefold()


def _parquet_identity(path: Path, dataset_root: Path) -> tuple[str, str, str | None, int | None]:
    relative = _relative(path, dataset_root)
    top_level = Path(relative).parts[0]
    if top_level == "train_tracking":
        return relative, "train", "tracking", int(path.stem)
    if top_level == "test_tracking":
        return relative, "test", "tracking", int(path.stem)
    if top_level == "train_annotation":
        return relative, "train", "annotation", int(path.stem)
    return relative, "unknown", None, None


def _source_file_record(path: Path, dataset_root: Path) -> dict[str, object]:
    relative = _relative(path, dataset_root)
    split = "train" if path.name == "train.csv" else "test" if path.name == "test.csv" else "other"
    kind = "metadata" if path.name in {"train.csv", "test.csv"} else "submission_template"
    lab_id: str | None = None
    video_id: int | None = None
    if path.suffix == ".parquet":
        relative, split, parsed_kind, video_id = _parquet_identity(path, dataset_root)
        kind = parsed_kind or "unknown"
        lab_id = path.parent.name
        parquet_file = pq.ParquetFile(path)
        row_count = parquet_file.metadata.num_rows
        schema_json = json.dumps(
            [(field.name, str(field.type)) for field in parquet_file.schema_arrow],
            separators=(",", ":"),
        )
    else:
        with path.open("rb") as stream:
            row_count = max(sum(1 for _ in stream) - 1, 0)
        header = path.open("r", encoding="utf-8-sig").readline().strip().split(",")
        schema_json = json.dumps(header, separators=(",", ":"))
    return {
        "relative_path": relative,
        "split": split,
        "kind": kind,
        "lab_id": lab_id,
        "video_id": video_id,
        "size_bytes": path.stat().st_size,
        "row_count": row_count,
        "schema_json": schema_json,
        "sha256": sha256_file(path),
    }


def build_source_inventory(dataset_root: Path) -> tuple[pl.DataFrame, str]:
    """Hash every source file and retain its row count and physical schema."""
    files = sorted(
        [dataset_root / "train.csv", dataset_root / "test.csv", dataset_root / "sample_submission.csv"]
        + list((dataset_root / "train_tracking").rglob("*.parquet"))
        + list((dataset_root / "test_tracking").rglob("*.parquet"))
        + list((dataset_root / "train_annotation").rglob("*.parquet")),
        key=lambda path: _relative(path, dataset_root),
    )
    workers = min(8, max(os.cpu_count() or 1, 1))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        records = list(executor.map(lambda path: _source_file_record(path, dataset_root), files))
    records.sort(key=lambda record: str(record["relative_path"]))
    return pl.DataFrame(records), inventory_fingerprint(records)


def _tracking_statistics(
    paths: Sequence[Path],
    *,
    extreme_jump_pixels_per_frame: float,
    cache_root: Path | None = None,
    batch_size: int = 250,
) -> pl.DataFrame:
    """Scan tracking rows in bounded batches and compute one record per file."""
    results: list[pl.DataFrame] = []
    if cache_root is not None:
        cache_root.mkdir(parents=True, exist_ok=True)
    for offset in range(0, len(paths), batch_size):
        batch = paths[offset : offset + batch_size]
        sources = [str(path.resolve()) for path in batch]
        cache_payload = json.dumps(
            [
                (str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns)
                for path in batch
            ]
            + [("extreme_jump_pixels_per_frame", extreme_jump_pixels_per_frame)],
            separators=(",", ":"),
        )
        cache_key = hashlib.sha256(cache_payload.encode()).hexdigest()[:20]
        cache_path = cache_root / f"tracking_stats_{offset:05d}_{cache_key}.parquet" if cache_root else None
        if cache_path is not None and cache_path.is_file():
            results.append(pl.read_parquet(cache_path))
            continue
        frame = pl.scan_parquet(
            sources,
            schema={
                "video_frame": pl.Int64,
                "mouse_id": pl.Int64,
                "bodypart": pl.String,
                "x": pl.Float64,
                "y": pl.Float64,
            },
            include_file_paths="tracking_source_path",
            cast_options=pl.ScanCastOptions(integer_cast="upcast", float_cast="upcast"),
            hive_partitioning=False,
        ).select(
            pl.col("video_frame").cast(pl.Int64),
            pl.col("mouse_id").cast(pl.Int64),
            pl.col("bodypart").cast(pl.String),
            pl.col("x").cast(pl.Float64),
            pl.col("y").cast(pl.Float64),
            pl.col("tracking_source_path").cast(pl.String),
        )
        jump_groups = ["tracking_source_path", "mouse_id", "bodypart"]
        enriched = frame.with_columns(
            pl.col("video_frame").diff().over("tracking_source_path").alias("_file_frame_delta"),
            pl.col("video_frame").diff().over(jump_groups).alias("_frame_delta"),
            pl.col("x").diff().over(jump_groups).alias("_dx"),
            pl.col("y").diff().over(jump_groups).alias("_dy"),
            (pl.col("x").is_null() | pl.col("y").is_null()).alias("_missing_coordinate"),
            (
                (pl.col("x").is_not_null() & ~pl.col("x").is_finite())
                | (pl.col("y").is_not_null() & ~pl.col("y").is_finite())
            ).alias("_non_finite_coordinate"),
        ).with_columns(
            (
                (pl.col("_frame_delta") > 0)
                & pl.col("_dx").is_finite()
                & pl.col("_dy").is_finite()
            ).alias("_valid_transition"),
            (
                (pl.col("_frame_delta") > 0)
                & (
                    (pl.col("_dx").pow(2) + pl.col("_dy").pow(2)).sqrt()
                    / pl.col("_frame_delta")
                    > extreme_jump_pixels_per_frame
                )
            ).fill_null(False).alias("_extreme_jump"),
        )
        grouped = enriched.group_by("tracking_source_path", maintain_order=True).agg(
            pl.len().alias("tracking_row_count"),
            pl.col("video_frame").min().alias("observed_start_frame"),
            (pl.col("video_frame").max() + 1).alias("observed_stop_frame"),
            pl.col("video_frame").n_unique().alias("observed_frame_count"),
            pl.col("mouse_id").unique().sort().alias("mouse_numbers"),
            pl.col("bodypart").unique().sort().alias("body_parts_observed"),
            pl.col("_missing_coordinate").sum().alias("missing_coordinate_count"),
            pl.col("_non_finite_coordinate").sum().alias("non_finite_coordinate_count"),
            pl.col("_valid_transition").sum().alias("valid_transition_count"),
            pl.col("_extreme_jump").sum().alias("extreme_jump_count"),
            (pl.col("_file_frame_delta") < 0).sum().alias("tracking_order_violation_count"),
        ).with_columns(
            (
                1
                - pl.col("observed_frame_count")
                / (pl.col("observed_stop_frame") - pl.col("observed_start_frame"))
            ).alias("missing_frame_rate"),
            (pl.col("missing_coordinate_count") / pl.col("tracking_row_count")).alias(
                "missing_coordinate_rate"
            ),
            (pl.col("non_finite_coordinate_count") / pl.col("tracking_row_count")).alias(
                "non_finite_coordinate_rate"
            ),
            pl.when(pl.col("valid_transition_count") > 0)
            .then(pl.col("extreme_jump_count") / pl.col("valid_transition_count"))
            .otherwise(0.0)
            .alias("extreme_jump_rate"),
        )
        result = grouped.collect(engine="streaming")
        if cache_path is not None:
            result.write_parquet(cache_path, compression="zstd", statistics=True)
        results.append(result)
    if not results:
        return pl.DataFrame()
    return pl.concat(results, how="vertical_relaxed").sort("tracking_source_path")


def _parse_metadata_masks(
    metadata_rows: list[dict[str, Any]],
) -> tuple[
    dict[tuple[str, str, int], ParsedBehaviorMask],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    parsed: dict[tuple[str, str, int], ParsedBehaviorMask] = {}
    triplet_rows: list[dict[str, object]] = []
    malformed_rows: list[dict[str, object]] = []
    for row in metadata_rows:
        raw_mask = row["behaviors_labeled"]
        if raw_mask is None:
            continue
        key = (str(row["split"]), str(row["lab_id"]), int(row["video_id"]))
        try:
            mask = parse_behaviors_labeled(str(raw_mask))
        except BehaviorMaskError as error:
            malformed_rows.append(
                {
                    "split": key[0],
                    "lab_id": key[1],
                    "video_id": key[2],
                    "raw_mask": raw_mask,
                    "error": str(error),
                }
            )
            continue
        parsed[key] = mask
        canonical_mask = mask.canonical_json
        for entry in mask.entries:
            triplet_rows.append(
                {
                    "split": key[0],
                    "lab_id": key[1],
                    "video_id": key[2],
                    "entry_index": entry.index,
                    "raw_mask": raw_mask,
                    "canonical_mask": canonical_mask,
                    "raw_entry": entry.raw_entry,
                    "raw_agent_id": entry.raw_agent_id,
                    "raw_target_id": entry.raw_target_id,
                    "raw_action": entry.raw_action,
                    "agent_id": entry.triplet.agent_id,
                    "target_id": entry.triplet.target_id,
                    "action": entry.triplet.action,
                    "canonical_entry": entry.triplet.canonical_entry,
                    "is_self": entry.triplet.is_self,
                    "was_repaired": entry.was_repaired,
                    "repair_reasons": list(entry.repair_reasons),
                    "is_duplicate": entry.is_duplicate,
                }
            )
    return parsed, triplet_rows, malformed_rows


def _load_annotations(paths: Sequence[Path]) -> list[dict[str, Any]]:
    frame = scan_annotation_files(paths).collect(engine="streaming").with_row_index("annotation_row_id")
    rows: list[dict[str, Any]] = []
    for row in frame.to_dicts():
        source = Path(str(row["source_path"])).resolve()
        rows.append(
            {
                **row,
                "source_path": str(source),
                "lab_id": source.parent.name,
                "video_id": int(source.stem),
            }
        )
    rows.sort(key=lambda row: (str(row["lab_id"]), int(row["video_id"]), int(row["annotation_row_id"])))
    return rows


def _union_length(intervals: Iterable[tuple[int, int]]) -> int:
    ordered = sorted((start, stop) for start, stop in intervals if start < stop)
    if not ordered:
        return 0
    total = 0
    current_start, current_stop = ordered[0]
    for start, stop in ordered[1:]:
        if start > current_stop:
            total += current_stop - current_start
            current_start, current_stop = start, stop
        else:
            current_stop = max(current_stop, stop)
    return total + current_stop - current_start


def _overlap_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, object]]:
    grouped: defaultdict[tuple[str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if int(row["start_frame"]) < int(row["stop_frame"]):
            grouped[
                (str(row["lab_id"]), int(row["video_id"]), str(row["agent_id"]), str(row["target_id"]))
            ].append(row)
    overlaps: list[dict[str, object]] = []
    for key, group in sorted(grouped.items()):
        active: list[dict[str, Any]] = []
        for row in sorted(group, key=lambda item: (int(item["start_frame"]), int(item["stop_frame"]))):
            start = int(row["start_frame"])
            active = [previous for previous in active if int(previous["stop_frame"]) > start]
            for previous in active:
                overlap_start = max(start, int(previous["start_frame"]))
                overlap_stop = min(int(row["stop_frame"]), int(previous["stop_frame"]))
                if overlap_start < overlap_stop:
                    overlaps.append(
                        {
                            "lab_id": key[0],
                            "video_id": key[1],
                            "agent_id": key[2],
                            "target_id": key[3],
                            "left_row_id": int(previous["annotation_row_id"]),
                            "right_row_id": int(row["annotation_row_id"]),
                            "left_action": str(previous["action"]),
                            "right_action": str(row["action"]),
                            "same_action": previous["action"] == row["action"],
                            "overlap_start_frame": overlap_start,
                            "overlap_stop_frame": overlap_stop,
                            "overlap_frame_count": overlap_stop - overlap_start,
                        }
                    )
            active.append(row)
    return overlaps


def audit_annotation_rows(
    rows: Sequence[dict[str, Any]],
    *,
    tracking_by_video: dict[tuple[str, int], dict[str, Any]],
    parsed_masks: dict[tuple[str, str, int], ParsedBehaviorMask],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, object]],
    dict[tuple[str, int], int],
    dict[tuple[str, int], int],
    dict[tuple[str, int], int],
]:
    """Classify annotation anomalies and compute eligible union-frame counts."""
    seen: set[tuple[object, ...]] = set()
    audited: list[dict[str, Any]] = []
    anomalies: list[dict[str, object]] = []
    annotation_counts: defaultdict[tuple[str, int], int] = defaultdict(int)
    quarantine_counts: defaultdict[tuple[str, int], int] = defaultdict(int)
    positive_intervals: defaultdict[tuple[str, int, str, str, str], list[tuple[int, int]]] = defaultdict(list)

    for row in rows:
        lab_id = str(row["lab_id"])
        video_id = int(row["video_id"])
        video_key = (lab_id, video_id)
        annotation_counts[video_key] += 1
        start = int(row["start_frame"])
        stop = int(row["stop_frame"])
        agent = str(row["agent_id"])
        target = str(row["target_id"])
        action = str(row["action"])
        mask_target = "self" if target == agent else target
        canonical = (agent, mask_target, action)
        tracking = tracking_by_video.get(video_key)
        observed_mice = set(tracking.get("mouse_ids", [])) if tracking else set()
        mask = parsed_masks.get(("train", lab_id, video_id))
        allowed = {(item.agent_id, item.target_id, item.action) for item in mask.triplets} if mask else set()

        invalid_interval = start >= stop
        out_of_bounds = (
            tracking is None
            or start < int(tracking["observed_start_frame"])
            or stop > int(tracking["observed_stop_frame"])
        )
        missing_mouse = agent not in observed_mice or target not in observed_mice
        triplet_absent_from_mask = canonical not in allowed
        duplicate_key = (lab_id, video_id, agent, target, action, start, stop)
        duplicate_interval = duplicate_key in seen
        seen.add(duplicate_key)
        quarantined = invalid_interval or out_of_bounds or missing_mouse or triplet_absent_from_mask
        if quarantined:
            quarantine_counts[video_key] += 1

        audited_row = {
            **row,
            "mask_target_id": mask_target,
            "invalid_interval": invalid_interval,
            "out_of_tracking_bounds": out_of_bounds,
            "missing_mouse": missing_mouse,
            "triplet_absent_from_mask": triplet_absent_from_mask,
            "duplicate_interval": duplicate_interval,
            "quarantined": quarantined,
        }
        audited.append(audited_row)
        if invalid_interval or out_of_bounds or missing_mouse or triplet_absent_from_mask or duplicate_interval:
            anomalies.append(audited_row)
        if not quarantined:
            positive_intervals[(lab_id, video_id, agent, mask_target, action)].append((start, stop))

    positive_counts: defaultdict[tuple[str, int], int] = defaultdict(int)
    for (lab_id, video_id, _agent, _target, _action), intervals in positive_intervals.items():
        positive_counts[(lab_id, video_id)] += _union_length(intervals)
    return audited, anomalies, dict(annotation_counts), dict(quarantine_counts), dict(positive_counts)


def _mouse_metadata(row: dict[str, Any]) -> list[dict[str, object]]:
    mice: list[dict[str, object]] = []
    for mouse in MOUSE_SLOTS:
        values = {field: row.get(f"{mouse}_{field}") for field in MOUSE_METADATA_FIELDS}
        if any(value is not None for value in values.values()):
            mice.append({"slot": mouse, **values})
    return mice


def _json_list(raw: object, *, field: str) -> list[str]:
    if raw is None:
        return []
    decoded: object = json.loads(str(raw))
    if not isinstance(decoded, list) or not all(isinstance(item, str) for item in decoded):
        raise ValueError(f"{field} must be a JSON list of strings")
    return sorted(set(decoded))


def _label_state(*, split: str, lab_id: str, has_mask: bool, annotation_exists: bool) -> str:
    if split == "test":
        return "test"
    if not has_mask and lab_id.startswith("MABe22"):
        return "unlabeled_mabe22"
    if not has_mask:
        return "unlabeled_missing_mask"
    if not annotation_exists:
        return "ambiguous_missing_annotation"
    return "labeled"


def _build_video_manifest(
    metadata_rows: Sequence[dict[str, Any]],
    *,
    dataset_root: Path,
    tracking_paths: dict[tuple[str, str, int], Path],
    annotation_paths: dict[tuple[str, int], Path],
    tracking_stats: dict[str, dict[str, Any]],
    parsed_masks: dict[tuple[str, str, int], ParsedBehaviorMask],
    annotation_counts: dict[tuple[str, int], int],
    quarantine_counts: dict[tuple[str, int], int],
    positive_counts: dict[tuple[str, int], int],
) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for metadata_row in metadata_rows:
        split = str(metadata_row["split"])
        lab_id = str(metadata_row["lab_id"])
        video_id = int(metadata_row["video_id"])
        key = (split, lab_id, video_id)
        video_key = (lab_id, video_id)
        tracking_path = tracking_paths.get(key)
        annotation_path = annotation_paths.get(video_key) if split == "train" else None
        stats = tracking_stats.get(_path_key(tracking_path)) if tracking_path else None
        raw_mask = metadata_row.get("behaviors_labeled")
        mask = parsed_masks.get(key)
        declared_body_parts = _json_list(metadata_row.get("body_parts_tracked"), field="body_parts_tracked")
        observed_body_parts = list(stats.get("body_parts_observed", [])) if stats else []
        skeleton_source = "|".join(observed_body_parts or declared_body_parts)
        skeleton_id = hashlib.sha256(skeleton_source.encode()).hexdigest()[:16]
        mouse_ids = list(stats.get("mouse_ids", [])) if stats else []
        fps = metadata_row.get("frames_per_second")
        duration = metadata_row.get("video_duration_sec")
        declared_frames = round(float(fps) * float(duration)) if fps is not None and duration is not None else None
        state = _label_state(
            split=split,
            lab_id=lab_id,
            has_mask=raw_mask is not None,
            annotation_exists=annotation_path is not None,
        )
        manifest.append(
            {
                "split": split,
                "lab_id": lab_id,
                "video_id": video_id,
                "frames_per_second": fps,
                "video_duration_sec": duration,
                "declared_frame_count": declared_frames,
                "observed_start_frame": stats.get("observed_start_frame") if stats else None,
                "observed_stop_frame": stats.get("observed_stop_frame") if stats else None,
                "observed_frame_count": stats.get("observed_frame_count") if stats else None,
                "tracking_row_count": stats.get("tracking_row_count") if stats else None,
                "video_width_pix": metadata_row.get("video_width_pix"),
                "video_height_pix": metadata_row.get("video_height_pix"),
                "arena_width_cm": metadata_row.get("arena_width_cm"),
                "arena_height_cm": metadata_row.get("arena_height_cm"),
                "arena_shape": metadata_row.get("arena_shape"),
                "arena_type": metadata_row.get("arena_type"),
                "pix_per_cm_approx": metadata_row.get("pix_per_cm_approx"),
                "tracking_method": metadata_row.get("tracking_method"),
                "body_parts_tracked_raw": metadata_row.get("body_parts_tracked"),
                "body_parts_declared": declared_body_parts,
                "body_parts_observed": observed_body_parts,
                "skeleton_id": skeleton_id,
                "mouse_ids": mouse_ids,
                "mouse_count": len(mouse_ids),
                "mouse_metadata_json": json.dumps(_mouse_metadata(metadata_row), sort_keys=True, separators=(",", ":")),
                "tracking_path": _relative(tracking_path, dataset_root) if tracking_path else None,
                "tracking_exists": tracking_path is not None,
                "annotation_path": _relative(annotation_path, dataset_root) if annotation_path else None,
                "annotation_exists": annotation_path is not None,
                "behaviors_labeled_raw": raw_mask,
                "has_behavior_mask": raw_mask is not None,
                "parsed_triplet_count": len(mask.triplets) if mask else 0,
                "annotation_count": annotation_counts.get(video_key, 0),
                "positive_frame_count": positive_counts.get(video_key, 0),
                "label_state": state,
                "supervised_eligible": state == "labeled",
                "quarantined_annotation_count": quarantine_counts.get(video_key, 0),
                "missing_frame_rate": stats.get("missing_frame_rate") if stats else None,
                "missing_coordinate_rate": stats.get("missing_coordinate_rate") if stats else None,
                "non_finite_coordinate_rate": stats.get("non_finite_coordinate_rate") if stats else None,
                "extreme_jump_rate": stats.get("extreme_jump_rate") if stats else None,
                "tracking_order_violation_count": stats.get("tracking_order_violation_count") if stats else None,
            }
        )
    return manifest


def _statistics_rows(
    videos: Sequence[dict[str, Any]],
    annotations: Sequence[dict[str, Any]],
) -> list[dict[str, object]]:
    video_lookup = {(str(row["lab_id"]), int(row["video_id"])): row for row in videos}
    groups: defaultdict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"videos": set(), "durations": [], "positive_frames": 0, "annotations": 0}
    )

    def add_video(dimension: str, key: str, video: tuple[str, int]) -> None:
        groups[(dimension, key)]["videos"].add(video)

    for row in videos:
        video = (str(row["lab_id"]), int(row["video_id"]))
        add_video("lab", video[0], video)
        add_video("video", f"{video[0]}/{video[1]}", video)
        add_video("fps", str(row["frames_per_second"]), video)
        add_video("skeleton", str(row["skeleton_id"]), video)
        add_video("tracker", str(row["tracking_method"]), video)
        add_video("mouse_count", str(row["mouse_count"]), video)

    for row in annotations:
        start, stop = int(row["start_frame"]), int(row["stop_frame"])
        if start >= stop:
            continue
        duration = stop - start
        video = (str(row["lab_id"]), int(row["video_id"]))
        metadata_row = video_lookup[video]
        keys = {
            "lab": video[0],
            "video": f"{video[0]}/{video[1]}",
            "action": str(row["action"]),
            "directed_pair": f"{row['agent_id']}->{row['target_id']}",
            "fps": str(metadata_row["frames_per_second"]),
            "skeleton": str(metadata_row["skeleton_id"]),
            "tracker": str(metadata_row["tracking_method"]),
            "mouse_count": str(metadata_row["mouse_count"]),
        }
        for dimension, key in keys.items():
            group = groups[(dimension, key)]
            group["videos"].add(video)
            group["durations"].append(duration)
            group["positive_frames"] += duration
            group["annotations"] += 1

    results: list[dict[str, object]] = []
    for (dimension, key), group in sorted(groups.items()):
        durations = np.asarray(group["durations"], dtype=np.float64)
        quantiles = np.quantile(durations, [0.25, 0.5, 0.75, 0.95, 0.99]).tolist() if len(durations) else [None] * 5
        results.append(
            {
                "dimension": dimension,
                "key": key,
                "video_count": len(group["videos"]),
                "annotation_count": group["annotations"],
                "positive_frame_count": group["positive_frames"],
                "duration_q25_frames": quantiles[0],
                "duration_q50_frames": quantiles[1],
                "duration_q75_frames": quantiles[2],
                "duration_q95_frames": quantiles[3],
                "duration_q99_frames": quantiles[4],
            }
        )
    return results


def _self_target_rows(
    triplet_rows: Sequence[dict[str, object]],
    annotations: Sequence[dict[str, Any]],
) -> list[dict[str, object]]:
    mask_counts: defaultdict[str, int] = defaultdict(int)
    annotation_counts: defaultdict[str, int] = defaultdict(int)
    invalid_counts: defaultdict[str, int] = defaultdict(int)
    for row in triplet_rows:
        if bool(row["is_self"]) and not bool(row["is_duplicate"]):
            mask_counts[str(row["action"])] += 1
    for row in annotations:
        if row["agent_id"] == row["target_id"]:
            annotation_counts[str(row["action"])] += 1
        elif str(row["action"]) in mask_counts:
            invalid_counts[str(row["action"])] += 1
    return [
        {
            "action": action,
            "mask_self_triplet_count": mask_counts[action],
            "annotation_self_row_count": annotation_counts[action],
            "annotation_nonself_row_count": invalid_counts[action],
            "mask_representation": "target_id=self",
            "annotation_representation": "agent_id_equals_target_id",
            "verified": annotation_counts[action] > 0 and invalid_counts[action] == 0,
        }
        for action in sorted(mask_counts)
    ]


def _frame(records: Sequence[dict[str, Any]], schema: pl.Schema | None = None) -> pl.DataFrame:
    if records:
        return pl.DataFrame(records, infer_schema_length=None)
    return pl.DataFrame(schema=schema or {})


def _write_parquet(frame: pl.DataFrame, path: Path, *, sort_by: Sequence[str] = ()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.sort(list(sort_by)) if sort_by and frame.height else frame
    output.write_parquet(path, compression="zstd", statistics=True)


def _artifact_sidecar(
    path: Path,
    *,
    source_fingerprint: str,
    row_count: int | None,
    schema_hash: str | None,
) -> None:
    try:
        code_version = metadata.version("mabe-behavior")
    except metadata.PackageNotFoundError:
        code_version = "0.1.0"
    sidecar = {
        "artifact": path.name,
        "artifact_sha256": sha256_file(path),
        "code_version": code_version,
        "created_at": datetime.now(UTC).isoformat(),
        "row_count": row_count,
        "schema_sha256": schema_hash,
        "source_data_fingerprint": source_fingerprint,
    }
    Path(f"{path}.meta.json").write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _markdown_report(report: dict[str, Any]) -> str:
    facts = report["facts"]
    anomalies = report["anomaly_counts"]
    label_states = report["label_state_counts"]
    lines = [
        "# MABe data audit",
        "",
        f"Source fingerprint: `{report['source_fingerprint']}`",
        "",
        "## Verified facts",
        "",
        "| Fact | Observed | Expected | Match |",
        "|---|---:|---:|:---:|",
    ]
    for key in sorted(facts):
        value = facts[key]
        expected = EXPECTED_FACTS.get(key)
        expected_text = expected if expected is not None else "n/a"
        matches = "yes" if expected is None or value == expected else "no"
        lines.append(f"| `{key}` | {value} | {expected_text} | {matches} |")
    lines.extend(["", "## Label states", "", "| State | Videos |", "|---|---:|"])
    for key, value in sorted(label_states.items()):
        lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "## Classified anomalies", "", "| Anomaly | Count |", "|---|---:|"])
    for key, value in sorted(anomalies.items()):
        lines.append(f"| `{key}` | {value} |")
    lines.extend(
        [
            "",
            "## Decisions",
            "",
            "- `PleasantMeerkat/1375833299` is quarantined as `ambiguous_missing_annotation`.",
            "- Invalid or out-of-bounds annotation rows are retained in anomaly tables and marked "
            "`quarantined`; they are never silently repaired.",
            "- Solo masks use `target_id=self`; annotation parquet represents the same event with "
            "`agent_id == target_id`.",
            "- Ground-truth overlaps are reported and retained. Prediction-time exclusivity is a "
            "later decoder concern.",
            "",
        ]
    )
    return "\n".join(lines)


def _data_dictionary() -> str:
    return """# Phase 2 data dictionary

## `manifests/videos.parquet`

One row per metadata video across train and visible test. Frame ranges are
half-open: `observed_start_frame` is inclusive and `observed_stop_frame` is
exclusive. `positive_frame_count` is the union of eligible intervals within
each canonical `(agent, target, action)` key. Coordinate quality rates use raw
tracking rows. `extreme_jump_rate` uses displacements greater than the audited
pixel-per-frame threshold. `supervised_eligible` is false for every unlabeled,
ambiguous, or test state; annotation rows marked `quarantined` must also be
excluded individually.

## `manifests/triplets.parquet`

One row per raw behavior-mask entry. Raw strings and the full raw mask are
retained beside canonical mouse/action IDs, repair reasons, duplicate status,
and the deduplicated canonical mask. Mask self-targets are stored as `self`.

## `manifests/source_files.parquet`

One row per immutable source CSV/parquet with relative path, physical schema,
row count, byte size, and exact SHA-256. The aggregate source fingerprint is a
hash over these records.

## Anomaly tables

- `annotation_anomalies.parquet`: every annotation row with any validation flag.
- `invalid_intervals.parquet`: the strict `start_frame >= stop_frame` subset.
- `overlapping_annotations.parquet`: interval pairs that overlap for the same
  directed agent-target pair; empty tables still retain their schema.
- `mask_repairs.parquet`: repaired or deduplicated raw mask entries.
- `malformed_masks.parquet`: unrecoverable masks (the audit fails if non-empty).
- `missing_annotations.parquet`: masked training videos without an annotation file.
- `self_target_representation.parquet`: per-action proof of mask versus annotation
  self-target conventions.

All suspicious source records are reported without rewriting source data.
"""


def run_data_audit(
    config: RunConfig,
    *,
    extreme_jump_pixels_per_frame: float = DEFAULT_EXTREME_JUMP_PIXELS_PER_FRAME,
) -> AuditOutputs:
    """Run the complete Phase 2 audit and write deterministic data artifacts."""
    dataset_root = config.paths.dataset_root
    config.require_train_csv()
    train_metadata = scan_metadata(dataset_root / "train.csv").collect().with_columns(pl.lit("train").alias("split"))
    test_metadata = scan_metadata(dataset_root / "test.csv").collect().with_columns(pl.lit("test").alias("split"))
    metadata_frame = pl.concat([train_metadata, test_metadata], how="vertical_relaxed").sort(
        ["split", "lab_id", "video_id"]
    )
    metadata_rows = metadata_frame.to_dicts()

    train_tracking = sorted((dataset_root / "train_tracking").rglob("*.parquet"))
    test_tracking = sorted((dataset_root / "test_tracking").rglob("*.parquet"))
    annotation_paths_list = sorted((dataset_root / "train_annotation").rglob("*.parquet"))
    tracking_paths: dict[tuple[str, str, int], Path] = {}
    duplicate_tracking_keys: list[tuple[str, str, int]] = []
    for split, paths in (("train", train_tracking), ("test", test_tracking)):
        for path in paths:
            tracking_key = (split, path.parent.name, int(path.stem))
            if tracking_key in tracking_paths:
                duplicate_tracking_keys.append(tracking_key)
            tracking_paths[tracking_key] = path
    annotation_paths: dict[tuple[str, int], Path] = {}
    duplicate_annotation_keys: list[tuple[str, int]] = []
    for path in annotation_paths_list:
        annotation_key = (path.parent.name, int(path.stem))
        if annotation_key in annotation_paths:
            duplicate_annotation_keys.append(annotation_key)
        annotation_paths[annotation_key] = path

    parsed_masks, triplet_rows, malformed_masks = _parse_metadata_masks(metadata_rows)
    tracking_stats_frame = _tracking_statistics(
        [*train_tracking, *test_tracking],
        extreme_jump_pixels_per_frame=extreme_jump_pixels_per_frame,
        cache_root=config.paths.cache_root / "phase2_tracking_stats",
    )
    tracking_stats: dict[str, dict[str, Any]] = {}
    tracking_by_video: dict[tuple[str, int], dict[str, Any]] = {}
    for row in tracking_stats_frame.to_dicts():
        row["mouse_ids"] = [f"mouse{number}" for number in row.pop("mouse_numbers")]
        stats_key = _path_key(str(row["tracking_source_path"]))
        tracking_stats[stats_key] = row
        source = Path(str(row["tracking_source_path"]))
        tracking_by_video[(source.parent.name, int(source.stem))] = row

    annotation_rows = _load_annotations(annotation_paths_list)
    audited_annotations, annotation_anomalies, annotation_counts, quarantine_counts, positive_counts = (
        audit_annotation_rows(
            annotation_rows,
            tracking_by_video=tracking_by_video,
            parsed_masks=parsed_masks,
        )
    )
    overlap_rows = _overlap_rows(audited_annotations)
    video_rows = _build_video_manifest(
        metadata_rows,
        dataset_root=dataset_root,
        tracking_paths=tracking_paths,
        annotation_paths=annotation_paths,
        tracking_stats=tracking_stats,
        parsed_masks=parsed_masks,
        annotation_counts=annotation_counts,
        quarantine_counts=quarantine_counts,
        positive_counts=positive_counts,
    )

    metadata_keys = {(str(row["split"]), str(row["lab_id"]), int(row["video_id"])) for row in metadata_rows}
    missing_tracking = sorted(metadata_keys - set(tracking_paths))
    extra_tracking = sorted(set(tracking_paths) - metadata_keys)
    train_metadata_keys = {
        (str(row["lab_id"]), int(row["video_id"]))
        for row in metadata_rows
        if row["split"] == "train"
    }
    annotation_without_metadata = sorted(set(annotation_paths) - train_metadata_keys)
    missing_annotations = [
        row for row in video_rows if row["label_state"] == "ambiguous_missing_annotation"
    ]

    train_triplets = [row for row in triplet_rows if row["split"] == "train" and not row["is_duplicate"]]
    actions = sorted({str(row["action"]) for row in train_triplets})
    self_actions = sorted({str(row["action"]) for row in train_triplets if row["is_self"]})
    pair_actions = sorted(set(actions) - set(self_actions))
    facts = {
        "train_metadata_rows": train_metadata.height,
        "test_metadata_rows": test_metadata.height,
        "train_tracking_files": len(train_tracking),
        "test_tracking_files": len(test_tracking),
        "annotation_files": len(annotation_paths_list),
        "annotation_rows": len(annotation_rows),
        "unlabeled_mabe22_rows": sum(
            row["split"] == "train"
            and str(row["lab_id"]).startswith("MABe22")
            and row["behaviors_labeled"] is None
            for row in metadata_rows
        ),
        "canonical_action_count": len(actions),
        "self_action_count": len(self_actions),
        "pair_action_count": len(pair_actions),
    }

    source_files_frame, source_fingerprint = build_source_inventory(dataset_root)
    videos_frame = _frame(video_rows).select(VIDEO_MANIFEST_COLUMNS)
    triplets_frame = _frame(triplet_rows).select(TRIPLET_MANIFEST_COLUMNS)
    anomalies_frame = _frame(annotation_anomalies)
    overlaps_frame = _frame(overlap_rows, OVERLAP_SCHEMA)
    repairs_frame = triplets_frame.filter(pl.col("was_repaired") | pl.col("is_duplicate"))
    malformed_frame = _frame(
        malformed_masks,
        pl.Schema(
            {
                "split": pl.String,
                "lab_id": pl.String,
                "video_id": pl.Int64,
                "raw_mask": pl.String,
                "error": pl.String,
            }
        ),
    )
    missing_annotations_frame = _frame(missing_annotations)
    invalid_intervals_frame = anomalies_frame.filter(pl.col("invalid_interval"))
    self_target_frame = _frame(_self_target_rows(triplet_rows, audited_annotations))
    statistics_frame = _frame(_statistics_rows(video_rows, audited_annotations))

    manifests_root = config.paths.artifact_root / "manifests"
    reports_root = config.paths.artifact_root / "reports"
    videos_path = manifests_root / "videos.parquet"
    triplets_path = manifests_root / "triplets.parquet"
    source_files_path = manifests_root / "source_files.parquet"
    artifact_frames = {
        videos_path: (videos_frame, ("split", "lab_id", "video_id")),
        triplets_path: (triplets_frame, ("split", "lab_id", "video_id", "entry_index")),
        source_files_path: (source_files_frame, ("relative_path",)),
        reports_root / "annotation_anomalies.parquet": (
            anomalies_frame,
            ("lab_id", "video_id", "annotation_row_id"),
        ),
        reports_root / "invalid_intervals.parquet": (
            invalid_intervals_frame,
            ("lab_id", "video_id", "annotation_row_id"),
        ),
        reports_root / "overlapping_annotations.parquet": (
            overlaps_frame,
            ("lab_id", "video_id", "agent_id", "target_id", "overlap_start_frame"),
        ),
        reports_root / "mask_repairs.parquet": (
            repairs_frame,
            ("split", "lab_id", "video_id", "entry_index"),
        ),
        reports_root / "malformed_masks.parquet": (malformed_frame, ("split", "lab_id", "video_id")),
        reports_root / "missing_annotations.parquet": (
            missing_annotations_frame,
            ("lab_id", "video_id"),
        ),
        reports_root / "self_target_representation.parquet": (self_target_frame, ("action",)),
        reports_root / "data_statistics.parquet": (statistics_frame, ("dimension", "key")),
    }
    for path, (frame, sort_by) in artifact_frames.items():
        _write_parquet(frame, path, sort_by=sort_by)

    source_tracking_rows = int(
        source_files_frame.filter(pl.col("kind") == "tracking")["row_count"].sum()
    )
    scanned_tracking_rows = int(tracking_stats_frame["tracking_row_count"].sum())
    source_annotation_rows = int(
        source_files_frame.filter(pl.col("kind") == "annotation")["row_count"].sum()
    )

    label_state_counts: dict[str, int] = defaultdict(int)
    for row in video_rows:
        label_state_counts[str(row["label_state"])] += 1
    anomaly_counts = {
        "malformed_masks": len(malformed_masks),
        "mask_repairs_or_duplicates": repairs_frame.height,
        "ambiguous_missing_annotations": len(missing_annotations),
        "invalid_intervals": sum(bool(row["invalid_interval"]) for row in audited_annotations),
        "out_of_tracking_bounds": sum(bool(row["out_of_tracking_bounds"]) for row in audited_annotations),
        "missing_mice": sum(bool(row["missing_mouse"]) for row in audited_annotations),
        "triplets_absent_from_mask": sum(bool(row["triplet_absent_from_mask"]) for row in audited_annotations),
        "duplicate_intervals": sum(bool(row["duplicate_interval"]) for row in audited_annotations),
        "same_action_overlaps": sum(bool(row["same_action"]) for row in overlap_rows),
        "different_action_overlaps": sum(not bool(row["same_action"]) for row in overlap_rows),
        "tracking_order_violations": int(
            tracking_stats_frame["tracking_order_violation_count"].sum()
        ),
        "videos_with_missing_frames": sum(
            float(row["missing_frame_rate"] or 0.0) > 0.0 for row in video_rows
        ),
        "videos_with_missing_coordinates": sum(
            float(row["missing_coordinate_rate"] or 0.0) > 0.0 for row in video_rows
        ),
        "videos_with_non_finite_coordinates": sum(
            float(row["non_finite_coordinate_rate"] or 0.0) > 0.0 for row in video_rows
        ),
        "videos_with_extreme_jumps": sum(
            float(row["extreme_jump_rate"] or 0.0) > 0.0 for row in video_rows
        ),
        "declared_observed_skeleton_mismatches": sum(
            row["body_parts_declared"] != row["body_parts_observed"] for row in video_rows
        ),
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "source_fingerprint": source_fingerprint,
        "facts": facts,
        "expected_facts": EXPECTED_FACTS,
        "fact_matches": {key: facts[key] == value for key, value in EXPECTED_FACTS.items()},
        "reconciliation": {
            "missing_tracking_keys": missing_tracking,
            "extra_tracking_keys": extra_tracking,
            "duplicate_tracking_keys": duplicate_tracking_keys,
            "annotation_without_metadata": annotation_without_metadata,
            "duplicate_annotation_keys": duplicate_annotation_keys,
            "row_counts": {
                "source_tracking_rows": source_tracking_rows,
                "scanned_tracking_rows": scanned_tracking_rows,
                "source_annotation_rows": source_annotation_rows,
                "scanned_annotation_rows": len(annotation_rows),
                "raw_mask_entries": len(triplet_rows),
                "triplet_manifest_rows": triplets_frame.height,
            },
        },
        "label_state_counts": dict(sorted(label_state_counts.items())),
        "canonical_actions": actions,
        "self_actions": self_actions,
        "pair_actions": pair_actions,
        "self_target_rule": {
            "mask": "target_id=self",
            "annotation": "agent_id_equals_target_id",
        },
        "anomaly_counts": anomaly_counts,
        "extreme_jump_pixels_per_frame": extreme_jump_pixels_per_frame,
        "quarantine_policy": {
            "video_states_excluded": [
                "unlabeled_mabe22",
                "unlabeled_missing_mask",
                "ambiguous_missing_annotation",
                "test",
            ],
            "annotation_flags_excluded": [
                "invalid_interval",
                "out_of_tracking_bounds",
                "missing_mouse",
                "triplet_absent_from_mask",
            ],
            "overlaps_retained": True,
        },
    }
    report_json_path = reports_root / "data_audit.json"
    report_markdown_path = reports_root / "data_audit.md"
    data_dictionary_path = reports_root / "data_dictionary.md"
    report_json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_markdown_path.write_text(_markdown_report(report), encoding="utf-8")
    data_dictionary_path.write_text(_data_dictionary(), encoding="utf-8")

    for path, (frame, _sort_by) in artifact_frames.items():
        _artifact_sidecar(
            path,
            source_fingerprint=source_fingerprint,
            row_count=frame.height,
            schema_hash=schema_fingerprint(frame),
        )
    for path in (report_json_path, report_markdown_path, data_dictionary_path):
        _artifact_sidecar(path, source_fingerprint=source_fingerprint, row_count=None, schema_hash=None)

    failed_facts = [key for key, expected in EXPECTED_FACTS.items() if facts[key] != expected]
    reconciliation_failures = {
        "missing_tracking": missing_tracking,
        "extra_tracking": extra_tracking,
        "duplicate_tracking": duplicate_tracking_keys,
        "annotation_without_metadata": annotation_without_metadata,
        "duplicate_annotations": duplicate_annotation_keys,
    }
    if failed_facts:
        raise AssertionError(f"expected dataset facts do not match: {failed_facts}")
    if any(reconciliation_failures.values()):
        raise AssertionError(f"source reconciliation failed: {reconciliation_failures}")
    if malformed_masks:
        raise BehaviorMaskError(f"unrecoverable behavior masks found: {len(malformed_masks)}")
    if source_tracking_rows != scanned_tracking_rows:
        raise AssertionError("tracking source row counts do not reconcile with the manifest scan")
    if source_annotation_rows != len(annotation_rows):
        raise AssertionError("annotation source row counts do not reconcile with the audit scan")
    if sum(label_state_counts.values()) != metadata_frame.height:
        raise AssertionError("not every metadata row received exactly one label state")
    if len(missing_annotations) != 1 or int(missing_annotations[0]["video_id"]) != 1_375_833_299:
        raise AssertionError("ambiguous missing-annotation video does not match the audited convention")

    return AuditOutputs(
        videos_path=videos_path,
        triplets_path=triplets_path,
        source_files_path=source_files_path,
        report_json_path=report_json_path,
        report_markdown_path=report_markdown_path,
        source_fingerprint=source_fingerprint,
        facts=facts,
    )
