"""Deterministic, bounded-memory, offline inference for Phase 12."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
import psutil
import xgboost as xgb

from mabe.audit import _build_video_manifest, _parse_metadata_masks
from mabe.decode import centered_smooth_array, decode_probabilities_deployment
from mabe.feature_stream import (
    iter_feature_windows,
    selected_feature_frame,
    selected_feature_names,
)
from mabe.features.common import load_feature_config, round_seconds_to_frames
from mabe.features.storage import load_category_registry
from mabe.fingerprints import sha256_file
from mabe.io import scan_metadata
from mabe.masks import parse_behaviors_labeled
from mabe.phase9_config import load_phase9_config
from mabe.phase9_pipeline import infer_temporal_video
from mabe.phase12_config import Phase12Config, phase12_config_fingerprint
from mabe.pose import load_pose_config
from mabe.probability_stacker import stacker_feature_matrix
from mabe.schemas import (
    SUBMISSION_COLUMNS,
    SUBMISSION_SCHEMA,
    TRIPLET_MANIFEST_COLUMNS,
    VIDEO_MANIFEST_COLUMNS,
)
from mabe.submission import safe_fallback_submission, validate_submission
from mabe.temporal_data import SequenceArtifact, build_sequence_artifact
from mabe.xgb_config import load_xgboost_config
from mabe.xgb_model import load_fold_model

PROBABILITY_KEYS = (
    "lab_id",
    "video_id",
    "video_frame",
    "agent_id",
    "target_id",
    "action",
)


@dataclass(frozen=True, slots=True)
class InferenceManifests:
    videos_path: Path
    triplets_path: Path
    videos: pl.DataFrame
    triplets: pl.DataFrame


@dataclass(frozen=True, slots=True)
class InferenceOutputs:
    submission_path: Path
    report_path: Path
    videos_path: Path
    triplets_path: Path
    video_count: int
    interval_count: int
    runtime_seconds: float
    projected_seconds_200_videos: float
    headroom_fraction: float


@dataclass(frozen=True, slots=True)
class VideoInferenceResult:
    """One video's decoded output plus non-fatal stage diagnostics."""

    decoded: pl.DataFrame
    fallback: pl.DataFrame
    inference_route: str
    stage_seconds: dict[str, float]
    failures: tuple[dict[str, str], ...]


def _dataset_roots_below(input_root: Path) -> list[Path]:
    """Return valid dataset roots below either legacy or nested Kaggle mounts."""
    if not input_root.is_dir():
        return []
    return sorted(
        {
            test_csv.parent.resolve()
            for test_csv in input_root.rglob("test.csv")
            if _valid_dataset_root(test_csv.parent)
        },
        key=lambda path: ("mabe" not in path.name.lower(), path.as_posix()),
    )


def _valid_dataset_root(path: Path) -> bool:
    return path.is_dir() and (path / "test.csv").is_file() and (path / "test_tracking").is_dir()


def _dataset_root_evidence(path: Path) -> tuple[int, int]:
    """Return metadata and tracking counts used to distinguish public/private mounts."""
    try:
        with (path / "test.csv").open("rb") as stream:
            metadata_rows = max(sum(1 for _ in stream) - 1, 0)
        tracking_files = sum(1 for _ in (path / "test_tracking").rglob("*.parquet"))
    except OSError:
        return (0, 0)
    return (metadata_rows, tracking_files)


def discover_dataset_root(explicit: str | Path | None = None) -> Path:
    """Discover a Kaggle or local dataset without assuming the public-test row count."""
    if explicit is not None:
        selected = Path(explicit).expanduser().resolve()
        if not _valid_dataset_root(selected):
            raise FileNotFoundError(f"dataset root must contain test.csv and test_tracking: {selected}")
        return selected
    candidates: list[Path] = []
    environment = os.environ.get("MABE_DATASET_ROOT")
    if environment:
        candidates.append(Path(environment).expanduser())
    candidates.extend((Path.cwd(), Path.cwd() / "MABe-mouse-behavior-detection"))
    kaggle_input = Path("/kaggle/input")
    # Kaggle's current mounts may include ownership/type components, for
    # example /kaggle/input/competitions/<competition-slug>/test.csv.
    candidates.extend(_dataset_roots_below(kaggle_input))
    valid = {path.resolve() for path in candidates if _valid_dataset_root(path)}
    if not valid:
        raise FileNotFoundError(
            "could not discover the competition dataset; pass --dataset-root or set MABE_DATASET_ROOT"
        )
    # A code-competition rerun can expose both the one-video public mount and
    # the much larger private replacement below /kaggle/input.  Selecting by
    # path name can silently run the public video and score 0.0 because none of
    # its video IDs exist in the private solution.  The private mount has the
    # strongest direct evidence: more test metadata rows and tracking files.
    return max(
        valid,
        key=lambda path: (
            *_dataset_root_evidence(path),
            "mabe" in path.name.lower(),
            path.as_posix(),
        ),
    )


def _path_key(path: str | Path) -> str:
    return str(Path(path).resolve()).casefold()


def _tracking_statistics(paths: tuple[Path, ...]) -> dict[str, dict[str, Any]]:
    """Collect inference facts one file at a time so hidden scale stays bounded.

    Test tracking files come from multiple laboratories and need not have
    identical physical parquet types.  Scanning the entire hidden collection as
    one lazy frame lets a single schema anomaly abort every video and can make
    Polars retain metadata for hundreds of files.  Per-file scans bound memory
    and turn an unreadable file into one quarantined video.
    """
    records: dict[str, dict[str, Any]] = {}
    total = len(paths)
    for position, path in enumerate(paths, start=1):
        resolved = path.resolve()
        try:
            frame = (
                pl.scan_parquet(resolved, hive_partitioning=False)
                .select(
                    pl.col("video_frame").cast(pl.Int64),
                    pl.col("mouse_id").cast(pl.String),
                    pl.col("bodypart").cast(pl.String),
                )
                .select(
                    pl.len().alias("tracking_row_count"),
                    pl.col("video_frame").min().alias("observed_start_frame"),
                    (pl.col("video_frame").max() + 1).alias("observed_stop_frame"),
                    pl.col("video_frame").n_unique().alias("observed_frame_count"),
                    pl.col("mouse_id").drop_nulls().unique().sort().implode().alias("raw_mouse_ids"),
                    pl.col("bodypart")
                    .drop_nulls()
                    .unique()
                    .sort()
                    .implode()
                    .alias("body_parts_observed"),
                )
                .with_columns(
                    (
                        1
                        - pl.col("observed_frame_count")
                        / (pl.col("observed_stop_frame") - pl.col("observed_start_frame"))
                    ).alias("missing_frame_rate")
                )
                .collect(engine="streaming")
            )
            row = frame.row(0, named=True)
            if int(row["tracking_row_count"]) <= 0 or row["observed_start_frame"] is None:
                raise ValueError("tracking parquet contains no rows")
            mouse_ids = sorted({_raw_mouse_id(value) for value in row.pop("raw_mouse_ids")})
            if not mouse_ids:
                raise ValueError("tracking parquet contains no mouse IDs")
            row.update(
                {
                    "tracking_source_path": str(resolved),
                    "mouse_ids": mouse_ids,
                    "missing_coordinate_rate": None,
                    "non_finite_coordinate_rate": None,
                    "extreme_jump_rate": None,
                    "tracking_order_violation_count": None,
                }
            )
            records[_path_key(resolved)] = row
        except Exception as error:
            print(
                "Phase 12 manifest: quarantined unreadable tracking file "
                f"{path.as_posix()} ({type(error).__name__}: {str(error)[:300]})",
                flush=True,
            )
        if position % 25 == 0 or position == total:
            print(
                f"Phase 12 manifest: scanned {position}/{total} tracking files; "
                f"usable={len(records)}",
                flush=True,
            )
            gc.collect()
    return records


def build_test_manifests_from_sources(
    *,
    metadata_path: Path,
    tracking_root: Path,
    dataset_root: Path,
    output_root: Path,
    allowed_labs: set[str] | frozenset[str] | None = None,
) -> InferenceManifests:
    """Build test-style manifests while quarantining individual bad videos."""
    metadata = scan_metadata(metadata_path).collect().with_columns(pl.lit("test").alias("split"))
    if allowed_labs is not None:
        metadata = metadata.filter(pl.col("lab_id").is_in(sorted(allowed_labs)))
    if metadata.is_empty():
        raise ValueError(f"{metadata_path.name} contains no eligible videos")

    input_metadata_rows = metadata.height
    unique_metadata = metadata.unique(
        subset=["lab_id", "video_id"],
        keep="first",
        maintain_order=True,
    ).sort("lab_id", "video_id")
    duplicate_metadata_rows = input_metadata_rows - unique_metadata.height
    metadata_rows = unique_metadata.to_dicts()
    parsed_masks, triplet_rows, malformed = _parse_metadata_masks(metadata_rows)
    malformed_keys = {
        ("test", str(row["lab_id"]), int(str(row["video_id"])))
        for row in malformed
    }
    if malformed_keys:
        metadata_rows = [
            row
            for row in metadata_rows
            if ("test", str(row["lab_id"]), int(str(row["video_id"]))) not in malformed_keys
        ]
        triplet_rows = [
            row
            for row in triplet_rows
            if ("test", str(row["lab_id"]), int(str(row["video_id"]))) not in malformed_keys
        ]

    metadata_keys = {
        ("test", str(row["lab_id"]), int(row["video_id"]))
        for row in metadata_rows
    }
    tracking_paths: dict[tuple[str, str, int], Path] = {}
    duplicate_tracking_routes = 0
    unparseable_tracking_names = 0
    for path in sorted(tracking_root.rglob("*.parquet")):
        try:
            video_id = int(path.stem)
        except ValueError:
            unparseable_tracking_names += 1
            continue
        key = ("test", path.parent.name, video_id)
        if key not in metadata_keys:
            continue
        if key in tracking_paths:
            duplicate_tracking_routes += 1
            continue
        tracking_paths[key] = path
    missing_tracking_rows = len(metadata_keys - set(tracking_paths))
    metadata_rows = [
        row
        for row in metadata_rows
        if ("test", str(row["lab_id"]), int(row["video_id"])) in tracking_paths
    ]

    stats = _tracking_statistics(tuple(tracking_paths[key] for key in sorted(tracking_paths)))
    failed_tracking_statistics = len(tracking_paths) - len(stats)
    usable_tracking_keys = {
        key for key, path in tracking_paths.items() if _path_key(path) in stats
    }
    metadata_rows = [
        row
        for row in metadata_rows
        if ("test", str(row["lab_id"]), int(row["video_id"])) in usable_tracking_keys
    ]

    video_rows: list[dict[str, Any]] = []
    malformed_video_metadata = 0
    for metadata_row in metadata_rows:
        try:
            video_rows.extend(
                _build_video_manifest(
                    [metadata_row],
                    dataset_root=dataset_root,
                    tracking_paths=tracking_paths,
                    annotation_paths={},
                    tracking_stats=stats,
                    parsed_masks=parsed_masks,
                    annotation_counts={},
                    quarantine_counts={},
                    positive_counts={},
                )
            )
        except Exception as error:
            malformed_video_metadata += 1
            print(
                "Phase 12 manifest: quarantined malformed video metadata "
                f"{metadata_row['lab_id']}/{metadata_row['video_id']} "
                f"({type(error).__name__}: {str(error)[:300]})",
                flush=True,
            )
    if not video_rows or not triplet_rows:
        raise ValueError("no hidden videos with readable tracking and behavior masks remain")

    videos = pl.DataFrame(video_rows, infer_schema_length=None).select(VIDEO_MANIFEST_COLUMNS)
    triplets = pl.DataFrame(triplet_rows, infer_schema_length=None).select(TRIPLET_MANIFEST_COLUMNS)
    usable_video_keys = videos.select("lab_id", "video_id")
    triplets = triplets.join(usable_video_keys, on=["lab_id", "video_id"], how="semi")

    # Keep the argument list compatible with Kaggle's competition image.  The
    # image predates Polars' ``empty_as_null`` keyword; every retained video is
    # guaranteed to have at least one observed mouse before this point.
    observed_mice = videos.select("lab_id", "video_id", "mouse_ids").explode("mouse_ids")
    valid_agents = triplets.join(
        observed_mice.rename({"mouse_ids": "agent_id"}),
        on=["lab_id", "video_id", "agent_id"],
        how="semi",
    )
    self_triplets = valid_agents.filter(pl.col("target_id") == "self")
    pair_triplets = valid_agents.filter(pl.col("target_id") != "self").join(
        observed_mice.rename({"mouse_ids": "target_id"}),
        on=["lab_id", "video_id", "target_id"],
        how="semi",
    )
    triplets = pl.concat([self_triplets, pair_triplets], how="vertical_relaxed").sort(
        "lab_id", "video_id", "entry_index"
    )
    pre_candidate_video_count = videos.height
    candidate_counts = (
        triplets.filter(~pl.col("is_duplicate"))
        .group_by("lab_id", "video_id")
        .len()
        .rename({"len": "usable_triplet_count"})
    )
    videos_with_candidates = videos.drop("parsed_triplet_count").join(
        candidate_counts, on=["lab_id", "video_id"], how="inner"
    )
    no_usable_triplet_rows = pre_candidate_video_count - videos_with_candidates.height
    invalid_geometry_rows = videos_with_candidates.filter(
        (pl.col("observed_frame_count") <= 0)
        | pl.col("observed_frame_count").is_null()
        | (pl.col("frames_per_second") <= 0)
        | pl.col("frames_per_second").is_null()
    ).height
    videos = (
        videos_with_candidates
        .rename({"usable_triplet_count": "parsed_triplet_count"})
        .filter(
            (pl.col("parsed_triplet_count") > 0)
            & (pl.col("observed_frame_count") > 0)
            & (pl.col("frames_per_second") > 0)
        )
        .select(VIDEO_MANIFEST_COLUMNS)
        .sort("lab_id", "video_id")
    )
    triplets = triplets.join(
        videos.select("lab_id", "video_id"),
        on=["lab_id", "video_id"],
        how="semi",
    ).select(TRIPLET_MANIFEST_COLUMNS)
    if videos.is_empty() or triplets.filter(~pl.col("is_duplicate")).is_empty():
        raise ValueError("no valid hidden videos/triplets remain after quarantine")

    output_root.mkdir(parents=True, exist_ok=True)
    videos_path = output_root / "test_videos.parquet"
    triplets_path = output_root / "test_triplets.parquet"
    videos.write_parquet(videos_path, compression="zstd", statistics=True)
    triplets.write_parquet(triplets_path, compression="zstd", statistics=True)
    report = {
        "schema_version": 1,
        "metadata_path": str(metadata_path),
        "tracking_root": str(tracking_root),
        "input_metadata_rows": input_metadata_rows,
        "output_video_rows": videos.height,
        "output_triplet_rows": triplets.height,
        "output_lab_count": videos["lab_id"].n_unique(),
        "quarantine_counts": {
            "duplicate_metadata_rows": duplicate_metadata_rows,
            "malformed_behavior_masks": len(malformed_keys),
            "missing_tracking_rows": missing_tracking_rows,
            "failed_tracking_statistics": failed_tracking_statistics,
            "malformed_video_metadata": malformed_video_metadata,
            "duplicate_tracking_routes": duplicate_tracking_routes,
            "unparseable_tracking_names": unparseable_tracking_names,
            "no_usable_triplet_rows": no_usable_triplet_rows,
            "invalid_geometry_rows": invalid_geometry_rows,
        },
    }
    (output_root / "manifest_build_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Phase 12 manifest: retained {videos.height}/{input_metadata_rows} videos "
        f"across {report['output_lab_count']} labs",
        flush=True,
    )
    return InferenceManifests(videos_path, triplets_path, videos, triplets)


def build_test_manifests(dataset_root: Path, output_root: Path) -> InferenceManifests:
    """Build hidden-test manifests dynamically, with exact behavior-mask semantics."""
    return build_test_manifests_from_sources(
        metadata_path=dataset_root / "test.csv",
        tracking_root=dataset_root / "test_tracking",
        dataset_root=dataset_root,
        output_root=output_root,
    )


def compose_crossfit_decoder_parameters(config: Phase12Config) -> pl.DataFrame:
    """Combine Phase 11 probability-transform and interval-decoder lineage."""
    source = pl.read_parquet(config.asset_path(config.assets.decoder_parameters))
    selection = pl.read_parquet(config.asset_path(config.assets.probability_decoder_selection))
    expected_folds = set(config.ensemble.folds)
    if set(int(value) for value in selection["held_out_fold"]) != expected_folds:
        raise ValueError("probability-decoder selection does not cover every configured fold")
    if selection["used_held_out_labels"].any():
        raise ValueError("probability-decoder selection used held-out labels")
    transform = selection.select(
        "held_out_fold",
        "additional_smoothing_seconds",
        "hysteresis_delta",
        pl.col("candidate").alias("probability_decoder_candidate"),
        pl.col("calibration_folds").alias("probability_calibration_folds"),
        pl.col("used_held_out_labels").alias("probability_used_held_out_labels"),
    )
    return source.drop("additional_smoothing_seconds", "hysteresis_delta").join(
        transform,
        on="held_out_fold",
        how="inner",
        validate="m:1",
    )


def freeze_deployment_parameters(config: Phase12Config) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Collapse five cross-fitted calibrations into deterministic hidden-test parameters."""
    source_weights = pl.read_parquet(config.asset_path(config.assets.blend_weights)).filter(
        pl.col("strategy") == config.ensemble.weight_strategy
    )
    expected_folds = set(config.ensemble.folds)
    for action_group in source_weights.partition_by("action"):
        if set(int(value) for value in action_group["held_out_fold"]) != expected_folds:
            raise ValueError(f"incomplete cross-fitted weights for {action_group['action'][0]}")
    weights = (
        source_weights.group_by("action")
        .agg(
            pl.col("gbdt_weight").mean(),
            pl.col("temporal_phase9_weight").mean(),
            pl.col("held_out_fold").n_unique().alias("source_fold_count"),
        )
        .with_columns(
            pl.lit("mean_of_cross_fitted_action_weights").alias("aggregation"),
            pl.lit(False).alias("used_test_labels"),
        )
        .sort("action")
    )
    source_decoder = compose_crossfit_decoder_parameters(config)
    numeric = (
        "threshold",
        "baseline_smoothing_seconds",
        "additional_smoothing_seconds",
        "hysteresis_delta",
        "gap_fill_seconds",
        "minimum_run_seconds",
    )
    for group in source_decoder.partition_by("lab_id", "action"):
        if set(int(value) for value in group["held_out_fold"]) != expected_folds:
            raise ValueError(
                f"incomplete cross-fitted decoder parameters for {group['lab_id'][0]}/{group['action'][0]}"
            )
    decoder = (
        source_decoder.group_by("lab_id", "action")
        .agg(
            *(pl.col(column).median().alias(column) for column in numeric),
            pl.col("held_out_fold").n_unique().alias("source_fold_count"),
        )
        .with_columns(
            pl.lit("median_of_cross_fitted_parameters").alias("postprocess_source"),
            pl.lit(False).alias("used_test_labels"),
        )
        .sort("lab_id", "action")
    )
    weight_path = config.asset_path(config.assets.deployment_weights)
    decoder_path = config.asset_path(config.assets.deployment_decoder_parameters)
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    weights.write_parquet(weight_path, compression="zstd", statistics=True)
    decoder.write_parquet(decoder_path, compression="zstd", statistics=True)
    return weights, decoder


def _model_path(value: object, project_root: Path) -> Path | None:
    if value is None:
        return None
    path = Path(str(value))
    if path.is_file():
        return path
    parts = path.parts
    lowered = [part.lower() for part in parts]
    if "artifacts" in lowered:
        candidate = project_root.joinpath(*parts[lowered.index("artifacts") :])
    else:
        candidate = project_root / path
    if not candidate.is_file():
        raise FileNotFoundError(f"missing packaged XGBoost model: {candidate}")
    return candidate


def _predict_gbdt_video(
    row: dict[str, object],
    candidates: pl.DataFrame,
    registry: pl.DataFrame,
    *,
    config: Phase12Config,
    dataset_root: Path,
    folds: tuple[int, ...] | None = None,
) -> pl.DataFrame:
    lab = str(row["lab_id"])
    video_id = int(cast(int, row["video_id"]))
    registry_lab = registry.filter(pl.col("lab_id") == lab)
    if registry_lab.is_empty():
        raise KeyError(f"no trained XGBoost routes exist for hidden lab {lab!r}")
    supported = candidates.join(registry_lab.select("action").unique(), on="action", how="semi")
    if supported.is_empty():
        raise ValueError(f"no supported candidate action exists for {lab}/{video_id}")
    routes = frozenset("solo" if bool(value) else "pair" for value in supported["is_self"].unique())
    pose_config = load_pose_config(config.asset_path(config.assets.pose_config))
    feature_config = load_feature_config(config.asset_path(config.assets.feature_config))
    xgb_config = load_xgboost_config(config.asset_path(config.assets.xgboost_config))
    categories = load_category_registry(config.asset_path(config.assets.category_registry))
    model_cache: dict[str, xgb.XGBClassifier] = {}
    outputs: list[pl.DataFrame] = []
    selected_folds = config.ensemble.folds if folds is None else folds
    for window in iter_feature_windows(
        row,
        dataset_root=dataset_root,
        pose_config=pose_config,
        feature_config=feature_config,
        model_feature_config=xgb_config.features,
        categories=categories,
        routes=routes,
    ):
        for route, table, groups in (
            ("solo", window.solo, xgb_config.features.solo_groups),
            ("pair", window.pair, xgb_config.features.pair_groups),
        ):
            if table is None:
                continue
            names = selected_feature_names(table, groups)
            features = selected_feature_frame(table, names)
            if route == "solo":
                features = features.with_columns(pl.lit("self").alias("target_id"))
            route_candidates = supported.filter(pl.col("is_self") if route == "solo" else ~pl.col("is_self"))
            for action_group in route_candidates.partition_by("action", maintain_order=True):
                action = str(action_group["action"][0])
                action_features = features.join(
                    action_group.select("agent_id", "target_id").unique(),
                    on=["agent_id", "target_id"],
                    how="semi",
                ).sort("agent_id", "target_id", "video_frame")
                fold_probabilities: list[np.ndarray[Any, np.dtype[np.float32]]] = []
                action_registry = registry_lab.filter(pl.col("action") == action)
                for fold in selected_folds:
                    entries = action_registry.filter(pl.col("held_out_fold") == fold)
                    if entries.height != 1:
                        raise ValueError(f"expected one XGBoost route for {lab}/{action}/fold_{fold}")
                    path = _model_path(entries.row(0, named=True).get("model_path"), config.project_root)
                    if path is None:
                        fold_probabilities.append(np.zeros(action_features.height, dtype=np.float32))
                        continue
                    key = str(path.resolve())
                    if key not in model_cache:
                        model_cache[key] = load_fold_model(path)
                    values = action_features.select(names).to_numpy().astype(np.float32, copy=False)
                    fold_probabilities.append(model_cache[key].predict_proba(values)[:, 1].astype(np.float32))
                mean_probability = np.mean(np.stack(fold_probabilities, axis=0), axis=0, dtype=np.float32)
                predicted = action_features.select("video_id", "video_frame", "agent_id", "target_id").with_columns(
                    pl.lit(action).alias("action"),
                    pl.Series("probability", mean_probability, dtype=pl.Float32),
                )
                radius = round_seconds_to_frames(
                    xgb_config.decoder.smoothing_seconds,
                    float(cast(float, row["frames_per_second"])),
                )
                smoothed: list[pl.DataFrame] = []
                for pair in predicted.partition_by("agent_id", "target_id", maintain_order=True):
                    values = centered_smooth_array(
                        pair["video_frame"].to_numpy().astype(np.int64, copy=False),
                        pair["probability"].to_numpy().astype(np.float32, copy=False),
                        radius_frames=radius,
                    )
                    smoothed.append(pair.with_columns(pl.Series("probability", values)))
                outputs.append(
                    pl.concat(smoothed, how="vertical")
                    .filter(pl.col("video_frame").is_between(window.core_start, window.core_stop, closed="left"))
                    .with_columns(pl.lit(lab).alias("lab_id"), pl.lit(1).alias("loss_mask"))
                )
    del model_cache
    gc.collect()
    return (
        pl.concat(outputs, how="vertical").select(*PROBABILITY_KEYS, "loss_mask", "probability").sort(*PROBABILITY_KEYS)
    )


def _temporal_cache_fingerprint(row: dict[str, object], dataset_root: Path, config: Phase12Config) -> str:
    tracking = Path(str(row["tracking_path"]))
    if not tracking.is_absolute():
        tracking = dataset_root / tracking
    payload = {
        "configuration_sha256": phase12_config_fingerprint(config.config_path),
        "tracking_path": str(tracking.resolve()),
        "tracking_size": tracking.stat().st_size,
        "tracking_mtime_ns": tracking.stat().st_mtime_ns,
        "behavior_mask": row.get("behaviors_labeled_raw"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _predict_temporal_video(
    row: dict[str, object],
    *,
    config: Phase12Config,
    dataset_root: Path,
    manifests: InferenceManifests,
    cache_root: Path,
    folds: tuple[int, ...] | None = None,
) -> pl.DataFrame:
    lab = str(row["lab_id"])
    video_id = int(cast(int, row["video_id"]))
    lab_payload = json.loads(config.asset_path(config.assets.temporal_lab_registry).read_text(encoding="utf-8"))
    lab_to_index = cast(dict[str, int], lab_payload["lab_to_index"])
    if lab not in lab_to_index:
        raise KeyError(f"no temporal lab embedding exists for hidden lab {lab!r}")
    temporal_config_path = config.asset_path(config.assets.temporal_config)
    temporal_config = load_phase9_config(temporal_config_path)
    sequence = build_sequence_artifact(
        row,
        fold=-1,
        lab_index=int(lab_to_index[lab]),
        dataset_root=dataset_root,
        videos_path=manifests.videos_path,
        triplets_path=manifests.triplets_path,
        pose_config_path=config.asset_path(config.assets.pose_config),
        vocabulary_path=config.asset_path(config.assets.vocabulary),
        output_root=cache_root / "sequences",
        config=temporal_config,
        configuration_sha256=_temporal_cache_fingerprint(row, dataset_root, config),
        split="test",
    )
    artifact = SequenceArtifact(sequence.artifact_path)
    total: np.ndarray[Any, np.dtype[np.float32]] | None = None
    keys: tuple[tuple[str, str, str], ...] | None = None
    checkpoint_root = config.asset_path(config.assets.temporal_checkpoint_root)
    selected_folds = config.ensemble.folds if folds is None else folds
    for fold in selected_folds:
        prediction = infer_temporal_video(
            artifact,
            checkpoint_path=checkpoint_root / f"fold_{fold}" / "model.pt",
            temporal_config_path=temporal_config_path,
            vocabulary_path=config.asset_path(config.assets.vocabulary),
        )
        if keys is None:
            keys = prediction.keys
            total = prediction.probabilities.astype(np.float32)
        else:
            if prediction.keys != keys:
                raise AssertionError("temporal output keys changed between fold checkpoints")
            assert total is not None
            total += prediction.probabilities.astype(np.float32)
    assert keys is not None and total is not None
    total /= np.float32(len(selected_folds))
    original = np.asarray(artifact.array("original_frame_ids"), dtype=np.int64)
    mapping = np.asarray(artifact.array("original_to_resampled"), dtype=np.int32)
    records: list[pl.DataFrame] = []
    for column, (agent, target, action) in enumerate(keys):
        records.append(
            pl.DataFrame(
                {
                    "lab_id": pl.Series([lab] * original.size, dtype=pl.String),
                    "video_id": pl.Series(np.full(original.size, video_id, dtype=np.int64), dtype=pl.Int64),
                    "video_frame": pl.Series(original, dtype=pl.Int64),
                    "agent_id": pl.Series([agent] * original.size, dtype=pl.String),
                    "target_id": pl.Series([target] * original.size, dtype=pl.String),
                    "action": pl.Series([action] * original.size, dtype=pl.String),
                    "probability": pl.Series(total[mapping, column], dtype=pl.Float32),
                }
            )
        )
    del artifact, total
    gc.collect()
    return pl.concat(records, how="vertical").sort(*PROBABILITY_KEYS)


def blend_probabilities(gbdt: pl.DataFrame, temporal: pl.DataFrame, weights: pl.DataFrame) -> pl.DataFrame:
    """Align exact frame/triplet keys before convex probability arithmetic."""
    if gbdt.select(PROBABILITY_KEYS).is_duplicated().any():
        raise ValueError("GBDT inference produced duplicate probability keys")
    if temporal.select(PROBABILITY_KEYS).is_duplicated().any():
        raise ValueError("temporal inference produced duplicate probability keys")
    joined = gbdt.rename({"probability": "gbdt_probability"}).join(
        temporal.rename({"probability": "temporal_probability"}),
        on=list(PROBABILITY_KEYS),
        how="left",
        validate="1:1",
    )
    if joined["temporal_probability"].null_count():
        raise ValueError("temporal probabilities do not cover every supported GBDT candidate")
    joined = joined.join(
        weights.select("action", "gbdt_weight", "temporal_phase9_weight"),
        on="action",
        how="left",
        validate="m:1",
    )
    if joined["gbdt_weight"].null_count():
        raise ValueError("deployment weights do not cover every supported candidate action")
    return joined.with_columns(
        (
            pl.col("gbdt_weight") * pl.col("gbdt_probability")
            + pl.col("temporal_phase9_weight") * pl.col("temporal_probability")
        )
        .clip(0.0, 1.0)
        .cast(pl.Float32)
        .alias("probability")
    ).select(*PROBABILITY_KEYS, "loss_mask", "probability")


def stack_probabilities(
    gbdt: pl.DataFrame,
    temporal: pl.DataFrame,
    registry: pl.DataFrame,
    *,
    project_root: Path,
) -> pl.DataFrame:
    """Apply deployment probability stackers after exact base-key alignment."""
    joined = gbdt.rename({"probability": "gbdt_probability"}).join(
        temporal.rename({"probability": "temporal_probability"}),
        on=list(PROBABILITY_KEYS),
        how="left",
        validate="1:1",
    )
    if joined["temporal_probability"].null_count():
        raise ValueError("temporal probabilities do not cover the GBDT stacker rows")
    deployment = registry.filter(pl.col("held_out_fold") == -1)
    output: list[pl.DataFrame] = []
    cache: dict[str, xgb.XGBClassifier] = {}
    for action_frame in joined.partition_by("action", maintain_order=True):
        action = str(action_frame["action"][0])
        entry = deployment.filter(pl.col("action") == action)
        if entry.height != 1:
            raise ValueError(f"missing unique deployment stacker for action {action!r}")
        base = action_frame["gbdt_probability"].to_numpy().astype(
            np.float32, copy=False
        )
        neural = action_frame["temporal_probability"].to_numpy().astype(
            np.float32, copy=False
        )
        path = _model_path(entry.row(0, named=True).get("model_path"), project_root)
        if path is None:
            probability = 0.5 * (base + neural)
        else:
            key = str(path.resolve())
            if key not in cache:
                cache[key] = load_fold_model(path)
            probability = cache[key].predict_proba(
                stacker_feature_matrix(base, neural)
            )[:, 1].astype(np.float32)
        output.append(
            action_frame.with_columns(
                pl.Series("probability", probability, dtype=pl.Float32)
            ).select(*PROBABILITY_KEYS, "loss_mask", "probability")
        )
    return pl.concat(output, how="vertical").sort(*PROBABILITY_KEYS)


def selected_probability_model(config: Phase12Config) -> str:
    if config.ensemble.probability_model != "auto":
        return config.ensemble.probability_model
    if config.assets.final_model_registry is None:
        raise ValueError("automatic probability-model selection requires a final registry")
    payload = json.loads(
        config.asset_path(config.assets.final_model_registry).read_text(encoding="utf-8")
    )
    selected = str(payload.get("selected_probability_model", ""))
    if selected not in {"blend", "stacker"}:
        raise ValueError(f"unsupported selected probability model: {selected!r}")
    return selected


def _stage_failure(stage: str, error: Exception) -> dict[str, str]:
    message = str(error).replace("\n", " ").strip()
    return {
        "stage": stage,
        "exception_type": type(error).__name__,
        "message": message[:1_000],
    }


def _temporal_probability_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """Give temporal-only predictions the decoder's canonical probability schema."""
    return frame.with_columns(pl.lit(1, dtype=pl.UInt8).alias("loss_mask")).select(
        *PROBABILITY_KEYS,
        "loss_mask",
        "probability",
    )


def predict_decode_video_resilient(
    row: dict[str, object],
    candidates: pl.DataFrame,
    registry: pl.DataFrame,
    weights: pl.DataFrame,
    decoder: pl.DataFrame,
    *,
    config: Phase12Config,
    dataset_root: Path,
    manifests: InferenceManifests,
    cache_root: Path,
    folds: tuple[int, ...] | None = None,
    stacker_registry: pl.DataFrame | None = None,
) -> VideoInferenceResult:
    """Run both model families independently and degrade without aborting a video."""
    failures: list[dict[str, str]] = []
    timings = {"gbdt": 0.0, "temporal": 0.0, "blend_decode": 0.0}
    gbdt: pl.DataFrame | None = None
    temporal: pl.DataFrame | None = None

    stage_started = time.perf_counter()
    try:
        predicted = _predict_gbdt_video(
            row,
            candidates,
            registry,
            config=config,
            dataset_root=dataset_root,
            folds=folds,
        )
        if predicted.is_empty():
            raise ValueError("XGBoost produced no probability rows")
        gbdt = predicted
    except Exception as error:
        failures.append(_stage_failure("gbdt", error))
        _release_accelerator_memory()
    finally:
        timings["gbdt"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    try:
        predicted = _predict_temporal_video(
            row,
            config=config,
            dataset_root=dataset_root,
            manifests=manifests,
            cache_root=cache_root,
            folds=folds,
        )
        if predicted.is_empty():
            raise ValueError("temporal model produced no probability rows")
        temporal = predicted
    except Exception as error:
        failures.append(_stage_failure("temporal", error))
        _release_accelerator_memory()
    finally:
        timings["temporal"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    probability: pl.DataFrame | None = None
    route = "skipped"
    if gbdt is not None and temporal is not None:
        try:
            if selected_probability_model(config) == "stacker":
                if stacker_registry is None:
                    raise ValueError("stacker inference was selected without a registry")
                probability = stack_probabilities(
                    gbdt,
                    temporal,
                    stacker_registry,
                    project_root=config.project_root,
                )
                route = "stacker"
            else:
                probability = blend_probabilities(gbdt, temporal, weights)
                route = "blend"
        except Exception as error:
            failures.append(_stage_failure("blend", error))
            probability = gbdt
            route = "gbdt_only"
    elif gbdt is not None:
        probability = gbdt
        route = "gbdt_only"
    elif temporal is not None:
        probability = _temporal_probability_frame(temporal)
        route = "temporal_only"

    decoded = _empty_submission()
    fallback = pl.DataFrame()
    if probability is not None:
        fallback = _fallback_candidate(probability, decoder)
        try:
            video_id = int(cast(int, row["video_id"]))
            fps = {video_id: float(cast(float, row["frames_per_second"]))}
            decoded_pd = decode_probabilities_deployment(probability, decoder, fps_by_video=fps)
            decoded = pl.DataFrame(decoded_pd) if not decoded_pd.empty else _empty_submission()
            if not decoded.is_empty():
                decoded = decoded.select(SUBMISSION_COLUMNS).with_columns(pl.col("row_id").cast(pl.Int64))
        except Exception as error:
            failures.append(_stage_failure("decode", error))
            decoded = _empty_submission()
            route = "skipped"
    timings["blend_decode"] = time.perf_counter() - stage_started
    return VideoInferenceResult(
        decoded=decoded,
        fallback=fallback,
        inference_route=route,
        stage_seconds=timings,
        failures=tuple(failures),
    )


def _empty_submission() -> pl.DataFrame:
    return pl.DataFrame(schema=SUBMISSION_SCHEMA).select(SUBMISSION_COLUMNS)


def _empty_fallback_candidate() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "video_id": pl.Int64,
            "video_frame": pl.Int64,
            "agent_id": pl.String,
            "target_id": pl.String,
            "action": pl.String,
            "confidence": pl.Float64,
        }
    )


def _fallback_candidate(probabilities: pl.DataFrame, decoder: pl.DataFrame) -> pl.DataFrame:
    candidate = (
        probabilities.join(
            decoder.select("lab_id", "action", "threshold"),
            on=["lab_id", "action"],
            how="inner",
        )
        .with_columns((pl.col("probability") / pl.col("threshold")).alias("confidence"))
        .sort(
            "confidence",
            "video_id",
            "video_frame",
            "agent_id",
            "target_id",
            "action",
            descending=[True, False, False, False, False, False],
        )
        .head(1)
    )
    if candidate.is_empty():
        return _empty_fallback_candidate()
    return candidate.select("video_id", "video_frame", "agent_id", "target_id", "action", "confidence")


def _fallback_submission(candidate: pl.DataFrame) -> pl.DataFrame:
    row = candidate.row(0, named=True)
    frame = int(cast(int, row["video_frame"]))
    return pl.DataFrame(
        {
            "row_id": pl.Series([0], dtype=pl.Int64),
            "video_id": pl.Series([int(cast(int, row["video_id"]))], dtype=pl.Int64),
            "agent_id": pl.Series([str(row["agent_id"])], dtype=pl.String),
            "target_id": pl.Series([str(row["target_id"])], dtype=pl.String),
            "action": pl.Series([str(row["action"])], dtype=pl.String),
            "start_frame": pl.Series([frame], dtype=pl.Int64),
            "stop_frame": pl.Series([frame + 1], dtype=pl.Int64),
        }
    ).select(SUBMISSION_COLUMNS)


def _release_accelerator_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    gc.collect()


def _run_fingerprint(dataset_root: Path, config: Phase12Config) -> str:
    payload = (
        sha256_file(dataset_root / "test.csv")
        + phase12_config_fingerprint(config.config_path)
        + sha256_file(config.asset_path(config.assets.deployment_weights))
        + sha256_file(config.asset_path(config.assets.deployment_decoder_parameters))
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def assemble_resilient_submission(
    interval_parts: list[pl.DataFrame],
    fallback_parts: list[pl.DataFrame],
    manifests: InferenceManifests,
) -> tuple[pl.DataFrame, bool]:
    """Assemble successful videos and retain one valid fallback only if all are empty."""
    nonempty = [part for part in interval_parts if not part.is_empty()]
    if nonempty:
        submission = pl.concat(nonempty, how="vertical_relaxed").drop("row_id")
        return submission.with_row_index("row_id").select(SUBMISSION_COLUMNS), False
    available = [part for part in fallback_parts if not part.is_empty()]
    if available:
        best = (
            pl.concat(available, how="vertical_relaxed")
            .sort(
                "confidence",
                "video_id",
                "video_frame",
                "agent_id",
                "target_id",
                "action",
                descending=[True, False, False, False, False, False],
            )
            .head(1)
        )
        return _fallback_submission(best), True
    return safe_fallback_submission(manifests.videos, manifests.triplets, split="test"), True


def _raw_mouse_id(value: object) -> str:
    text = str(value)
    if text.lower().startswith("mouse"):
        return text.lower()
    return f"mouse{int(float(text))}"


def write_emergency_submission(dataset_root: Path, submission_path: Path) -> Path:
    """Write one mask-valid interval before expensive hidden inference begins."""
    metadata = pl.read_csv(dataset_root / "test.csv").select(
        pl.col("lab_id").cast(pl.String),
        pl.col("video_id").cast(pl.Int64),
        pl.col("behaviors_labeled").cast(pl.String),
    )
    tracking_root = dataset_root / "test_tracking"
    for row in metadata.sort("lab_id", "video_id").iter_rows(named=True):
        lab = str(row["lab_id"])
        video_id = int(cast(int, row["video_id"]))
        tracking_path = tracking_root / lab / f"{video_id}.parquet"
        if not tracking_path.is_file():
            matches = sorted(tracking_root.rglob(f"{video_id}.parquet"))
            if len(matches) != 1:
                continue
            tracking_path = matches[0]
        tracking = (
            pl.scan_parquet(tracking_path)
            .select(
                pl.col("video_frame").cast(pl.Int64),
                pl.col("mouse_id"),
            )
            .collect(engine="streaming")
        )
        if tracking.is_empty():
            continue
        start = int(cast(int, tracking["video_frame"].min()))
        observed_mice = {_raw_mouse_id(value) for value in tracking["mouse_id"].unique().to_list()}
        try:
            mask = parse_behaviors_labeled(str(row["behaviors_labeled"]))
        except Exception:
            continue
        for entry in mask.entries:
            triplet = entry.triplet
            if entry.is_duplicate or triplet.agent_id not in observed_mice:
                continue
            if triplet.target_id != "self" and triplet.target_id not in observed_mice:
                continue
            fallback = pl.DataFrame(
                {
                    "row_id": pl.Series([0], dtype=pl.Int64),
                    "video_id": pl.Series([video_id], dtype=pl.Int64),
                    "agent_id": pl.Series([triplet.agent_id], dtype=pl.String),
                    "target_id": pl.Series([triplet.target_id], dtype=pl.String),
                    "action": pl.Series([triplet.action], dtype=pl.String),
                    "start_frame": pl.Series([start], dtype=pl.Int64),
                    "stop_frame": pl.Series([start + 1], dtype=pl.Int64),
                }
            ).select(SUBMISSION_COLUMNS)
            submission_path.parent.mkdir(parents=True, exist_ok=True)
            fallback.write_csv(submission_path)
            return submission_path
    raise ValueError("could not construct an emergency submission from any hidden test video")


def recover_submission_after_failure(
    dataset_root: Path,
    work_root: Path,
    submission_path: Path,
) -> Path:
    """Salvage decoded checkpoint CSVs, retaining the prewritten fallback otherwise."""
    parts: list[pl.DataFrame] = []
    for path in sorted((work_root / "video_predictions").rglob("intervals.csv")):
        try:
            frame = pl.read_csv(path, schema_overrides=dict(SUBMISSION_SCHEMA)).select(SUBMISSION_COLUMNS)
        except Exception:
            continue
        valid = frame.filter(pl.col("start_frame") < pl.col("stop_frame"))
        if not valid.is_empty():
            parts.append(valid)
    if parts:
        recovered = (
            pl.concat(parts, how="vertical_relaxed")
            .drop("row_id")
            .unique(maintain_order=True)
            .sort("video_id", "start_frame", "agent_id", "target_id", "action", "stop_frame")
            .with_row_index("row_id")
            .select(SUBMISSION_COLUMNS)
        )
        submission_path.parent.mkdir(parents=True, exist_ok=True)
        recovered.write_csv(submission_path)
        return submission_path
    if submission_path.is_file():
        try:
            existing = pl.read_csv(submission_path, schema_overrides=dict(SUBMISSION_SCHEMA)).select(
                SUBMISSION_COLUMNS
            )
            if not existing.is_empty():
                return submission_path
        except Exception:
            pass
    return write_emergency_submission(dataset_root, submission_path)


def apply_adaptable_snail_25fps_policy(
    submission: pl.DataFrame,
    videos: pl.DataFrame,
    *,
    policy: str,
) -> pl.DataFrame:
    """Apply the documented private-test 25 FPS policy after raw-bound validation."""
    if policy == "none" or submission.is_empty():
        return submission
    affected = videos.filter(
        (pl.col("lab_id") == "AdaptableSnail")
        & (pl.col("frames_per_second").cast(pl.Float64) - 25.0).abs().lt(1e-6)
    )["video_id"].cast(pl.Int64)
    if affected.is_empty():
        return submission
    affected_ids = affected.to_list()
    if policy == "drop":
        filtered = submission.filter(~pl.col("video_id").is_in(affected_ids))
        return (
            filtered.drop("row_id")
            .with_row_index("row_id")
            .select(SUBMISSION_COLUMNS)
        )
    if policy != "scale_30_over_25":
        raise ValueError(f"unknown AdaptableSnail 25 FPS policy: {policy}")
    scaled = submission.with_columns(
        pl.when(pl.col("video_id").is_in(affected_ids))
        .then((pl.col("start_frame") * 6) // 5)
        .otherwise(pl.col("start_frame"))
        .cast(pl.Int64)
        .alias("start_frame"),
        pl.when(pl.col("video_id").is_in(affected_ids))
        .then((pl.col("stop_frame") * 6) // 5)
        .otherwise(pl.col("stop_frame"))
        .cast(pl.Int64)
        .alias("stop_frame"),
    )
    return scaled.drop("row_id").with_row_index("row_id").select(SUBMISSION_COLUMNS)


def run_offline_inference(
    config: Phase12Config,
    *,
    dataset_root: Path,
    work_root: Path,
    submission_path: Path,
) -> InferenceOutputs:
    """Run hidden inference one video at a time and write a strictly validated CSV."""
    started = time.perf_counter()
    work_root.mkdir(parents=True, exist_ok=True)
    manifests = build_test_manifests(dataset_root, work_root / "manifests")
    weights = pl.read_parquet(config.asset_path(config.assets.deployment_weights))
    decoder = pl.read_parquet(config.asset_path(config.assets.deployment_decoder_parameters))
    registry = pl.read_parquet(config.asset_path(config.assets.xgboost_registry))
    stacker_registry = (
        pl.read_parquet(config.asset_path(config.assets.stacker_registry))
        if config.assets.stacker_registry is not None
        else None
    )
    fingerprint = _run_fingerprint(dataset_root, config)
    stage_seconds: dict[str, float] = {
        "gbdt": 0.0,
        "temporal": 0.0,
        "blend_decode": 0.0,
    }
    video_reports: list[dict[str, object]] = []
    interval_parts: list[pl.DataFrame] = []
    fallback_parts: list[pl.DataFrame] = []
    progress_root = work_root / "video_predictions"
    rows = manifests.videos.sort("lab_id", "video_id").to_dicts()
    for position, row in enumerate(rows, start=1):
        lab = str(row["lab_id"])
        video_id = int(row["video_id"])
        route_root = progress_root / lab / str(video_id)
        interval_path = route_root / "intervals.csv"
        fallback_path = route_root / "fallback.parquet"
        done_path = route_root / "done.json"
        if config.runtime.resume and interval_path.is_file() and fallback_path.is_file() and done_path.is_file():
            done = json.loads(done_path.read_text(encoding="utf-8"))
            if done.get("run_fingerprint") == fingerprint:
                interval_parts.append(pl.read_csv(interval_path, schema_overrides=dict(SUBMISSION_SCHEMA)))
                fallback_parts.append(pl.read_parquet(fallback_path))
                video_reports.append(cast(dict[str, object], done["video_report"]))
                print(f"Phase 12 inference: {position}/{len(rows)} resumed {lab}/{video_id}", flush=True)
                continue
        route_root.mkdir(parents=True, exist_ok=True)
        candidates = (
            manifests.triplets.filter(
                (pl.col("lab_id") == lab) & (pl.col("video_id") == video_id) & ~pl.col("is_duplicate")
            )
            .select("agent_id", "target_id", "action", "is_self")
            .unique()
        )
        video_started = time.perf_counter()
        try:
            result = predict_decode_video_resilient(
                row,
                candidates,
                registry,
                weights,
                decoder,
                config=config,
                dataset_root=dataset_root,
                manifests=manifests,
                cache_root=work_root / "cache",
                stacker_registry=stacker_registry,
            )
        except Exception as error:
            result = VideoInferenceResult(
                decoded=_empty_submission(),
                fallback=_empty_fallback_candidate(),
                inference_route="skipped",
                stage_seconds={"gbdt": 0.0, "temporal": 0.0, "blend_decode": 0.0},
                failures=(_stage_failure("video", error),),
            )
        for stage, seconds in result.stage_seconds.items():
            stage_seconds[stage] += seconds
        decoded = result.decoded
        fallback = result.fallback
        decoded.write_csv(interval_path)
        fallback.write_parquet(fallback_path, compression="zstd", statistics=True)
        memory_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        video_report: dict[str, object] = {
            "lab_id": lab,
            "video_id": video_id,
            "frames": int(row["observed_frame_count"]),
            "mouse_count": int(row["mouse_count"]),
            "skeleton_id": str(row["skeleton_id"]),
            "gbdt_seconds": result.stage_seconds["gbdt"],
            "temporal_seconds": result.stage_seconds["temporal"],
            "blend_decode_seconds": result.stage_seconds["blend_decode"],
            "runtime_seconds": time.perf_counter() - video_started,
            "peak_observed_rss_mb": memory_mb,
            "interval_count": decoded.height,
            "inference_route": result.inference_route,
            "failure_count": len(result.failures),
            "failures": list(result.failures),
        }
        done_path.write_text(
            json.dumps(
                {"run_fingerprint": fingerprint, "video_report": video_report},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        interval_parts.append(decoded)
        fallback_parts.append(fallback)
        video_reports.append(video_report)
        if position == 1 or position % 5 == 0 or position == len(rows):
            partial, _ = assemble_resilient_submission(interval_parts, fallback_parts, manifests)
            checked_partial = validate_submission(
                partial,
                videos=manifests.videos,
                triplets=manifests.triplets,
                allowed_splits=("test",),
            )
            checked_partial = apply_adaptable_snail_25fps_policy(
                checked_partial,
                manifests.videos,
                policy=config.special_cases.adaptable_snail_25fps_policy,
            )
            submission_path.parent.mkdir(parents=True, exist_ok=True)
            partial_path = submission_path.with_suffix(".partial.csv")
            checked_partial.write_csv(partial_path)
            partial_path.replace(submission_path)
        del result, decoded, fallback
        _release_accelerator_memory()
        print(
            f"Phase 12 inference: {position}/{len(rows)} completed {lab}/{video_id} "
            f"via {video_report['inference_route']} in {video_report['runtime_seconds']:.1f}s",
            flush=True,
        )
    successful_video_count = sum(
        report.get("inference_route") != "skipped" for report in video_reports
    )
    if successful_video_count == 0:
        raise RuntimeError(
            "all test videos failed model inference; refusing to score the emergency fallback"
        )
    submission, used_confidence_fallback = assemble_resilient_submission(
        interval_parts,
        fallback_parts,
        manifests,
    )
    checked = validate_submission(
        submission,
        videos=manifests.videos,
        triplets=manifests.triplets,
        allowed_splits=("test",),
    )
    checked = apply_adaptable_snail_25fps_policy(
        checked,
        manifests.videos,
        policy=config.special_cases.adaptable_snail_25fps_policy,
    )
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    checked.write_csv(submission_path)
    runtime = time.perf_counter() - started
    projected = runtime / len(rows) * config.runtime.hidden_video_count
    budget_with_headroom = config.runtime.kaggle_limit_seconds * (1.0 - config.runtime.required_headroom_fraction)
    headroom = 1.0 - projected / config.runtime.kaggle_limit_seconds
    route_counts: dict[str, int] = {}
    for video_report in video_reports:
        route = str(video_report.get("inference_route", "legacy"))
        route_counts[route] = route_counts.get(route, 0) + 1
    report = {
        "schema_version": 1,
        "inference_version": config.inference_version,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_root": str(dataset_root),
        "test_csv_sha256": sha256_file(dataset_root / "test.csv"),
        "run_fingerprint": fingerprint,
        "video_count": len(rows),
        "interval_count": checked.height,
        "used_highest_confidence_fallback": used_confidence_fallback,
        "failed_video_count": route_counts.get("skipped", 0),
        "successful_video_count": successful_video_count,
        "degraded_video_count": route_counts.get("gbdt_only", 0) + route_counts.get("temporal_only", 0),
        "inference_route_counts": route_counts,
        "validation_error_count": 0,
        "adaptable_snail_25fps_policy": (
            config.special_cases.adaptable_snail_25fps_policy
        ),
        "runtime_seconds": runtime,
        "stage_seconds": stage_seconds,
        "projected_seconds_200_videos": projected,
        "kaggle_limit_seconds": config.runtime.kaggle_limit_seconds,
        "budget_with_20_percent_headroom_seconds": budget_with_headroom,
        "projected_headroom_fraction": headroom,
        "runtime_gate_passed": projected <= budget_with_headroom,
        "device": "cuda" if _cuda_available() else "cpu",
        "submission_sha256": sha256_file(submission_path),
        "videos": video_reports,
    }
    report_path = work_root / "inference_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return InferenceOutputs(
        submission_path=submission_path,
        report_path=report_path,
        videos_path=manifests.videos_path,
        triplets_path=manifests.triplets_path,
        video_count=len(rows),
        interval_count=checked.height,
        runtime_seconds=runtime,
        projected_seconds_200_videos=projected,
        headroom_fraction=headroom,
    )


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False
