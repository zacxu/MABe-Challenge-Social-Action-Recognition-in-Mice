"""Phase 7 end-to-end sampled training, full OOF inference, decoding, and scoring."""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb

from mabe.decode import centered_smooth_array, decode_probabilities
from mabe.diagnostics import metric_breakdown
from mabe.feature_stream import (
    extract_training_features,
    iter_feature_windows,
    selected_feature_frame,
    selected_feature_names,
)
from mabe.features.common import (
    CategoryRegistry,
    FeatureConfig,
    load_feature_config,
    round_seconds_to_frames,
)
from mabe.features.storage import load_category_registry
from mabe.fingerprints import schema_fingerprint, sha256_file
from mabe.labels import LabelStore, build_validation_solution
from mabe.metric import mouse_fbeta
from mabe.pose import PoseConfig, load_pose_config
from mabe.schemas import OOF_COLUMNS, OOF_SCHEMA, SUBMISSION_COLUMNS
from mabe.submission import safe_fallback_submission, validate_submission
from mabe.training_data import build_sampling_plan, filter_feasible_triplets
from mabe.xgb_config import (
    XGBoostBaselineConfig,
    load_xgboost_config,
    xgboost_config_fingerprint,
)
from mabe.xgb_model import TrainedFoldModel, choose_model_route, fit_fold_model, load_fold_model


@dataclass(frozen=True, slots=True)
class Phase7Outputs:
    model_registry_path: Path
    oof_manifest_path: Path
    cv_report_path: Path
    submission_path: Path
    fake_submission_path: Path
    overall_fbeta: float
    model_count: int
    oof_row_count: int


def _write_sidecar(
    path: Path,
    *,
    source_fingerprint: str,
    configuration_sha256: str,
    row_count: int | None,
    schema_hash: str | None,
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


def _manifest_lookup(videos: pl.DataFrame) -> dict[tuple[str, int], dict[str, object]]:
    return {
        (str(row["lab_id"]), int(row["video_id"])): row
        for row in videos.iter_rows(named=True)
    }


def _build_training_caches(
    primary_videos: pl.DataFrame,
    sampling_plan: pl.DataFrame,
    *,
    manifests: dict[tuple[str, int], dict[str, object]],
    cache_root: Path,
    dataset_root: Path,
    pose_config: PoseConfig,
    feature_config: FeatureConfig,
    baseline_config: XGBoostBaselineConfig,
    categories: CategoryRegistry,
) -> tuple[dict[str, Path], dict[str, tuple[str, ...]]]:
    cache_root.mkdir(parents=True, exist_ok=True)
    lab_paths: dict[str, Path] = {}
    route_features: dict[str, tuple[str, ...]] = {}
    for lab in sorted(str(value) for value in primary_videos["lab_id"].unique()):
        lab_plan = sampling_plan.filter(pl.col("lab_id") == lab)
        video_outputs: list[pl.DataFrame] = []
        for video_id in sorted(int(value) for value in lab_plan["video_id"].unique()):
            video_plan = lab_plan.filter(pl.col("video_id") == video_id)
            features, names = extract_training_features(
                manifests[(lab, video_id)],
                video_plan,
                dataset_root=dataset_root,
                pose_config=pose_config,
                feature_config=feature_config,
                model_feature_config=baseline_config.features,
                categories=categories,
            )
            for route, route_names in names.items():
                if route in route_features and route_features[route] != route_names:
                    raise AssertionError(f"{route} model feature schema changed across labs")
                route_features[route] = route_names
            video_outputs.append(features)
        if not video_outputs:
            continue
        lab_frame = pl.concat(video_outputs, how="diagonal_relaxed").sort(
            "action", "fold", "video_id", "video_frame", "agent_id", "target_id"
        )
        path = cache_root / f"{lab}.parquet"
        lab_frame.write_parquet(path, compression="zstd", statistics=True)
        lab_paths[lab] = path
    return lab_paths, route_features


def _action_route(triplets: pl.DataFrame, lab: str, action: str) -> str:
    routes = triplets.filter(
        (pl.col("lab_id") == lab) & (pl.col("action") == action)
    )["is_self"].unique()
    if routes.len() != 1:
        raise ValueError(f"action {action!r} has inconsistent solo/pair semantics in {lab}")
    return "solo" if bool(routes[0]) else "pair"


def _fake_fold_gate(
    *,
    held_out_fold: int,
    folds: pl.DataFrame,
    videos: pl.DataFrame,
    triplets: pl.DataFrame,
    videos_path: Path,
    dataset_root: Path,
    output_path: Path,
) -> float:
    keys = [
        (str(row["lab_id"]), int(row["video_id"]))
        for row in folds.filter(
            pl.col("is_primary_validation") & (pl.col("fold") == held_out_fold)
        ).iter_rows(named=True)
    ]
    solution = build_validation_solution(
        videos_path=videos_path,
        dataset_root=dataset_root,
        validation_videos=keys,
    )
    fake = solution[
        ["video_id", "agent_id", "target_id", "action", "start_frame", "stop_frame"]
    ].copy()
    fake.insert(0, "row_id", np.arange(len(fake), dtype=np.int64))
    checked = validate_submission(fake, videos=videos, triplets=triplets, allowed_splits=("train",))
    checked.write_csv(output_path)
    return float(mouse_fbeta(solution, fake, beta=1.0))


def _train_registry(
    primary_videos: pl.DataFrame,
    primary_triplets: pl.DataFrame,
    lab_paths: dict[str, Path],
    route_features: dict[str, tuple[str, ...]],
    *,
    model_root: Path,
    config: XGBoostBaselineConfig,
    fake_gate: Callable[[], None],
) -> pl.DataFrame:
    lab_actions = sorted(
        (str(row["lab_id"]), str(row["action"]))
        for row in primary_triplets.select("lab_id", "action").unique().iter_rows(named=True)
    )
    lab_frames = {lab: pl.read_parquet(path) for lab, path in lab_paths.items()}
    pooled_support = (
        pl.concat(list(lab_frames.values()), how="diagonal_relaxed")
        .filter(pl.col("target") == 1)
        .group_by("action")
        .len(name="positive_rows")
    )
    pooled_counts = {
        str(row["action"]): int(row["positive_rows"])
        for row in pooled_support.iter_rows(named=True)
    }
    records: list[dict[str, object]] = []
    pooled_requests: set[tuple[int, str, str]] = set()
    fold_order = tuple(config.execution.folds)
    for fold_position, held_fold in enumerate(fold_order):
        for lab, action in lab_actions:
            lab_frame = lab_frames[lab].filter(pl.col("action") == action)
            outer = lab_frame.filter(pl.col("fold") != held_fold)
            positives = outer.filter(pl.col("target") == 1)
            decision = choose_model_route(
                direct_positive_rows=positives.height,
                direct_positive_videos=positives["video_id"].n_unique(),
                pooled_positive_rows=pooled_counts.get(action, 0),
                auxiliary_positive_rows=0,
                action=action,
                support=config.support,
            )
            route = _action_route(primary_triplets, lab, action)
            base: dict[str, object] = {
                "lab_id": lab,
                "action": action,
                "route_kind": route,
                "held_out_fold": held_fold,
                "routing": decision.route,
                "routing_reason": decision.reason,
                "direct_positive_rows": positives.height,
                "direct_positive_videos": positives["video_id"].n_unique(),
            }
            if decision.route == "lab_action":
                trained = fit_fold_model(
                    lab_frame,
                    route_features[route],
                    held_out_fold=held_fold,
                    scope=lab,
                    action=action,
                    output_root=model_root,
                    config=config,
                )
                records.append(
                    {
                        **base,
                        "model_scope": lab,
                        "model_path": trained.model_path.as_posix(),
                        "best_iteration": trained.best_iteration,
                        "training_rows": trained.training_rows,
                        "inner_validation_fold": trained.inner_validation_fold,
                        "device": trained.device,
                    }
                )
            elif decision.route == "pooled_action":
                pooled_requests.add((held_fold, action, route))
                records.append(
                    {
                        **base,
                        "model_scope": "pooled",
                        "model_path": None,
                        "best_iteration": None,
                        "training_rows": None,
                        "inner_validation_fold": None,
                        "device": None,
                    }
                )
            else:
                records.append(
                    {
                        **base,
                        "model_scope": "none",
                        "model_path": None,
                        "best_iteration": None,
                        "training_rows": 0,
                        "inner_validation_fold": None,
                        "device": None,
                    }
                )
        if fold_position == 0:
            fake_gate()

    pooled_models: dict[tuple[int, str, str], TrainedFoldModel] = {}
    all_samples = pl.concat(list(lab_frames.values()), how="diagonal_relaxed")
    for held_fold, action, route in sorted(pooled_requests):
        pooled_frame = all_samples.filter(
            (pl.col("action") == action) & (pl.col("route") == route)
        )
        trained = fit_fold_model(
            pooled_frame,
            route_features[route],
            held_out_fold=held_fold,
            scope="pooled",
            action=action,
            output_root=model_root,
            config=config,
        )
        pooled_models[(held_fold, action, route)] = trained
    for record in records:
        if record["routing"] != "pooled_action":
            continue
        key = (int(str(record["held_out_fold"])), str(record["action"]), str(record["route_kind"]))
        trained = pooled_models[key]
        record.update(
            {
                "model_path": trained.model_path.as_posix(),
                "best_iteration": trained.best_iteration,
                "training_rows": trained.training_rows,
                "inner_validation_fold": trained.inner_validation_fold,
                "device": trained.device,
            }
        )
    return pl.DataFrame(records).sort("held_out_fold", "lab_id", "action")


def _target_for_frames(
    frame_ids: np.ndarray[tuple[int], np.dtype[np.int64]],
    intervals: tuple[tuple[int, int], ...],
) -> np.ndarray[tuple[int], np.dtype[np.uint8]]:
    target = np.zeros(frame_ids.size, dtype=np.uint8)
    for start, stop in intervals:
        target[(frame_ids >= start) & (frame_ids < stop)] = 1
    return target


def _predict_oof(
    primary_videos: pl.DataFrame,
    primary_triplets: pl.DataFrame,
    registry: pl.DataFrame,
    *,
    videos_path: Path,
    triplets_path: Path,
    dataset_root: Path,
    oof_root: Path,
    pose_config: PoseConfig,
    feature_config: FeatureConfig,
    config: XGBoostBaselineConfig,
    categories: CategoryRegistry,
) -> tuple[list[Path], dict[int, list[Path]], int]:
    manifests = _manifest_lookup(primary_videos)
    model_cache: dict[str, xgb.XGBClassifier] = {}
    all_paths: list[Path] = []
    video_paths: dict[int, list[Path]] = defaultdict(list)
    row_count = 0
    for row in primary_videos.sort("lab_id", "video_id").iter_rows(named=True):
        lab = str(row["lab_id"])
        video_id = int(row["video_id"])
        held_fold = int(row["fold"])
        candidates = primary_triplets.filter(
            (pl.col("lab_id") == lab) & (pl.col("video_id") == video_id)
        ).select("agent_id", "target_id", "action", "is_self").unique()
        routes = frozenset(
            "solo" if bool(value) else "pair" for value in candidates["is_self"].unique()
        )
        store = LabelStore.from_manifests(
            lab_id=lab,
            video_id=video_id,
            dataset_root=dataset_root,
            videos_path=videos_path,
            triplets_path=triplets_path,
        )
        intervals = store.positive_intervals()
        assert isinstance(intervals, dict)
        registry_video = registry.filter(
            (pl.col("lab_id") == lab) & (pl.col("held_out_fold") == held_fold)
        )
        part = 0
        for window in iter_feature_windows(
            manifests[(lab, video_id)],
            dataset_root=dataset_root,
            pose_config=pose_config,
            feature_config=feature_config,
            model_feature_config=config.features,
            categories=categories,
            routes=routes,
        ):
            outputs: list[pl.DataFrame] = []
            for route, table, groups in (
                ("solo", window.solo, config.features.solo_groups),
                ("pair", window.pair, config.features.pair_groups),
            ):
                if table is None:
                    continue
                names = selected_feature_names(table, groups)
                feature_frame = selected_feature_frame(table, names)
                if route == "solo":
                    feature_frame = feature_frame.with_columns(pl.lit("self").alias("target_id"))
                route_candidates = candidates.filter(
                    pl.col("is_self") if route == "solo" else ~pl.col("is_self")
                )
                for action_group in route_candidates.partition_by("action", maintain_order=True):
                    action = str(action_group["action"][0])
                    pairs = action_group.select("agent_id", "target_id").unique()
                    action_features = feature_frame.join(
                        pairs, on=["agent_id", "target_id"], how="semi"
                    ).sort("agent_id", "target_id", "video_frame")
                    entry = registry_video.filter(pl.col("action") == action).row(0, named=True)
                    path_value = entry.get("model_path")
                    if path_value is None:
                        raw_probability = np.zeros(action_features.height, dtype=np.float32)
                    else:
                        model_path_string = str(path_value)
                        if model_path_string not in model_cache:
                            model_cache[model_path_string] = load_fold_model(model_path_string)
                        raw_probability = model_cache[model_path_string].predict_proba(
                            action_features.select(names).to_numpy().astype(np.float32, copy=False)
                        )[:, 1].astype(np.float32)
                    predicted = action_features.select(
                        "video_id", "video_frame", "agent_id", "target_id"
                    ).with_columns(
                        pl.lit(action).alias("action"),
                        pl.Series("probability", raw_probability, dtype=pl.Float32),
                    )
                    smoothed_parts: list[pl.DataFrame] = []
                    radius = round_seconds_to_frames(
                        config.decoder.smoothing_seconds,
                        float(str(row["frames_per_second"])),
                    )
                    for pair_frame in predicted.partition_by(
                        ["agent_id", "target_id"], maintain_order=True
                    ):
                        smoothed = centered_smooth_array(
                            pair_frame["video_frame"].to_numpy().astype(np.int64, copy=False),
                            pair_frame["probability"].to_numpy().astype(np.float32, copy=False),
                            radius_frames=radius,
                        )
                        smoothed_parts.append(
                            pair_frame.with_columns(pl.Series("probability", smoothed))
                        )
                    core = pl.concat(smoothed_parts, how="vertical").filter(
                        pl.col("video_frame").is_between(
                            window.core_start, window.core_stop, closed="left"
                        )
                    )
                    targets: list[pl.DataFrame] = []
                    for pair_frame in core.partition_by(
                        ["agent_id", "target_id"], maintain_order=True
                    ):
                        agent = str(pair_frame["agent_id"][0])
                        target_id = str(pair_frame["target_id"][0])
                        triplet_key = next(
                            key
                            for key in intervals
                            if key.agent_id == agent
                            and key.target_id == target_id
                            and key.action == action
                        )
                        target_values = _target_for_frames(
                            pair_frame["video_frame"].to_numpy().astype(np.int64, copy=False),
                            intervals[triplet_key],
                        )
                        targets.append(
                            pair_frame.with_columns(
                                pl.lit(lab).alias("lab_id"),
                                pl.lit(held_fold, dtype=pl.Int8).alias("fold"),
                                pl.lit(1, dtype=pl.UInt8).alias("loss_mask"),
                                pl.Series("target", target_values, dtype=pl.UInt8),
                                pl.lit(config.model_version).alias("model_id"),
                                pl.lit(feature_config.feature_version).alias("feature_version"),
                            ).select(OOF_COLUMNS)
                        )
                    outputs.extend(targets)
            chunk = pl.concat(outputs, how="vertical").sort(
                "video_frame", "agent_id", "target_id", "action"
            )
            chunk = chunk.select(
                pl.col(name).cast(dtype, strict=True) for name, dtype in OOF_SCHEMA.items()
            ).select(OOF_COLUMNS)
            output_path = (
                oof_root
                / f"fold_{held_fold}"
                / lab
                / str(video_id)
                / f"part-{part:05d}.parquet"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            chunk.write_parquet(output_path, compression="zstd", statistics=True)
            all_paths.append(output_path)
            video_paths[video_id].append(output_path)
            row_count += chunk.height
            part += 1
    return all_paths, dict(video_paths), row_count


def _threshold_counts(paths: list[Path], thresholds: tuple[float, ...]) -> pl.DataFrame:
    scan = pl.scan_parquet([str(path) for path in paths])
    expressions: list[pl.Expr] = [pl.col("target").sum().alias("positive_frames")]
    for index, threshold in enumerate(thresholds):
        predicted = pl.col("probability") >= threshold
        expressions.extend(
            (
                predicted.sum().alias(f"predicted_{index}"),
                (predicted & (pl.col("target") == 1)).sum().alias(f"tp_{index}"),
            )
        )
    return scan.group_by("lab_id", "action", "fold").agg(expressions).collect(engine="streaming")


def _best_count_threshold(frame: pl.DataFrame, thresholds: tuple[float, ...]) -> float:
    scored: list[tuple[float, float]] = []
    positive = int(str(frame["positive_frames"].sum())) if frame.height else 0
    for index, threshold in enumerate(thresholds):
        predicted = int(str(frame[f"predicted_{index}"].sum())) if frame.height else 0
        tp = int(str(frame[f"tp_{index}"].sum())) if frame.height else 0
        denominator = positive + predicted
        score = 0.0 if denominator == 0 else 2 * tp / denominator
        scored.append((score, threshold))
    return float(max(scored)[1])


def _nested_thresholds(
    counts: pl.DataFrame,
    requested: pl.DataFrame,
    config: XGBoostBaselineConfig,
) -> pl.DataFrame:
    records: list[dict[str, object]] = []
    for held_fold in config.execution.folds:
        calibration = counts.filter(pl.col("fold") != held_fold)
        global_threshold = _best_count_threshold(calibration, config.decoder.threshold_grid)
        for item in requested.iter_rows(named=True):
            lab = str(item["lab_id"])
            action = str(item["action"])
            lab_action = calibration.filter(
                (pl.col("lab_id") == lab) & (pl.col("action") == action)
            )
            action_frame = calibration.filter(pl.col("action") == action)
            if lab_action.height and int(str(lab_action["positive_frames"].sum())) > 0:
                threshold = _best_count_threshold(lab_action, config.decoder.threshold_grid)
                source = "lab_action"
            elif action_frame.height and int(str(action_frame["positive_frames"].sum())) > 0:
                threshold = _best_count_threshold(action_frame, config.decoder.threshold_grid)
                source = "action"
            else:
                threshold = global_threshold or config.decoder.fallback_threshold
                source = "global"
            records.append(
                {
                    "held_out_fold": held_fold,
                    "lab_id": lab,
                    "action": action,
                    "threshold": threshold,
                    "threshold_source": source,
                    "calibration_folds": json.dumps(
                        [fold for fold in config.execution.folds if fold != held_fold]
                    ),
                    "used_held_out_labels": False,
                }
            )
    return pl.DataFrame(records).sort("held_out_fold", "lab_id", "action")


def _decode_oof(
    video_paths: dict[int, list[Path]],
    primary_videos: pl.DataFrame,
    thresholds: pl.DataFrame,
) -> pd.DataFrame:
    decoded: list[pd.DataFrame] = []
    fold_lookup = {
        int(row["video_id"]): (str(row["lab_id"]), int(row["fold"]))
        for row in primary_videos.iter_rows(named=True)
    }
    for video_id, paths in sorted(video_paths.items()):
        lab, fold = fold_lookup[video_id]
        selected_thresholds = thresholds.filter(
            (pl.col("held_out_fold") == fold) & (pl.col("lab_id") == lab)
        )
        frame = pl.read_parquet(paths)
        part = decode_probabilities(frame, selected_thresholds, smoothing_seconds=0.0)
        if not part.empty:
            decoded.append(part)
    result = pd.concat(decoded, ignore_index=True) if decoded else pd.DataFrame(columns=SUBMISSION_COLUMNS)
    result["row_id"] = np.arange(len(result), dtype=np.int64)
    return result[list(SUBMISSION_COLUMNS)]


def build_phase7_artifacts(
    *,
    videos_path: str | Path,
    triplets_path: str | Path,
    folds_path: str | Path,
    dataset_root: str | Path,
    pose_config_path: str | Path,
    feature_config_path: str | Path,
    category_registry_path: str | Path,
    xgboost_config_path: str | Path,
    artifact_root: str | Path,
    cache_root: str | Path,
    model_root: str | Path,
    source_fingerprint: str,
) -> Phase7Outputs:
    started = time.perf_counter()
    videos_path = Path(videos_path)
    triplets_path = Path(triplets_path)
    dataset_root = Path(dataset_root)
    videos = pl.read_parquet(videos_path)
    triplets = pl.read_parquet(triplets_path)
    folds = pl.read_parquet(folds_path)
    primary_membership = folds.filter(pl.col("is_primary_validation")).select(
        "lab_id", "video_id", "fold"
    )
    primary_videos = videos.join(primary_membership, on=["lab_id", "video_id"], how="inner")
    primary_triplets = filter_feasible_triplets(
        triplets.filter((pl.col("split") == "train") & ~pl.col("is_duplicate"))
        .join(primary_membership.select("lab_id", "video_id"), on=["lab_id", "video_id"], how="inner")
        .unique(subset=["lab_id", "video_id", "agent_id", "target_id", "action"]),
        primary_videos,
    )
    pose_config = load_pose_config(pose_config_path)
    feature_config = load_feature_config(feature_config_path)
    config = load_xgboost_config(xgboost_config_path)
    config_sha = xgboost_config_fingerprint(xgboost_config_path)
    categories = load_category_registry(category_registry_path)
    output_root = Path(artifact_root) / "xgb" / config.model_version
    output_root.mkdir(parents=True, exist_ok=True)
    plan_path = output_root / f"training_sampling_plan_{config_sha[:12]}.parquet"
    if plan_path.is_file():
        training_plan = pl.read_parquet(plan_path)
    else:
        training_plan = build_sampling_plan(
            primary_videos,
            triplets_path,
            videos_path=videos_path,
            dataset_root=dataset_root,
            config=config.sampler,
            seed=config.execution.seed,
            fold_count=len(config.execution.folds),
        )
        training_plan.write_parquet(plan_path, compression="zstd", statistics=True)
    feasible_keys = primary_triplets.select(
        "lab_id", "video_id", "agent_id", "target_id", "action"
    )
    feasible_training_plan = training_plan.join(
        feasible_keys,
        on=["lab_id", "video_id", "agent_id", "target_id", "action"],
        how="semi",
    )
    if feasible_training_plan.height != training_plan.height:
        training_plan = feasible_training_plan
        training_plan.write_parquet(plan_path, compression="zstd", statistics=True)
    manifests = _manifest_lookup(primary_videos)
    training_cache_root = Path(cache_root) / feature_config.feature_version / "training_samples"
    lab_paths, route_features = _build_training_caches(
        primary_videos,
        training_plan,
        manifests=manifests,
        cache_root=training_cache_root,
        dataset_root=dataset_root,
        pose_config=pose_config,
        feature_config=feature_config,
        baseline_config=config,
        categories=categories,
    )
    feature_list_path = output_root / "model_features.json"
    feature_list_path.write_text(
        json.dumps({key: list(value) for key, value in route_features.items()}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    fake_path = output_root / "fake_held_out_submission.csv"
    gate_state: dict[str, float] = {}

    def fake_gate() -> None:
        score = _fake_fold_gate(
            held_out_fold=config.execution.folds[0],
            folds=folds,
            videos=videos,
            triplets=triplets,
            videos_path=videos_path,
            dataset_root=dataset_root,
            output_path=fake_path,
        )
        gate_state["score"] = score
        if not math.isclose(score, 1.0):
            raise AssertionError(
                f"fake held-out metric gate returned {score}, expected exactly 1.0"
            )

    registry = _train_registry(
        primary_videos,
        primary_triplets,
        lab_paths,
        route_features,
        model_root=Path(model_root) / config.model_version,
        config=config,
        fake_gate=fake_gate,
    )
    if "score" not in gate_state:
        raise AssertionError("fake held-out metric gate did not run")
    registry_path = output_root / "model_registry.parquet"
    registry.write_parquet(registry_path, compression="zstd", statistics=True)
    oof_root = Path(artifact_root) / "oof" / config.model_version / config_sha[:12]
    oof_paths, video_paths, oof_rows = _predict_oof(
        primary_videos,
        primary_triplets,
        registry,
        videos_path=videos_path,
        triplets_path=triplets_path,
        dataset_root=dataset_root,
        oof_root=oof_root,
        pose_config=pose_config,
        feature_config=feature_config,
        config=config,
        categories=categories,
    )
    feasible_counts = primary_triplets.group_by("lab_id", "video_id").len(
        name="feasible_triplet_count"
    )
    expected_oof_rows = int(
        str(
            primary_videos.join(feasible_counts, on=["lab_id", "video_id"], how="left")
            .select(
                (
                    pl.col("observed_frame_count")
                    * pl.col("feasible_triplet_count").fill_null(0)
                ).sum()
            )
            .item()
        )
    )
    if oof_rows != expected_oof_rows:
        raise AssertionError(f"OOF rows {oof_rows} do not reconcile with mask rows {expected_oof_rows}")
    oof_manifest = {
        "schema_version": 1,
        "model_version": config.model_version,
        "feature_version": feature_config.feature_version,
        "probability_stage": "centered_smoothed",
        "smoothing_seconds": config.decoder.smoothing_seconds,
        "row_count": oof_rows,
        "expected_masked_row_count": expected_oof_rows,
        "columns": list(OOF_COLUMNS),
        "schema": {name: str(dtype) for name, dtype in OOF_SCHEMA.items()},
        "parts": [path.as_posix() for path in oof_paths],
    }
    oof_manifest_path = oof_root / "manifest.json"
    oof_manifest_path.write_text(
        json.dumps(oof_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    stable_oof_manifest_path = output_root / "oof_manifest.json"
    stable_oof_manifest_path.write_text(
        json.dumps(oof_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    counts = _threshold_counts(oof_paths, config.decoder.threshold_grid)
    counts_path = output_root / "threshold_counts.parquet"
    counts.write_parquet(counts_path, compression="zstd", statistics=True)
    requested = primary_triplets.select("lab_id", "action").unique()
    nested_thresholds = _nested_thresholds(counts, requested, config)
    thresholds_path = output_root / "decoder_parameters.parquet"
    nested_thresholds.write_parquet(thresholds_path, compression="zstd", statistics=True)
    submission = _decode_oof(video_paths, primary_videos, nested_thresholds)
    if submission.empty:
        submission = safe_fallback_submission(videos, triplets, split="train").to_pandas()
    validated_submission = validate_submission(
        submission, videos=videos, triplets=triplets, allowed_splits=("train",)
    )
    submission_path = output_root / "held_out_submission.csv"
    validated_submission.write_csv(submission_path)
    validation_keys = [
        (str(row["lab_id"]), int(row["video_id"]))
        for row in primary_videos.iter_rows(named=True)
    ]
    solution = build_validation_solution(
        videos_path=videos_path,
        dataset_root=dataset_root,
        validation_videos=validation_keys,
    )
    breakdown = metric_breakdown(solution, validated_submission.to_pandas(), beta=1.0)
    by_lab_path = output_root / "cv_by_lab.csv"
    by_action_path = output_root / "cv_by_lab_action.csv"
    breakdown.by_lab.to_csv(by_lab_path, index=False)
    breakdown.by_lab_action.to_csv(by_action_path, index=False)
    report = {
        "schema_version": 1,
        "model_version": config.model_version,
        "configuration_sha256": config_sha,
        "editable_configuration": str(Path(xgboost_config_path).resolve()),
        "overall_fbeta": breakdown.overall,
        "fake_fold_gate_fbeta": gate_state["score"],
        "model_count": registry.filter(pl.col("model_path").is_not_null()).height,
        "direct_model_routes": registry.filter(pl.col("routing") == "lab_action").height,
        "pooled_model_routes": registry.filter(pl.col("routing") == "pooled_action").height,
        "no_prediction_routes": registry.filter(pl.col("routing") == "no_prediction").height,
        "oof_row_count": oof_rows,
        "oof_schema_valid": True,
        "submission_rows": validated_submission.height,
        "runtime_seconds": time.perf_counter() - started,
        "early_stopping_policy": "inner_video_fold_then_refit_all_outer_training",
        "held_out_labels_used_for_early_stopping": False,
        "decoder_calibration": "other_oof_folds_only",
    }
    cv_report_path = output_root / "cv_report.json"
    cv_report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for path, row_count, schema_hash in (
        (plan_path, training_plan.height, schema_fingerprint(training_plan)),
        (registry_path, registry.height, schema_fingerprint(registry)),
        (counts_path, counts.height, schema_fingerprint(counts)),
        (thresholds_path, nested_thresholds.height, schema_fingerprint(nested_thresholds)),
        (submission_path, validated_submission.height, None),
        (fake_path, None, None),
        (feature_list_path, None, None),
        (cv_report_path, None, None),
        (oof_manifest_path, oof_rows, None),
        (stable_oof_manifest_path, oof_rows, None),
    ):
        _write_sidecar(
            path,
            source_fingerprint=source_fingerprint,
            configuration_sha256=config_sha,
            row_count=row_count,
            schema_hash=schema_hash,
        )
    return Phase7Outputs(
        model_registry_path=registry_path,
        oof_manifest_path=oof_manifest_path,
        cv_report_path=cv_report_path,
        submission_path=submission_path,
        fake_submission_path=fake_path,
        overall_fbeta=breakdown.overall,
        model_count=int(str(report["model_count"])),
        oof_row_count=oof_rows,
    )
