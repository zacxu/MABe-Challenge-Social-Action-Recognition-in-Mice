"""Halo-aware bounded feature windows for Phase 7 training and inference."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from mabe.features.common import CategoryRegistry, FeatureConfig, FeatureTable, MetadataContext
from mabe.features.pair import build_pair_features
from mabe.features.solo import build_solo_features
from mabe.pose import PoseConfig, load_clean_pose_from_manifest
from mabe.xgb_config import ModelFeatureConfig


@dataclass(frozen=True, slots=True)
class FeatureWindow:
    core_start: int
    core_stop: int
    solo: FeatureTable | None
    pair: FeatureTable | None


def selected_feature_names(table: FeatureTable, groups: tuple[str, ...]) -> tuple[str, ...]:
    selected = tuple(name for name in table.feature_names if table.groups[name] in groups)
    if not selected:
        raise ValueError(f"no {table.kind} features selected for groups {groups}")
    return selected


def selected_feature_frame(table: FeatureTable, names: tuple[str, ...]) -> pl.DataFrame:
    return table.index.hstack(
        pl.DataFrame({name: pl.Series(name, table.values[name], dtype=pl.Float32) for name in names})
    )


def _core_span(mouse_count: int, config: ModelFeatureConfig) -> int:
    solo_entities = mouse_count
    pair_entities = mouse_count * max(1, mouse_count - 1)
    return max(128, config.maximum_rows_per_chunk // max(solo_entities, pair_entities))


def iter_feature_windows(
    manifest: dict[str, object],
    *,
    dataset_root: str | Path,
    pose_config: PoseConfig,
    feature_config: FeatureConfig,
    model_feature_config: ModelFeatureConfig,
    categories: CategoryRegistry,
    routes: frozenset[str],
    requested_frames: tuple[int, ...] | None = None,
) -> Iterator[FeatureWindow]:
    """Yield bounded core windows with a full maximum-temporal-window halo."""
    unsupported = routes - {"solo", "pair"}
    if unsupported or not routes:
        raise ValueError(f"unsupported or empty feature routes: {sorted(unsupported)}")
    start = int(str(manifest["observed_start_frame"]))
    stop = int(str(manifest["observed_stop_frame"]))
    fps = float(str(manifest["frames_per_second"]))
    mouse_count = int(str(manifest["mouse_count"]))
    span = _core_span(mouse_count, model_feature_config)
    block_count = math.ceil((stop - start) / span)
    if requested_frames is None:
        blocks = tuple(range(block_count))
    else:
        blocks = tuple(
            sorted(
                {
                    (frame - start) // span
                    for frame in requested_frames
                    if start <= frame < stop
                }
            )
        )
    halo = int(math.ceil(max(feature_config.temporal_windows_seconds) * fps)) + 2
    context = MetadataContext.from_manifest_row(manifest)
    for block in blocks:
        core_start = start + block * span
        core_stop = min(stop, core_start + span)
        pose = load_clean_pose_from_manifest(
            manifest,
            dataset_root=dataset_root,
            config=pose_config,
            start_frame=max(start, core_start - halo),
            stop_frame=min(stop, core_stop + halo),
        )
        solo = (
            build_solo_features(pose, context, feature_config, categories)
            if "solo" in routes
            else None
        )
        pair = (
            build_pair_features(pose, context, feature_config, categories)
            if "pair" in routes
            else None
        )
        yield FeatureWindow(core_start=core_start, core_stop=core_stop, solo=solo, pair=pair)


def extract_training_features(
    manifest: dict[str, object],
    plan: pl.DataFrame,
    *,
    dataset_root: str | Path,
    pose_config: PoseConfig,
    feature_config: FeatureConfig,
    model_feature_config: ModelFeatureConfig,
    categories: CategoryRegistry,
) -> tuple[pl.DataFrame, dict[str, tuple[str, ...]]]:
    """Build features only for blocks containing preselected training rows."""
    if plan.is_empty():
        return pl.DataFrame(), {}
    routes = frozenset(str(value) for value in plan["route"].unique())
    requested_frames = tuple(int(value) for value in plan["video_frame"].unique().sort())
    outputs: list[pl.DataFrame] = []
    feature_names: dict[str, tuple[str, ...]] = {}
    for window in iter_feature_windows(
        manifest,
        dataset_root=dataset_root,
        pose_config=pose_config,
        feature_config=feature_config,
        model_feature_config=model_feature_config,
        categories=categories,
        routes=routes,
        requested_frames=requested_frames,
    ):
        for route, table, groups in (
            ("solo", window.solo, model_feature_config.solo_groups),
            ("pair", window.pair, model_feature_config.pair_groups),
        ):
            if table is None:
                continue
            names = selected_feature_names(table, groups)
            feature_names.setdefault(route, names)
            if feature_names[route] != names:
                raise AssertionError(f"{route} feature schema changed between windows")
            features = selected_feature_frame(table, names).filter(
                pl.col("video_frame").is_between(
                    window.core_start, window.core_stop, closed="left"
                )
            )
            if route == "solo":
                features = features.with_columns(pl.lit("self").alias("target_id"))
            selected_plan = plan.filter(pl.col("route") == route).select(
                "lab_id",
                "video_id",
                "video_frame",
                "agent_id",
                "target_id",
                "action",
                "route",
                "fold",
                "target",
                "sampling_kind",
            )
            joined = selected_plan.join(
                features,
                on=["video_id", "video_frame", "agent_id", "target_id"],
                how="inner",
            )
            outputs.append(joined)
    if not outputs:
        raise ValueError(f"sampling plan produced no feature rows for video {manifest['video_id']}")
    result = pl.concat(outputs, how="diagonal_relaxed").sort(
        "action", "fold", "video_frame", "agent_id", "target_id"
    )
    if result.height != plan.height:
        missing = plan.join(
            result.select("video_id", "video_frame", "agent_id", "target_id", "action"),
            on=["video_id", "video_frame", "agent_id", "target_id", "action"],
            how="anti",
        )
        raise AssertionError(
            f"feature extraction lost {plan.height - result.height} sampled rows: {missing.head(5)}"
        )
    return result, feature_names
