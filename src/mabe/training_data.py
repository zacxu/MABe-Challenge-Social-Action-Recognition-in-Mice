"""Leakage-safe Phase 7 sampling plans over valid sparse supervision."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import polars as pl

from mabe.labels import LabelStore
from mabe.xgb_config import SamplerConfig

SAMPLING_SCHEMA = {
    "lab_id": pl.String,
    "video_id": pl.Int64,
    "video_frame": pl.Int64,
    "agent_id": pl.String,
    "target_id": pl.String,
    "action": pl.String,
    "route": pl.String,
    "fold": pl.Int8,
    "target": pl.UInt8,
    "sampling_kind": pl.String,
    "priority": pl.UInt64,
}


def filter_feasible_triplets(triplets: pl.DataFrame, videos: pl.DataFrame) -> pl.DataFrame:
    """Drop mask triplets that reference mice absent from a video's tracking data.

    Some raw behavior masks enumerate a nominal mouse (for example ``mouse3``)
    even when that video tracks a different set (for example ``mouse1``,
    ``mouse2``, and ``mouse4``).  Those rows cannot have pose features and must
    not be interpreted as supervised negatives.
    """
    triplet_required = {"lab_id", "video_id", "agent_id", "target_id", "is_self"}
    video_required = {"lab_id", "video_id", "mouse_ids"}
    if missing := triplet_required - set(triplets.columns):
        raise ValueError(f"triplet table is missing columns: {sorted(missing)}")
    if missing := video_required - set(videos.columns):
        raise ValueError(f"video table is missing columns: {sorted(missing)}")
    mice = videos.select("lab_id", "video_id", "mouse_ids").unique(
        subset=["lab_id", "video_id"]
    )
    return (
        triplets.join(mice, on=["lab_id", "video_id"], how="inner")
        .filter(
            pl.col("mouse_ids").list.contains(pl.col("agent_id"))
            & (
                pl.col("is_self")
                | pl.col("mouse_ids").list.contains(pl.col("target_id"))
            )
        )
        .drop("mouse_ids")
    )


def _priority(*parts: object, seed: int) -> int:
    payload = "|".join([str(seed), *(str(part) for part in parts)]).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _positive_mask(
    frame_ids: np.ndarray[tuple[int], np.dtype[np.int64]],
    intervals: tuple[tuple[int, int], ...],
) -> np.ndarray[tuple[int], np.dtype[np.bool_]]:
    result = np.zeros(frame_ids.size, dtype=bool)
    for start, stop in intervals:
        left = int(np.searchsorted(frame_ids, start, side="left"))
        right = int(np.searchsorted(frame_ids, stop, side="left"))
        result[left:right] = True
    return result


def _sample_negative_indices(
    frame_ids: np.ndarray[tuple[int], np.dtype[np.int64]],
    positive: np.ndarray[tuple[int], np.dtype[np.bool_]],
    intervals: tuple[tuple[int, int], ...],
    *,
    fps: float,
    config: SamplerConfig,
    seed: int,
) -> tuple[np.ndarray[tuple[int], np.dtype[np.int64]], np.ndarray[tuple[int], np.dtype[np.int64]]]:
    radius = int(math.ceil(config.near_boundary_seconds * fps))
    boundary = np.zeros(frame_ids.size, dtype=bool)
    for start, stop in intervals:
        boundary |= ((frame_ids >= start - radius) & (frame_ids < start)) | (
            (frame_ids >= stop) & (frame_ids < stop + radius)
        )
    boundary &= ~positive
    boundary_indices = np.flatnonzero(boundary).astype(np.int64, copy=False)
    negative_indices = np.flatnonzero(~positive).astype(np.int64, copy=False)
    if negative_indices.size == 0 or config.background_blocks_per_video_triplet == 0:
        return boundary_indices, np.empty(0, dtype=np.int64)
    generator = np.random.default_rng(seed)
    anchor_count = min(config.background_blocks_per_video_triplet, negative_indices.size)
    anchors = generator.choice(negative_indices, size=anchor_count, replace=False)
    background: list[np.ndarray[tuple[int], np.dtype[np.int64]]] = []
    for anchor in anchors.tolist():
        stop = min(frame_ids.size, int(anchor) + config.background_block_frames)
        block = np.arange(int(anchor), stop, dtype=np.int64)
        background.append(block[~positive[block] & ~boundary[block]])
    combined = np.unique(np.concatenate(background)) if background else np.empty(0, dtype=np.int64)
    return boundary_indices, combined.astype(np.int64, copy=False)


def cap_sampling_plan(
    plan: pl.DataFrame,
    config: SamplerConfig,
    *,
    fold_count: int,
) -> pl.DataFrame:
    """Cap each lab/action/fold so every outer fold remains represented."""
    if plan.is_empty():
        return pl.DataFrame(schema=SAMPLING_SCHEMA)
    positive_cap = math.ceil(config.max_positive_rows_per_lab_action / fold_count)
    row_cap = math.ceil(config.max_rows_per_lab_action / fold_count)
    minimum_negative = math.ceil(config.minimum_negative_rows_per_lab_action / fold_count)
    parts: list[pl.DataFrame] = []
    for group in plan.partition_by(["lab_id", "action", "fold"], maintain_order=True):
        positives = group.filter(pl.col("target") == 1).sort("priority").head(positive_cap)
        negative_count = min(
            max(0, row_cap - positives.height),
            max(minimum_negative, math.ceil(positives.height * config.negative_to_positive_ratio)),
        )
        negatives = (
            group.filter(pl.col("target") == 0)
            .with_columns(
                pl.when(pl.col("sampling_kind") == "near_boundary")
                .then(pl.lit(0))
                .otherwise(pl.lit(1))
                .alias("kind_order")
            )
            .sort("kind_order", "priority")
            .head(negative_count)
            .drop("kind_order")
        )
        parts.extend((positives, negatives))
    return (
        pl.concat(parts, how="vertical")
        .unique(
            subset=["lab_id", "video_id", "video_frame", "agent_id", "target_id", "action"],
            keep="first",
        )
        .sort("lab_id", "action", "fold", "video_id", "video_frame", "agent_id", "target_id")
        .select(*SAMPLING_SCHEMA)
    )


def build_sampling_plan(
    videos: pl.DataFrame,
    triplets_path: str | Path,
    *,
    videos_path: str | Path,
    dataset_root: str | Path,
    config: SamplerConfig,
    seed: int,
    fold_count: int,
) -> pl.DataFrame:
    """Sample only valid supervised triplets; annotations never enter feature inference."""
    required = {"lab_id", "video_id", "fold", "frames_per_second"}
    if missing := required - set(videos.columns):
        raise ValueError(f"sampling video table is missing columns: {sorted(missing)}")
    records: list[dict[str, object]] = []
    manifests_path = Path(videos_path)
    for row in videos.sort("lab_id", "video_id").iter_rows(named=True):
        store = LabelStore.from_manifests(
            lab_id=str(row["lab_id"]),
            video_id=int(row["video_id"]),
            dataset_root=dataset_root,
            videos_path=manifests_path,
            triplets_path=triplets_path,
        )
        if not store.is_supervised:
            continue
        frame_ids = store.tracking_frame_ids
        intervals_by_triplet = store.positive_intervals()
        assert isinstance(intervals_by_triplet, dict)
        observed_mice = {str(mouse_id) for mouse_id in row["mouse_ids"]}
        for triplet in store.allowed_triplets:
            if triplet.agent_id not in observed_mice or (
                not triplet.is_self and triplet.target_id not in observed_mice
            ):
                continue
            intervals = intervals_by_triplet[triplet]
            positive = _positive_mask(frame_ids, intervals)
            triplet_seed = _priority(
                row["lab_id"], row["video_id"], triplet.agent_id, triplet.target_id, triplet.action, seed=seed
            )
            boundary, background = _sample_negative_indices(
                frame_ids,
                positive,
                intervals,
                fps=float(row["frames_per_second"]),
                config=config,
                seed=triplet_seed,
            )
            route = "solo" if triplet.is_self else "pair"
            for kind, indices, target in (
                ("positive", np.flatnonzero(positive), 1),
                ("near_boundary", boundary, 0),
                ("background", background, 0),
            ):
                for index in indices.tolist():
                    video_frame = int(frame_ids[index])
                    records.append(
                        {
                            "lab_id": str(row["lab_id"]),
                            "video_id": int(row["video_id"]),
                            "video_frame": video_frame,
                            "agent_id": triplet.agent_id,
                            "target_id": triplet.target_id,
                            "action": triplet.action,
                            "route": route,
                            "fold": int(row["fold"]),
                            "target": target,
                            "sampling_kind": kind,
                            "priority": _priority(
                                row["lab_id"],
                                row["video_id"],
                                video_frame,
                                triplet.agent_id,
                                triplet.target_id,
                                triplet.action,
                                kind,
                                seed=seed,
                            ),
                        }
                    )
    raw = pl.DataFrame(records, schema=SAMPLING_SCHEMA) if records else pl.DataFrame(schema=SAMPLING_SCHEMA)
    return cap_sampling_plan(raw, config, fold_count=fold_count)
