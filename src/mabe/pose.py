"""Canonical pose loading, cleanup, geometry, resampling, and caching."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import polars as pl
import yaml
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator
from scipy.optimize import linear_sum_assignment

from mabe.fingerprints import sha256_file
from mabe.io import scan_tracking

CORE_LANDMARKS = ("nose", "ear_left", "ear_right", "body_center", "tail_base")
OPTIONAL_LANDMARKS = ("neck", "lateral_left", "lateral_right")


class ScalingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    min_pixels_per_cm: float = Field(gt=0)
    max_pixels_per_cm: float = Field(gt=0)
    geometry_ratio_min: float = Field(gt=0)
    geometry_ratio_max: float = Field(gt=0)


class CleanupConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_interpolation_gap_seconds: float = Field(ge=0)
    max_speed_cm_per_second: float = Field(gt=0)
    max_jump_pixels_per_frame: float = Field(gt=0)
    identity_swap_cost_ratio: float = Field(gt=0, lt=1)


class ResamplingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    target_fps: float = Field(gt=0)
    max_interpolation_gap_seconds: float = Field(ge=0)


class PoseConfig(BaseModel):
    """Reviewed canonical skeleton and physical cleanup settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(ge=1)
    pose_version: str = Field(min_length=1)
    canonical_landmarks: tuple[str, ...]
    bodypart_synonyms: dict[str, dict[str, str]]
    scaling: ScalingConfig
    cleanup: CleanupConfig
    resampling: ResamplingConfig
    adaptable_snail_frame_corrections: dict[int, float]

    @field_validator("canonical_landmarks")
    @classmethod
    def validate_landmarks(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("canonical_landmarks must be unique")
        missing = sorted(set(CORE_LANDMARKS + OPTIONAL_LANDMARKS) - set(value))
        if missing:
            raise ValueError(f"canonical_landmarks are missing required channels: {missing}")
        return value


def load_pose_config(path: str | Path = Path("configs/pose.yaml")) -> PoseConfig:
    source = Path(path)
    loaded: object = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"pose configuration must be a mapping: {source}")
    return PoseConfig.model_validate(cast(dict[str, Any], loaded))


@dataclass(frozen=True, slots=True)
class CleanPose:
    """Finite canonical arrays with explicit observation and availability masks."""

    lab_id: str
    video_id: int
    fps: float
    frame_ids: NDArray[np.int64]
    timestamps: NDArray[np.float64]
    mouse_ids: tuple[str, ...]
    landmarks: tuple[str, ...]
    raw_pixels: NDArray[np.float32]
    coordinates: NDArray[np.float32]
    coordinate_unit: Literal["cm", "pixel"]
    pixels_per_cm: float
    scale_valid: bool
    observed_mask: NDArray[np.bool_]
    availability_mask: NDArray[np.bool_]
    interpolated_mask: NDArray[np.bool_]
    invalid_mask: NDArray[np.bool_]
    jump_mask: NDArray[np.bool_]
    long_gap_mask: NDArray[np.bool_]
    derived_body_center: NDArray[np.bool_]
    identity_swap_mask: NDArray[np.bool_]
    center: NDArray[np.float32]
    center_mask: NDArray[np.bool_]
    body_axis: NDArray[np.float32]
    body_axis_mask: NDArray[np.bool_]
    body_length: NDArray[np.float32]
    confidence: NDArray[np.float32]
    pose_version: str

    def __post_init__(self) -> None:
        frames = self.frame_ids.size
        mice = len(self.mouse_ids)
        landmarks = len(self.landmarks)
        coordinate_shape = (frames, mice, landmarks, 2)
        mask_shape = (frames, mice, landmarks)
        for name in ("raw_pixels", "coordinates"):
            value = getattr(self, name)
            if value.shape != coordinate_shape or value.dtype != np.float32:
                raise ValueError(f"{name} must have float32 shape {coordinate_shape}")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite; missing values belong in masks")
        for name in (
            "observed_mask",
            "availability_mask",
            "interpolated_mask",
            "invalid_mask",
            "jump_mask",
            "long_gap_mask",
        ):
            value = getattr(self, name)
            if value.shape != mask_shape or value.dtype != np.bool_:
                raise ValueError(f"{name} must have boolean shape {mask_shape}")
        if self.derived_body_center.shape != (frames, mice):
            raise ValueError("derived_body_center has the wrong shape")
        if self.identity_swap_mask.shape != (frames, mice):
            raise ValueError("identity_swap_mask has the wrong shape")
        if self.center.shape != (frames, mice, 2) or self.body_axis.shape != (frames, mice, 2):
            raise ValueError("center/body_axis have the wrong shape")
        if self.body_length.shape != (frames, mice) or self.confidence.shape != (frames, mice):
            raise ValueError("body_length/confidence have the wrong shape")
        if self.frame_ids.dtype != np.int64 or self.timestamps.dtype != np.float64:
            raise TypeError("frame IDs and timestamps must use int64 and float64")
        if frames > 1 and np.any(np.diff(self.frame_ids) <= 0):
            raise ValueError("frame IDs must be strictly increasing")


@dataclass(frozen=True, slots=True)
class TimeMap:
    """Reversible nearest-time mapping between original frames and a regular grid."""

    original_frame_ids: NDArray[np.int64]
    original_times: NDArray[np.float64]
    resampled_times: NDArray[np.float64]
    original_to_resampled: NDArray[np.int64]
    resampled_to_original: NDArray[np.int64]

    def predictions_to_original(self, predictions: NDArray[Any]) -> NDArray[Any]:
        if predictions.shape[0] != self.resampled_times.size:
            raise ValueError("prediction length does not match the resampled time grid")
        result: NDArray[Any] = predictions[self.original_to_resampled]
        return result

    def labels_to_resampled(
        self,
        labels: NDArray[Any],
        *,
        method: Literal["nearest", "mode"] = "nearest",
    ) -> NDArray[Any]:
        if labels.shape[0] != self.original_times.size:
            raise ValueError("label length does not match original frames")
        if method == "nearest":
            result: NDArray[Any] = labels[self.resampled_to_original]
            return result
        if method != "mode":
            raise ValueError("label mapping method must be nearest or mode")
        output_shape = (self.resampled_times.size, *labels.shape[1:])
        output = np.empty(output_shape, dtype=labels.dtype)
        if self.resampled_times.size == 1:
            output[0] = labels[self.resampled_to_original[0]]
            return output
        edges = np.empty(self.resampled_times.size + 1, dtype=np.float64)
        edges[1:-1] = (self.resampled_times[:-1] + self.resampled_times[1:]) / 2
        edges[0] = self.resampled_times[0] - (edges[1] - self.resampled_times[0])
        edges[-1] = self.resampled_times[-1] + (self.resampled_times[-1] - edges[-2])
        for index in range(self.resampled_times.size):
            members = np.flatnonzero(
                (self.original_times >= edges[index]) & (self.original_times < edges[index + 1])
            )
            if members.size == 0:
                output[index] = labels[self.resampled_to_original[index]]
                continue
            values = labels[members]
            if values.ndim == 1:
                unique, counts = np.unique(values, return_counts=True)
                output[index] = unique[np.argmax(counts)]
            else:
                flattened = values.reshape(values.shape[0], -1)
                mode_values = []
                for column in range(flattened.shape[1]):
                    unique, counts = np.unique(flattened[:, column], return_counts=True)
                    mode_values.append(unique[np.argmax(counts)])
                output[index] = np.asarray(mode_values, dtype=labels.dtype).reshape(labels.shape[1:])
        return output


@dataclass(frozen=True, slots=True)
class ResampledPose:
    timestamps: NDArray[np.float64]
    coordinates: NDArray[np.float32]
    raw_pixels: NDArray[np.float32]
    availability_mask: NDArray[np.bool_]
    center: NDArray[np.float32]
    center_mask: NDArray[np.bool_]
    body_axis: NDArray[np.float32]
    body_axis_mask: NDArray[np.bool_]
    body_length: NDArray[np.float32]
    confidence: NDArray[np.float32]
    time_map: TimeMap
    target_fps: float


def _credible_scale(
    pixels_per_cm: float | None,
    *,
    video_width_pix: int | None,
    video_height_pix: int | None,
    arena_width_cm: float | None,
    arena_height_cm: float | None,
    config: ScalingConfig,
) -> bool:
    if pixels_per_cm is None or not np.isfinite(pixels_per_cm):
        return False
    if not config.min_pixels_per_cm <= pixels_per_cm <= config.max_pixels_per_cm:
        return False
    estimates: list[float] = []
    for pixels, centimeters in (
        (video_width_pix, arena_width_cm),
        (video_height_pix, arena_height_cm),
    ):
        if pixels is not None and centimeters is not None and pixels > 0 and centimeters > 0:
            estimates.append(float(pixels) / float(centimeters))
    if estimates:
        reference = float(np.median(estimates))
        ratio = pixels_per_cm / reference
        if not config.geometry_ratio_min <= ratio <= config.geometry_ratio_max:
            return False
    return True


def _jump_and_isolated_masks(
    values: NDArray[np.float32],
    observed: NDArray[np.bool_],
    frame_ids: NDArray[np.int64],
    *,
    fps: float,
    threshold_per_second: float,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    jump = np.zeros(observed.shape, dtype=bool)
    isolated = np.zeros(observed.shape, dtype=bool)
    for mouse in range(observed.shape[1]):
        for landmark in range(observed.shape[2]):
            indices = np.flatnonzero(observed[:, mouse, landmark])
            if indices.size < 2:
                continue
            coordinates = values[indices, mouse, landmark].astype(np.float64)
            elapsed = np.diff(frame_ids[indices]).astype(np.float64) / fps
            speeds = np.linalg.norm(np.diff(coordinates, axis=0), axis=1) / elapsed
            bad = speeds > threshold_per_second
            jump[indices[1:][bad], mouse, landmark] = True
            for position in range(1, indices.size - 1):
                if not (bad[position - 1] and bad[position]):
                    continue
                left = indices[position - 1]
                right = indices[position + 1]
                duration = (frame_ids[right] - frame_ids[left]) / fps
                direct_speed = (
                    np.linalg.norm(values[right, mouse, landmark] - values[left, mouse, landmark])
                    / duration
                )
                if direct_speed <= threshold_per_second:
                    isolated[indices[position], mouse, landmark] = True
    return jump, isolated


def _interpolate_short_gaps(
    values: NDArray[np.float32],
    availability: NDArray[np.bool_],
    frame_ids: NDArray[np.int64],
    *,
    fps: float,
    max_gap_seconds: float,
) -> NDArray[np.bool_]:
    interpolated = np.zeros(availability.shape, dtype=bool)
    for mouse in range(availability.shape[1]):
        for landmark in range(availability.shape[2]):
            valid = np.flatnonzero(availability[:, mouse, landmark])
            for left, right in zip(valid[:-1], valid[1:], strict=True):
                if right - left <= 1:
                    continue
                duration = (frame_ids[right] - frame_ids[left]) / fps
                if duration > max_gap_seconds:
                    continue
                missing = np.arange(left + 1, right)
                weights = (
                    (frame_ids[missing] - frame_ids[left])
                    / (frame_ids[right] - frame_ids[left])
                ).astype(np.float32)
                values[missing, mouse, landmark] = (
                    values[left, mouse, landmark][None, :] * (1 - weights[:, None])
                    + values[right, mouse, landmark][None, :] * weights[:, None]
                )
                availability[missing, mouse, landmark] = True
                interpolated[missing, mouse, landmark] = True
    return interpolated


def _proxy_centers(
    values: NDArray[np.float32],
    availability: NDArray[np.bool_],
    landmark_index: dict[str, int],
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    frames, mice = availability.shape[:2]
    output = np.zeros((frames, mice, 2), dtype=np.float32)
    mask = np.zeros((frames, mice), dtype=bool)
    body_index = landmark_index["body_center"]
    direct = availability[:, :, body_index]
    output[direct] = values[:, :, body_index][direct]
    mask |= direct
    source_indices = [
        landmark_index[name]
        for name in ("nose", "ear_left", "ear_right", "tail_base", "head")
        if name in landmark_index
    ]
    for frame in range(frames):
        for mouse in range(mice):
            if mask[frame, mouse]:
                continue
            usable = [index for index in source_indices if availability[frame, mouse, index]]
            if usable:
                output[frame, mouse] = values[frame, mouse, usable].mean(axis=0)
                mask[frame, mouse] = True
    return output, mask


def _identity_swap_mask(
    centers: NDArray[np.float32],
    center_mask: NDArray[np.bool_],
    frame_ids: NDArray[np.int64],
    *,
    fps: float,
    speed_threshold: float,
    cost_ratio: float,
) -> NDArray[np.bool_]:
    frames, mice = center_mask.shape
    output = np.zeros((frames, mice), dtype=bool)
    if mice < 2:
        return output
    for frame in range(1, frames):
        if not (center_mask[frame - 1].all() and center_mask[frame].all()):
            continue
        elapsed = (frame_ids[frame] - frame_ids[frame - 1]) / fps
        if elapsed <= 0:
            continue
        distances = np.linalg.norm(
            centers[frame - 1, :, None, :] - centers[frame, None, :, :], axis=-1
        )
        identity_cost = float(np.trace(distances))
        if float(np.max(np.diag(distances))) / elapsed <= speed_threshold:
            continue
        previous_indices, current_indices = linear_sum_assignment(distances)
        optimal_cost = float(distances[previous_indices, current_indices].sum())
        if optimal_cost >= cost_ratio * identity_cost:
            continue
        for previous, current in zip(previous_indices.tolist(), current_indices.tolist(), strict=True):
            if previous != current:
                output[frame, previous] = True
                output[frame, current] = True
    return output


def _derive_body_center(
    values: NDArray[np.float32],
    availability: NDArray[np.bool_],
    landmark_index: dict[str, int],
) -> NDArray[np.bool_]:
    frames, mice = availability.shape[:2]
    derived = np.zeros((frames, mice), dtype=bool)
    center_index = landmark_index["body_center"]
    tail_index = landmark_index["tail_base"]
    for frame in range(frames):
        for mouse in range(mice):
            if availability[frame, mouse, center_index] or not availability[frame, mouse, tail_index]:
                continue
            head: NDArray[np.float32] | None = None
            for name in ("nose", "head"):
                index = landmark_index.get(name)
                if index is not None and availability[frame, mouse, index]:
                    head = values[frame, mouse, index]
                    break
            if head is None:
                left = landmark_index["ear_left"]
                right = landmark_index["ear_right"]
                if availability[frame, mouse, left] and availability[frame, mouse, right]:
                    head = (values[frame, mouse, left] + values[frame, mouse, right]) / 2
            if head is None:
                continue
            values[frame, mouse, center_index] = (
                head + values[frame, mouse, tail_index]
            ) / 2
            availability[frame, mouse, center_index] = True
            derived[frame, mouse] = True
    return derived


def _geometry(
    coordinates: NDArray[np.float32],
    availability: NDArray[np.bool_],
    landmarks: tuple[str, ...],
) -> tuple[
    NDArray[np.float32],
    NDArray[np.bool_],
    NDArray[np.float32],
    NDArray[np.bool_],
    NDArray[np.float32],
    NDArray[np.float32],
]:
    index = {name: position for position, name in enumerate(landmarks)}
    frames, mice = availability.shape[:2]
    center = coordinates[:, :, index["body_center"]].copy()
    center_mask = availability[:, :, index["body_center"]].copy()
    head = np.zeros((frames, mice, 2), dtype=np.float32)
    head_mask = np.zeros((frames, mice), dtype=bool)
    for name in ("nose", "head"):
        landmark = index.get(name)
        if landmark is None:
            continue
        use = ~head_mask & availability[:, :, landmark]
        head[use] = coordinates[:, :, landmark][use]
        head_mask[use] = True
    ears = availability[:, :, index["ear_left"]] & availability[:, :, index["ear_right"]]
    use_ears = ~head_mask & ears
    ear_midpoint = (
        coordinates[:, :, index["ear_left"]] + coordinates[:, :, index["ear_right"]]
    ) / 2
    head[use_ears] = ear_midpoint[use_ears]
    head_mask[use_ears] = True
    tail = coordinates[:, :, index["tail_base"]]
    axis_mask = head_mask & availability[:, :, index["tail_base"]]
    axis = np.zeros((frames, mice, 2), dtype=np.float32)
    axis[axis_mask] = head[axis_mask] - tail[axis_mask]
    length = np.zeros((frames, mice), dtype=np.float32)
    length[axis_mask] = np.linalg.norm(axis[axis_mask], axis=1).astype(np.float32)
    core_indices = [index[name] for name in CORE_LANDMARKS]
    confidence = availability[:, :, core_indices].mean(axis=2).astype(np.float32)
    center[~center_mask] = 0
    return center, center_mask, axis, axis_mask, length, confidence


def load_clean_pose(
    tracking_path: str | Path,
    *,
    lab_id: str,
    video_id: int,
    fps: float,
    pixels_per_cm: float | None,
    config: PoseConfig,
    mouse_ids: tuple[str, ...] | list[str] | None = None,
    video_width_pix: int | None = None,
    video_height_pix: int | None = None,
    arena_width_cm: float | None = None,
    arena_height_cm: float | None = None,
    start_frame: int | None = None,
    stop_frame: int | None = None,
) -> CleanPose:
    """Load one full video or frame window on its original, possibly sparse frame IDs."""
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    source = scan_tracking(tracking_path)
    if start_frame is not None:
        source = source.filter(pl.col("video_frame") >= start_frame)
    if stop_frame is not None:
        source = source.filter(pl.col("video_frame") < stop_frame)
    frame_ids = source.select("video_frame").unique().sort("video_frame").collect()[
        "video_frame"
    ].to_numpy().astype(np.int64, copy=False)
    if frame_ids.size == 0:
        raise ValueError("requested tracking window contains no frames")
    observed_mice = tuple(source.select("mouse_id").unique().sort("mouse_id").collect()["mouse_id"])
    selected_mice = tuple(str(value) for value in (mouse_ids or observed_mice))
    aliases = config.bodypart_synonyms.get(lab_id, {})
    canonical_expression = pl.col("bodypart").replace_strict(
        aliases, default=pl.col("bodypart"), return_dtype=pl.String
    )
    rows = (
        source.with_columns(canonical_expression.alias("canonical_landmark"))
        .filter(pl.col("canonical_landmark").is_in(config.canonical_landmarks))
        .group_by("video_frame", "mouse_id", "canonical_landmark")
        .agg(pl.col("x").mean(), pl.col("y").mean())
        .sort("video_frame", "mouse_id", "canonical_landmark")
        .collect(engine="streaming")
    )
    landmarks = config.canonical_landmarks
    shape = (frame_ids.size, len(selected_mice), len(landmarks), 2)
    raw_pixels = np.zeros(shape, dtype=np.float32)
    observed = np.zeros(shape[:-1], dtype=bool)
    nonfinite = np.zeros(shape[:-1], dtype=bool)
    mouse_index = {mouse: index for index, mouse in enumerate(selected_mice)}
    landmark_index = {landmark: index for index, landmark in enumerate(landmarks)}
    if rows.height:
        row_frames = rows["video_frame"].to_numpy().astype(np.int64, copy=False)
        frame_index = np.searchsorted(frame_ids, row_frames)
        row_mice = np.asarray([mouse_index.get(str(value), -1) for value in rows["mouse_id"]])
        row_landmarks = np.asarray(
            [landmark_index[str(value)] for value in rows["canonical_landmark"]], dtype=np.int64
        )
        xy = rows.select("x", "y").to_numpy().astype(np.float32, copy=False)
        included = row_mice >= 0
        finite = np.isfinite(xy).all(axis=1) & included
        raw_pixels[frame_index[finite], row_mice[finite], row_landmarks[finite]] = xy[finite]
        observed[frame_index[finite], row_mice[finite], row_landmarks[finite]] = True
        invalid_rows = ~np.isfinite(xy).all(axis=1) & included
        nonfinite[
            frame_index[invalid_rows], row_mice[invalid_rows], row_landmarks[invalid_rows]
        ] = True

    scale_valid = _credible_scale(
        pixels_per_cm,
        video_width_pix=video_width_pix,
        video_height_pix=video_height_pix,
        arena_width_cm=arena_width_cm,
        arena_height_cm=arena_height_cm,
        config=config.scaling,
    )
    scale = float(pixels_per_cm) if scale_valid and pixels_per_cm is not None else 1.0
    detection_values = (raw_pixels / scale).astype(np.float32)
    speed_threshold = (
        config.cleanup.max_speed_cm_per_second
        if scale_valid
        else config.cleanup.max_jump_pixels_per_frame * fps
    )
    jump, isolated = _jump_and_isolated_masks(
        detection_values,
        observed,
        frame_ids,
        fps=fps,
        threshold_per_second=speed_threshold,
    )
    availability = observed & ~isolated
    invalid = nonfinite | isolated
    raw_pixels[isolated] = 0
    proxy, proxy_mask = _proxy_centers(raw_pixels, availability, landmark_index)
    identity_swaps = _identity_swap_mask(
        proxy / scale,
        proxy_mask,
        frame_ids,
        fps=fps,
        speed_threshold=speed_threshold,
        cost_ratio=config.cleanup.identity_swap_cost_ratio,
    )
    interpolated = _interpolate_short_gaps(
        raw_pixels,
        availability,
        frame_ids,
        fps=fps,
        max_gap_seconds=config.cleanup.max_interpolation_gap_seconds,
    )
    derived = _derive_body_center(raw_pixels, availability, landmark_index)
    long_gap = ~availability
    raw_pixels[~availability] = 0
    coordinates = (raw_pixels / scale).astype(np.float32)
    coordinates[~availability] = 0
    center, center_mask, axis, axis_mask, length, confidence = _geometry(
        coordinates, availability, landmarks
    )
    timestamps = (frame_ids - frame_ids[0]).astype(np.float64) / fps
    return CleanPose(
        lab_id=str(lab_id),
        video_id=int(video_id),
        fps=float(fps),
        frame_ids=frame_ids,
        timestamps=timestamps,
        mouse_ids=selected_mice,
        landmarks=landmarks,
        raw_pixels=raw_pixels,
        coordinates=coordinates,
        coordinate_unit="cm" if scale_valid else "pixel",
        pixels_per_cm=float(pixels_per_cm) if pixels_per_cm is not None else float("nan"),
        scale_valid=scale_valid,
        observed_mask=observed,
        availability_mask=availability,
        interpolated_mask=interpolated,
        invalid_mask=invalid,
        jump_mask=jump,
        long_gap_mask=long_gap,
        derived_body_center=derived,
        identity_swap_mask=identity_swaps,
        center=center,
        center_mask=center_mask,
        body_axis=axis,
        body_axis_mask=axis_mask,
        body_length=length,
        confidence=confidence,
        pose_version=config.pose_version,
    )


def load_clean_pose_from_manifest(
    row: dict[str, Any],
    *,
    dataset_root: str | Path,
    config: PoseConfig,
    start_frame: int | None = None,
    stop_frame: int | None = None,
) -> CleanPose:
    path = Path(str(row["tracking_path"]))
    if not path.is_absolute():
        path = Path(dataset_root) / path
    return load_clean_pose(
        path,
        lab_id=str(row["lab_id"]),
        video_id=int(row["video_id"]),
        fps=float(row["frames_per_second"]),
        pixels_per_cm=(
            float(row["pix_per_cm_approx"])
            if row.get("pix_per_cm_approx") is not None
            else None
        ),
        config=config,
        mouse_ids=tuple(str(value) for value in (row.get("mouse_ids") or ())),
        video_width_pix=(int(row["video_width_pix"]) if row.get("video_width_pix") else None),
        video_height_pix=(int(row["video_height_pix"]) if row.get("video_height_pix") else None),
        arena_width_cm=(float(row["arena_width_cm"]) if row.get("arena_width_cm") else None),
        arena_height_cm=(float(row["arena_height_cm"]) if row.get("arena_height_cm") else None),
        start_frame=start_frame,
        stop_frame=stop_frame,
    )


def build_time_map(pose: CleanPose, *, target_fps: float) -> TimeMap:
    if not np.isfinite(target_fps) or target_fps <= 0:
        raise ValueError("target_fps must be finite and positive")
    duration = float(pose.timestamps[-1])
    count = int(np.floor(duration * target_fps + 1e-9)) + 1
    resampled_times = np.arange(count, dtype=np.float64) / target_fps
    resampled_times = resampled_times[resampled_times <= duration + 1e-12]

    def nearest(source: NDArray[np.float64], targets: NDArray[np.float64]) -> NDArray[np.int64]:
        right = np.searchsorted(source, targets, side="left")
        right = np.clip(right, 0, source.size - 1)
        left = np.clip(right - 1, 0, source.size - 1)
        choose_right = np.abs(source[right] - targets) < np.abs(source[left] - targets)
        return np.where(choose_right, right, left).astype(np.int64)

    return TimeMap(
        original_frame_ids=pose.frame_ids.copy(),
        original_times=pose.timestamps.copy(),
        resampled_times=resampled_times,
        original_to_resampled=nearest(resampled_times, pose.timestamps),
        resampled_to_original=nearest(pose.timestamps, resampled_times),
    )


def _resample_continuous(
    values: NDArray[np.float32],
    mask: NDArray[np.bool_],
    source_times: NDArray[np.float64],
    target_times: NDArray[np.float64],
    *,
    max_gap_seconds: float,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    trailing = mask.shape[1:]
    dimensions = int(np.prod(trailing, dtype=np.int64))
    value_tail = values.shape[mask.ndim:]
    flattened_values = values.reshape(values.shape[0], dimensions, *value_tail)
    flattened_mask = mask.reshape(mask.shape[0], dimensions)
    output = np.zeros((target_times.size, dimensions, *value_tail), dtype=np.float32)
    output_mask = np.zeros((target_times.size, dimensions), dtype=bool)
    for dimension in range(dimensions):
        valid = np.flatnonzero(flattened_mask[:, dimension])
        if valid.size == 0:
            continue
        valid_times = source_times[valid]
        right_positions = np.searchsorted(valid_times, target_times, side="left")
        for target_index, right_position in enumerate(right_positions.tolist()):
            if right_position < valid.size and np.isclose(
                valid_times[right_position], target_times[target_index], atol=1e-10
            ):
                output[target_index, dimension] = flattened_values[valid[right_position], dimension]
                output_mask[target_index, dimension] = True
                continue
            if right_position == 0 or right_position >= valid.size:
                continue
            left_index = valid[right_position - 1]
            right_index = valid[right_position]
            gap = source_times[right_index] - source_times[left_index]
            if gap > max_gap_seconds:
                continue
            weight = (target_times[target_index] - source_times[left_index]) / gap
            output[target_index, dimension] = (
                flattened_values[left_index, dimension] * (1 - weight)
                + flattened_values[right_index, dimension] * weight
            )
            output_mask[target_index, dimension] = True
    return (
        output.reshape(target_times.size, *trailing, *value_tail),
        output_mask.reshape(target_times.size, *trailing),
    )


def resample_pose(
    pose: CleanPose,
    *,
    target_fps: float = 30.0,
    max_gap_seconds: float = 0.2,
) -> ResampledPose:
    """Linearly resample available pose values without edge extrapolation."""
    time_map = build_time_map(pose, target_fps=target_fps)
    coordinates, availability = _resample_continuous(
        pose.coordinates,
        pose.availability_mask,
        pose.timestamps,
        time_map.resampled_times,
        max_gap_seconds=max_gap_seconds,
    )
    raw_pixels, raw_mask = _resample_continuous(
        pose.raw_pixels,
        pose.availability_mask,
        pose.timestamps,
        time_map.resampled_times,
        max_gap_seconds=max_gap_seconds,
    )
    availability &= raw_mask
    coordinates[~availability] = 0
    raw_pixels[~availability] = 0
    center, center_mask, axis, axis_mask, length, confidence = _geometry(
        coordinates, availability, pose.landmarks
    )
    return ResampledPose(
        timestamps=time_map.resampled_times,
        coordinates=coordinates,
        raw_pixels=raw_pixels,
        availability_mask=availability,
        center=center,
        center_mask=center_mask,
        body_axis=axis,
        body_axis_mask=axis_mask,
        body_length=length,
        confidence=confidence,
        time_map=time_map,
        target_fps=target_fps,
    )


def save_clean_pose(
    pose: CleanPose,
    path: str | Path,
    *,
    source_sha256: str,
    configuration_sha256: str,
    frame_window: tuple[int, int] | None = None,
) -> Path:
    """Write a float32/boolean NPZ cache and auditable JSON sidecar."""
    output = Path(path)
    if output.suffix != ".npz":
        raise ValueError("pose cache path must end in .npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        lab_id=np.asarray(pose.lab_id),
        video_id=np.asarray(pose.video_id, dtype=np.int64),
        fps=np.asarray(pose.fps, dtype=np.float64),
        frame_ids=pose.frame_ids,
        timestamps=pose.timestamps,
        mouse_ids=np.asarray(pose.mouse_ids),
        landmarks=np.asarray(pose.landmarks),
        raw_pixels=pose.raw_pixels,
        coordinates=pose.coordinates,
        coordinate_unit=np.asarray(pose.coordinate_unit),
        pixels_per_cm=np.asarray(pose.pixels_per_cm, dtype=np.float64),
        scale_valid=np.asarray(pose.scale_valid),
        observed_mask=pose.observed_mask,
        availability_mask=pose.availability_mask,
        interpolated_mask=pose.interpolated_mask,
        invalid_mask=pose.invalid_mask,
        jump_mask=pose.jump_mask,
        long_gap_mask=pose.long_gap_mask,
        derived_body_center=pose.derived_body_center,
        identity_swap_mask=pose.identity_swap_mask,
        center=pose.center,
        center_mask=pose.center_mask,
        body_axis=pose.body_axis,
        body_axis_mask=pose.body_axis_mask,
        body_length=pose.body_length,
        confidence=pose.confidence,
        pose_version=np.asarray(pose.pose_version),
    )
    sidecar = {
        "artifact": output.name,
        "artifact_sha256": sha256_file(output),
        "configuration_sha256": configuration_sha256,
        "created_at": datetime.now(UTC).isoformat(),
        "dtypes": {"coordinates": "float32", "masks": "bool"},
        "frame_count": int(pose.frame_ids.size),
        "frame_window": list(frame_window) if frame_window else None,
        "landmark_count": len(pose.landmarks),
        "mouse_count": len(pose.mouse_ids),
        "pose_version": pose.pose_version,
        "source_data_fingerprint": source_sha256,
    }
    Path(f"{output}.meta.json").write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def load_cached_pose(path: str | Path) -> CleanPose:
    source = Path(path)
    with np.load(source, allow_pickle=False) as arrays:
        return CleanPose(
            lab_id=str(arrays["lab_id"].item()),
            video_id=int(arrays["video_id"].item()),
            fps=float(arrays["fps"].item()),
            frame_ids=arrays["frame_ids"],
            timestamps=arrays["timestamps"],
            mouse_ids=tuple(str(value) for value in arrays["mouse_ids"].tolist()),
            landmarks=tuple(str(value) for value in arrays["landmarks"].tolist()),
            raw_pixels=arrays["raw_pixels"],
            coordinates=arrays["coordinates"],
            coordinate_unit=cast(Literal["cm", "pixel"], str(arrays["coordinate_unit"].item())),
            pixels_per_cm=float(arrays["pixels_per_cm"].item()),
            scale_valid=bool(arrays["scale_valid"].item()),
            observed_mask=arrays["observed_mask"],
            availability_mask=arrays["availability_mask"],
            interpolated_mask=arrays["interpolated_mask"],
            invalid_mask=arrays["invalid_mask"],
            jump_mask=arrays["jump_mask"],
            long_gap_mask=arrays["long_gap_mask"],
            derived_body_center=arrays["derived_body_center"],
            identity_swap_mask=arrays["identity_swap_mask"],
            center=arrays["center"],
            center_mask=arrays["center_mask"],
            body_axis=arrays["body_axis"],
            body_axis_mask=arrays["body_axis_mask"],
            body_length=arrays["body_length"],
            confidence=arrays["confidence"],
            pose_version=str(arrays["pose_version"].item()),
        )


def pose_config_fingerprint(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
