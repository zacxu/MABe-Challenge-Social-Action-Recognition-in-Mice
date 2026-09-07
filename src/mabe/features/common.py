"""Shared schemas, metadata encoding, masks, and physical-time operations."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import polars as pl
import yaml
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator

from mabe.pose import CleanPose

FloatArray = NDArray[np.float32]
BoolArray = NDArray[np.bool_]
FeatureKind = Literal["solo", "pair"]
MISSING_CATEGORY = "__MISSING__"
UNKNOWN_CATEGORY = "__UNKNOWN__"
CATEGORY_FIELDS = (
    "lab",
    "tracker",
    "skeleton",
    "arena_type",
    "arena_shape",
    "strain",
    "color",
    "sex",
    "age",
    "condition",
)


class FeatureConfig(BaseModel):
    """Versioned feature semantics and physical-time scales."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(ge=1)
    feature_version: str = Field(min_length=1)
    temporal_windows_seconds: tuple[float, ...]
    maximum_derivative_gap_seconds: float = Field(gt=0)
    pause_speed_cm_per_second: float = Field(ge=0)
    contact_distances_cm: tuple[float, ...]
    nearby_mouse_distance_cm: float = Field(gt=0)
    between_mouse_distance_cm: float = Field(gt=0)
    chunk_rows: int = Field(gt=0)
    audit_max_frames: int = Field(gt=1)

    @field_validator("temporal_windows_seconds", "contact_distances_cm")
    @classmethod
    def validate_positive_sorted(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if not value or any(item <= 0 or not math.isfinite(item) for item in value):
            raise ValueError("temporal/contact scales must be finite and positive")
        if tuple(sorted(set(value))) != value:
            raise ValueError("temporal/contact scales must be unique and sorted")
        return value


def load_feature_config(path: str | Path = Path("configs/features.yaml")) -> FeatureConfig:
    source = Path(path)
    loaded: object = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"feature configuration must be a mapping: {source}")
    return FeatureConfig.model_validate(cast(dict[str, Any], loaded))


def feature_config_fingerprint(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class MouseMetadata:
    slot: str
    strain: str | None = None
    color: str | None = None
    sex: str | None = None
    age: str | None = None
    condition: str | None = None


@dataclass(frozen=True, slots=True)
class MetadataContext:
    """Only label-free metadata available identically in train and inference."""

    lab_id: str
    video_id: int
    tracker: str | None
    skeleton: str | None
    arena_type: str | None
    arena_shape: str | None
    fps: float
    mouse_count: int
    video_width_pix: float | None
    video_height_pix: float | None
    arena_width_cm: float | None
    arena_height_cm: float | None
    mice: dict[str, MouseMetadata]

    @classmethod
    def from_manifest_row(cls, row: dict[str, object]) -> MetadataContext:
        raw_mice = json.loads(str(row.get("mouse_metadata_json") or "[]"))
        mice: dict[str, MouseMetadata] = {}
        if isinstance(raw_mice, list):
            for value in raw_mice:
                if not isinstance(value, dict) or not value.get("slot"):
                    continue
                slot = str(value["slot"])
                mice[slot] = MouseMetadata(
                    slot=slot,
                    strain=_optional_string(value.get("strain")),
                    color=_optional_string(value.get("color")),
                    sex=_optional_string(value.get("sex")),
                    age=_optional_string(value.get("age")),
                    condition=_optional_string(value.get("condition")),
                )
        return cls(
            lab_id=str(row["lab_id"]),
            video_id=int(cast(int, row["video_id"])),
            tracker=_optional_string(row.get("tracking_method")),
            skeleton=_optional_string(row.get("skeleton_id")),
            arena_type=_optional_string(row.get("arena_type")),
            arena_shape=_optional_string(row.get("arena_shape")),
            fps=float(cast(float, row["frames_per_second"])),
            mouse_count=int(cast(int, row["mouse_count"])),
            video_width_pix=_optional_float(row.get("video_width_pix")),
            video_height_pix=_optional_float(row.get("video_height_pix")),
            arena_width_cm=_optional_float(row.get("arena_width_cm")),
            arena_height_cm=_optional_float(row.get("arena_height_cm")),
            mice=mice,
        )


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text and text.lower() != "nan" else None


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    numeric = float(cast(float, value))
    return numeric if math.isfinite(numeric) else None


@dataclass(frozen=True, slots=True)
class CategoryRegistry:
    """Stable train/inference categorical codes with missing and unknown buckets."""

    categories: dict[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        if tuple(sorted(self.categories)) != tuple(sorted(CATEGORY_FIELDS)):
            raise ValueError("category registry has the wrong fields")
        for values in self.categories.values():
            if values[:2] != (MISSING_CATEGORY, UNKNOWN_CATEGORY):
                raise ValueError("every category field must begin with missing and unknown")
            if len(values) != len(set(values)):
                raise ValueError("category values must be unique")

    def encode(self, field: str, value: str | None) -> np.float32:
        values = self.categories[field]
        selected = normalize_category(field, value)
        try:
            return np.float32(values.index(selected))
        except ValueError:
            return np.float32(1.0)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "missing_category": MISSING_CATEGORY,
            "unknown_category": UNKNOWN_CATEGORY,
            "categories": {name: list(values) for name, values in self.categories.items()},
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> CategoryRegistry:
        raw = cast(dict[str, list[str]], value["categories"])
        return cls(categories={name: tuple(items) for name, items in raw.items()})


def fit_category_registry(videos: pl.DataFrame) -> CategoryRegistry:
    """Fit label-free vocabularies once; unseen hidden values map to ``__UNKNOWN__``."""
    values: dict[str, set[str]] = {field: set() for field in CATEGORY_FIELDS}
    direct = {
        "lab": "lab_id",
        "tracker": "tracking_method",
        "skeleton": "skeleton_id",
        "arena_type": "arena_type",
        "arena_shape": "arena_shape",
    }
    for field, column in direct.items():
        values[field].update(
            str(item) for item in videos[column].drop_nulls().unique().to_list() if str(item)
        )
    for raw in videos["mouse_metadata_json"].drop_nulls().to_list():
        decoded = json.loads(str(raw))
        if not isinstance(decoded, list):
            continue
        for mouse in decoded:
            if not isinstance(mouse, dict):
                continue
            for field in ("strain", "color", "sex", "age", "condition"):
                item = normalize_category(field, _optional_string(mouse.get(field)))
                if item != MISSING_CATEGORY:
                    values[field].add(item)
    return CategoryRegistry(
        categories={
            field: (MISSING_CATEGORY, UNKNOWN_CATEGORY, *tuple(sorted(values[field])))
            for field in CATEGORY_FIELDS
        }
    )


def normalize_category(field: str, value: str | None) -> str:
    """Collapse condition timestamps to semantics and retain other reviewed values."""
    if value is None:
        return MISSING_CATEGORY
    text = value.strip()
    if not text:
        return MISSING_CATEGORY
    if field != "condition":
        return text
    lowered = re.sub(r"\s+", " ", text.lower())
    if "lights off" in lowered:
        return "lights_off"
    if "lights on" in lowered:
        return "lights_on"
    if "wireless" in lowered:
        return "wireless_device"
    if "cable" in lowered or "cabled" in lowered:
        return "cabled_device"
    return lowered


@dataclass(frozen=True, slots=True)
class FeatureTable:
    """Typed index plus finite float32 model columns and per-column group tags."""

    kind: FeatureKind
    index: pl.DataFrame
    values: dict[str, FloatArray]
    groups: dict[str, str]
    feature_version: str

    def __post_init__(self) -> None:
        if tuple(self.values) != tuple(self.groups):
            raise ValueError("feature values and group registry have different column order")
        row_count = self.index.height
        for name, value in self.values.items():
            if value.shape != (row_count,) or value.dtype != np.float32:
                raise ValueError(f"{name} must be float32 with shape {(row_count,)}")
            if not np.isfinite(value).all():
                raise ValueError(f"non-finite feature values in {name}")

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(self.values)

    @property
    def schema_hash(self) -> str:
        payload = {
            "feature_version": self.feature_version,
            "kind": self.kind,
            "index": [(name, str(dtype)) for name, dtype in self.index.schema.items()],
            "features": [(name, "Float32", self.groups[name]) for name in self.values],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def to_frame(self) -> pl.DataFrame:
        feature_frame = pl.DataFrame(
            {name: pl.Series(name, value, dtype=pl.Float32) for name, value in self.values.items()}
        )
        return self.index.hstack(feature_frame)

    def registry_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "feature_version": self.feature_version,
            "kind": self.kind,
            "schema_sha256": self.schema_hash,
            "index_columns": [
                {"name": name, "dtype": str(dtype)} for name, dtype in self.index.schema.items()
            ],
            "feature_columns": [
                {"name": name, "dtype": "Float32", "group": self.groups[name]}
                for name in self.values
            ],
        }


def round_seconds_to_frames(seconds: float, fps: float) -> int:
    """Round a physical duration half-up and always retain at least one frame."""
    if seconds <= 0 or fps <= 0 or not math.isfinite(seconds) or not math.isfinite(fps):
        raise ValueError("seconds and FPS must be finite and positive")
    return max(1, int(math.floor(seconds * fps + 0.5)))


def window_token(seconds: float) -> str:
    return f"{seconds:g}s".replace(".", "p")


def safe_norm(values: FloatArray, *, axis: int = -1) -> FloatArray:
    return np.linalg.norm(values, axis=axis).astype(np.float32)


def masked_distance(
    left: FloatArray,
    right: FloatArray,
    valid: BoolArray,
) -> FloatArray:
    output = safe_norm(left - right)
    output[~valid] = 0.0
    return output


def temporal_derivative(
    values: FloatArray,
    timestamps: NDArray[np.float64],
    valid: BoolArray,
    *,
    maximum_gap_seconds: float,
) -> tuple[FloatArray, BoolArray]:
    """Centered derivative with one-sided edges and no derivative across long gaps."""
    if values.shape[0] != timestamps.size or valid.shape != values.shape[:-1]:
        raise ValueError("derivative shapes do not share the time dimension")
    output = np.zeros_like(values, dtype=np.float32)
    previous_ok = np.zeros(valid.shape, dtype=bool)
    following_ok = np.zeros(valid.shape, dtype=bool)
    elapsed = np.diff(timestamps)
    short_gap = elapsed <= maximum_gap_seconds
    gap_shape = (elapsed.size, *([1] * (valid.ndim - 1)))
    previous_ok[1:] = valid[1:] & valid[:-1] & short_gap.reshape(gap_shape)
    following_ok[:-1] = valid[:-1] & valid[1:] & short_gap.reshape(gap_shape)
    centered = previous_ok & following_ok
    if values.shape[0] > 2:
        centered_elapsed = (timestamps[2:] - timestamps[:-2]).reshape(
            (-1, *([1] * (values.ndim - 1)))
        )
        centered_values = (values[2:] - values[:-2]) / centered_elapsed
        output[1:-1] = np.where(
            centered[1:-1][..., None], centered_values, output[1:-1]
        )
    previous_only = previous_ok & ~following_ok
    previous_elapsed = elapsed.reshape((-1, *([1] * (values.ndim - 1))))
    previous_values = (values[1:] - values[:-1]) / previous_elapsed
    output[1:] = np.where(
        previous_only[1:][..., None], previous_values, output[1:]
    )
    following_only = following_ok & ~previous_ok
    output[:-1] = np.where(
        following_only[:-1][..., None], previous_values, output[:-1]
    )
    output_valid = previous_ok | following_ok
    output[~output_valid] = 0.0
    return output.astype(np.float32), output_valid


def scalar_derivative(
    values: FloatArray,
    timestamps: NDArray[np.float64],
    valid: BoolArray,
    *,
    maximum_gap_seconds: float,
) -> tuple[FloatArray, BoolArray]:
    derivative, output_valid = temporal_derivative(
        values[..., None], timestamps, valid, maximum_gap_seconds=maximum_gap_seconds
    )
    return derivative[..., 0], output_valid


def angular_derivative(
    angles: FloatArray,
    timestamps: NDArray[np.float64],
    valid: BoolArray,
    *,
    maximum_gap_seconds: float,
) -> tuple[FloatArray, BoolArray]:
    vectors = np.stack((np.cos(angles), np.sin(angles)), axis=-1).astype(np.float32)
    vector_rate, output_valid = temporal_derivative(
        vectors, timestamps, valid, maximum_gap_seconds=maximum_gap_seconds
    )
    tangent = np.stack((-np.sin(angles), np.cos(angles)), axis=-1).astype(np.float32)
    rate = np.sum(vector_rate * tangent, axis=-1).astype(np.float32)
    rate[~output_valid] = 0.0
    return rate, output_valid


def centered_mean(
    values: FloatArray,
    radius: int,
    valid: BoolArray | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Centered masked mean and valid proportion, including truncated video edges."""
    if radius < 0:
        raise ValueError("window radius cannot be negative")
    mask = np.ones(values.shape, dtype=bool) if valid is None else valid
    if mask.shape != values.shape:
        raise ValueError("rolling mask must match values")
    weighted = np.where(mask, values, 0.0).astype(np.float64)
    counts = mask.astype(np.int64)
    cumulative = np.concatenate(
        (np.zeros((1, *values.shape[1:]), dtype=np.float64), np.cumsum(weighted, axis=0)),
        axis=0,
    )
    cumulative_counts = np.concatenate(
        (np.zeros((1, *values.shape[1:]), dtype=np.int64), np.cumsum(counts, axis=0)),
        axis=0,
    )
    indices = np.arange(values.shape[0])
    starts = np.maximum(0, indices - radius)
    stops = np.minimum(values.shape[0], indices + radius + 1)
    sums = cumulative[stops] - cumulative[starts]
    usable = cumulative_counts[stops] - cumulative_counts[starts]
    output = np.divide(sums, usable, out=np.zeros_like(sums), where=usable > 0)
    window_lengths = (stops - starts).reshape((-1, *([1] * (values.ndim - 1))))
    proportion = usable / window_lengths
    return output.astype(np.float32), proportion.astype(np.float32)


def centered_sum(
    values: FloatArray,
    radius: int,
    valid: BoolArray | None = None,
) -> FloatArray:
    mean, proportion = centered_mean(values, radius, valid)
    indices = np.arange(values.shape[0])
    lengths = np.minimum(values.shape[0], indices + radius + 1) - np.maximum(0, indices - radius)
    reshape = lengths.reshape((-1, *([1] * (values.ndim - 1))))
    output: FloatArray = (mean * proportion * reshape).astype(np.float32)
    return output


def lagged_displacement(
    positions: FloatArray,
    valid: BoolArray,
    lag: int,
) -> tuple[FloatArray, BoolArray]:
    output = np.zeros(valid.shape, dtype=np.float32)
    output_valid = np.zeros(valid.shape, dtype=bool)
    if lag < positions.shape[0]:
        usable = valid[lag:] & valid[:-lag]
        output[lag:][usable] = safe_norm(positions[lag:] - positions[:-lag])[usable]
        output_valid[lag:] = usable
    return output, output_valid


def past_future_change(
    values: FloatArray,
    valid: BoolArray,
    radius: int,
) -> tuple[FloatArray, FloatArray]:
    """Future mean minus past mean, excluding the current frame."""
    weighted = np.where(valid, values, 0.0).astype(np.float64)
    cumulative = np.concatenate(
        (np.zeros((1, *values.shape[1:]), dtype=np.float64), np.cumsum(weighted, axis=0)),
        axis=0,
    )
    cumulative_counts = np.concatenate(
        (
            np.zeros((1, *values.shape[1:]), dtype=np.int64),
            np.cumsum(valid.astype(np.int64), axis=0),
        ),
        axis=0,
    )
    indices = np.arange(values.shape[0])
    past_starts = np.maximum(0, indices - radius)
    future_stops = np.minimum(values.shape[0], indices + radius + 1)
    past_sums = cumulative[indices] - cumulative[past_starts]
    future_sums = cumulative[future_stops] - cumulative[indices + 1]
    past_counts = cumulative_counts[indices] - cumulative_counts[past_starts]
    future_counts = cumulative_counts[future_stops] - cumulative_counts[indices + 1]
    past_mean = np.divide(
        past_sums, past_counts, out=np.zeros_like(past_sums), where=past_counts > 0
    )
    future_mean = np.divide(
        future_sums,
        future_counts,
        out=np.zeros_like(future_sums),
        where=future_counts > 0,
    )
    usable = (past_counts > 0) & (future_counts > 0)
    output = np.where(usable, future_mean - past_mean, 0.0)
    support = np.minimum(past_counts, future_counts) / max(1, radius)
    return output.astype(np.float32), support.astype(np.float32)


def unit_vectors(values: FloatArray, valid: BoolArray) -> tuple[FloatArray, BoolArray]:
    norms = safe_norm(values)
    output_valid = valid & (norms > 1e-6)
    output = np.divide(
        values,
        norms[..., None],
        out=np.zeros_like(values),
        where=output_valid[..., None],
    ).astype(np.float32)
    return output, output_valid


def egocentric_coordinates(
    relative: FloatArray,
    heading: FloatArray,
    valid: BoolArray,
) -> tuple[FloatArray, FloatArray, BoolArray]:
    unit, heading_valid = unit_vectors(heading, valid)
    output_valid = valid & heading_valid
    forward = np.sum(relative * unit, axis=-1).astype(np.float32)
    lateral = (unit[..., 0] * relative[..., 1] - unit[..., 1] * relative[..., 0]).astype(
        np.float32
    )
    forward[~output_valid] = 0.0
    lateral[~output_valid] = 0.0
    return forward, lateral, output_valid


def add_feature(
    values: dict[str, FloatArray],
    groups: dict[str, str],
    name: str,
    value: NDArray[Any],
    group: str,
) -> None:
    if name in values:
        raise ValueError(f"duplicate feature column: {name}")
    converted = np.asarray(value, dtype=np.float32)
    if not np.isfinite(converted).all():
        raise ValueError(f"non-finite values in computed feature {name}")
    values[name] = converted.reshape(-1)
    groups[name] = group


def slice_clean_pose(pose: CleanPose, stop: int) -> CleanPose:
    """Take a deterministic leading feature-audit window without changing frame IDs."""
    count = min(stop, pose.frame_ids.size)
    names = (
        "frame_ids",
        "timestamps",
        "raw_pixels",
        "coordinates",
        "observed_mask",
        "availability_mask",
        "interpolated_mask",
        "invalid_mask",
        "jump_mask",
        "long_gap_mask",
        "derived_body_center",
        "identity_swap_mask",
        "center",
        "center_mask",
        "body_axis",
        "body_axis_mask",
        "body_length",
        "confidence",
    )
    updates = {name: getattr(pose, name)[:count].copy() for name in names}
    return replace(pose, **updates)
