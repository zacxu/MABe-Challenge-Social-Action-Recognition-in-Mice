"""Fold-safe lab normalization fitted only on training-video membership."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from mabe.validation import primary_split

NORMALIZATION_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class FeatureMoments:
    count: int
    mean: float
    std: float

    def to_dict(self) -> dict[str, float | int]:
        return {"count": self.count, "mean": self.mean, "std": self.std}


@dataclass(frozen=True, slots=True)
class FoldNormalization:
    held_out_fold: int
    feature_names: tuple[str, ...]
    global_stats: dict[str, FeatureMoments]
    lab_stats: dict[str, dict[str, FeatureMoments]]
    training_video_count: int
    training_keys_sha256: str
    includes_auxiliary: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": NORMALIZATION_SCHEMA_VERSION,
            "held_out_fold": self.held_out_fold,
            "feature_names": list(self.feature_names),
            "global_stats": {
                feature: moments.to_dict() for feature, moments in self.global_stats.items()
            },
            "lab_stats": {
                lab: {feature: moments.to_dict() for feature, moments in features.items()}
                for lab, features in self.lab_stats.items()
            },
            "training_video_count": self.training_video_count,
            "training_keys_sha256": self.training_keys_sha256,
            "includes_auxiliary": self.includes_auxiliary,
        }


def _moments(frame: pl.DataFrame, features: tuple[str, ...]) -> dict[str, FeatureMoments]:
    result: dict[str, FeatureMoments] = {}
    for feature in features:
        finite = frame.filter(pl.col(feature).is_finite() & pl.col(feature).is_not_null())[feature]
        count = finite.len()
        mean = float(str(finite.mean())) if count else 0.0
        std_value = float(str(finite.std(ddof=0))) if count else 1.0
        std = std_value if std_value > 1e-12 else 1.0
        result[feature] = FeatureMoments(count=count, mean=mean, std=std)
    return result


def fit_fold_normalization(
    records: pl.DataFrame,
    folds: pl.DataFrame,
    *,
    held_out_fold: int,
    feature_names: tuple[str, ...] | list[str],
    include_auxiliary: bool = False,
) -> FoldNormalization:
    """Fit lab/global moments after joining to the fixed training membership."""
    features = tuple(feature_names)
    required = {"lab_id", "video_id", *features}
    if missing := required - set(records.columns):
        raise ValueError(f"normalization records are missing columns: {sorted(missing)}")
    split = primary_split(
        folds,
        held_out_fold=held_out_fold,
        include_auxiliary=include_auxiliary,
    )
    train_keys = split.train.select("lab_id", "video_id").unique()
    held_keys = split.validation.select("lab_id", "video_id").unique()
    overlap = train_keys.join(held_keys, on=["lab_id", "video_id"], how="inner")
    if overlap.height:
        raise AssertionError("held-out videos entered normalization training membership")
    training_records = records.join(train_keys, on=["lab_id", "video_id"], how="inner")
    if training_records.is_empty():
        raise ValueError("normalization training records are empty")
    stable_keys = sorted(
        f"{row[0]}:{int(row[1])}" for row in train_keys.iter_rows()
    )
    key_hash = hashlib.sha256("|".join(stable_keys).encode()).hexdigest()
    lab_stats: dict[str, dict[str, FeatureMoments]] = {}
    for lab in sorted(str(value) for value in training_records["lab_id"].unique()):
        lab_stats[lab] = _moments(
            training_records.filter(pl.col("lab_id") == lab), features
        )
    return FoldNormalization(
        held_out_fold=held_out_fold,
        feature_names=features,
        global_stats=_moments(training_records, features),
        lab_stats=lab_stats,
        training_video_count=train_keys.height,
        training_keys_sha256=key_hash,
        includes_auxiliary=include_auxiliary,
    )


def apply_fold_normalization(
    records: pl.DataFrame,
    normalization: FoldNormalization,
) -> pl.DataFrame:
    """Append z-scored columns, falling back to training-global moments."""
    output = records
    labs = records["lab_id"].to_list()
    for feature in normalization.feature_names:
        means = []
        standard_deviations = []
        for lab_value in labs:
            lab = str(lab_value)
            moments = normalization.lab_stats.get(lab, normalization.global_stats)[feature]
            means.append(moments.mean)
            standard_deviations.append(moments.std)
        output = output.with_columns(
            ((pl.col(feature) - pl.Series(means)) / pl.Series(standard_deviations)).alias(
                f"{feature}_z"
            )
        )
    return output


def save_fold_normalization(
    normalization: FoldNormalization,
    path: str | Path,
    *,
    source_fingerprint: str,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(normalization.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sidecar = {
        "artifact": output.name,
        "artifact_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "configuration": {
            "held_out_fold": normalization.held_out_fold,
            "features": list(normalization.feature_names),
            "includes_auxiliary": normalization.includes_auxiliary,
        },
        "created_at": datetime.now(UTC).isoformat(),
        "row_count": normalization.training_video_count,
        "schema_sha256": hashlib.sha256(
            json.dumps(sorted(normalization.to_dict()), separators=(",", ":")).encode()
        ).hexdigest(),
        "source_data_fingerprint": source_fingerprint,
    }
    Path(f"{output}.meta.json").write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def manifest_normalization_records(videos: pl.DataFrame) -> pl.DataFrame:
    """Build physical video/context values usable before learned pose features exist."""
    return videos.filter(pl.col("split") == "train").select(
        "lab_id",
        "video_id",
        pl.col("frames_per_second").alias("fps"),
        pl.col("pix_per_cm_approx").alias("pixels_per_cm"),
        "arena_width_cm",
        "arena_height_cm",
        (pl.col("video_width_pix") / pl.col("pix_per_cm_approx")).alias("video_width_cm"),
        (pl.col("video_height_pix") / pl.col("pix_per_cm_approx")).alias("video_height_cm"),
    )
