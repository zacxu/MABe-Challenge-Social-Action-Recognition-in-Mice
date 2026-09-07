"""Strict Kaggle submission validation and safe non-empty fallback handling."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import polars as pl

from mabe.schemas import SUBMISSION_COLUMNS, SUBMISSION_SCHEMA


def _polars(frame: pl.DataFrame | pd.DataFrame) -> pl.DataFrame:
    return frame if isinstance(frame, pl.DataFrame) else pl.DataFrame(frame)


def validate_submission(
    submission: pl.DataFrame | pd.DataFrame,
    *,
    videos: pl.DataFrame,
    triplets: pl.DataFrame,
    allowed_splits: Iterable[str] = ("train", "test"),
) -> pl.DataFrame:
    """Validate schema, masks, bounds, self convention, row IDs, and overlap."""
    frame = _polars(submission)
    if tuple(frame.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"submission columns must be exactly {list(SUBMISSION_COLUMNS)}")
    if frame.is_empty():
        raise ValueError("submission must contain at least one interval")
    try:
        frame = frame.select(
            pl.col(name).cast(dtype, strict=True) for name, dtype in SUBMISSION_SCHEMA.items()
        ).select(SUBMISSION_COLUMNS)
    except Exception as error:
        raise ValueError("submission columns have invalid dtypes or values") from error
    if frame["row_id"].n_unique() != frame.height:
        raise ValueError("submission row IDs must be unique")
    if frame["row_id"].sort().to_list() != list(range(frame.height)):
        raise ValueError("submission row IDs must be contiguous from zero")
    if frame.filter(pl.col("start_frame") >= pl.col("stop_frame")).height:
        raise ValueError("submission intervals must be non-empty half-open intervals")

    selected_videos = videos.filter(pl.col("split").is_in(tuple(allowed_splits))).select(
        "split",
        "lab_id",
        "video_id",
        "observed_start_frame",
        "observed_stop_frame",
    )
    checked = frame.join(selected_videos, on="video_id", how="left")
    if checked["lab_id"].null_count():
        raise ValueError("submission contains an unknown video")
    if checked.filter(
        (pl.col("start_frame") < pl.col("observed_start_frame"))
        | (pl.col("stop_frame") > pl.col("observed_stop_frame"))
    ).height:
        raise ValueError("submission interval lies outside observed tracking bounds")

    allowed = (
        triplets.filter(
            pl.col("split").is_in(tuple(allowed_splits)) & ~pl.col("is_duplicate")
        )
        .select("lab_id", "video_id", "agent_id", "target_id", "action")
        .unique()
    )
    invalid = checked.join(
        allowed.with_columns(pl.lit(True).alias("allowed")),
        on=["lab_id", "video_id", "agent_id", "target_id", "action"],
        how="left",
    ).filter(pl.col("allowed").is_null())
    if invalid.height:
        raise ValueError("submission contains a triplet outside behaviors_labeled")
    if checked.filter(
        ((pl.col("target_id") == "self") & (pl.col("agent_id") == "self"))
        | ((pl.col("target_id") != "self") & (pl.col("agent_id") == pl.col("target_id")))
    ).height:
        raise ValueError("submission violates the canonical self-target convention")

    for group in frame.sort("start_frame", "stop_frame").partition_by(
        ["video_id", "agent_id", "target_id"], maintain_order=True
    ):
        starts = group["start_frame"].to_numpy()
        stops = group["stop_frame"].to_numpy()
        if starts.size > 1 and np.any(starts[1:] < np.maximum.accumulate(stops[:-1])):
            raise ValueError("submission has overlapping actions for one directed pair")
    return frame


def safe_fallback_submission(videos: pl.DataFrame, triplets: pl.DataFrame, *, split: str) -> pl.DataFrame:
    """Create one valid one-frame interval when a decoder emits no positives."""
    candidates = (
        triplets.filter((pl.col("split") == split) & ~pl.col("is_duplicate"))
        .join(
            videos.filter(pl.col("split") == split).select(
                "video_id", "observed_start_frame", "observed_stop_frame"
            ),
            on="video_id",
        )
        .filter(pl.col("observed_start_frame") < pl.col("observed_stop_frame"))
        .sort("video_id", "agent_id", "target_id", "action")
    )
    if candidates.is_empty():
        raise ValueError(f"no valid {split} triplet exists for a fallback row")
    row = candidates.row(0, named=True)
    start = int(str(row["observed_start_frame"]))
    return pl.DataFrame(
        {
            "row_id": pl.Series([0], dtype=pl.Int64),
            "video_id": pl.Series([int(str(row["video_id"]))], dtype=pl.Int64),
            "agent_id": pl.Series([str(row["agent_id"])], dtype=pl.String),
            "target_id": pl.Series([str(row["target_id"])], dtype=pl.String),
            "action": pl.Series([str(row["action"])], dtype=pl.String),
            "start_frame": pl.Series([start], dtype=pl.Int64),
            "stop_frame": pl.Series([start + 1], dtype=pl.Int64),
        }
    ).select(SUBMISSION_COLUMNS)
