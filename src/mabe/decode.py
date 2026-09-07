"""Probability smoothing, directed-pair conflict resolution, and interval decoding."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import polars as pl

from mabe.features.common import round_seconds_to_frames
from mabe.intervals import frames_to_intervals


def centered_smooth_array(
    frame_ids: np.ndarray[tuple[int], np.dtype[np.int64]],
    probabilities: np.ndarray[tuple[int], np.dtype[np.float32]],
    *,
    radius_frames: int,
) -> np.ndarray[tuple[int], np.dtype[np.float32]]:
    """Centered mean on actual frame IDs without crossing a temporal hole."""
    if frame_ids.shape != probabilities.shape:
        raise ValueError("frame IDs and probabilities must have equal one-dimensional shapes")
    if radius_frames <= 0 or frame_ids.size == 0:
        return probabilities.astype(np.float32, copy=True)
    cumulative = np.concatenate(([0.0], np.cumsum(probabilities, dtype=np.float64)))
    starts = np.searchsorted(frame_ids, frame_ids - radius_frames, side="left")
    stops = np.searchsorted(frame_ids, frame_ids + radius_frames, side="right")
    result: np.ndarray[tuple[int], np.dtype[np.float32]] = (
        (cumulative[stops] - cumulative[starts]) / (stops - starts)
    ).astype(np.float32)
    return result


def smooth_probability_frame(
    probabilities: pl.DataFrame,
    *,
    fps_by_video: Mapping[int, float],
    smoothing_seconds: float,
) -> pl.DataFrame:
    """Smooth independently by candidate triplet using physical-time FPS."""
    if smoothing_seconds <= 0 or probabilities.is_empty():
        return probabilities
    groups: list[pl.DataFrame] = []
    keys = ["video_id", "agent_id", "target_id", "action"]
    for group in probabilities.sort(*keys, "video_frame").partition_by(keys, maintain_order=True):
        video_id = int(str(group["video_id"][0]))
        fps = fps_by_video.get(video_id)
        if fps is None:
            raise KeyError(f"missing FPS for video {video_id}")
        radius = round_seconds_to_frames(smoothing_seconds, fps)
        smoothed = centered_smooth_array(
            group["video_frame"].to_numpy().astype(np.int64, copy=False),
            group["probability"].to_numpy().astype(np.float32, copy=False),
            radius_frames=radius,
        )
        groups.append(group.with_columns(pl.Series("probability", smoothed)))
    return pl.concat(groups, how="vertical").sort(*keys, "video_frame")


def hysteresis_mask(
    frame_ids: np.ndarray[tuple[int], np.dtype[np.int64]],
    probabilities: np.ndarray[tuple[int], np.dtype[np.float32]],
    *,
    high_threshold: float,
    delta: float,
) -> np.ndarray[tuple[int], np.dtype[np.bool_]]:
    """Keep contiguous low-threshold runs only when they contain a high-threshold seed."""
    if frame_ids.shape != probabilities.shape:
        raise ValueError("frame IDs and probabilities must have equal one-dimensional shapes")
    if not 0 <= high_threshold <= 1 or delta < 0:
        raise ValueError("hysteresis requires 0 <= high_threshold <= 1 and delta >= 0")
    high = probabilities >= high_threshold
    if delta == 0 or frame_ids.size == 0:
        return high
    low_threshold = max(0.0, high_threshold - delta)
    low = probabilities >= low_threshold
    active = np.zeros(frame_ids.size, dtype=bool)
    start = 0
    while start < frame_ids.size:
        if not low[start]:
            start += 1
            continue
        stop = start + 1
        while stop < frame_ids.size and low[stop] and frame_ids[stop] == frame_ids[stop - 1] + 1:
            stop += 1
        if high[start:stop].any():
            active[start:stop] = True
        start = stop
    return active


def temporal_binary_postprocess(
    frame_ids: np.ndarray[tuple[int], np.dtype[np.int64]],
    active: np.ndarray[tuple[int], np.dtype[np.bool_]],
    *,
    maximum_gap_frames: int,
    minimum_run_frames: int,
) -> np.ndarray[tuple[int], np.dtype[np.bool_]]:
    """Fill bounded internal gaps and remove short runs without crossing missing frames."""
    if frame_ids.shape != active.shape:
        raise ValueError("frame IDs and active mask must have equal one-dimensional shapes")
    if maximum_gap_frames < 0 or minimum_run_frames < 1:
        raise ValueError("gap must be non-negative and minimum run must be positive")
    result = active.astype(bool, copy=True)
    if maximum_gap_frames > 0:
        index = 0
        while index < result.size:
            if result[index]:
                index += 1
                continue
            start = index
            while (
                index < result.size
                and not result[index]
                and (index == start or frame_ids[index] == frame_ids[index - 1] + 1)
            ):
                index += 1
            bounded = start > 0 and index < result.size and result[start - 1] and result[index]
            contiguous = (
                bounded
                and frame_ids[start] == frame_ids[start - 1] + 1
                and frame_ids[index] == frame_ids[index - 1] + 1
            )
            if bounded and contiguous and index - start <= maximum_gap_frames:
                result[start:index] = True
    index = 0
    while index < result.size:
        if not result[index]:
            index += 1
            continue
        start = index
        while (
            index < result.size
            and result[index]
            and (index == start or frame_ids[index] == frame_ids[index - 1] + 1)
        ):
            index += 1
        if index - start < minimum_run_frames:
            result[start:index] = False
    return result


def decode_probabilities_advanced(
    probabilities: pl.DataFrame,
    parameters: pl.DataFrame,
    *,
    fps_by_video: Mapping[int, float],
) -> pd.DataFrame:
    """Decode per-action threshold/smoothing/hysteresis/gap/run parameters."""
    required_parameters = {
        "lab_id",
        "action",
        "threshold",
        "additional_smoothing_seconds",
        "hysteresis_delta",
        "gap_fill_seconds",
        "minimum_run_seconds",
    }
    if missing := required_parameters - set(parameters.columns):
        raise ValueError(f"advanced decoder parameters are missing columns: {sorted(missing)}")
    smoothed_parts: list[pl.DataFrame] = []
    keys = ["lab_id", "video_id", "agent_id", "target_id", "action"]
    joined = probabilities.join(parameters.select(*required_parameters), on=["lab_id", "action"])
    for group in joined.sort(*keys, "video_frame").partition_by(keys, maintain_order=True):
        video_id = int(group["video_id"][0])
        fps = fps_by_video[video_id]
        frame_ids = group["video_frame"].to_numpy().astype(np.int64, copy=False)
        values = group["probability"].to_numpy().astype(np.float32, copy=False)
        smoothing = float(group["additional_smoothing_seconds"][0])
        if smoothing > 0:
            values = centered_smooth_array(
                frame_ids,
                values,
                radius_frames=round_seconds_to_frames(smoothing, fps),
            )
        threshold = float(group["threshold"][0])
        active = hysteresis_mask(
            frame_ids,
            values,
            high_threshold=threshold,
            delta=float(group["hysteresis_delta"][0]),
        )
        gap_seconds = float(group["gap_fill_seconds"][0])
        run_seconds = float(group["minimum_run_seconds"][0])
        gap_frames = 0 if gap_seconds <= 0 else round_seconds_to_frames(gap_seconds, fps)
        run_frames = 1 if run_seconds <= 0 else round_seconds_to_frames(run_seconds, fps)
        active = temporal_binary_postprocess(
            frame_ids,
            active,
            maximum_gap_frames=gap_frames,
            minimum_run_frames=run_frames,
        )
        smoothed_parts.append(
            group.with_columns(
                pl.Series("probability", values, dtype=pl.Float32),
                pl.Series("active", active, dtype=pl.Boolean),
            )
        )
    active_frame = pl.concat(smoothed_parts, how="vertical").filter(
        (pl.col("loss_mask") == 1) & pl.col("active")
    ).with_columns((pl.col("probability") / pl.col("threshold")).alias("margin")).sort(
        "video_id",
        "video_frame",
        "agent_id",
        "target_id",
        "margin",
        "action",
        descending=[False, False, False, False, True, False],
    ).unique(
        subset=["video_id", "video_frame", "agent_id", "target_id"],
        keep="first",
        maintain_order=True,
    )
    records: list[dict[str, object]] = []
    for group in active_frame.partition_by(
        ["video_id", "agent_id", "target_id", "action"], maintain_order=True
    ):
        first = group.row(0, named=True)
        for start, stop in frames_to_intervals(group["video_frame"].sort().to_list()):
            records.append(
                {
                    "video_id": int(first["video_id"]),
                    "agent_id": str(first["agent_id"]),
                    "target_id": str(first["target_id"]),
                    "action": str(first["action"]),
                    "start_frame": start,
                    "stop_frame": stop,
                }
            )
    records.sort(
        key=lambda row: (
            int(str(row["video_id"])),
            int(str(row["start_frame"])),
            str(row["agent_id"]),
            str(row["target_id"]),
            str(row["action"]),
        )
    )
    for row_id, record in enumerate(records):
        record["row_id"] = row_id
    return pd.DataFrame.from_records(
        records,
        columns=[
            "row_id",
            "video_id",
            "agent_id",
            "target_id",
            "action",
            "start_frame",
            "stop_frame",
        ],
    )


def decode_probabilities_deployment(
    probabilities: pl.DataFrame,
    parameters: pl.DataFrame,
    *,
    fps_by_video: Mapping[int, float],
) -> pd.DataFrame:
    """Reproduce the selected Phase 11 probability and interval decoder in order."""
    from mabe.phase8_decoder import IntervalCalibrationStore

    required_parameters = {
        "lab_id",
        "action",
        "threshold",
        "additional_smoothing_seconds",
        "hysteresis_delta",
        "gap_fill_seconds",
        "minimum_run_seconds",
    }
    if missing := required_parameters - set(parameters.columns):
        raise ValueError(f"deployment decoder parameters are missing columns: {sorted(missing)}")
    joined = probabilities.join(
        parameters.select(*required_parameters),
        on=["lab_id", "action"],
        how="inner",
    )
    active_parts: list[pl.DataFrame] = []
    keys = ["lab_id", "video_id", "agent_id", "target_id", "action"]
    for group in joined.sort(*keys, "video_frame").partition_by(keys, maintain_order=True):
        video_id = int(group["video_id"][0])
        fps = fps_by_video[video_id]
        frame_ids = group["video_frame"].to_numpy().astype(np.int64, copy=False)
        values = group["probability"].to_numpy().astype(np.float32, copy=False)
        smoothing = float(group["additional_smoothing_seconds"][0])
        if smoothing > 0:
            values = centered_smooth_array(
                frame_ids,
                values,
                radius_frames=round_seconds_to_frames(smoothing, fps),
            )
        threshold = float(group["threshold"][0])
        active = hysteresis_mask(
            frame_ids,
            values,
            high_threshold=threshold,
            delta=float(group["hysteresis_delta"][0]),
        ) & group["loss_mask"].to_numpy().astype(bool, copy=False)
        if active.any():
            active_parts.append(
                group.select("video_id", "video_frame", "agent_id", "target_id", "action")
                .with_columns(
                    pl.Series("margin", values / threshold, dtype=pl.Float32),
                    pl.Series("active", active, dtype=pl.Boolean),
                )
                .filter(pl.col("active"))
                .drop("active")
            )
    columns = [
        "row_id",
        "video_id",
        "agent_id",
        "target_id",
        "action",
        "start_frame",
        "stop_frame",
    ]
    if not active_parts:
        return pd.DataFrame(columns=columns)
    selected = (
        pl.concat(active_parts, how="vertical")
        .sort(
            "video_id",
            "video_frame",
            "agent_id",
            "target_id",
            "margin",
            "action",
            descending=[False, False, False, False, True, False],
        )
        .unique(
            subset=["video_id", "video_frame", "agent_id", "target_id"],
            keep="first",
            maintain_order=True,
        )
        .sort("video_id", "agent_id", "target_id", "action", "video_frame")
    )
    initial_records: list[dict[str, object]] = []
    for group in selected.partition_by(
        ["video_id", "agent_id", "target_id", "action"], maintain_order=True
    ):
        first = group.row(0, named=True)
        for start, stop in frames_to_intervals(group["video_frame"].to_list()):
            initial_records.append(
                {
                    "video_id": int(first["video_id"]),
                    "agent_id": str(first["agent_id"]),
                    "target_id": str(first["target_id"]),
                    "action": str(first["action"]),
                    "start_frame": start,
                    "stop_frame": stop,
                }
            )
    initial = pd.DataFrame.from_records(initial_records)
    metadata_rows = [
        {
            "lab_id": str(row["lab_id"]),
            "video_id": int(row["video_id"]),
            "frames_per_second": fps_by_video[int(row["video_id"])],
        }
        for row in probabilities.select("lab_id", "video_id").unique().iter_rows(named=True)
    ]
    empty_solution = pd.DataFrame(
        columns=[
            "lab_id",
            "video_id",
            "agent_id",
            "target_id",
            "action",
            "start_frame",
            "stop_frame",
        ]
    )
    store = IntervalCalibrationStore(empty_solution, initial, pl.DataFrame(metadata_rows))
    parameter_lookup = {
        (str(row["lab_id"]), str(row["action"])): row
        for row in parameters.iter_rows(named=True)
    }
    records: list[dict[str, object]] = []
    for component in sorted(store.predictions):
        row = parameter_lookup.get(component)
        if row is None:
            raise KeyError(f"missing deployment decoder route for {component[0]}/{component[1]}")
        records.extend(
            store.rows_for_component(
                component,
                minimum_run_seconds=float(row["minimum_run_seconds"]),
                gap_fill_seconds=float(row["gap_fill_seconds"]),
            )
        )
    records.sort(
        key=lambda row: (
            int(str(row["video_id"])),
            int(str(row["start_frame"])),
            str(row["agent_id"]),
            str(row["target_id"]),
            str(row["action"]),
        )
    )
    for row_id, record in enumerate(records):
        record["row_id"] = row_id
    return pd.DataFrame.from_records(records, columns=columns)


def decode_probabilities(
    probabilities: pl.DataFrame,
    thresholds: pl.DataFrame,
    *,
    fps_by_video: Mapping[int, float] | None = None,
    smoothing_seconds: float = 0.0,
) -> pd.DataFrame:
    """Choose at most one action per directed pair/frame and run-length encode."""
    required = {
        "lab_id",
        "video_id",
        "video_frame",
        "agent_id",
        "target_id",
        "action",
        "loss_mask",
        "probability",
    }
    if missing := required - set(probabilities.columns):
        raise ValueError(f"probabilities are missing columns: {sorted(missing)}")
    frame = probabilities
    if smoothing_seconds > 0:
        if fps_by_video is None:
            raise ValueError("FPS mapping is required for physical-time smoothing")
        frame = smooth_probability_frame(
            frame,
            fps_by_video=fps_by_video,
            smoothing_seconds=smoothing_seconds,
        )
    selected = (
        frame.join(
            thresholds.select("lab_id", "action", "threshold"),
            on=["lab_id", "action"],
            how="left",
        )
        .filter(
            (pl.col("loss_mask") == 1)
            & pl.col("threshold").is_not_null()
            & (pl.col("probability") >= pl.col("threshold"))
        )
        .with_columns((pl.col("probability") / pl.col("threshold")).alias("margin"))
        .sort(
            "video_id",
            "video_frame",
            "agent_id",
            "target_id",
            "margin",
            "action",
            descending=[False, False, False, False, True, False],
        )
        .unique(
            subset=["video_id", "video_frame", "agent_id", "target_id"],
            keep="first",
            maintain_order=True,
        )
    )
    records: list[dict[str, object]] = []
    keys = ["video_id", "agent_id", "target_id", "action"]
    for group in selected.partition_by(keys, maintain_order=True):
        first = group.row(0, named=True)
        for start, stop in frames_to_intervals(group["video_frame"].sort().to_list()):
            records.append(
                {
                    "video_id": int(str(first["video_id"])),
                    "agent_id": str(first["agent_id"]),
                    "target_id": str(first["target_id"]),
                    "action": str(first["action"]),
                    "start_frame": start,
                    "stop_frame": stop,
                }
            )
    records.sort(
        key=lambda row: (
            int(str(row["video_id"])),
            int(str(row["start_frame"])),
            str(row["agent_id"]),
            str(row["target_id"]),
            str(row["action"]),
        )
    )
    for row_id, row in enumerate(records):
        row["row_id"] = row_id
    return pd.DataFrame.from_records(
        records,
        columns=[
            "row_id",
            "video_id",
            "agent_id",
            "target_id",
            "action",
            "start_frame",
            "stop_frame",
        ],
    )
