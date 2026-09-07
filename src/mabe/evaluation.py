"""OOF validation, nested decoder calibration, and exact metric evaluation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from mabe.diagnostics import MetricBreakdown, metric_breakdown
from mabe.intervals import frames_to_intervals
from mabe.metric import mouse_fbeta
from mabe.schemas import OOF_COLUMNS, OOF_SCHEMA


@dataclass(frozen=True, slots=True)
class NestedEvaluationResult:
    """Leakage-safe decoded predictions and exact official diagnostics."""

    breakdown: MetricBreakdown
    submission: pd.DataFrame
    by_fold: pd.DataFrame
    calibration: pd.DataFrame


def validate_oof(oof: pl.DataFrame, folds: pl.DataFrame) -> pl.DataFrame:
    """Validate and canonicalize one probability stream against fixed folds."""
    missing = set(OOF_COLUMNS) - set(oof.columns)
    if missing:
        raise ValueError(f"OOF data is missing columns: {sorted(missing)}")
    frame = oof.select(
        pl.col(name).cast(dtype, strict=True) for name, dtype in OOF_SCHEMA.items()
    ).select(OOF_COLUMNS)
    if frame.filter(~pl.col("probability").is_finite()).height:
        raise ValueError("OOF probabilities must be finite")
    if frame.filter(~pl.col("probability").is_between(0.0, 1.0, closed="both")).height:
        raise ValueError("OOF probabilities must be in [0, 1]")
    for column in ("loss_mask", "target"):
        if frame.filter(~pl.col(column).is_in([0, 1])).height:
            raise ValueError(f"{column} must be binary")
    if frame.filter((pl.col("target") == 1) & (pl.col("loss_mask") == 0)).height:
        raise ValueError("a positive target cannot be masked out")
    unique_key = [
        "lab_id",
        "video_id",
        "video_frame",
        "agent_id",
        "target_id",
        "action",
        "model_id",
        "feature_version",
    ]
    if frame.unique(subset=unique_key).height != frame.height:
        raise ValueError("OOF rows must be unique per candidate/model/feature version")
    fold_lookup = folds.select(
        "lab_id",
        "video_id",
        pl.col("fold").alias("expected_fold"),
        "is_primary_validation",
        "source_role",
    )
    checked = frame.join(fold_lookup, on=["lab_id", "video_id"], how="left")
    if checked["expected_fold"].null_count():
        raise ValueError("OOF contains a video without a supervised fold")
    if checked.filter(pl.col("fold") != pl.col("expected_fold")).height:
        raise ValueError("OOF fold values disagree with the fixed fold artifact")
    return checked.select(*OOF_COLUMNS, "is_primary_validation", "source_role")


def _fbeta_at_threshold(
    probabilities: np.ndarray[Any, np.dtype[np.float64]],
    targets: np.ndarray[Any, np.dtype[np.uint8]],
    threshold: float,
    *,
    beta: float,
) -> float:
    predicted = probabilities >= threshold
    positive = targets.astype(bool)
    tp = int(np.count_nonzero(predicted & positive))
    fp = int(np.count_nonzero(predicted & ~positive))
    fn = int(np.count_nonzero(~predicted & positive))
    denominator = (1 + beta**2) * tp + beta**2 * fn + fp
    return 0.0 if denominator == 0 else (1 + beta**2) * tp / denominator


def _best_threshold(
    frame: pl.DataFrame,
    threshold_grid: Sequence[float],
    *,
    beta: float,
) -> float:
    eligible = frame.filter(pl.col("loss_mask") == 1)
    if eligible.height == 0 or int(eligible["target"].sum()) == 0:
        return float(max(threshold_grid))
    probabilities = eligible["probability"].cast(pl.Float64).to_numpy()
    targets = eligible["target"].to_numpy().astype(np.uint8, copy=False)
    scored = [
        (_fbeta_at_threshold(probabilities, targets, threshold, beta=beta), threshold)
        for threshold in threshold_grid
    ]
    return float(max(scored)[1])


def tune_thresholds(
    calibration_oof: pl.DataFrame,
    requested_pairs: Iterable[tuple[str, str]],
    *,
    threshold_grid: Sequence[float] = tuple(np.linspace(0.1, 0.9, 17).tolist()),
    beta: float = 1.0,
) -> pl.DataFrame:
    """Tune requested lab/actions using calibration rows only, with safe fallbacks."""
    grid = tuple(float(value) for value in threshold_grid)
    if not grid or any(value < 0 or value > 1 for value in grid):
        raise ValueError("threshold_grid must contain values in [0, 1]")
    calibration = calibration_oof.filter(pl.col("loss_mask") == 1)
    global_threshold = _best_threshold(calibration, grid, beta=beta)
    action_thresholds: dict[str, float] = {}
    for action in calibration["action"].unique().sort().to_list():
        action_thresholds[str(action)] = _best_threshold(
            calibration.filter(pl.col("action") == action), grid, beta=beta
        )
    lab_action_thresholds: dict[tuple[str, str], float] = {}
    for row in calibration.select("lab_id", "action").unique().iter_rows(named=True):
        key = (str(row["lab_id"]), str(row["action"]))
        subset = calibration.filter(
            (pl.col("lab_id") == key[0]) & (pl.col("action") == key[1])
        )
        if int(subset["target"].sum()) > 0:
            lab_action_thresholds[key] = _best_threshold(subset, grid, beta=beta)
    records: list[dict[str, object]] = []
    for lab_id, action in sorted(set(requested_pairs)):
        key = (str(lab_id), str(action))
        if key in lab_action_thresholds:
            threshold = lab_action_thresholds[key]
            source = "lab_action"
        elif action in action_thresholds:
            threshold = action_thresholds[action]
            source = "action"
        else:
            threshold = global_threshold
            source = "global"
        records.append(
            {
                "lab_id": key[0],
                "action": key[1],
                "threshold": threshold,
                "threshold_source": source,
            }
        )
    return pl.DataFrame(
        records,
        schema={
            "lab_id": pl.String,
            "action": pl.String,
            "threshold": pl.Float64,
            "threshold_source": pl.String,
        },
    )


def decode_oof_probabilities(
    probabilities: pl.DataFrame,
    thresholds: pl.DataFrame,
) -> pd.DataFrame:
    """Decode masked frame probabilities with one action per pair/frame."""
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
    selected = (
        probabilities.join(
            thresholds.select("lab_id", "action", "threshold"),
            on=["lab_id", "action"],
            how="left",
        )
        .filter(
            (pl.col("loss_mask") == 1)
            & pl.col("threshold").is_not_null()
            & (pl.col("probability") >= pl.col("threshold"))
        )
        .with_columns(
            pl.when(pl.col("threshold") > 0)
            .then(pl.col("probability") / pl.col("threshold"))
            .otherwise(pl.col("probability"))
            .alias("margin")
        )
        .sort(
            ["video_id", "video_frame", "agent_id", "target_id", "margin", "action"],
            descending=[False, False, False, False, True, False],
        )
        .unique(
            subset=["video_id", "video_frame", "agent_id", "target_id"],
            keep="first",
            maintain_order=True,
        )
    )
    records: list[dict[str, object]] = []
    group_columns = ["video_id", "agent_id", "target_id", "action"]
    for group in selected.partition_by(group_columns, maintain_order=True):
        first = group.row(0, named=True)
        for start, stop in frames_to_intervals(group["video_frame"].to_list()):
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
    for row_id, row in enumerate(records):
        row["row_id"] = row_id
    columns = [
        "row_id",
        "video_id",
        "agent_id",
        "target_id",
        "action",
        "start_frame",
        "stop_frame",
    ]
    return pd.DataFrame.from_records(records, columns=columns)


def _filter_solution(solution: pd.DataFrame, keys: set[tuple[str, int]]) -> pd.DataFrame:
    mask = [
        (str(row.lab_id), int(row.video_id)) in keys
        for row in solution[["lab_id", "video_id"]].itertuples(index=False)
    ]
    return solution.loc[mask].reset_index(drop=True)


def nested_decoder_evaluation(
    oof: pl.DataFrame,
    folds: pl.DataFrame,
    solution: pd.DataFrame,
    *,
    threshold_grid: Sequence[float] = tuple(np.linspace(0.1, 0.9, 17).tolist()),
    beta: float = 1.0,
) -> NestedEvaluationResult:
    """Calibrate every held fold using only the other OOF folds, then score exactly."""
    checked = validate_oof(oof, folds).filter(pl.col("is_primary_validation"))
    if checked["model_id"].n_unique() != 1 or checked["feature_version"].n_unique() != 1:
        raise ValueError("nested evaluation accepts exactly one model/feature probability stream")
    held_folds = sorted(int(value) for value in checked["fold"].unique())
    decoded_parts: list[pd.DataFrame] = []
    fold_records: list[dict[str, object]] = []
    calibration_records: list[dict[str, object]] = []
    for held_fold in held_folds:
        calibration = checked.filter(pl.col("fold") != held_fold)
        held = checked.filter(pl.col("fold") == held_fold)
        calibration_folds = sorted(int(value) for value in calibration["fold"].unique())
        if held_fold in calibration_folds:
            raise AssertionError("held fold entered decoder calibration")
        requested = [
            (str(row["lab_id"]), str(row["action"]))
            for row in held.select("lab_id", "action").unique().iter_rows(named=True)
        ]
        thresholds = tune_thresholds(
            calibration,
            requested,
            threshold_grid=threshold_grid,
            beta=beta,
        )
        for row in thresholds.iter_rows(named=True):
            calibration_records.append(
                {
                    "held_out_fold": held_fold,
                    "lab_id": str(row["lab_id"]),
                    "action": str(row["action"]),
                    "threshold": float(row["threshold"]),
                    "threshold_source": str(row["threshold_source"]),
                    "calibration_folds": json_list(calibration_folds),
                    "calibration_row_count": calibration.height,
                    "used_held_out_labels": False,
                }
            )
        decoder_input = held.select(
            "lab_id",
            "video_id",
            "video_frame",
            "agent_id",
            "target_id",
            "action",
            "loss_mask",
            "probability",
        )
        decoded = decode_oof_probabilities(decoder_input, thresholds)
        decoded_parts.append(decoded)
        held_keys = {
            (str(row["lab_id"]), int(row["video_id"]))
            for row in held.select("lab_id", "video_id").unique().iter_rows(named=True)
        }
        held_solution = _filter_solution(solution, held_keys)
        fold_score = (
            mouse_fbeta(held_solution, decoded, beta=beta)
            if not held_solution.empty and not decoded.empty
            else 0.0
        )
        fold_records.append(
            {
                "fold": held_fold,
                "fbeta": fold_score,
                "solution_rows": len(held_solution),
                "submission_rows": len(decoded),
                "calibration_folds": json_list(calibration_folds),
            }
        )
    submission = pd.concat(decoded_parts, ignore_index=True) if decoded_parts else pd.DataFrame()
    if submission.empty:
        raise ValueError("nested decoder produced an empty submission")
    submission["row_id"] = np.arange(len(submission), dtype=np.int64)
    primary_keys = {
        (str(row["lab_id"]), int(row["video_id"]))
        for row in checked.select("lab_id", "video_id").unique().iter_rows(named=True)
    }
    primary_solution = _filter_solution(solution, primary_keys)
    breakdown = metric_breakdown(primary_solution, submission, beta=beta)
    return NestedEvaluationResult(
        breakdown=breakdown,
        submission=submission,
        by_fold=pd.DataFrame(fold_records),
        calibration=pd.DataFrame(calibration_records),
    )


def json_list(values: Sequence[int]) -> str:
    """Stable JSON storage for fold provenance."""
    return "[" + ",".join(str(int(value)) for value in values) + "]"
