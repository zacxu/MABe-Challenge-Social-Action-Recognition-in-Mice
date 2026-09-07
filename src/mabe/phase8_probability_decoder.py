"""Exact, cross-fitted probability-decoder screening for Phase 8."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from mabe.decode import centered_smooth_array, hysteresis_mask
from mabe.features.common import round_seconds_to_frames
from mabe.intervals import frames_to_intervals
from mabe.metric import mouse_fbeta
from mabe.phase8_config import Phase8DecoderConfig
from mabe.schemas import SUBMISSION_COLUMNS


@dataclass(frozen=True, slots=True)
class ProbabilityDecoderCandidate:
    candidate: str
    threshold_policy: str
    global_threshold: float | None
    additional_smoothing_seconds: float
    hysteresis_delta: float
    complexity: float


@dataclass(frozen=True, slots=True)
class ProbabilityDecoderScreenResult:
    submission: pd.DataFrame
    route_parameters: pl.DataFrame
    search_summary: pl.DataFrame
    nested_selection: pl.DataFrame
    overall_fbeta: float


def probability_decoder_candidates(
    config: Phase8DecoderConfig,
) -> tuple[ProbabilityDecoderCandidate, ...]:
    """Cover the configured decoder grid without an intractable full Cartesian product."""
    candidates: list[ProbabilityDecoderCandidate] = []
    for smoothing in config.additional_smoothing_seconds:
        for hysteresis in config.hysteresis_delta:
            candidates.append(
                ProbabilityDecoderCandidate(
                    candidate=(
                        f"nested__smooth_{smoothing:.10g}__hysteresis_{hysteresis:.10g}"
                    ),
                    threshold_policy="nested_lab_action",
                    global_threshold=None,
                    additional_smoothing_seconds=smoothing,
                    hysteresis_delta=hysteresis,
                    complexity=smoothing + hysteresis,
                )
            )
    for threshold in config.threshold_grid:
        candidates.append(
            ProbabilityDecoderCandidate(
                candidate=f"global_threshold_{threshold:.10g}",
                threshold_policy="global",
                global_threshold=threshold,
                additional_smoothing_seconds=0.0,
                hysteresis_delta=0.0,
                complexity=1.0 + abs(threshold - 0.5),
            )
        )
    return tuple(candidates)


def _interval_submission(active: pl.DataFrame) -> pd.DataFrame:
    if active.is_empty():
        return pd.DataFrame(columns=SUBMISSION_COLUMNS)
    selected = (
        active.sort(
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
    records: list[dict[str, object]] = []
    for group in selected.partition_by(
        ["video_id", "agent_id", "target_id", "action"], maintain_order=True
    ):
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
    return pd.DataFrame.from_records(records, columns=SUBMISSION_COLUMNS)


def _decode_video_grid(
    probabilities: pl.DataFrame,
    parameters: pl.DataFrame,
    *,
    fps: float,
    candidates: tuple[ProbabilityDecoderCandidate, ...],
) -> dict[str, pd.DataFrame]:
    joined = probabilities.join(
        parameters.select("lab_id", "action", "threshold"),
        on=["lab_id", "action"],
        how="inner",
    )
    active_parts: defaultdict[str, list[pl.DataFrame]] = defaultdict(list)
    keys = ["lab_id", "video_id", "agent_id", "target_id", "action"]
    for group in joined.sort(*keys, "video_frame").partition_by(keys, maintain_order=True):
        frame_ids = group["video_frame"].to_numpy().astype(np.int64, copy=False)
        raw = group["probability"].to_numpy().astype(np.float32, copy=False)
        loss_mask = group["loss_mask"].to_numpy().astype(bool, copy=False)
        smoothed: dict[float, np.ndarray[tuple[int], np.dtype[np.float32]]] = {}
        for seconds in {candidate.additional_smoothing_seconds for candidate in candidates}:
            smoothed[seconds] = (
                raw
                if seconds <= 0
                else centered_smooth_array(
                    frame_ids,
                    raw,
                    radius_frames=round_seconds_to_frames(seconds, fps),
                )
            )
        nested_threshold = float(group["threshold"][0])
        base = group.select(
            "video_id", "video_frame", "agent_id", "target_id", "action"
        )
        for candidate in candidates:
            threshold = (
                nested_threshold
                if candidate.global_threshold is None
                else candidate.global_threshold
            )
            values = smoothed[candidate.additional_smoothing_seconds]
            active = hysteresis_mask(
                frame_ids,
                values,
                high_threshold=threshold,
                delta=candidate.hysteresis_delta,
            ) & loss_mask
            if not active.any():
                continue
            active_parts[candidate.candidate].append(
                base.with_columns(
                    pl.Series("margin", values / threshold, dtype=pl.Float32),
                    pl.Series("active", active, dtype=pl.Boolean),
                )
                .filter(pl.col("active"))
                .drop("active")
            )
    outputs: dict[str, pd.DataFrame] = {}
    for candidate in candidates:
        parts = active_parts.get(candidate.candidate, [])
        outputs[candidate.candidate] = _interval_submission(
            pl.concat(parts, how="vertical") if parts else pl.DataFrame()
        )
    return outputs


def _exact_score(solution: pd.DataFrame, submission: pd.DataFrame) -> float:
    if solution.empty or submission.empty:
        return 0.0
    return float(mouse_fbeta(solution, submission))


def select_nested_probability_candidates(
    search_summary: pl.DataFrame,
    candidates: tuple[ProbabilityDecoderCandidate, ...],
    folds: tuple[int, ...],
) -> pl.DataFrame:
    """Select every reported fold using exact scores from the other folds only."""
    lookup = {candidate.candidate: candidate for candidate in candidates}
    records: list[dict[str, object]] = []
    for held_fold in folds:
        calibration_folds = tuple(fold for fold in folds if fold != held_fold)
        means = (
            search_summary.filter(pl.col("evaluated_fold").is_in(calibration_folds))
            .group_by("candidate")
            .agg(pl.col("exact_fbeta").mean().alias("calibration_exact_fbeta"))
            .with_columns(
                pl.col("candidate")
                .replace_strict(
                    {name: candidate.complexity for name, candidate in lookup.items()},
                    return_dtype=pl.Float64,
                )
                .alias("complexity")
            )
            .sort(
                ["calibration_exact_fbeta", "complexity", "candidate"],
                descending=[True, False, False],
            )
        )
        winner = str(means["candidate"][0])
        candidate = lookup[winner]
        records.append(
            {
                "held_out_fold": held_fold,
                "candidate": winner,
                "threshold_policy": candidate.threshold_policy,
                "global_threshold": candidate.global_threshold,
                "additional_smoothing_seconds": candidate.additional_smoothing_seconds,
                "hysteresis_delta": candidate.hysteresis_delta,
                "calibration_exact_fbeta": float(means["calibration_exact_fbeta"][0]),
                "calibration_folds": json.dumps(calibration_folds),
                "used_held_out_labels": False,
            }
        )
    return pl.DataFrame(records).sort("held_out_fold")


def _route_parameters(
    baseline_parameters: pl.DataFrame,
    selection: pl.DataFrame,
) -> pl.DataFrame:
    joined = baseline_parameters.join(
        selection.select(
            "held_out_fold",
            "threshold_policy",
            "global_threshold",
            "additional_smoothing_seconds",
            "hysteresis_delta",
            pl.col("candidate").alias("probability_decoder_candidate"),
            pl.col("calibration_folds").alias("probability_calibration_folds"),
            pl.col("used_held_out_labels").alias("probability_used_held_out_labels"),
        ),
        on="held_out_fold",
        how="inner",
    )
    return joined.with_columns(
        pl.when(pl.col("threshold_policy") == "global")
        .then(pl.col("global_threshold"))
        .otherwise(pl.col("threshold"))
        .alias("threshold")
    ).drop("global_threshold")


def screen_probability_decoder(
    *,
    solution: pd.DataFrame,
    oof_paths: tuple[str, ...],
    videos: pl.DataFrame,
    folds: pl.DataFrame,
    baseline_parameters: pl.DataFrame,
    config: Phase8DecoderConfig,
) -> ProbabilityDecoderScreenResult:
    """Evaluate configured probability transforms and select them by nested exact OOF."""
    candidates = probability_decoder_candidates(config)
    primary = folds.filter(pl.col("is_primary_validation")).select(
        "lab_id", "video_id", "fold"
    )
    membership = {
        int(row["video_id"]): (str(row["lab_id"]), int(row["fold"]))
        for row in primary.iter_rows(named=True)
    }
    fps_by_video = {
        int(row["video_id"]): float(row["frames_per_second"])
        for row in videos.iter_rows(named=True)
    }
    paths_by_video: defaultdict[int, list[str]] = defaultdict(list)
    for raw_path in oof_paths:
        path = Path(raw_path)
        paths_by_video[int(path.parent.name)].append(raw_path)
    submissions: defaultdict[str, list[pd.DataFrame]] = defaultdict(list)
    for video_id in sorted(paths_by_video):
        lab, held_fold = membership[video_id]
        parameters = baseline_parameters.filter(
            (pl.col("held_out_fold") == held_fold) & (pl.col("lab_id") == lab)
        )
        decoded = _decode_video_grid(
            pl.read_parquet(paths_by_video[video_id]),
            parameters,
            fps=fps_by_video[video_id],
            candidates=candidates,
        )
        for candidate_name, frame in decoded.items():
            if not frame.empty:
                submissions[candidate_name].append(frame)
    combined = {
        candidate.candidate: (
            pd.concat(submissions[candidate.candidate], ignore_index=True)
            if submissions[candidate.candidate]
            else pd.DataFrame(columns=SUBMISSION_COLUMNS)
        )
        for candidate in candidates
    }
    fold_by_video = {video_id: fold for video_id, (_, fold) in membership.items()}
    score_records: list[dict[str, object]] = []
    for candidate in candidates:
        submission = combined[candidate.candidate]
        for evaluated_fold in sorted(set(fold_by_video.values())):
            fold_videos = {
                video_id for video_id, fold in fold_by_video.items() if fold == evaluated_fold
            }
            fold_solution = solution[solution["video_id"].isin(fold_videos)].reset_index(
                drop=True
            )
            fold_submission = submission[
                submission["video_id"].isin(fold_videos)
            ].reset_index(drop=True)
            score_records.append(
                {
                    "candidate": candidate.candidate,
                    "threshold_policy": candidate.threshold_policy,
                    "global_threshold": candidate.global_threshold,
                    "additional_smoothing_seconds": (
                        candidate.additional_smoothing_seconds
                    ),
                    "hysteresis_delta": candidate.hysteresis_delta,
                    "evaluated_fold": evaluated_fold,
                    "exact_fbeta": _exact_score(fold_solution, fold_submission),
                    "submission_rows": len(fold_submission),
                }
            )
    search_summary = pl.DataFrame(score_records).sort("candidate", "evaluated_fold")
    fold_values = tuple(sorted(set(fold_by_video.values())))
    selection = select_nested_probability_candidates(
        search_summary, candidates, fold_values
    )
    final_parts: list[pd.DataFrame] = []
    for row in selection.iter_rows(named=True):
        held_fold = int(row["held_out_fold"])
        selected_videos = {
            video_id for video_id, fold in fold_by_video.items() if fold == held_fold
        }
        candidate_submission = combined[str(row["candidate"])]
        final_parts.append(
            candidate_submission[candidate_submission["video_id"].isin(selected_videos)]
        )
    final_submission = pd.concat(final_parts, ignore_index=True)
    final_submission["row_id"] = np.arange(len(final_submission), dtype=np.int64)
    final_submission = final_submission[list(SUBMISSION_COLUMNS)]
    return ProbabilityDecoderScreenResult(
        submission=final_submission,
        route_parameters=_route_parameters(baseline_parameters, selection),
        search_summary=search_summary,
        nested_selection=selection,
        overall_fbeta=_exact_score(solution, final_submission),
    )
