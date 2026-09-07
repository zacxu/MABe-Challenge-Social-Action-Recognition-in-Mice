"""Exact interval-domain nested post-processing for the Phase 8 GBDT system."""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl

from mabe.diagnostics import MetricBreakdown, metric_breakdown
from mabe.features.common import round_seconds_to_frames
from mabe.metric import mouse_fbeta
from mabe.phase8_config import Phase8DecoderConfig

Component = tuple[str, str]
PairKey = tuple[int, str, str]
LabelKey = tuple[int, str, str]
Interval = tuple[int, int]


@dataclass(frozen=True, slots=True)
class NestedPostprocessResult:
    submission: pd.DataFrame
    parameters: pl.DataFrame
    by_fold: pd.DataFrame
    breakdown: MetricBreakdown
    search_summary: pl.DataFrame


def _merged(intervals: Iterable[Interval]) -> tuple[Interval, ...]:
    ordered = sorted((int(start), int(stop)) for start, stop in intervals if start < stop)
    if not ordered:
        return ()
    output: list[list[int]] = [[ordered[0][0], ordered[0][1]]]
    for start, stop in ordered[1:]:
        if start <= output[-1][1]:
            output[-1][1] = max(output[-1][1], stop)
        else:
            output.append([start, stop])
    return tuple((start, stop) for start, stop in output)


def _intersection_length(left: tuple[Interval, ...], right: tuple[Interval, ...]) -> int:
    left_index = 0
    right_index = 0
    total = 0
    while left_index < len(left) and right_index < len(right):
        left_start, left_stop = left[left_index]
        right_start, right_stop = right[right_index]
        total += max(0, min(left_stop, right_stop) - max(left_start, right_start))
        if left_stop <= right_stop:
            left_index += 1
        else:
            right_index += 1
    return total


def _component_counts(
    truth: Mapping[LabelKey, tuple[Interval, ...]],
    predictions: Mapping[LabelKey, tuple[Interval, ...]],
) -> tuple[int, int, int]:
    tp = 0
    truth_frames = 0
    predicted_frames = 0
    for key in set(truth) | set(predictions):
        labeled = truth.get(key, ())
        predicted = predictions.get(key, ())
        tp += _intersection_length(labeled, predicted)
        truth_frames += sum(stop - start for start, stop in labeled)
        predicted_frames += sum(stop - start for start, stop in predicted)
    return tp, predicted_frames - tp, truth_frames - tp


def _f1(counts: tuple[int, int, int]) -> float:
    tp, fp, fn = counts
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else 2 * tp / denominator


class IntervalCalibrationStore:
    """Compact exact-count store for independent lab/action post-processing."""

    def __init__(
        self,
        solution: pd.DataFrame,
        submission: pd.DataFrame,
        video_metadata: pl.DataFrame,
    ) -> None:
        required_video = {"lab_id", "video_id", "frames_per_second"}
        if missing := required_video - set(video_metadata.columns):
            raise ValueError(f"video metadata is missing columns: {sorted(missing)}")
        self.video_lab = {
            int(row["video_id"]): str(row["lab_id"])
            for row in video_metadata.iter_rows(named=True)
        }
        self.video_fps = {
            int(row["video_id"]): float(row["frames_per_second"])
            for row in video_metadata.iter_rows(named=True)
        }
        truth_values: defaultdict[Component, defaultdict[LabelKey, list[Interval]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for row in solution.itertuples(index=False):
            component = (str(row.lab_id), str(row.action))
            key = (int(row.video_id), str(row.agent_id), str(row.target_id))
            truth_values[component][key].append((int(row.start_frame), int(row.stop_frame)))
        self.truth: dict[Component, dict[LabelKey, tuple[Interval, ...]]] = {
            component: {key: _merged(intervals) for key, intervals in values.items()}
            for component, values in truth_values.items()
        }
        predictions: defaultdict[Component, defaultdict[PairKey, list[Interval]]] = defaultdict(
            lambda: defaultdict(list)
        )
        blockers: defaultdict[PairKey, list[tuple[int, int, str]]] = defaultdict(list)
        for row in submission.itertuples(index=False):
            video_id = int(row.video_id)
            lab = self.video_lab[video_id]
            action = str(row.action)
            pair = (video_id, str(row.agent_id), str(row.target_id))
            interval = (int(row.start_frame), int(row.stop_frame))
            predictions[(lab, action)][pair].append(interval)
            blockers[pair].append((interval[0], interval[1], action))
        self.predictions = {
            component: {key: tuple(sorted(intervals)) for key, intervals in values.items()}
            for component, values in predictions.items()
        }
        self.blockers = {key: tuple(sorted(values)) for key, values in blockers.items()}
        self.blocker_starts = {
            key: tuple(value[0] for value in values) for key, values in self.blockers.items()
        }
        self._score_cache: dict[tuple[Component, float, float], float] = {}

    def support(self, component: Component) -> tuple[int, int]:
        values = self.truth.get(component, {})
        frames = sum(stop - start for intervals in values.values() for start, stop in intervals)
        videos = len({key[0] for key in values})
        return frames, videos

    def transformed(
        self,
        component: Component,
        *,
        minimum_run_seconds: float,
        gap_fill_seconds: float,
    ) -> dict[LabelKey, tuple[Interval, ...]]:
        action = component[1]
        output: dict[LabelKey, tuple[Interval, ...]] = {}
        for pair, intervals in self.predictions.get(component, {}).items():
            video_id = pair[0]
            fps = self.video_fps[video_id]
            minimum_frames = (
                1
                if minimum_run_seconds <= 0
                else round_seconds_to_frames(minimum_run_seconds, fps)
            )
            maximum_gap = (
                0 if gap_fill_seconds <= 0 else round_seconds_to_frames(gap_fill_seconds, fps)
            )
            kept = [(start, stop) for start, stop in intervals if stop - start >= minimum_frames]
            if not kept:
                continue
            merged: list[list[int]] = [[kept[0][0], kept[0][1]]]
            for start, stop in kept[1:]:
                gap_start = merged[-1][1]
                pair_blockers = self.blockers.get(pair, ())
                blocker_index = bisect_left(self.blocker_starts.get(pair, ()), gap_start)
                blocked = False
                while (
                    blocker_index < len(pair_blockers)
                    and pair_blockers[blocker_index][0] < start
                ):
                    other_start, other_stop, other_action = pair_blockers[blocker_index]
                    if other_action != action and other_start < start and other_stop > gap_start:
                        blocked = True
                        break
                    blocker_index += 1
                if start - gap_start <= maximum_gap and not blocked:
                    merged[-1][1] = stop
                else:
                    merged.append([start, stop])
            output[pair] = tuple((start, stop) for start, stop in merged)
        return output

    def score_component(
        self,
        component: Component,
        *,
        minimum_run_seconds: float,
        gap_fill_seconds: float,
    ) -> float:
        cache_key = (component, minimum_run_seconds, gap_fill_seconds)
        cached = self._score_cache.get(cache_key)
        if cached is not None:
            return cached
        score = _f1(
            _component_counts(
                self.truth.get(component, {}),
                self.transformed(
                    component,
                    minimum_run_seconds=minimum_run_seconds,
                    gap_fill_seconds=gap_fill_seconds,
                ),
            )
        )
        self._score_cache[cache_key] = score
        return score

    def rows_for_component(
        self,
        component: Component,
        *,
        minimum_run_seconds: float,
        gap_fill_seconds: float,
    ) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        for (video_id, agent_id, target_id), intervals in self.transformed(
            component,
            minimum_run_seconds=minimum_run_seconds,
            gap_fill_seconds=gap_fill_seconds,
        ).items():
            for start, stop in intervals:
                records.append(
                    {
                        "video_id": video_id,
                        "agent_id": agent_id,
                        "target_id": target_id,
                        "action": component[1],
                        "start_frame": start,
                        "stop_frame": stop,
                    }
                )
        return records


def _filter_by_folds(
    frame: pd.DataFrame,
    fold_by_video: Mapping[int, int],
    selected_folds: set[int],
) -> pd.DataFrame:
    return frame.loc[
        [fold_by_video.get(int(video_id)) in selected_folds for video_id in frame["video_id"]]
    ].reset_index(drop=True)


def _best_candidate(
    scored: Iterable[tuple[float, float, float]],
) -> tuple[float, float, float]:
    # Prefer the simpler (smaller) transformation on an exact tie.
    return max(scored, key=lambda value: (value[0], -value[1] - value[2], -value[1], -value[2]))


def nested_interval_postprocessing(
    *,
    solution: pd.DataFrame,
    baseline_submission: pd.DataFrame,
    videos: pl.DataFrame,
    folds: pl.DataFrame,
    baseline_parameters: pl.DataFrame,
    config: Phase8DecoderConfig,
    baseline_smoothing_seconds: float,
) -> NestedPostprocessResult:
    """Tune gap/minimum-run settings on other folds using exact metric components."""
    primary = folds.filter(pl.col("is_primary_validation")).select("lab_id", "video_id", "fold")
    metadata = videos.join(primary, on=["lab_id", "video_id"], how="inner")
    fold_by_video = {
        int(row["video_id"]): int(row["fold"]) for row in metadata.iter_rows(named=True)
    }
    held_folds = sorted(set(fold_by_video.values()))
    requested = sorted(
        (str(row["lab_id"]), str(row["action"]))
        for row in baseline_parameters.select("lab_id", "action").unique().iter_rows(named=True)
    )
    candidates = tuple(
        (float(minimum), float(gap))
        for minimum in config.minimum_run_seconds
        for gap in config.gap_fill_seconds
    )
    output_rows: list[dict[str, object]] = []
    parameter_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    search_rows: list[dict[str, object]] = []
    for held_fold in held_folds:
        calibration_folds = {fold for fold in held_folds if fold != held_fold}
        held_set = {held_fold}
        calibration_solution = _filter_by_folds(solution, fold_by_video, calibration_folds)
        calibration_submission = _filter_by_folds(
            baseline_submission, fold_by_video, calibration_folds
        )
        calibration_metadata = metadata.filter(pl.col("fold").is_in(calibration_folds))
        store = IntervalCalibrationStore(
            calibration_solution, calibration_submission, calibration_metadata
        )
        official_components = sorted(store.truth)
        global_scored: list[tuple[float, float, float]] = []
        for minimum, gap in candidates:
            by_lab: defaultdict[str, list[float]] = defaultdict(list)
            for component in official_components:
                by_lab[component[0]].append(
                    store.score_component(
                        component,
                        minimum_run_seconds=minimum,
                        gap_fill_seconds=gap,
                    )
                )
            score = float(np.mean([np.mean(values) for values in by_lab.values()]))
            global_scored.append((score, minimum, gap))
        global_score, global_minimum, global_gap = _best_candidate(global_scored)
        search_rows.append(
            {
                "held_out_fold": held_fold,
                "scope": "global",
                "scope_id": "global",
                "score": global_score,
                "minimum_run_seconds": global_minimum,
                "gap_fill_seconds": global_gap,
            }
        )
        action_parameters: dict[str, tuple[float, float, str]] = {}
        for action in sorted({component[1] for component in official_components}):
            action_components = [component for component in official_components if component[1] == action]
            action_frames = sum(store.support(component)[0] for component in action_components)
            if action_frames < config.minimum_action_positive_frames:
                action_parameters[action] = (global_minimum, global_gap, "global")
                continue
            scored = []
            for minimum, gap in candidates:
                score = float(
                    np.mean(
                        [
                            store.score_component(
                                component,
                                minimum_run_seconds=minimum,
                                gap_fill_seconds=gap,
                            )
                            for component in action_components
                        ]
                    )
                )
                scored.append((score, minimum, gap))
            score, minimum, gap = _best_candidate(scored)
            action_parameters[action] = (minimum, gap, "action")
            search_rows.append(
                {
                    "held_out_fold": held_fold,
                    "scope": "action",
                    "scope_id": action,
                    "score": score,
                    "minimum_run_seconds": minimum,
                    "gap_fill_seconds": gap,
                }
            )
        selected: dict[Component, tuple[float, float, str, int, int]] = {}
        for component in requested:
            positive_frames, positive_videos = store.support(component)
            action_minimum, action_gap, action_source = action_parameters.get(
                component[1], (global_minimum, global_gap, "global")
            )
            if (
                positive_frames >= config.minimum_lab_action_positive_frames
                and positive_videos >= config.minimum_lab_action_positive_videos
            ):
                scored = [
                    (
                        store.score_component(
                            component,
                            minimum_run_seconds=minimum,
                            gap_fill_seconds=gap,
                        ),
                        minimum,
                        gap,
                    )
                    for minimum, gap in candidates
                ]
                score, minimum, gap = _best_candidate(scored)
                source = "lab_action"
                search_rows.append(
                    {
                        "held_out_fold": held_fold,
                        "scope": "lab_action",
                        "scope_id": f"{component[0]}::{component[1]}",
                        "score": score,
                        "minimum_run_seconds": minimum,
                        "gap_fill_seconds": gap,
                    }
                )
            else:
                minimum, gap, source = action_minimum, action_gap, action_source
            selected[component] = (
                minimum,
                gap,
                source,
                positive_frames,
                positive_videos,
            )
        held_solution = _filter_by_folds(solution, fold_by_video, held_set)
        held_submission = _filter_by_folds(baseline_submission, fold_by_video, held_set)
        held_metadata = metadata.filter(pl.col("fold") == held_fold)
        held_store = IntervalCalibrationStore(held_solution, held_submission, held_metadata)
        fold_output_rows: list[dict[str, object]] = []
        for component, (minimum, gap, source, positive_frames, positive_videos) in selected.items():
            component_rows = held_store.rows_for_component(
                component,
                minimum_run_seconds=minimum,
                gap_fill_seconds=gap,
            )
            fold_output_rows.extend(component_rows)
            output_rows.extend(component_rows)
            baseline_row = baseline_parameters.filter(
                (pl.col("held_out_fold") == held_fold)
                & (pl.col("lab_id") == component[0])
                & (pl.col("action") == component[1])
            )
            threshold = float(baseline_row["threshold"][0]) if baseline_row.height else 0.5
            parameter_rows.append(
                {
                    "held_out_fold": held_fold,
                    "lab_id": component[0],
                    "action": component[1],
                    "threshold": threshold,
                    "baseline_smoothing_seconds": baseline_smoothing_seconds,
                    "additional_smoothing_seconds": 0.0,
                    "hysteresis_delta": 0.0,
                    "gap_fill_seconds": gap,
                    "minimum_run_seconds": minimum,
                    "postprocess_source": source,
                    "calibration_positive_frames": positive_frames,
                    "calibration_positive_videos": positive_videos,
                    "calibration_folds": "[" + ",".join(map(str, sorted(calibration_folds))) + "]",
                    "used_held_out_labels": False,
                }
            )
        held_decoded = pd.DataFrame(fold_output_rows)
        fold_score = (
            mouse_fbeta(held_solution, held_decoded, beta=1.0)
            if not held_solution.empty and not held_decoded.empty
            else 0.0
        )
        fold_rows.append(
            {
                "fold": held_fold,
                "fbeta": fold_score,
                "submission_rows": len(held_decoded),
                "calibration_folds": "[" + ",".join(map(str, sorted(calibration_folds))) + "]",
                "used_held_out_labels": False,
            }
        )
    submission = pd.DataFrame(output_rows)
    if submission.empty:
        raise ValueError("nested Phase 8 post-processing produced an empty submission")
    submission = submission.sort_values(
        ["video_id", "start_frame", "agent_id", "target_id", "action"]
    ).reset_index(drop=True)
    submission.insert(0, "row_id", np.arange(len(submission), dtype=np.int64))
    breakdown = metric_breakdown(solution, submission, beta=1.0)
    parameters = pl.DataFrame(parameter_rows).sort("held_out_fold", "lab_id", "action")
    if parameters.filter(pl.col("used_held_out_labels")).height:
        raise AssertionError("held-out labels entered Phase 8 decoder calibration")
    return NestedPostprocessResult(
        submission=submission,
        parameters=parameters,
        by_fold=pd.DataFrame(fold_rows),
        breakdown=breakdown,
        search_summary=pl.DataFrame(search_rows).sort("held_out_fold", "scope", "scope_id"),
    )
