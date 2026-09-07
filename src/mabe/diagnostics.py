"""Detailed diagnostics that mirror the official MABe metric."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import pandas as pd
import polars as pl

from mabe.metric import HostVisibleError, mouse_fbeta


@dataclass(frozen=True, slots=True)
class MetricBreakdown:
    """Official score plus its lab and lab/action components."""

    overall: float
    by_lab: pd.DataFrame
    by_lab_action: pd.DataFrame


def _with_key(frame: pl.DataFrame, alias: str) -> pl.DataFrame:
    return frame.with_columns(
        pl.concat_str(
            [
                pl.col("video_id").cast(pl.Utf8),
                pl.col("agent_id").cast(pl.Utf8),
                pl.col("target_id").cast(pl.Utf8),
                pl.col("action"),
            ],
            separator="_",
        ).alias(alias)
    )


def _lab_counts(
    lab_solution: pl.DataFrame,
    lab_submission: pl.DataFrame,
    *,
    beta: float,
) -> list[dict[str, Any]]:
    label_frames: defaultdict[str, set[int]] = defaultdict(set)
    prediction_frames: defaultdict[str, set[int]] = defaultdict(set)

    for row in lab_solution.to_dicts():
        label_frames[row["label_key"]].update(range(row["start_frame"], row["stop_frame"]))

    for video in lab_solution["video_id"].unique():
        active_raw = lab_solution.filter(pl.col("video_id") == video)["behaviors_labeled"].first()
        if not isinstance(active_raw, str):
            raise TypeError("behaviors_labeled must contain JSON strings")
        active_labels = set(json.loads(active_raw))
        predicted_mouse_pairs: defaultdict[str, set[int]] = defaultdict(set)

        for row in lab_submission.filter(pl.col("video_id") == video).to_dicts():
            triplet = ",".join([str(row["agent_id"]), str(row["target_id"]), row["action"]])
            if triplet not in active_labels:
                continue

            new_frames = set(range(row["start_frame"], row["stop_frame"]))
            new_frames.difference_update(prediction_frames[row["prediction_key"]])
            prediction_pair = ",".join([str(row["agent_id"]), str(row["target_id"])])
            if predicted_mouse_pairs[prediction_pair].intersection(new_frames):
                raise HostVisibleError("Multiple predictions for the same frame from one agent/target pair")
            prediction_frames[row["prediction_key"]].update(new_frames)
            predicted_mouse_pairs[prediction_pair].update(new_frames)

    tps: defaultdict[str, int] = defaultdict(int)
    fns: defaultdict[str, int] = defaultdict(int)
    fps: defaultdict[str, int] = defaultdict(int)
    for key, predicted in prediction_frames.items():
        action = key.split("_")[-1]
        labeled = label_frames[key]
        tps[action] += len(predicted.intersection(labeled))
        fns[action] += len(labeled.difference(predicted))
        fps[action] += len(predicted.difference(labeled))

    distinct_actions: set[str] = set()
    for key, labeled in label_frames.items():
        action = key.split("_")[-1]
        distinct_actions.add(action)
        if key not in prediction_frames:
            fns[action] += len(labeled)

    rows: list[dict[str, Any]] = []
    for action in sorted(distinct_actions):
        tp = tps[action]
        fp = fps[action]
        fn = fns[action]
        denominator = (1 + beta**2) * tp + beta**2 * fn + fp
        fbeta = 0.0 if denominator == 0 else (1 + beta**2) * tp / denominator
        precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
        rows.append(
            {
                "action": action,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "ground_truth_frames": tp + fn,
                "predicted_frames": tp + fp,
                "precision": precision,
                "recall": recall,
                "fbeta": fbeta,
            }
        )
    return rows


def metric_breakdown(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    *,
    beta: float = 1,
    row_id_column_name: str = "row_id",
) -> MetricBreakdown:
    """Return exact-score diagnostics while asserting parity with the official oracle."""
    clean_solution = solution.drop(row_id_column_name, axis="columns", errors="ignore")
    clean_submission = submission.drop(row_id_column_name, axis="columns", errors="ignore")
    official_score = mouse_fbeta(clean_solution, clean_submission, beta=beta)

    solution_pl = pl.DataFrame(clean_solution)
    submission_pl = pl.DataFrame(clean_submission)
    solution_videos = set(solution_pl["video_id"].unique())
    submission_pl = submission_pl.filter(pl.col("video_id").is_in(solution_videos))
    solution_pl = _with_key(solution_pl, "label_key")
    submission_pl = _with_key(submission_pl, "prediction_key")

    lab_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    for lab in solution_pl["lab_id"].unique():
        lab_solution = solution_pl.filter(pl.col("lab_id") == lab).clone()
        lab_videos = set(lab_solution["video_id"].unique())
        lab_submission = submission_pl.filter(pl.col("video_id").is_in(lab_videos)).clone()
        rows = _lab_counts(lab_solution, lab_submission, beta=beta)
        for row in rows:
            action_rows.append({"lab_id": lab, **row})
        lab_fbeta = math.fsum(float(row["fbeta"]) for row in rows) / len(rows)
        lab_rows.append({"lab_id": lab, "n_actions": len(rows), "fbeta": lab_fbeta})

    reconstructed = math.fsum(float(row["fbeta"]) for row in lab_rows) / len(lab_rows)
    if not math.isclose(reconstructed, official_score, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError(
            f"diagnostic score {reconstructed} does not match official score {official_score}"
        )

    return MetricBreakdown(
        overall=official_score,
        by_lab=pd.DataFrame(lab_rows),
        by_lab_action=pd.DataFrame(action_rows),
    )
