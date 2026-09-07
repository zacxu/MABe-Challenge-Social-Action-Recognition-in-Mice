from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from mabe.decode import hysteresis_mask, temporal_binary_postprocess
from mabe.phase8_config import load_phase8_config
from mabe.phase8_decoder import IntervalCalibrationStore, nested_interval_postprocessing
from mabe.phase8_probability_decoder import (
    probability_decoder_candidates,
    select_nested_probability_candidates,
)
from mabe.phase8_study import _cap_training_frame

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "phase8.yaml"


def test_phase8_configuration_is_valid_and_nested() -> None:
    config = load_phase8_config(CONFIG)
    assert config.phase8_version == "gbdt-phase8-v1"
    assert config.screen.held_out_folds == (0, 1)
    assert 0.0 in config.decoder.minimum_run_seconds
    assert config.decoder.minimum_lab_action_positive_videos >= 2


def test_screen_row_cap_preserves_both_classes_in_every_fold() -> None:
    rows: list[dict[str, object]] = []
    for fold in range(5):
        for target, count in ((1, 200), (0, 20)):
            for index in range(count):
                rows.append(
                    {
                        "lab_id": "AdaptableSnail",
                        "video_id": fold,
                        "video_frame": index,
                        "agent_id": "mouse1",
                        "target_id": "self",
                        "action": "rear",
                        "route": "solo",
                        "fold": fold,
                        "target": target,
                        "sampling_kind": "positive" if target else "background",
                    }
                )
    capped = _cap_training_frame(
        pl.DataFrame(rows),
        maximum_rows=100,
        negative_to_positive_ratio=3.0,
        seed=2025,
    )
    support = capped.group_by("fold").agg(pl.col("target").n_unique().alias("classes"))
    assert capped.height == 100
    assert support["classes"].to_list() == [2] * 5
    assert 0 < capped["target"].sum() < capped.height


def test_probability_decoder_grid_and_selection_exclude_held_fold_score() -> None:
    config = load_phase8_config(CONFIG).decoder
    candidates = probability_decoder_candidates(config)
    assert len(candidates) == 13
    baseline = candidates[0].candidate
    global_candidate = next(
        candidate.candidate
        for candidate in candidates
        if candidate.threshold_policy == "global"
    )
    rows = []
    for fold in range(3):
        rows.extend(
            [
                {"candidate": baseline, "evaluated_fold": fold, "exact_fbeta": 0.5},
                {
                    "candidate": global_candidate,
                    "evaluated_fold": fold,
                    "exact_fbeta": 1.0 if fold == 0 else 0.0,
                },
            ]
        )
    selection = select_nested_probability_candidates(
        pl.DataFrame(rows), candidates, (0, 1, 2)
    )
    held_zero = selection.filter(pl.col("held_out_fold") == 0).row(0, named=True)
    assert held_zero["candidate"] == baseline
    assert held_zero["calibration_folds"] == "[1, 2]"
    assert held_zero["used_held_out_labels"] is False


def test_hysteresis_gap_fill_and_minimum_run_respect_temporal_holes() -> None:
    frames = np.asarray([0, 1, 2, 3, 10, 11], dtype=np.int64)
    probabilities = np.asarray([0.1, 0.55, 0.8, 0.55, 0.55, 0.8], dtype=np.float32)
    active = hysteresis_mask(
        frames,
        probabilities,
        high_threshold=0.7,
        delta=0.2,
    )
    assert active.tolist() == [False, True, True, True, True, True]
    processed = temporal_binary_postprocess(
        frames,
        np.asarray([True, False, True, False, True, True], dtype=np.bool_),
        maximum_gap_frames=1,
        minimum_run_frames=2,
    )
    assert processed.tolist() == [True, True, True, False, True, True]


def test_interval_postprocessing_does_not_bridge_another_action() -> None:
    solution = pd.DataFrame(
        [
            {
                "lab_id": "Lab",
                "video_id": 1,
                "agent_id": "mouse1",
                "target_id": "mouse2",
                "action": "attack",
                "start_frame": 0,
                "stop_frame": 5,
            }
        ]
    )
    submission = pd.DataFrame(
        [
            {
                "video_id": 1,
                "agent_id": "mouse1",
                "target_id": "mouse2",
                "action": "attack",
                "start_frame": 0,
                "stop_frame": 2,
            },
            {
                "video_id": 1,
                "agent_id": "mouse1",
                "target_id": "mouse2",
                "action": "chase",
                "start_frame": 2,
                "stop_frame": 3,
            },
            {
                "video_id": 1,
                "agent_id": "mouse1",
                "target_id": "mouse2",
                "action": "attack",
                "start_frame": 3,
                "stop_frame": 5,
            },
        ]
    )
    videos = pl.DataFrame(
        {"lab_id": ["Lab"], "video_id": [1], "frames_per_second": [30.0]}
    )
    store = IntervalCalibrationStore(solution, submission, videos)
    transformed = store.transformed(
        ("Lab", "attack"), minimum_run_seconds=0.0, gap_fill_seconds=1.0
    )
    assert transformed[(1, "mouse1", "mouse2")] == ((0, 2), (3, 5))


def test_nested_postprocessing_never_uses_held_fold_labels() -> None:
    solution_rows = []
    submission_rows = []
    video_rows = []
    fold_rows = []
    parameter_rows = []
    for fold in range(2):
        video_id = fold + 1
        solution_rows.append(
            {
                "lab_id": "Lab",
                "video_id": video_id,
                "agent_id": "mouse1",
                "target_id": "mouse2",
                "action": "attack",
                "start_frame": 0,
                "stop_frame": 10,
                "behaviors_labeled": '["mouse1,mouse2,attack"]',
            }
        )
        submission_rows.append(
            {
                "row_id": fold,
                "video_id": video_id,
                "agent_id": "mouse1",
                "target_id": "mouse2",
                "action": "attack",
                "start_frame": 0,
                "stop_frame": 10,
            }
        )
        video_rows.append(
            {"lab_id": "Lab", "video_id": video_id, "frames_per_second": 30.0}
        )
        fold_rows.append(
            {
                "lab_id": "Lab",
                "video_id": video_id,
                "fold": fold,
                "is_primary_validation": True,
            }
        )
        parameter_rows.append(
            {
                "held_out_fold": fold,
                "lab_id": "Lab",
                "action": "attack",
                "threshold": 0.5,
            }
        )
    config = load_phase8_config(CONFIG).decoder.model_copy(
        update={
            "gap_fill_seconds": (0.0,),
            "minimum_run_seconds": (0.0,),
            "minimum_lab_action_positive_frames": 1,
            "minimum_lab_action_positive_videos": 1,
            "minimum_action_positive_frames": 1,
        }
    )
    result = nested_interval_postprocessing(
        solution=pd.DataFrame(solution_rows),
        baseline_submission=pd.DataFrame(submission_rows),
        videos=pl.DataFrame(video_rows),
        folds=pl.DataFrame(fold_rows),
        baseline_parameters=pl.DataFrame(parameter_rows),
        config=config,
        baseline_smoothing_seconds=0.1,
    )
    assert result.breakdown.overall == 1.0
    assert not result.parameters["used_held_out_labels"].any()
