from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from mabe.decode import centered_smooth_array, decode_probabilities
from mabe.submission import safe_fallback_submission, validate_submission
from mabe.training_data import SAMPLING_SCHEMA, cap_sampling_plan, filter_feasible_triplets
from mabe.xgb_config import load_xgboost_config
from mabe.xgb_model import (
    _positive_weight,
    choose_model_route,
    fit_fold_model,
    load_fold_model,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "xgboost.yaml"


def test_initial_model_configuration_is_central_and_valid() -> None:
    config = load_xgboost_config(CONFIG_PATH)
    assert config.xgboost.tree_method == "hist"
    assert config.xgboost.early_stopping_rounds > 0
    assert config.execution.folds == (0, 1, 2, 3, 4)
    assert config.sampler.max_rows_per_lab_action >= config.sampler.max_positive_rows_per_lab_action


def test_sampling_caps_by_fold_and_prioritizes_boundary_negatives() -> None:
    rows: list[dict[str, object]] = []
    for fold in range(5):
        for index in range(20):
            target = int(index < 5)
            rows.append(
                {
                    "lab_id": "Lab",
                    "video_id": fold,
                    "video_frame": index,
                    "agent_id": "mouse1",
                    "target_id": "mouse2",
                    "action": "approach",
                    "route": "pair",
                    "fold": fold,
                    "target": target,
                    "sampling_kind": (
                        "positive" if target else "near_boundary" if index < 10 else "background"
                    ),
                    "priority": index,
                }
            )
    config = load_xgboost_config(CONFIG_PATH).sampler.model_copy(
        update={
            "max_positive_rows_per_lab_action": 10,
            "minimum_negative_rows_per_lab_action": 10,
            "max_rows_per_lab_action": 20,
            "negative_to_positive_ratio": 1.0,
        }
    )
    result = cap_sampling_plan(
        pl.DataFrame(rows, schema=SAMPLING_SCHEMA), config, fold_count=5
    )
    counts = result.group_by("fold").agg(
        pl.col("target").sum().alias("positives"), pl.len().alias("rows")
    )
    assert counts["positives"].to_list() == [2] * 5
    assert counts["rows"].to_list() == [4] * 5
    assert result.filter(pl.col("target") == 0)["sampling_kind"].unique().to_list() == [
        "near_boundary"
    ]


def test_mask_triplets_for_unobserved_mice_are_not_supervision() -> None:
    videos = pl.DataFrame(
        {
            "lab_id": ["Lab"],
            "video_id": [1],
            "mouse_ids": [["mouse1", "mouse2", "mouse4"]],
        }
    )
    triplets = pl.DataFrame(
        {
            "lab_id": ["Lab"] * 4,
            "video_id": [1] * 4,
            "agent_id": ["mouse1", "mouse2", "mouse3", "mouse4"],
            "target_id": ["self", "mouse4", "mouse1", "mouse3"],
            "action": ["rear", "attack", "chase", "avoid"],
            "is_self": [True, False, False, False],
        }
    )
    result = filter_feasible_triplets(triplets, videos)
    assert result.select("agent_id", "target_id").rows() == [
        ("mouse1", "self"),
        ("mouse2", "mouse4"),
    ]


def test_model_routing_hierarchy() -> None:
    support = load_xgboost_config(CONFIG_PATH).support.model_copy(
        update={"auxiliary_labs_by_action": {"rare": ("CRIM13",)}}
    )
    assert choose_model_route(
        direct_positive_rows=100,
        direct_positive_videos=2,
        pooled_positive_rows=0,
        auxiliary_positive_rows=0,
        action="rare",
        support=support,
    ).route == "lab_action"
    assert choose_model_route(
        direct_positive_rows=5,
        direct_positive_videos=1,
        pooled_positive_rows=200,
        auxiliary_positive_rows=500,
        action="rare",
        support=support,
    ).route == "pooled_action"
    assert choose_model_route(
        direct_positive_rows=5,
        direct_positive_videos=1,
        pooled_positive_rows=5,
        auxiliary_positive_rows=100,
        action="rare",
        support=support,
    ).route == "auxiliary_action"
    assert choose_model_route(
        direct_positive_rows=0,
        direct_positive_videos=0,
        pooled_positive_rows=0,
        auxiliary_positive_rows=0,
        action="unknown",
        support=support,
    ).route == "no_prediction"


def test_performance_routing_suppresses_zero_direct_support_and_sqrt_weight() -> None:
    support = load_xgboost_config(
        ROOT / "configs" / "xgboost-performance.yaml"
    ).support
    decision = choose_model_route(
        direct_positive_rows=0,
        direct_positive_videos=0,
        pooled_positive_rows=10_000,
        auxiliary_positive_rows=0,
        action="rear",
        support=support,
    )
    assert decision.route == "no_prediction"
    target = np.asarray([1] * 4 + [0] * 36, dtype=np.uint8)
    assert _positive_weight(target, "sqrt_auto") == 3.0


def test_early_stopping_uses_inner_fold_and_refits_outer_training(tmp_path: Path) -> None:
    records: list[dict[str, object]] = []
    for fold in range(5):
        for index in range(20):
            target = int(index >= 10)
            records.append(
                {
                    "video_id": fold * 10 + index // 10,
                    "fold": fold,
                    "target": target,
                    "feature_a": np.float32(target + index / 100),
                    "feature_b": np.float32((index % 3) / 3),
                }
            )
    frame = pl.DataFrame(records).with_columns(
        pl.col("fold").cast(pl.Int8),
        pl.col("target").cast(pl.UInt8),
        pl.col("feature_a").cast(pl.Float32),
        pl.col("feature_b").cast(pl.Float32),
    )
    config = load_xgboost_config(CONFIG_PATH)
    config = config.model_copy(
        update={
            "xgboost": config.xgboost.model_copy(
                update={
                    "n_estimators": 30,
                    "early_stopping_rounds": 5,
                    "n_jobs": 1,
                    "min_child_weight": 1.0,
                    "learning_rate": 0.2,
                }
            )
        }
    )
    result = fit_fold_model(
        frame,
        ("feature_a", "feature_b"),
        held_out_fold=0,
        scope="Lab",
        action="approach",
        output_root=tmp_path,
        config=config,
    )
    metadata = json.loads(result.metadata_path.read_text())
    assert result.inner_validation_fold != 0
    assert not metadata["held_out_labels_used_for_early_stopping"]
    assert result.training_rows == 80
    probability = load_fold_model(result.model_path).predict_proba(
        np.asarray([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    )[:, 1]
    assert probability[1] > probability[0]


def test_centered_smoothing_and_directed_pair_conflict_resolution() -> None:
    frames = np.arange(5, dtype=np.int64)
    probabilities = np.asarray([0, 0, 1, 0, 0], dtype=np.float32)
    assert np.allclose(
        centered_smooth_array(frames, probabilities, radius_frames=1),
        [0, 1 / 3, 1 / 3, 1 / 3, 0],
    )
    rows = []
    for target in ("mouse2", "mouse3"):
        for action, probability in (("attack", 0.8), ("chase", 0.7)):
            rows.append(
                {
                    "lab_id": "Lab",
                    "video_id": 1,
                    "video_frame": 10,
                    "agent_id": "mouse1",
                    "target_id": target,
                    "action": action,
                    "loss_mask": 1,
                    "probability": probability,
                }
            )
    thresholds = pl.DataFrame(
        {"lab_id": ["Lab", "Lab"], "action": ["attack", "chase"], "threshold": [0.5, 0.5]}
    )
    decoded = decode_probabilities(pl.DataFrame(rows), thresholds)
    assert len(decoded) == 2
    assert set(decoded["target_id"]) == {"mouse2", "mouse3"}
    assert set(decoded["action"]) == {"attack"}
    assert (decoded["stop_frame"] - decoded["start_frame"]).tolist() == [1, 1]


def _submission_manifests() -> tuple[pl.DataFrame, pl.DataFrame]:
    videos = pl.DataFrame(
        {
            "split": ["test"],
            "lab_id": ["Lab"],
            "video_id": [1],
            "observed_start_frame": [0],
            "observed_stop_frame": [100],
        }
    )
    triplets = pl.DataFrame(
        {
            "split": ["test", "test"],
            "lab_id": ["Lab", "Lab"],
            "video_id": [1, 1],
            "agent_id": ["mouse1", "mouse1"],
            "target_id": ["self", "mouse2"],
            "action": ["rear", "attack"],
            "is_duplicate": [False, False],
        }
    )
    return videos, triplets


def test_submission_validation_and_fallback() -> None:
    videos, triplets = _submission_manifests()
    fallback = safe_fallback_submission(videos, triplets, split="test")
    assert validate_submission(fallback, videos=videos, triplets=triplets).height == 1
    valid = pd.DataFrame(
        [
            {
                "row_id": 0,
                "video_id": 1,
                "agent_id": "mouse1",
                "target_id": "mouse2",
                "action": "attack",
                "start_frame": 1,
                "stop_frame": 3,
            }
        ]
    )
    assert validate_submission(valid, videos=videos, triplets=triplets).height == 1
    overlap = pd.concat(
        [valid, valid.assign(row_id=1, start_frame=2, stop_frame=4)], ignore_index=True
    )
    try:
        validate_submission(overlap, videos=videos, triplets=triplets)
    except ValueError as error:
        assert "overlapping" in str(error)
    else:
        raise AssertionError("overlapping actions must be rejected")
