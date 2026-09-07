from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pytest

import mabe.phase12_inference as phase12_inference
from mabe.decode import decode_probabilities_deployment
from mabe.phase12_config import load_phase12_config
from mabe.phase12_inference import (
    InferenceManifests,
    _dataset_roots_below,
    _fallback_candidate,
    _fallback_submission,
    _tracking_statistics,
    apply_adaptable_snail_25fps_policy,
    assemble_resilient_submission,
    blend_probabilities,
    build_test_manifests_from_sources,
    discover_dataset_root,
    freeze_deployment_parameters,
    predict_decode_video_resilient,
    recover_submission_after_failure,
    write_emergency_submission,
)
from mabe.schemas import METADATA_COLUMNS, METADATA_SCHEMA, SUBMISSION_COLUMNS, SUBMISSION_SCHEMA
from mabe.submission import validate_submission


def _probability_frame(probability: float) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "lab_id": ["lab"],
            "video_id": [7],
            "video_frame": [11],
            "agent_id": ["mouse1"],
            "target_id": ["self"],
            "action": ["rear"],
            "loss_mask": [1],
            "probability": [probability],
        }
    )


def _decoder_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "lab_id": ["lab"],
            "action": ["rear"],
            "threshold": [0.5],
            "additional_smoothing_seconds": [0.0],
            "hysteresis_delta": [0.0],
            "gap_fill_seconds": [0.0],
            "minimum_run_seconds": [0.0],
        }
    )


def test_adaptable_snail_policy_scales_or_drops_only_25fps_rows() -> None:
    submission = pl.DataFrame(
        {
            "row_id": [0, 1],
            "video_id": [7, 8],
            "agent_id": ["mouse1", "mouse1"],
            "target_id": ["self", "self"],
            "action": ["rear", "rear"],
            "start_frame": [3, 3],
            "stop_frame": [8, 8],
        }
    )
    videos = pl.DataFrame(
        {
            "lab_id": ["AdaptableSnail", "AdaptableSnail"],
            "video_id": [7, 8],
            "frames_per_second": [25.0, 30.0],
        }
    )

    scaled = apply_adaptable_snail_25fps_policy(
        submission, videos, policy="scale_30_over_25"
    )
    assert scaled.filter(pl.col("video_id") == 7)["start_frame"][0] == 3
    assert scaled.filter(pl.col("video_id") == 7)["stop_frame"][0] == 9
    assert scaled.filter(pl.col("video_id") == 8)["stop_frame"][0] == 8
    dropped = apply_adaptable_snail_25fps_policy(
        submission, videos, policy="drop"
    )
    assert dropped["video_id"].to_list() == [8]
    assert dropped["row_id"].to_list() == [0]


def _video_inputs(tmp_path: Path) -> tuple[dict[str, object], pl.DataFrame, InferenceManifests]:
    row: dict[str, object] = {
        "lab_id": "lab",
        "video_id": 7,
        "frames_per_second": 30.0,
    }
    candidates = pl.DataFrame(
        {
            "agent_id": ["mouse1"],
            "target_id": ["self"],
            "action": ["rear"],
            "is_self": [True],
        }
    )
    videos = pl.DataFrame(
        {
            "split": ["test"],
            "lab_id": ["lab"],
            "video_id": [7],
            "observed_start_frame": [0],
            "observed_stop_frame": [20],
        }
    )
    triplets = pl.DataFrame(
        {
            "split": ["test"],
            "lab_id": ["lab"],
            "video_id": [7],
            "agent_id": ["mouse1"],
            "target_id": ["self"],
            "action": ["rear"],
            "is_duplicate": [False],
        }
    )
    manifests = InferenceManifests(tmp_path / "videos.parquet", tmp_path / "triplets.parquet", videos, triplets)
    return row, candidates, manifests


def _raise_runtime(*args: Any, **kwargs: Any) -> pl.DataFrame:
    raise RuntimeError("simulated model failure")


def test_discover_dataset_root_accepts_explicit_hidden_layout(tmp_path: Path) -> None:
    (tmp_path / "test.csv").write_text("video_id\n", encoding="utf-8")
    (tmp_path / "test_tracking").mkdir()
    assert discover_dataset_root(tmp_path) == tmp_path.resolve()


def test_nested_kaggle_dataset_discovery(tmp_path: Path) -> None:
    dataset = tmp_path / "competitions" / "MABe-mouse-behavior-detection"
    dataset.mkdir(parents=True)
    (dataset / "test.csv").write_text("video_id\n", encoding="utf-8")
    (dataset / "test_tracking").mkdir()
    assert _dataset_roots_below(tmp_path) == [dataset.resolve()]


def test_dataset_discovery_prefers_hidden_scale_over_public_mount(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    public = tmp_path / "competitions" / "MABe-mouse-behavior-detection"
    private = tmp_path / "mabe-mouse-behavior-detection"
    for root, video_ids in ((public, [7]), (private, [101, 102, 103])):
        tracking = root / "test_tracking" / "lab"
        tracking.mkdir(parents=True)
        pl.DataFrame({"video_id": video_ids}).write_csv(root / "test.csv")
        for video_id in video_ids:
            pl.DataFrame({"video_frame": [0]}).write_parquet(tracking / f"{video_id}.parquet")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MABE_DATASET_ROOT", str(public))

    assert discover_dataset_root() == private.resolve()


def test_tracking_statistics_isolates_files_and_physical_schemas(tmp_path: Path) -> None:
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    invalid = tmp_path / "invalid.parquet"
    pl.DataFrame(
        {
            "video_frame": pl.Series([0, 1], dtype=pl.Int32),
            "mouse_id": pl.Series([1, 1], dtype=pl.Int32),
            "bodypart": ["nose", "nose"],
        }
    ).write_parquet(first)
    pl.DataFrame(
        {
            "video_frame": pl.Series([10, 12], dtype=pl.Int64),
            "mouse_id": pl.Series([2, 2], dtype=pl.Int64),
            "bodypart": ["tail", "tail"],
        }
    ).write_parquet(second)
    pl.DataFrame({"video_frame": [0], "mouse_id": [1]}).write_parquet(invalid)

    statistics = _tracking_statistics((first, invalid, second))

    assert set(statistics) == {
        str(first.resolve()).casefold(),
        str(second.resolve()).casefold(),
    }
    assert statistics[str(first.resolve()).casefold()]["mouse_ids"] == ["mouse1"]
    assert statistics[str(second.resolve()).casefold()]["observed_stop_frame"] == 13
    assert statistics[str(second.resolve()).casefold()]["missing_frame_rate"] == pytest.approx(1 / 3)


def test_dynamic_manifest_quarantines_bad_videos_without_global_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    tracking_root = dataset / "train_tracking" / "lab"
    tracking_root.mkdir(parents=True)

    def metadata_row(video_id: int, mask: str) -> dict[str, object | None]:
        row: dict[str, object | None] = {column: None for column in METADATA_COLUMNS}
        row.update(
            {
                "lab_id": "lab",
                "video_id": video_id,
                "frames_per_second": 30.0,
                "video_duration_sec": 1.0,
                "body_parts_tracked": '["nose"]',
                "behaviors_labeled": mask,
            }
        )
        return row

    pl.DataFrame(
        [
            metadata_row(1, '["mouse1,self,rear"]'),
            metadata_row(2, "not-json"),
            metadata_row(3, '["mouse2,self,rear"]'),
        ],
        schema=METADATA_SCHEMA,
    ).write_csv(dataset / "train.csv")
    for video_id in (1, 2, 3):
        pl.DataFrame(
            {
                "video_frame": [0, 1],
                "mouse_id": [1, 1],
                "bodypart": ["nose", "nose"],
            }
        ).write_parquet(tracking_root / f"{video_id}.parquet")

    current_explode = pl.DataFrame.explode

    def kaggle_compatible_explode(
        frame: pl.DataFrame,
        columns: str,
        *more_columns: str,
    ) -> pl.DataFrame:
        return current_explode(frame, columns, *more_columns)

    monkeypatch.setattr(pl.DataFrame, "explode", kaggle_compatible_explode)

    manifests = build_test_manifests_from_sources(
        metadata_path=dataset / "train.csv",
        tracking_root=dataset / "train_tracking",
        dataset_root=dataset,
        output_root=tmp_path / "manifests",
    )

    assert manifests.videos["video_id"].to_list() == [1]
    assert manifests.triplets["video_id"].unique().to_list() == [1]
    report = (tmp_path / "manifests" / "manifest_build_report.json").read_text(encoding="utf-8")
    assert '"malformed_behavior_masks": 1' in report


def test_resilient_video_uses_gbdt_when_temporal_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    row, candidates, manifests = _video_inputs(tmp_path)
    monkeypatch.setattr(phase12_inference, "_predict_gbdt_video", lambda *args, **kwargs: _probability_frame(0.9))
    monkeypatch.setattr(phase12_inference, "_predict_temporal_video", _raise_runtime)
    result = predict_decode_video_resilient(
        row,
        candidates,
        pl.DataFrame(),
        pl.DataFrame(),
        _decoder_frame(),
        config=load_phase12_config("configs/inference.yaml"),
        dataset_root=tmp_path,
        manifests=manifests,
        cache_root=tmp_path / "cache",
    )
    assert result.inference_route == "gbdt_only"
    assert result.decoded.height == 1
    assert [failure["stage"] for failure in result.failures] == ["temporal"]


def test_resilient_video_uses_temporal_when_gbdt_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    row, candidates, manifests = _video_inputs(tmp_path)
    monkeypatch.setattr(phase12_inference, "_predict_gbdt_video", _raise_runtime)
    monkeypatch.setattr(
        phase12_inference,
        "_predict_temporal_video",
        lambda *args, **kwargs: _probability_frame(0.9).drop("loss_mask"),
    )
    result = predict_decode_video_resilient(
        row,
        candidates,
        pl.DataFrame(),
        pl.DataFrame(),
        _decoder_frame(),
        config=load_phase12_config("configs/inference.yaml"),
        dataset_root=tmp_path,
        manifests=manifests,
        cache_root=tmp_path / "cache",
    )
    assert result.inference_route == "temporal_only"
    assert result.decoded.height == 1
    assert [failure["stage"] for failure in result.failures] == ["gbdt"]


def test_resilient_video_skips_when_both_models_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    row, candidates, manifests = _video_inputs(tmp_path)
    monkeypatch.setattr(phase12_inference, "_predict_gbdt_video", _raise_runtime)
    monkeypatch.setattr(phase12_inference, "_predict_temporal_video", _raise_runtime)
    result = predict_decode_video_resilient(
        row,
        candidates,
        pl.DataFrame(),
        pl.DataFrame(),
        _decoder_frame(),
        config=load_phase12_config("configs/inference.yaml"),
        dataset_root=tmp_path,
        manifests=manifests,
        cache_root=tmp_path / "cache",
    )
    assert result.inference_route == "skipped"
    assert result.decoded.is_empty()
    assert result.fallback.is_empty()
    assert {failure["stage"] for failure in result.failures} == {"gbdt", "temporal"}


def test_partial_video_failure_preserves_successful_submission(tmp_path: Path) -> None:
    _, _, manifests = _video_inputs(tmp_path)
    successful = pl.DataFrame(
        {
            "row_id": [0],
            "video_id": [7],
            "agent_id": ["mouse1"],
            "target_id": ["self"],
            "action": ["rear"],
            "start_frame": [11],
            "stop_frame": [12],
        },
        schema=SUBMISSION_SCHEMA,
    )
    empty = pl.DataFrame(schema=SUBMISSION_SCHEMA)
    assembled, used_fallback = assemble_resilient_submission([empty, successful], [], manifests)
    checked = validate_submission(
        assembled,
        videos=manifests.videos,
        triplets=manifests.triplets,
        allowed_splits=("test",),
    )
    assert checked.height == 1
    assert not used_fallback


def test_all_video_failures_emit_one_valid_global_fallback(tmp_path: Path) -> None:
    _, _, manifests = _video_inputs(tmp_path)
    assembled, used_fallback = assemble_resilient_submission([], [], manifests)
    checked = validate_submission(
        assembled,
        videos=manifests.videos,
        triplets=manifests.triplets,
        allowed_splits=("test",),
    )
    assert checked.height == 1
    assert used_fallback


def test_emergency_submission_uses_observed_frame_and_allowed_triplet(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    tracking_root = dataset / "test_tracking" / "lab"
    tracking_root.mkdir(parents=True)
    pl.DataFrame(
        {
            "video_frame": [5, 6],
            "mouse_id": [1, 1],
        }
    ).write_parquet(tracking_root / "7.parquet")
    pl.DataFrame(
        {
            "lab_id": ["lab"],
            "video_id": [7],
            "behaviors_labeled": ['["mouse1,self,rear"]'],
        }
    ).write_csv(dataset / "test.csv")
    submission_path = tmp_path / "submission.csv"
    assert write_emergency_submission(dataset, submission_path) == submission_path
    submission = pl.read_csv(submission_path)
    assert submission.to_dicts() == [
        {
            "row_id": 0,
            "video_id": 7,
            "agent_id": "mouse1",
            "target_id": "self",
            "action": "rear",
            "start_frame": 5,
            "stop_frame": 6,
        }
    ]


def test_recovery_prefers_completed_video_checkpoints(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    tracking_root = dataset / "test_tracking" / "lab"
    tracking_root.mkdir(parents=True)
    pl.DataFrame({"video_frame": [5], "mouse_id": [1]}).write_parquet(tracking_root / "7.parquet")
    pl.DataFrame(
        {
            "lab_id": ["lab"],
            "video_id": [7],
            "behaviors_labeled": ['["mouse1,self,rear"]'],
        }
    ).write_csv(dataset / "test.csv")
    submission_path = tmp_path / "submission.csv"
    write_emergency_submission(dataset, submission_path)
    checkpoint_root = tmp_path / "work" / "video_predictions" / "lab" / "7"
    checkpoint_root.mkdir(parents=True)
    pl.DataFrame(
        {
            "row_id": [0, 1],
            "video_id": [7, 7],
            "agent_id": ["mouse1", "mouse1"],
            "target_id": ["self", "self"],
            "action": ["rear", "rear"],
            "start_frame": [5, 9],
            "stop_frame": [7, 11],
        },
        schema=SUBMISSION_SCHEMA,
    ).write_csv(checkpoint_root / "intervals.csv")
    recover_submission_after_failure(dataset, tmp_path / "work", submission_path)
    recovered = pl.read_csv(submission_path)
    assert recovered.height == 2
    assert recovered["row_id"].to_list() == [0, 1]


def test_blend_requires_exact_keys_and_uses_convex_weights() -> None:
    gbdt = _probability_frame(0.8)
    temporal = _probability_frame(0.2).drop("loss_mask")
    weights = pl.DataFrame(
        {
            "action": ["rear"],
            "gbdt_weight": [0.75],
            "temporal_phase9_weight": [0.25],
        }
    )
    blended = blend_probabilities(gbdt, temporal, weights)
    assert blended["probability"][0] == pytest.approx(0.65)
    with pytest.raises(ValueError, match="do not cover"):
        blend_probabilities(
            gbdt,
            temporal.with_columns(pl.lit(12).alias("video_frame")),
            weights,
        )


def test_fallback_uses_highest_normalized_confidence() -> None:
    probabilities = pl.concat(
        [
            _probability_frame(0.4),
            _probability_frame(0.7).with_columns(
                pl.lit(12).alias("video_frame"), pl.lit("run").alias("action")
            ),
        ],
        how="vertical_relaxed",
    )
    decoder = pl.DataFrame(
        {
            "lab_id": ["lab", "lab"],
            "action": ["rear", "run"],
            "threshold": [0.2, 0.7],
        }
    )
    candidate = _fallback_candidate(probabilities, decoder)
    assert candidate["action"][0] == "rear"
    fallback = _fallback_submission(candidate)
    assert tuple(fallback.columns) == SUBMISSION_COLUMNS
    assert fallback["start_frame"][0] == 11
    assert fallback["stop_frame"][0] == 12


def test_crossfit_deployment_parameters_are_complete_and_label_free() -> None:
    config = load_phase12_config("configs/inference.yaml")
    weights, decoder = freeze_deployment_parameters(config)
    assert weights.height == 34
    assert decoder.height == 102
    assert weights["source_fold_count"].unique().to_list() == [5]
    assert decoder["source_fold_count"].unique().to_list() == [5]
    assert decoder["additional_smoothing_seconds"].unique().to_list() == [0.2]
    assert decoder["hysteresis_delta"].unique().to_list() == [0.05]
    assert not weights["used_test_labels"].any()
    assert not decoder["used_test_labels"].any()
    sums = weights.select(
        (pl.col("gbdt_weight") + pl.col("temporal_phase9_weight")).alias("sum")
    )["sum"]
    assert all(value == pytest.approx(1.0) for value in sums)


def test_deployment_decoder_resolves_conflicts_before_gap_filling() -> None:
    probabilities = pl.DataFrame(
        {
            "lab_id": ["lab"] * 6,
            "video_id": [7] * 6,
            "video_frame": [0, 1, 2, 0, 1, 2],
            "agent_id": ["mouse1"] * 6,
            "target_id": ["self"] * 6,
            "action": ["a", "a", "a", "b", "b", "b"],
            "loss_mask": [1] * 6,
            "probability": [0.9, 0.1, 0.9, 0.1, 0.95, 0.1],
        }
    )
    parameters = pl.DataFrame(
        {
            "lab_id": ["lab", "lab"],
            "action": ["a", "b"],
            "threshold": [0.5, 0.5],
            "additional_smoothing_seconds": [0.0, 0.0],
            "hysteresis_delta": [0.0, 0.0],
            "gap_fill_seconds": [1.0, 0.0],
            "minimum_run_seconds": [0.0, 0.0],
        }
    )
    decoded = decode_probabilities_deployment(
        probabilities,
        parameters,
        fps_by_video={7: 1.0},
    )
    assert decoded.drop(columns="row_id").to_dict("records") == [
        {
            "video_id": 7,
            "agent_id": "mouse1",
            "target_id": "self",
            "action": "a",
            "start_frame": 0,
            "stop_frame": 1,
        },
        {
            "video_id": 7,
            "agent_id": "mouse1",
            "target_id": "self",
            "action": "b",
            "start_frame": 1,
            "stop_frame": 2,
        },
        {
            "video_id": 7,
            "agent_id": "mouse1",
            "target_id": "self",
            "action": "a",
            "start_frame": 2,
            "stop_frame": 3,
        },
    ]
