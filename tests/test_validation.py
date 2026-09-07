from __future__ import annotations

import hashlib
from pathlib import Path

import polars as pl

from mabe.validation import (
    assign_grouped_folds,
    build_duplicate_groups,
    build_video_action_support,
    classify_labs,
    normalized_trajectory_signature,
    primary_split,
    stress_split,
    support_distribution,
)


def _tracking(path: Path, *, transformed: bool = False, distinct: bool = False) -> None:
    records: list[dict[str, object]] = []
    for frame in range(10):
        points = {
            1: [(float(frame), 0.0), (float(frame), 2.0)],
            2: [(0.0, float(frame + 1)), (2.0, float(frame + 1))],
        }
        if distinct:
            points[2] = [(20.0 + frame**2, 5.0), (22.0 + frame**2, 5.0)]
        for mouse, positions in points.items():
            output_mouse = 3 - mouse if transformed else mouse
            for bodypart, (x, y) in zip(("nose", "tail"), positions, strict=True):
                if transformed:
                    x, y = 100.0 - 3.0 * y, -50.0 + 3.0 * x
                records.append(
                    {
                        "video_frame": frame,
                        "mouse_id": output_mouse,
                        "bodypart": bodypart,
                        "x": x,
                        "y": y,
                    }
                )
    pl.DataFrame(records).write_parquet(path)


def test_lab_classification_uses_auxiliary_source_conventions() -> None:
    videos = pl.DataFrame(
        {
            "split": ["train"] * 9,
            "lab_id": [
                "CompetitionA",
                "CompetitionB",
                "CalMS21_task1",
                "CalMS21_task2",
                "CalMS21_supplemental",
                "CRIM13",
                "MABe22_keypoints",
                "MABe22_movies",
                "CompetitionA",
            ],
        }
    )

    result = classify_labs(videos, validate_expected=False)

    assert result.competition_labs == ("CompetitionA", "CompetitionB")
    assert result.auxiliary_sources == (
        "CRIM13",
        "CalMS21_supplemental",
        "CalMS21_task1",
        "CalMS21_task2",
        "MABe22_keypoints",
        "MABe22_movies",
    )
    assert result.labs.filter(pl.col("lab_id") == "CalMS21_task1")["source_group"].item() == "calms21"


def test_normalized_signature_is_invariant_to_geometry_and_mouse_permutation(
    tmp_path: Path,
) -> None:
    original = tmp_path / "original.parquet"
    transformed = tmp_path / "transformed.parquet"
    _tracking(original)
    _tracking(transformed, transformed=True)

    left = normalized_trajectory_signature(
        original,
        observed_start_frame=0,
        observed_stop_frame=10,
        mouse_count=2,
        sample_count=10,
    )
    right = normalized_trajectory_signature(
        transformed,
        observed_start_frame=0,
        observed_stop_frame=10,
        mouse_count=2,
        sample_count=10,
    )

    assert left.strict == right.strict
    assert left.coarse == right.coarse
    assert left.feature_count == right.feature_count


def test_duplicate_groups_combine_exact_and_normalized_copies(tmp_path: Path) -> None:
    paths = [tmp_path / f"{index}.parquet" for index in range(4)]
    _tracking(paths[0])
    _tracking(paths[1], transformed=True)
    _tracking(paths[2], distinct=True)
    _tracking(paths[3], distinct=True)
    hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    videos = pl.DataFrame(
        {
            "split": ["train"] * 4,
            "lab_id": ["Lab"] * 4,
            "video_id": [1, 2, 3, 4],
            "supervised_eligible": [True] * 4,
            "skeleton_id": ["skeleton"] * 4,
            "mouse_count": [2] * 4,
            "observed_start_frame": [0] * 4,
            "observed_stop_frame": [10] * 4,
            "observed_frame_count": [10] * 4,
            "tracking_row_count": [40] * 4,
            "tracking_path": [str(path) for path in paths],
        }
    )
    sources = pl.DataFrame(
        {
            "split": ["train"] * 4,
            "kind": ["tracking"] * 4,
            "lab_id": ["Lab"] * 4,
            "video_id": [1, 2, 3, 4],
            "sha256": [hashes[0], hashes[1], hashes[2], hashes[2]],
        }
    )

    duplicates = build_duplicate_groups(
        videos,
        sources,
        dataset_root=tmp_path,
        sample_count=10,
        max_workers=1,
    )

    group_by_video = dict(
        duplicates.select("video_id", "duplicate_group_id").iter_rows()
    )
    assert group_by_video[1] == group_by_video[2]
    assert group_by_video[3] == group_by_video[4]
    assert group_by_video[1] != group_by_video[3]
    assert duplicates.filter(pl.col("video_id") == 4)["duplicate_type"].item() == "exact_file"


def test_video_action_support_unions_triplets_and_filters_invalid_rows(tmp_path: Path) -> None:
    annotation_path = tmp_path / "annotation.parquet"
    pl.DataFrame(
        {
            "agent_id": [1, 1, 1, 1],
            "target_id": [2, 2, 1, 2],
            "action": ["attack", "attack", "groom", "attack"],
            "start_frame": [1, 3, 5, 9],
            "stop_frame": [4, 6, 7, 9],
        }
    ).write_parquet(annotation_path)
    videos = pl.DataFrame(
        {
            "lab_id": ["Lab"],
            "video_id": [1],
            "supervised_eligible": [True],
            "annotation_path": [str(annotation_path)],
            "mouse_ids": [["mouse1", "mouse2"]],
            "observed_start_frame": [0],
            "observed_stop_frame": [10],
            "positive_frame_count": [7],
        }
    )
    triplets = pl.DataFrame(
        {
            "split": ["train", "train"],
            "lab_id": ["Lab", "Lab"],
            "video_id": [1, 1],
            "agent_id": ["mouse1", "mouse1"],
            "target_id": ["mouse2", "self"],
            "action": ["attack", "groom"],
            "is_duplicate": [False, False],
        }
    )

    support = build_video_action_support(videos, triplets, dataset_root=tmp_path)

    assert support.select("action", "positive_frames").to_dicts() == [
        {"action": "attack", "positive_frames": 5},
        {"action": "groom", "positive_frames": 2},
    ]


def test_grouped_fold_assignment_is_deterministic_and_spreads_feasible_support() -> None:
    videos = pl.DataFrame(
        {
            "lab_id": ["LabA"] * 10 + ["LabB"] * 10,
            "video_id": list(range(20)),
            "duplicate_group_id": ["shared"] * 2 + [f"group-{index}" for index in range(2, 20)],
        }
    )
    support = pl.DataFrame(
        {
            "lab_id": ["LabA"] * 10 + ["LabB"] * 10,
            "video_id": list(range(20)),
            "action": ["attack" if index % 2 == 0 else "chase" for index in range(20)],
            "positive_frames": [10 + index for index in range(20)],
        }
    )

    first = assign_grouped_folds(videos, support, seed=9)
    second = assign_grouped_folds(videos, support, seed=9)

    assert first == second
    assert set(first.values()) == set(range(5))
    joined = videos.with_columns(
        pl.col("duplicate_group_id").replace_strict(first, return_dtype=pl.Int8).alias("fold")
    )
    distribution = support.join(joined, on=["lab_id", "video_id"]).group_by(
        "lab_id", "action"
    ).agg(pl.col("fold").n_unique().alias("folds"))
    assert distribution["folds"].min() >= 2


def test_primary_and_stress_splits_enforce_source_policy() -> None:
    folds = pl.DataFrame(
        {
            "lab_id": ["LabA", "LabA", "LabB", "CalMS21_task1", "CRIM13", "MABe22_movies"],
            "video_id": [1, 2, 3, 4, 5, 6],
            "fold": [0, 1, 0, 2, 3, None],
            "duplicate_group_id": [f"g{index}" for index in range(6)],
            "label_state": ["labeled"] * 5 + ["unlabeled_mabe22"],
            "supervised_eligible": [True] * 5 + [False],
            "source_role": ["competition"] * 3 + ["auxiliary"] * 3,
            "source_group": ["competition"] * 3 + ["calms21", "crim13", "mabe22"],
            "is_primary_validation": [True] * 3 + [False] * 3,
            "positive_frame_count": [1] * 5 + [0],
            "positive_action_count": [1] * 5 + [0],
        }
    ).with_columns(pl.col("fold").cast(pl.Int8))
    videos = pl.DataFrame(
        {
            "split": ["train"] * 6,
            "lab_id": folds["lab_id"],
            "video_id": folds["video_id"],
            "skeleton_id": ["s1", "s1", "s2", "s1", "s2", "s3"],
            "tracking_method": ["DLC", "DLC", "MARS", "DLC", "MARS", "HRNet"],
        }
    )

    split = primary_split(folds, held_out_fold=0, include_auxiliary=True)

    assert set(split.validation["video_id"].to_list()) == {1, 3}
    assert set(split.train["video_id"].to_list()) == {2, 4, 5}
    assert 6 not in split.train["video_id"].to_list()
    per_lab = primary_split(folds, held_out_fold=0, target_lab="LabA")
    assert per_lab.train["video_id"].to_list() == [2]
    assert per_lab.validation["video_id"].to_list() == [1]
    stress = stress_split(folds, videos, dimension="tracker", value="DLC")
    assert set(stress.validation["video_id"].to_list()) == {1, 2}
    assert stress.train["video_id"].to_list() == [3]


def test_support_report_marks_only_intrinsically_rare_exceptions() -> None:
    folds = pl.DataFrame(
        {
            "lab_id": ["Lab", "Lab", "Lab"],
            "video_id": [1, 2, 3],
            "fold": pl.Series([0, 1, 2], dtype=pl.Int8),
            "duplicate_group_id": ["a", "b", "c"],
        }
    )
    support = pl.DataFrame(
        {
            "lab_id": ["Lab", "Lab", "Lab"],
            "video_id": [1, 2, 3],
            "action": ["common", "common", "rare"],
            "positive_frames": [10, 20, 3],
        }
    )

    report = support_distribution(folds, support)

    common = report.filter(pl.col("action") == "common")
    rare = report.filter(pl.col("action") == "rare")
    assert common["positive_fold_count"].unique().item() == 2
    assert common["coverage_exception"].null_count() == common.height
    assert set(rare["coverage_exception"].to_list()) == {"only_one_independent_positive_group"}


def test_real_dataset_confirms_expected_source_counts() -> None:
    videos = pl.read_parquet(Path("artifacts/manifests/videos.parquet"))
    result = classify_labs(videos)
    assert len(result.competition_labs) == 15
    assert len(result.auxiliary_sources) == 6


def test_real_fold_artifacts_pass_leakage_and_support_gates() -> None:
    folds = pl.read_parquet("artifacts/folds/folds.parquet")
    support = pl.read_parquet("artifacts/folds/support_distribution.parquet")

    crossings = (
        folds.filter(pl.col("fold").is_not_null())
        .group_by("duplicate_group_id")
        .agg(pl.col("fold").n_unique().alias("n_folds"))
        .filter(pl.col("n_folds") > 1)
    )
    assert crossings.is_empty()
    assert folds.filter(pl.col("is_primary_validation"))["lab_id"].n_unique() == 15
    assert folds.filter(
        pl.col("is_primary_validation") & (pl.col("source_role") != "competition")
    ).is_empty()
    assert folds.filter(
        (pl.col("source_group") == "mabe22") & pl.col("fold").is_not_null()
    ).is_empty()
    assert support.filter(
        pl.col("coverage_exception") == "assignment_coverage_shortfall"
    ).is_empty()
