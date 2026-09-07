from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from mabe.normalization import (
    apply_fold_normalization,
    fit_fold_normalization,
    save_fold_normalization,
)
from mabe.pose import (
    load_cached_pose,
    load_clean_pose,
    load_pose_config,
    pose_config_fingerprint,
    resample_pose,
    save_clean_pose,
)

ROOT = Path(__file__).resolve().parents[1]
POSE_CONFIG_PATH = ROOT / "configs" / "pose.yaml"


def _pose_rows(
    frames: list[int],
    *,
    mice: dict[int, list[tuple[float, float]]],
    transform: bool = False,
    omit: set[tuple[int, int, str]] | None = None,
    include_tail_midpoint: bool = False,
) -> list[dict[str, object]]:
    omitted = omit or set()
    rows: list[dict[str, object]] = []
    for mouse, centers in mice.items():
        for frame, (center_x, center_y) in zip(frames, centers, strict=True):
            parts = {
                "nose": (center_x + 2, center_y),
                "ear_left": (center_x + 1, center_y + 1),
                "ear_right": (center_x + 1, center_y - 1),
                "tail_base": (center_x - 2, center_y),
            }
            if include_tail_midpoint:
                parts["tail_midpoint"] = (center_x - 3, center_y)
            for bodypart, (x, y) in parts.items():
                if (frame, mouse, bodypart) in omitted:
                    continue
                if transform:
                    x, y = 50.0 - y, -20.0 + x
                rows.append(
                    {
                        "video_frame": frame,
                        "mouse_id": mouse,
                        "bodypart": bodypart,
                        "x": x,
                        "y": y,
                    }
                )
    return rows


def _load(
    path: Path,
    *,
    lab: str = "Lab",
    fps: float = 10.0,
    pixels_per_cm: float | None = 10.0,
    mice: tuple[str, ...] = ("mouse1",),
):
    return load_clean_pose(
        path,
        lab_id=lab,
        video_id=1,
        fps=fps,
        pixels_per_cm=pixels_per_cm,
        config=load_pose_config(POSE_CONFIG_PATH),
        mouse_ids=mice,
        video_width_pix=100,
        video_height_pix=100,
        arena_width_cm=10,
        arena_height_cm=10,
    )


def test_pose_loader_retains_noncontiguous_frames_derives_center_and_masks_gaps(
    tmp_path: Path,
) -> None:
    frames = [10, 11, 12, 13, 14, 20]
    omit = {(12, 1, "nose"), (14, 1, "nose")}
    rows = _pose_rows(
        frames,
        mice={1: [(float(frame), 5.0) for frame in frames]},
        omit=omit,
        include_tail_midpoint=True,
    )
    rows.append(
        {
            "video_frame": 13,
            "mouse_id": 1,
            "bodypart": "neck",
            "x": float("inf"),
            "y": 0.0,
        }
    )
    path = tmp_path / "tracking.parquet"
    pl.DataFrame(rows).write_parquet(path)

    pose = _load(path, lab="AdaptableSnail")
    index = {name: position for position, name in enumerate(pose.landmarks)}

    assert pose.frame_ids.tolist() == frames
    assert pose.coordinate_unit == "cm"
    assert pose.scale_valid
    assert pose.interpolated_mask[2, 0, index["nose"]]
    assert pose.availability_mask[2, 0, index["nose"]]
    assert not pose.availability_mask[4, 0, index["nose"]]
    assert pose.long_gap_mask[4, 0, index["nose"]]
    assert pose.invalid_mask[3, 0, index["neck"]]
    assert pose.availability_mask[:, 0, index["tail_mid"]].all()
    assert pose.derived_body_center[:, 0].all()
    assert np.isfinite(pose.coordinates).all()
    assert np.allclose(pose.body_length[[0, 1, 2, 3, 5], 0], 0.4)
    assert np.isclose(pose.body_length[4, 0], 0.3)


def test_synonym_mapping_is_not_applied_to_an_unreviewed_lab(tmp_path: Path) -> None:
    path = tmp_path / "tracking.parquet"
    pl.DataFrame(
        _pose_rows(
            [0, 1],
            mice={1: [(0.0, 0.0), (1.0, 0.0)]},
            include_tail_midpoint=True,
        )
    ).write_parquet(path)

    pose = _load(path, lab="UnreviewedLab")
    tail_mid = pose.landmarks.index("tail_mid")

    assert not pose.observed_mask[:, :, tail_mid].any()


def test_isolated_jump_is_removed_then_short_gap_interpolated(tmp_path: Path) -> None:
    frames = [0, 1, 2]
    rows = _pose_rows(frames, mice={1: [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]})
    for row in rows:
        if row["video_frame"] == 1 and row["bodypart"] == "nose":
            row["x"] = 10_000.0
    path = tmp_path / "jump.parquet"
    pl.DataFrame(rows).write_parquet(path)

    pose = _load(path, pixels_per_cm=None)
    nose = pose.landmarks.index("nose")

    assert pose.jump_mask[1, 0, nose]
    assert pose.invalid_mask[1, 0, nose]
    assert pose.interpolated_mask[1, 0, nose]
    assert np.allclose(pose.coordinates[1, 0, nose], [3.0, 0.0])
    assert pose.coordinate_unit == "pixel"


def test_potential_mouse_identity_swap_is_flagged_without_rewriting(tmp_path: Path) -> None:
    frames = [0, 1]
    rows = _pose_rows(
        frames,
        mice={
            1: [(0.0, 0.0), (1000.0, 0.0)],
            2: [(1000.0, 0.0), (0.0, 0.0)],
        },
    )
    path = tmp_path / "swap.parquet"
    pl.DataFrame(rows).write_parquet(path)

    pose = _load(path, fps=1.0, pixels_per_cm=None, mice=("mouse1", "mouse2"))

    assert pose.identity_swap_mask[1].all()
    assert pose.center[1, 0, 0] > pose.center[1, 1, 0]


def test_body_geometry_is_translation_and_rotation_invariant(tmp_path: Path) -> None:
    frames = [0, 1, 2]
    mice = {1: [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]}
    original_path = tmp_path / "original.parquet"
    transformed_path = tmp_path / "transformed.parquet"
    pl.DataFrame(_pose_rows(frames, mice=mice)).write_parquet(original_path)
    pl.DataFrame(_pose_rows(frames, mice=mice, transform=True)).write_parquet(transformed_path)

    original = _load(original_path, pixels_per_cm=None)
    transformed = _load(transformed_path, pixels_per_cm=None)

    assert np.allclose(original.body_length, transformed.body_length)
    assert np.allclose(transformed.body_axis[..., 0], -original.body_axis[..., 1])
    assert np.allclose(transformed.body_axis[..., 1], original.body_axis[..., 0])


def test_30hz_resampling_and_inverse_mapping_have_no_drift(tmp_path: Path) -> None:
    frames = [0, 1, 2]
    path = tmp_path / "tracking.parquet"
    pl.DataFrame(
        _pose_rows(frames, mice={1: [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]})
    ).write_parquet(path)
    pose = _load(path, fps=10.0, pixels_per_cm=None)

    resampled = resample_pose(pose, target_fps=30.0, max_gap_seconds=0.2)

    assert np.allclose(resampled.timestamps, np.arange(7) / 30)
    assert resampled.timestamps[-1] == pose.timestamps[-1]
    assert resampled.time_map.original_to_resampled.tolist() == [0, 3, 6]
    predictions = np.arange(7)
    assert resampled.time_map.predictions_to_original(predictions).tolist() == [0, 3, 6]
    labels = np.asarray([0, 1, 1], dtype=np.uint8)
    nearest = resampled.time_map.labels_to_resampled(labels, method="nearest")
    mode = resampled.time_map.labels_to_resampled(labels, method="mode")
    assert nearest[[0, 3, 6]].tolist() == labels.tolist()
    assert mode.shape == nearest.shape
    assert np.allclose(resampled.center[[0, 3, 6]], pose.center)


def test_pose_cache_round_trip_preserves_float32_and_boolean_masks(tmp_path: Path) -> None:
    tracking = tmp_path / "tracking.parquet"
    pl.DataFrame(
        _pose_rows([0, 1], mice={1: [(0.0, 0.0), (1.0, 0.0)]})
    ).write_parquet(tracking)
    pose = _load(tracking)
    cache = tmp_path / "pose.npz"

    save_clean_pose(
        pose,
        cache,
        source_sha256="source-hash",
        configuration_sha256=pose_config_fingerprint(POSE_CONFIG_PATH),
    )
    restored = load_cached_pose(cache)

    assert restored.coordinates.dtype == np.float32
    assert restored.availability_mask.dtype == np.bool_
    assert np.array_equal(restored.frame_ids, pose.frame_ids)
    assert np.array_equal(restored.coordinates, pose.coordinates)
    sidecar = json.loads(Path(f"{cache}.meta.json").read_text(encoding="utf-8"))
    assert sidecar["source_data_fingerprint"] == "source-hash"
    assert sidecar["dtypes"] == {"coordinates": "float32", "masks": "bool"}


def test_fold_normalization_excludes_held_video_and_serializes(tmp_path: Path) -> None:
    folds = pl.DataFrame(
        {
            "lab_id": ["Lab", "Lab", "Lab"],
            "video_id": [1, 2, 3],
            "fold": pl.Series([0, 1, 1], dtype=pl.Int8),
            "is_primary_validation": [True, True, True],
            "supervised_eligible": [True, True, True],
            "source_role": ["competition"] * 3,
            "source_group": ["competition"] * 3,
        }
    )
    records = pl.DataFrame(
        {
            "lab_id": ["Lab", "Lab", "Lab", "Unknown"],
            "video_id": [1, 2, 3, 99],
            "length": [100.0, 1.0, 3.0, 4.0],
        }
    )

    normalization = fit_fold_normalization(
        records,
        folds,
        held_out_fold=0,
        feature_names=["length"],
    )
    transformed = apply_fold_normalization(records, normalization)
    output = tmp_path / "fold_0.json"
    save_fold_normalization(normalization, output, source_fingerprint="data-hash")

    assert normalization.training_video_count == 2
    assert normalization.lab_stats["Lab"]["length"].mean == 2.0
    assert transformed.filter(pl.col("video_id") == 1)["length_z"].item() == 98.0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["held_out_fold"] == 0
    assert payload["training_video_count"] == 2


def test_real_pose_artifacts_cover_every_family_and_keep_25fps_uncorrected() -> None:
    quality = pl.read_parquet(ROOT / "artifacts" / "pose" / "tracking_quality.parquet")
    adaptable = pl.read_parquet(
        ROOT / "artifacts" / "pose" / "adaptable_snail_25fps.parquet"
    )

    families = quality.group_by("skeleton_id", "tracking_method").len()
    assert families.height == 13
    assert families["len"].min() >= 2
    assert quality["all_numeric_arrays_finite"].all()
    assert all(Path(path).is_file() for path in quality["cache_path"].to_list())
    assert adaptable.height == 4
    assert not adaptable["correction_enabled"].any()
    assert (
        adaptable["declared_fps_absolute_error_seconds"]
        < adaptable["thirty_fps_absolute_error_seconds"]
    ).all()
    for fold in range(5):
        normalization = ROOT / "artifacts" / "pose" / "normalization" / f"fold_{fold}.json"
        payload = json.loads(normalization.read_text(encoding="utf-8"))
        assert payload["held_out_fold"] == fold
        assert len(payload["lab_stats"]) == 15
        assert Path(f"{normalization}.meta.json").is_file()
