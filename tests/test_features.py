from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from mabe.features import (
    MetadataContext,
    build_feature_tables,
    fit_category_registry,
    load_feature_config,
)
from mabe.features.common import round_seconds_to_frames
from mabe.features.storage import save_feature_table, validate_feature_cache
from mabe.pose import load_clean_pose, load_pose_config

ROOT = Path(__file__).resolve().parents[1]
POSE_CONFIG_PATH = ROOT / "configs" / "pose.yaml"
FEATURE_CONFIG_PATH = ROOT / "configs" / "features.yaml"


def _transform(point: tuple[float, float], *, rotate: bool) -> tuple[float, float]:
    if not rotate:
        return point
    x, y = point
    return 1000.0 - y, 500.0 + x


def _tracking_rows(
    *,
    fps: float,
    seconds: float,
    mouse_count: int,
    rotate: bool = False,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    frame_count = int(seconds * fps) + 1
    for frame in range(frame_count):
        time_seconds = frame / fps
        for mouse in range(1, mouse_count + 1):
            center_x = 100.0 + 10.0 * time_seconds + 70.0 * (mouse - 1)
            center_y = 100.0 + 20.0 * (mouse - 1)
            parts = {
                "nose": (center_x + 20.0, center_y),
                "ear_left": (center_x + 10.0, center_y + 10.0),
                "ear_right": (center_x + 10.0, center_y - 10.0),
                "tail_base": (center_x - 20.0, center_y),
                "tail_mid": (center_x - 30.0, center_y),
                "neck": (center_x + 5.0, center_y),
                "lateral_left": (center_x, center_y + 10.0),
                "lateral_right": (center_x, center_y - 10.0),
            }
            for bodypart, point in parts.items():
                x, y = _transform(point, rotate=rotate)
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


def _manifest_row(
    *,
    fps: float,
    mouse_count: int,
    tracker: str | None = "DeepLabCut",
    condition: str | None = None,
) -> dict[str, object]:
    mice = [
        {
            "slot": f"mouse{mouse}",
            "strain": "strain-a" if mouse < 3 else "strain-b",
            "color": "white",
            "sex": "male" if mouse == 1 else "female",
            "age": "adult",
            "condition": condition,
        }
        for mouse in range(1, mouse_count + 1)
    ]
    return {
        "split": "train",
        "lab_id": "Lab",
        "video_id": 1,
        "frames_per_second": fps,
        "video_width_pix": 5000,
        "video_height_pix": 5000,
        "arena_width_cm": 500.0,
        "arena_height_cm": 500.0,
        "arena_type": None,
        "arena_shape": "square",
        "tracking_method": tracker,
        "skeleton_id": "skeleton-a",
        "mouse_count": mouse_count,
        "mouse_metadata_json": json.dumps(mice),
    }


def _registry(rows: list[dict[str, object]]):
    return fit_category_registry(pl.DataFrame(rows))


def _load_pose(path: Path, *, fps: float, mouse_count: int, rotate: bool = False):
    pl.DataFrame(
        _tracking_rows(fps=fps, seconds=2.0, mouse_count=mouse_count, rotate=rotate)
    ).write_parquet(path)
    return load_clean_pose(
        path,
        lab_id="Lab",
        video_id=1,
        fps=fps,
        pixels_per_cm=10.0,
        config=load_pose_config(POSE_CONFIG_PATH),
        mouse_ids=tuple(f"mouse{mouse}" for mouse in range(1, mouse_count + 1)),
        video_width_pix=5000,
        video_height_pix=5000,
        arena_width_cm=500.0,
        arena_height_cm=500.0,
    )


def _pair_row(frame: pl.DataFrame, agent: str, target: str, video_frame: int = 5) -> dict[str, object]:
    return frame.filter(
        (pl.col("video_frame") == video_frame)
        & (pl.col("agent_id") == agent)
        & (pl.col("target_id") == target)
    ).row(0, named=True)


def test_temporal_rounding_is_central_and_half_up() -> None:
    assert round_seconds_to_frames(0.1, 25.0) == 3
    assert round_seconds_to_frames(0.25, 30.0) == 8
    assert round_seconds_to_frames(0.01, 10.0) == 1


def test_distances_and_egocentric_geometry_are_rigid_transform_invariant(
    tmp_path: Path,
) -> None:
    row = _manifest_row(fps=10.0, mouse_count=2)
    categories = _registry([row])
    config = load_feature_config(FEATURE_CONFIG_PATH)
    original = _load_pose(tmp_path / "original.parquet", fps=10.0, mouse_count=2)
    transformed = _load_pose(
        tmp_path / "transformed.parquet", fps=10.0, mouse_count=2, rotate=True
    )
    _, pair_original = build_feature_tables(
        original, MetadataContext.from_manifest_row(row), config, categories
    )
    _, pair_transformed = build_feature_tables(
        transformed, MetadataContext.from_manifest_row(row), config, categories
    )
    left = pair_original.to_frame()
    right = pair_transformed.to_frame()
    invariant_columns = [
        "metric__center_distance_cm",
        "metric__agent_nose_to_target_nose_cm",
        "metric__agent_nose_to_target_tail_cm",
        "egocentric__target_forward_cm",
        "egocentric__target_lateral_cm",
        "egocentric__heading_alignment_cos",
    ]
    assert np.allclose(left.select(invariant_columns), right.select(invariant_columns), atol=1e-5)


def test_ordered_pair_directionality_and_group_masking(tmp_path: Path) -> None:
    row = _manifest_row(fps=10.0, mouse_count=2)
    pose = _load_pose(tmp_path / "dyad.parquet", fps=10.0, mouse_count=2)
    _, table = build_feature_tables(
        pose,
        MetadataContext.from_manifest_row(row),
        load_feature_config(FEATURE_CONFIG_PATH),
        _registry([row]),
    )
    frame = table.to_frame()
    forward = _pair_row(frame, "mouse1", "mouse2")
    reverse = _pair_row(frame, "mouse2", "mouse1")
    assert np.isclose(
        float(str(forward["metric__center_distance_cm"])),
        float(str(reverse["metric__center_distance_cm"])),
    )
    assert np.isclose(
        float(str(forward["raw__relative_x_pixels"])),
        -float(str(reverse["raw__relative_x_pixels"])),
    )
    assert float(str(forward["egocentric__target_forward_cm"])) > 0
    assert float(str(reverse["egocentric__target_forward_cm"])) < 0
    group_columns = [name for name in table.feature_names if name.startswith("group__")]
    assert np.count_nonzero(frame.select(group_columns).to_numpy()) == 0


def test_third_mouse_context_is_present_only_when_a_third_mouse_exists(tmp_path: Path) -> None:
    row = _manifest_row(fps=10.0, mouse_count=3)
    pose = _load_pose(tmp_path / "triad.parquet", fps=10.0, mouse_count=3)
    _, pair = build_feature_tables(
        pose,
        MetadataContext.from_manifest_row(row),
        load_feature_config(FEATURE_CONFIG_PATH),
        _registry([row]),
    )
    frame = pair.to_frame()
    assert frame["group__third_mouse_available"].min() == 1.0
    assert frame["group__closest_third_to_agent_cm"].min() > 0
    assert frame["group__pair_distance_rank"].is_between(0.0, 1.0).all()


def test_fps_aware_speed_matches_equivalent_physical_trajectories(tmp_path: Path) -> None:
    config = load_feature_config(FEATURE_CONFIG_PATH)
    medians: list[float] = []
    for fps in (10.0, 20.0):
        row = _manifest_row(fps=fps, mouse_count=2)
        pose = _load_pose(tmp_path / f"fps-{int(fps)}.parquet", fps=fps, mouse_count=2)
        solo, _ = build_feature_tables(
            pose, MetadataContext.from_manifest_row(row), config, _registry([row])
        )
        mouse = solo.to_frame().filter(pl.col("agent_id") == "mouse1")
        medians.append(float(str(mouse["metric__speed_cm_s"].median())))
    assert np.allclose(medians, [1.0, 1.0], atol=1e-5)


def test_train_inference_schema_and_float32_cache_are_identical(tmp_path: Path) -> None:
    train_row = _manifest_row(
        fps=10.0, mouse_count=2, tracker="DeepLabCut", condition="wireless device"
    )
    inference_row = _manifest_row(fps=10.0, mouse_count=2, tracker="unseen tracker")
    categories = _registry([train_row])
    pose = _load_pose(tmp_path / "schema.parquet", fps=10.0, mouse_count=2)
    config = load_feature_config(FEATURE_CONFIG_PATH)
    train_tables = build_feature_tables(
        pose, MetadataContext.from_manifest_row(train_row), config, categories
    )
    inference_tables = build_feature_tables(
        pose, MetadataContext.from_manifest_row(inference_row), config, categories
    )
    for train, inference in zip(train_tables, inference_tables, strict=True):
        assert train.schema_hash == inference.schema_hash
        assert train.to_frame().schema == inference.to_frame().schema
        assert all(dtype == pl.Float32 for dtype in train.to_frame().select(train.feature_names).dtypes)
        assert np.isfinite(train.to_frame().select(train.feature_names).to_numpy()).all()
        cache = save_feature_table(
            train,
            tmp_path / train.kind,
            chunk_rows=37,
            source_fingerprint="source",
            configuration_sha256="config",
        )
        validate_feature_cache(cache, train)
        assert cache.row_count == train.index.height
        assert len(cache.chunks) > 1
    pair_frame = train_tables[1].to_frame()
    assert pair_frame["metadata__agent_device_indicator"].max() == 1.0
    assert pair_frame["metadata__target_device_indicator"].max() == 1.0
    inference_pair = inference_tables[1].to_frame()
    unknown_columns = [
        name
        for name in inference_tables[1].feature_names
        if name.startswith("metadata__video_tracker_is_unknown")
    ]
    assert len(unknown_columns) == 1
    assert inference_pair[unknown_columns[0]].min() == 1.0
