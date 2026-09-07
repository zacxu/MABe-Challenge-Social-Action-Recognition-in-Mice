"""Per-mouse geometric, kinematic, temporal, boundary, and quality features."""

from __future__ import annotations

import numpy as np
import polars as pl

from mabe.features.common import (
    CategoryRegistry,
    FeatureConfig,
    FeatureTable,
    FloatArray,
    MetadataContext,
    add_feature,
    angular_derivative,
    centered_mean,
    centered_sum,
    lagged_displacement,
    masked_distance,
    round_seconds_to_frames,
    safe_norm,
    temporal_derivative,
    unit_vectors,
    window_token,
)
from mabe.features.metadata import add_metadata_features
from mabe.pose import CleanPose


def _landmark(pose: CleanPose, name: str) -> tuple[FloatArray, np.ndarray]:
    index = pose.landmarks.index(name)
    return pose.coordinates[:, :, index], pose.availability_mask[:, :, index]


def _angle_between(left: FloatArray, right: FloatArray, valid: np.ndarray) -> FloatArray:
    left_unit, left_valid = unit_vectors(left, valid)
    right_unit, right_valid = unit_vectors(right, valid)
    usable = valid & left_valid & right_valid
    cosine = np.clip(np.sum(left_unit * right_unit, axis=-1), -1.0, 1.0)
    output = np.arccos(cosine).astype(np.float32)
    output[~usable] = 0.0
    return output


def _index(pose: CleanPose) -> pl.DataFrame:
    frames = pose.frame_ids.size
    mice = len(pose.mouse_ids)
    return pl.DataFrame(
        {
            "video_id": pl.Series(
                "video_id", np.full(frames * mice, pose.video_id, dtype=np.int64), dtype=pl.Int64
            ),
            "video_frame": pl.Series(
                "video_frame", np.repeat(pose.frame_ids, mice), dtype=pl.Int64
            ),
            "agent_id": pl.Series("agent_id", np.tile(pose.mouse_ids, frames), dtype=pl.String),
            "target_id": pl.Series("target_id", np.tile(pose.mouse_ids, frames), dtype=pl.String),
            "agent_index": pl.Series(
                "agent_index", np.tile(np.arange(mice, dtype=np.int16), frames), dtype=pl.Int16
            ),
            "target_index": pl.Series(
                "target_index", np.tile(np.arange(mice, dtype=np.int16), frames), dtype=pl.Int16
            ),
        }
    )


def build_solo_features(
    pose: CleanPose,
    context: MetadataContext,
    config: FeatureConfig,
    categories: CategoryRegistry,
) -> FeatureTable:
    """Build deterministic time-major features for every mouse and original frame."""
    if context.video_id != pose.video_id or context.lab_id != pose.lab_id:
        raise ValueError("pose and metadata context identify different videos")
    frames, mice = pose.center_mask.shape
    shape = (frames, mice)
    values: dict[str, FloatArray] = {}
    groups: dict[str, str] = {}
    landmark_index = {name: index for index, name in enumerate(pose.landmarks)}
    center_raw = pose.raw_pixels[:, :, landmark_index["body_center"]]
    center_valid = pose.center_mask
    metric_valid = center_valid & pose.scale_valid
    metric_center = pose.center.copy() if pose.scale_valid else np.zeros_like(pose.center)

    add_feature(values, groups, "raw__center_x_px", center_raw[..., 0], "raw")
    add_feature(values, groups, "raw__center_y_px", center_raw[..., 1], "raw")
    add_feature(values, groups, "raw__center_valid", center_valid, "raw")
    add_feature(values, groups, "metric__center_x_cm", metric_center[..., 0], "metric")
    add_feature(values, groups, "metric__center_y_cm", metric_center[..., 1], "metric")
    add_feature(values, groups, "metric__scale_valid", np.full(shape, pose.scale_valid), "metric")

    velocity, velocity_valid = temporal_derivative(
        metric_center,
        pose.timestamps,
        metric_valid,
        maximum_gap_seconds=config.maximum_derivative_gap_seconds,
    )
    acceleration, acceleration_valid = temporal_derivative(
        velocity,
        pose.timestamps,
        velocity_valid,
        maximum_gap_seconds=config.maximum_derivative_gap_seconds,
    )
    jerk, jerk_valid = temporal_derivative(
        acceleration,
        pose.timestamps,
        acceleration_valid,
        maximum_gap_seconds=config.maximum_derivative_gap_seconds,
    )
    speed = safe_norm(velocity)
    speed[~velocity_valid] = 0.0
    acceleration_magnitude = safe_norm(acceleration)
    acceleration_magnitude[~acceleration_valid] = 0.0
    jerk_magnitude = safe_norm(jerk)
    jerk_magnitude[~jerk_valid] = 0.0
    for name, array in (
        ("velocity_x_cm_s", velocity[..., 0]),
        ("velocity_y_cm_s", velocity[..., 1]),
        ("speed_cm_s", speed),
        ("acceleration_x_cm_s2", acceleration[..., 0]),
        ("acceleration_y_cm_s2", acceleration[..., 1]),
        ("acceleration_cm_s2", acceleration_magnitude),
        ("jerk_x_cm_s3", jerk[..., 0]),
        ("jerk_y_cm_s3", jerk[..., 1]),
        ("jerk_cm_s3", jerk_magnitude),
    ):
        add_feature(values, groups, f"metric__{name}", array, "metric")
    add_feature(values, groups, "metric__kinematics_valid", velocity_valid, "metric")

    nose, nose_valid = _landmark(pose, "nose")
    tail, tail_valid = _landmark(pose, "tail_base")
    ear_left, ear_left_valid = _landmark(pose, "ear_left")
    ear_right, ear_right_valid = _landmark(pose, "ear_right")
    ear_valid = ear_left_valid & ear_right_valid
    ear_spread = masked_distance(ear_left, ear_right, ear_valid)
    body_length = pose.body_length.copy()
    body_length[~pose.body_axis_mask] = 0.0
    elongation_valid = pose.body_axis_mask & ear_valid & (ear_spread > 1e-6)
    elongation = np.divide(
        body_length,
        ear_spread,
        out=np.zeros_like(body_length),
        where=elongation_valid,
    ).astype(np.float32)
    curvature_valid = nose_valid & center_valid & tail_valid
    curvature = _angle_between(nose - pose.center, tail - pose.center, curvature_valid)
    head_angle = _angle_between(
        nose - pose.center,
        pose.body_axis,
        nose_valid & center_valid & pose.body_axis_mask,
    )
    tail_mid, tail_mid_valid = _landmark(pose, "tail_mid")
    tail_angle = _angle_between(
        tail_mid - tail,
        -pose.body_axis,
        tail_mid_valid & tail_valid & pose.body_axis_mask,
    )
    orientation = np.arctan2(pose.body_axis[..., 1], pose.body_axis[..., 0]).astype(np.float32)
    orientation[~pose.body_axis_mask] = 0.0
    angular_velocity, angular_valid = angular_derivative(
        orientation,
        pose.timestamps,
        pose.body_axis_mask,
        maximum_gap_seconds=config.maximum_derivative_gap_seconds,
    )
    movement_angle = np.arctan2(velocity[..., 1], velocity[..., 0]).astype(np.float32)
    turning_valid = velocity_valid & (speed > 1e-4)
    turning_rate, turning_rate_valid = angular_derivative(
        movement_angle,
        pose.timestamps,
        turning_valid,
        maximum_gap_seconds=config.maximum_derivative_gap_seconds,
    )
    morph_features = (
        ("body_length", body_length),
        ("ear_spread", ear_spread),
        ("elongation", elongation),
        ("curvature_rad", curvature),
        ("head_angle_rad", head_angle),
        ("tail_angle_rad", tail_angle),
        ("angular_velocity_rad_s", angular_velocity),
        ("turning_rate_rad_s", turning_rate),
    )
    for name, array in morph_features:
        add_feature(values, groups, f"invariant__{name}", array, "invariant")
    add_feature(values, groups, "raw__body_orientation_rad", orientation, "raw")
    add_feature(values, groups, "invariant__body_axis_valid", pose.body_axis_mask, "invariant")
    add_feature(values, groups, "invariant__elongation_valid", elongation_valid, "invariant")
    add_feature(values, groups, "invariant__turning_valid", turning_rate_valid, "invariant")

    width = context.video_width_pix
    height = context.video_height_pix
    geometry_known = width is not None and height is not None and width > 0 and height > 0
    if geometry_known:
        assert width is not None and height is not None
        relative_x = center_raw[..., 0] / width
        relative_y = center_raw[..., 1] / height
        boundary_pixels = np.minimum.reduce(
            (
                center_raw[..., 0],
                center_raw[..., 1],
                width - center_raw[..., 0],
                height - center_raw[..., 1],
            )
        ).astype(np.float32)
        boundary_pixels = np.maximum(boundary_pixels, 0.0)
        center_distance = np.sqrt((relative_x - 0.5) ** 2 + (relative_y - 0.5) ** 2).astype(
            np.float32
        )
    else:
        relative_x = np.zeros(shape, dtype=np.float32)
        relative_y = np.zeros(shape, dtype=np.float32)
        boundary_pixels = np.zeros(shape, dtype=np.float32)
        center_distance = np.zeros(shape, dtype=np.float32)
    boundary_cm = (
        boundary_pixels / pose.pixels_per_cm
        if geometry_known and pose.scale_valid
        else np.zeros(shape, dtype=np.float32)
    )
    arena_known = (
        pose.scale_valid
        and context.arena_width_cm is not None
        and context.arena_height_cm is not None
        and context.arena_width_cm > 0
        and context.arena_height_cm > 0
    )
    if arena_known:
        assert context.arena_width_cm is not None and context.arena_height_cm is not None
        arena_relative_x = metric_center[..., 0] / context.arena_width_cm
        arena_relative_y = metric_center[..., 1] / context.arena_height_cm
    else:
        arena_relative_x = np.zeros(shape, dtype=np.float32)
        arena_relative_y = np.zeros(shape, dtype=np.float32)
    for name, array in (
        ("video_relative_x", relative_x),
        ("video_relative_y", relative_y),
        ("video_center_distance", center_distance),
        ("boundary_distance_pixels", boundary_pixels),
        ("boundary_distance_cm", boundary_cm),
        ("arena_relative_x_assumed_origin", arena_relative_x),
        ("arena_relative_y_assumed_origin", arena_relative_y),
    ):
        add_feature(values, groups, f"boundary__{name}", array, "boundary")
    add_feature(
        values,
        groups,
        "boundary__video_geometry_known",
        np.full(shape, geometry_known),
        "boundary",
    )
    add_feature(
        values,
        groups,
        "boundary__arena_dimensions_known",
        np.full(shape, arena_known),
        "boundary",
    )
    add_feature(
        values,
        groups,
        "boundary__arena_origin_known",
        np.zeros(shape, dtype=np.float32),
        "boundary",
    )

    landmark_availability = pose.availability_mask.mean(axis=2).astype(np.float32)
    interpolated_fraction = pose.interpolated_mask.mean(axis=2).astype(np.float32)
    invalid_fraction = pose.invalid_mask.mean(axis=2).astype(np.float32)
    jump_fraction = pose.jump_mask.mean(axis=2).astype(np.float32)
    quality_features = (
        ("landmark_availability", landmark_availability),
        ("landmark_missingness", 1.0 - landmark_availability),
        ("interpolated_fraction", interpolated_fraction),
        ("invalid_fraction", invalid_fraction),
        ("jump_fraction", jump_fraction),
        ("confidence", pose.confidence),
        ("derived_body_center", pose.derived_body_center.astype(np.float32)),
        ("identity_swap_candidate", pose.identity_swap_mask.astype(np.float32)),
    )
    for name, array in quality_features:
        add_feature(values, groups, f"quality__{name}", array, "quality")

    for seconds in config.temporal_windows_seconds:
        radius = round_seconds_to_frames(seconds, pose.fps)
        token = window_token(seconds)
        speed_mean, speed_support = centered_mean(speed, radius, velocity_valid)
        speed_square_mean, _ = centered_mean(speed**2, radius, velocity_valid)
        speed_std = np.sqrt(np.maximum(speed_square_mean - speed_mean**2, 0)).astype(np.float32)
        path_length = centered_sum(
            (speed / pose.fps).astype(np.float32), radius, velocity_valid
        )
        pause = (speed <= config.pause_speed_cm_per_second) & velocity_valid
        pause_ratio, _ = centered_mean(pause.astype(np.float32), radius, velocity_valid)
        angular_mean, _ = centered_mean(
            np.abs(angular_velocity).astype(np.float32), radius, angular_valid
        )
        confidence_mean, _ = centered_mean(pose.confidence, radius)
        interpolation_mean, _ = centered_mean(interpolated_fraction, radius)
        missing_mean, _ = centered_mean((1.0 - landmark_availability).astype(np.float32), radius)
        displacement, displacement_valid = lagged_displacement(
            metric_center, metric_valid, radius
        )
        elapsed = radius / pose.fps
        lag_speed = displacement / elapsed
        temporal = (
            ("speed_mean", speed_mean),
            ("speed_std", speed_std),
            ("path_length_cm", path_length),
            ("pause_ratio", pause_ratio),
            ("abs_angular_velocity_mean", angular_mean),
            ("confidence_mean", confidence_mean),
            ("interpolation_ratio", interpolation_mean),
            ("missingness_ratio", missing_mean),
            ("lag_displacement_cm", displacement),
            ("lag_speed_cm_s", lag_speed),
            ("support", speed_support),
            ("lag_valid", displacement_valid.astype(np.float32)),
        )
        for name, array in temporal:
            add_feature(values, groups, f"temporal__{name}_{token}", array, "temporal")

    add_metadata_features(
        values,
        groups,
        context=context,
        registry=categories,
        agent_ids=pose.mouse_ids,
        target_ids=None,
        frames=frames,
    )
    return FeatureTable(
        kind="solo",
        index=_index(pose),
        values=values,
        groups=groups,
        feature_version=config.feature_version,
    )
