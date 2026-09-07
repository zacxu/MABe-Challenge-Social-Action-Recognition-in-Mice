"""Directed agent-target geometry, motion, interaction, and temporal features."""

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
    centered_mean,
    egocentric_coordinates,
    masked_distance,
    past_future_change,
    round_seconds_to_frames,
    safe_norm,
    scalar_derivative,
    temporal_derivative,
    unit_vectors,
    window_token,
)
from mabe.features.group import add_group_features
from mabe.features.metadata import add_metadata_features
from mabe.pose import CleanPose


def ordered_pairs(mouse_count: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (agent, target)
        for agent in range(mouse_count)
        for target in range(mouse_count)
        if agent != target
    )


def _index(pose: CleanPose, pairs: tuple[tuple[int, int], ...]) -> pl.DataFrame:
    frames = pose.frame_ids.size
    pair_count = len(pairs)
    agent_indices = np.asarray([agent for agent, _ in pairs], dtype=np.int16)
    target_indices = np.asarray([target for _, target in pairs], dtype=np.int16)
    agent_ids = tuple(pose.mouse_ids[index] for index in agent_indices)
    target_ids = tuple(pose.mouse_ids[index] for index in target_indices)
    return pl.DataFrame(
        {
            "video_id": pl.Series(
                "video_id",
                np.full(frames * pair_count, pose.video_id, dtype=np.int64),
                dtype=pl.Int64,
            ),
            "video_frame": pl.Series(
                "video_frame", np.repeat(pose.frame_ids, pair_count), dtype=pl.Int64
            ),
            "agent_id": pl.Series("agent_id", np.tile(agent_ids, frames), dtype=pl.String),
            "target_id": pl.Series("target_id", np.tile(target_ids, frames), dtype=pl.String),
            "agent_index": pl.Series(
                "agent_index", np.tile(agent_indices, frames), dtype=pl.Int16
            ),
            "target_index": pl.Series(
                "target_index", np.tile(target_indices, frames), dtype=pl.Int16
            ),
        }
    )


def _landmark_pair(
    pose: CleanPose,
    name: str,
    indices: np.ndarray,
) -> tuple[FloatArray, np.ndarray]:
    landmark = pose.landmarks.index(name)
    return pose.coordinates[:, indices, landmark], pose.availability_mask[:, indices, landmark]


def build_pair_features(
    pose: CleanPose,
    context: MetadataContext,
    config: FeatureConfig,
    categories: CategoryRegistry,
) -> FeatureTable:
    """Build time-major features for every ordered non-self pair."""
    if context.video_id != pose.video_id or context.lab_id != pose.lab_id:
        raise ValueError("pose and metadata context identify different videos")
    pairs = ordered_pairs(len(pose.mouse_ids))
    frames = pose.frame_ids.size
    agent_indices = np.asarray([agent for agent, _ in pairs], dtype=np.int64)
    target_indices = np.asarray([target for _, target in pairs], dtype=np.int64)
    agent_ids = tuple(pose.mouse_ids[index] for index in agent_indices)
    target_ids = tuple(pose.mouse_ids[index] for index in target_indices)
    values: dict[str, FloatArray] = {}
    groups: dict[str, str] = {}

    agent_center_raw = pose.raw_pixels[:, agent_indices, pose.landmarks.index("body_center")]
    target_center_raw = pose.raw_pixels[:, target_indices, pose.landmarks.index("body_center")]
    raw_pair_valid = pose.center_mask[:, agent_indices] & pose.center_mask[:, target_indices]
    raw_distance = masked_distance(agent_center_raw, target_center_raw, raw_pair_valid)
    add_feature(values, groups, "raw__center_distance_pixels", raw_distance, "raw")
    add_feature(values, groups, "raw__relative_x_pixels", target_center_raw[..., 0] - agent_center_raw[..., 0], "raw")
    add_feature(values, groups, "raw__relative_y_pixels", target_center_raw[..., 1] - agent_center_raw[..., 1], "raw")

    agent_center = pose.center[:, agent_indices]
    target_center = pose.center[:, target_indices]
    pair_valid = raw_pair_valid & pose.scale_valid
    if not pose.scale_valid:
        agent_center = np.zeros_like(agent_center)
        target_center = np.zeros_like(target_center)
    relative = (target_center - agent_center).astype(np.float32)
    center_distance = masked_distance(agent_center, target_center, pair_valid)
    add_feature(values, groups, "metric__center_distance_cm", center_distance, "metric")
    add_feature(values, groups, "metric__pair_center_valid", pair_valid, "metric")

    cross_specs = (
        ("agent_nose_to_target_nose", "nose", "nose"),
        ("agent_nose_to_target_body", "nose", "body_center"),
        ("agent_nose_to_target_lateral_left", "nose", "lateral_left"),
        ("agent_nose_to_target_lateral_right", "nose", "lateral_right"),
        ("agent_nose_to_target_tail", "nose", "tail_base"),
        ("agent_tail_to_target_nose", "tail_base", "nose"),
        ("agent_body_to_target_nose", "body_center", "nose"),
    )
    cross_values: list[FloatArray] = []
    cross_masks: list[np.ndarray] = []
    for name, agent_landmark, target_landmark in cross_specs:
        agent_point, agent_valid = _landmark_pair(pose, agent_landmark, agent_indices)
        target_point, target_valid = _landmark_pair(pose, target_landmark, target_indices)
        valid = agent_valid & target_valid & pose.scale_valid
        distance = masked_distance(agent_point, target_point, valid)
        cross_values.append(distance)
        cross_masks.append(valid)
        add_feature(values, groups, f"metric__{name}_cm", distance, "metric")
        add_feature(values, groups, f"quality__{name}_valid", valid, "quality")
    stacked_cross = np.stack(cross_values, axis=-1)
    stacked_masks = np.stack(cross_masks, axis=-1)
    minimum_cross = np.min(np.where(stacked_masks, stacked_cross, np.inf), axis=-1)
    minimum_cross[~np.any(stacked_masks, axis=-1)] = 0.0
    add_feature(values, groups, "metric__minimum_cross_keypoint_distance_cm", minimum_cross, "metric")

    agent_heading = pose.body_axis[:, agent_indices]
    target_heading = pose.body_axis[:, target_indices]
    heading_valid = pose.body_axis_mask[:, agent_indices] & pair_valid
    forward, lateral, egocentric_valid = egocentric_coordinates(
        relative, agent_heading, heading_valid
    )
    body_scale = np.maximum(pose.body_length[:, agent_indices], 1e-6)
    normalized_valid = egocentric_valid & (pose.body_length[:, agent_indices] > 1e-6)
    normalized_forward = np.divide(
        forward,
        body_scale,
        out=np.zeros_like(forward),
        where=normalized_valid,
    )
    normalized_lateral = np.divide(
        lateral,
        body_scale,
        out=np.zeros_like(lateral),
        where=normalized_valid,
    )
    for name, array in (
        ("target_forward_cm", forward),
        ("target_lateral_cm", lateral),
        ("target_forward_body_lengths", normalized_forward),
        ("target_lateral_body_lengths", normalized_lateral),
        ("valid", egocentric_valid),
    ):
        add_feature(values, groups, f"egocentric__{name}", array, "egocentric")

    all_velocity, all_velocity_valid = temporal_derivative(
        pose.center if pose.scale_valid else np.zeros_like(pose.center),
        pose.timestamps,
        pose.center_mask & pose.scale_valid,
        maximum_gap_seconds=config.maximum_derivative_gap_seconds,
    )
    agent_velocity = all_velocity[:, agent_indices]
    target_velocity = all_velocity[:, target_indices]
    relative_velocity = (target_velocity - agent_velocity).astype(np.float32)
    velocity_valid = all_velocity_valid[:, agent_indices] & all_velocity_valid[:, target_indices]
    relative_acceleration, acceleration_valid = temporal_derivative(
        relative_velocity,
        pose.timestamps,
        velocity_valid,
        maximum_gap_seconds=config.maximum_derivative_gap_seconds,
    )
    relative_jerk, jerk_valid = temporal_derivative(
        relative_acceleration,
        pose.timestamps,
        acceleration_valid,
        maximum_gap_seconds=config.maximum_derivative_gap_seconds,
    )
    radial_unit, radial_valid = unit_vectors(relative, pair_valid)
    radial_velocity = np.sum(relative_velocity * radial_unit, axis=-1).astype(np.float32)
    tangential_velocity = (
        radial_unit[..., 0] * relative_velocity[..., 1]
        - radial_unit[..., 1] * relative_velocity[..., 0]
    ).astype(np.float32)
    motion_valid = velocity_valid & radial_valid
    radial_velocity[~motion_valid] = 0.0
    tangential_velocity[~motion_valid] = 0.0
    separation_rate, separation_valid = scalar_derivative(
        center_distance,
        pose.timestamps,
        pair_valid,
        maximum_gap_seconds=config.maximum_derivative_gap_seconds,
    )
    motion_features = (
        ("relative_velocity_x_cm_s", relative_velocity[..., 0]),
        ("relative_velocity_y_cm_s", relative_velocity[..., 1]),
        ("relative_speed_cm_s", safe_norm(relative_velocity)),
        ("relative_acceleration_x_cm_s2", relative_acceleration[..., 0]),
        ("relative_acceleration_y_cm_s2", relative_acceleration[..., 1]),
        ("relative_acceleration_cm_s2", safe_norm(relative_acceleration)),
        ("relative_jerk_x_cm_s3", relative_jerk[..., 0]),
        ("relative_jerk_y_cm_s3", relative_jerk[..., 1]),
        ("relative_jerk_cm_s3", safe_norm(relative_jerk)),
        ("radial_velocity_cm_s", radial_velocity),
        ("tangential_velocity_cm_s", tangential_velocity),
        ("separation_rate_cm_s", separation_rate),
        ("approach_speed_cm_s", -separation_rate),
    )
    for name, array in motion_features:
        add_feature(values, groups, f"metric__{name}", array, "metric")
    add_feature(values, groups, "quality__relative_velocity_valid", velocity_valid, "quality")
    add_feature(values, groups, "quality__relative_acceleration_valid", acceleration_valid, "quality")
    add_feature(values, groups, "quality__relative_jerk_valid", jerk_valid, "quality")
    add_feature(values, groups, "quality__separation_rate_valid", separation_valid, "quality")

    agent_unit, agent_heading_valid = unit_vectors(
        agent_heading, pose.body_axis_mask[:, agent_indices]
    )
    target_unit, target_heading_valid = unit_vectors(
        target_heading, pose.body_axis_mask[:, target_indices]
    )
    toward_target = radial_unit
    agent_facing = np.sum(agent_unit * toward_target, axis=-1).astype(np.float32)
    target_facing_agent = np.sum(target_unit * -toward_target, axis=-1).astype(np.float32)
    heading_alignment = np.sum(agent_unit * target_unit, axis=-1).astype(np.float32)
    facing_valid = radial_valid & agent_heading_valid & target_heading_valid
    for array in (agent_facing, target_facing_agent, heading_alignment):
        array[~facing_valid] = 0.0
    following_score = (
        np.maximum(agent_facing, 0.0)
        * np.maximum(heading_alignment, 0.0)
        * (forward > 0)
    ).astype(np.float32)
    agent_facing_angle = np.arccos(np.clip(agent_facing, -1.0, 1.0)).astype(np.float32)
    target_facing_angle = np.arccos(np.clip(target_facing_agent, -1.0, 1.0)).astype(np.float32)
    heading_alignment_angle = np.arccos(np.clip(heading_alignment, -1.0, 1.0)).astype(np.float32)
    for array in (agent_facing_angle, target_facing_angle, heading_alignment_angle):
        array[~facing_valid] = 0.0
    for name, array in (
        ("agent_facing_target_cos", agent_facing),
        ("target_facing_agent_cos", target_facing_agent),
        ("heading_alignment_cos", heading_alignment),
        ("agent_facing_target_angle_rad", agent_facing_angle),
        ("target_facing_agent_angle_rad", target_facing_angle),
        ("heading_alignment_angle_rad", heading_alignment_angle),
        ("following_score", following_score),
        ("facing_valid", facing_valid),
    ):
        add_feature(values, groups, f"egocentric__{name}", array, "egocentric")

    pair_availability = (
        pose.availability_mask[:, agent_indices].mean(axis=2)
        * pose.availability_mask[:, target_indices].mean(axis=2)
    ).astype(np.float32)
    pair_interpolation = np.maximum(
        pose.interpolated_mask[:, agent_indices].mean(axis=2),
        pose.interpolated_mask[:, target_indices].mean(axis=2),
    ).astype(np.float32)
    add_feature(values, groups, "quality__pair_landmark_availability", pair_availability, "quality")
    add_feature(values, groups, "quality__pair_interpolated_fraction", pair_interpolation, "quality")
    add_feature(
        values,
        groups,
        "quality__identity_swap_candidate",
        pose.identity_swap_mask[:, agent_indices] | pose.identity_swap_mask[:, target_indices],
        "quality",
    )

    for distance_threshold in config.contact_distances_cm:
        distance_token = f"{distance_threshold:g}cm".replace(".", "p")
        contact = (center_distance <= distance_threshold) & pair_valid
        add_feature(
            values,
            groups,
            f"metric__contact_within_{distance_token}",
            contact,
            "metric",
        )
        for seconds in config.temporal_windows_seconds:
            radius = round_seconds_to_frames(seconds, pose.fps)
            token = window_token(seconds)
            contact_ratio, contact_support = centered_mean(
                contact.astype(np.float32), radius, pair_valid
            )
            add_feature(
                values,
                groups,
                f"temporal__contact_ratio_{distance_token}_{token}",
                contact_ratio,
                "temporal",
            )
            add_feature(
                values,
                groups,
                f"temporal__contact_support_{distance_token}_{token}",
                contact_support,
                "temporal",
            )
    for seconds in config.temporal_windows_seconds:
        radius = round_seconds_to_frames(seconds, pose.fps)
        token = window_token(seconds)
        distance_mean, distance_support = centered_mean(center_distance, radius, pair_valid)
        approach_mean, _ = centered_mean(-separation_rate, radius, separation_valid)
        distance_change, change_support = past_future_change(center_distance, pair_valid, radius)
        forward_change, _ = past_future_change(forward, egocentric_valid, radius)
        for name, array in (
            ("center_distance_mean", distance_mean),
            ("approach_speed_mean", approach_mean),
            ("future_minus_past_distance", distance_change),
            ("future_minus_past_forward", forward_change),
            ("distance_support", distance_support),
            ("change_support", change_support),
        ):
            add_feature(values, groups, f"temporal__{name}_{token}", array, "temporal")

    add_group_features(values, groups, pose=pose, pair_indices=pairs, config=config)
    add_metadata_features(
        values,
        groups,
        context=context,
        registry=categories,
        agent_ids=agent_ids,
        target_ids=target_ids,
        frames=frames,
    )
    return FeatureTable(
        kind="pair",
        index=_index(pose, pairs),
        values=values,
        groups=groups,
        feature_version=config.feature_version,
    )
