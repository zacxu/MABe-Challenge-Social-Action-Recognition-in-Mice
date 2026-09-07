"""Third-mouse and whole-group context for ordered pairs."""

from __future__ import annotations

import numpy as np

from mabe.features.common import (
    FeatureConfig,
    FloatArray,
    add_feature,
    egocentric_coordinates,
    safe_norm,
)
from mabe.pose import CleanPose


def add_group_features(
    values: dict[str, FloatArray],
    groups: dict[str, str],
    *,
    pose: CleanPose,
    pair_indices: tuple[tuple[int, int], ...],
    config: FeatureConfig,
) -> None:
    """Append masked group context; every group value is zero in dyadic videos."""
    frames = pose.frame_ids.size
    pairs = len(pair_indices)
    shape = (frames, pairs)
    closest_agent = np.zeros(shape, dtype=np.float32)
    closest_target = np.zeros(shape, dtype=np.float32)
    closest_midpoint = np.zeros(shape, dtype=np.float32)
    nearby_count = np.zeros(shape, dtype=np.float32)
    between_count = np.zeros(shape, dtype=np.float32)
    distance_rank = np.zeros(shape, dtype=np.float32)
    centroid_forward = np.zeros(shape, dtype=np.float32)
    centroid_lateral = np.zeros(shape, dtype=np.float32)
    dispersion = np.zeros(shape, dtype=np.float32)
    third_available = np.zeros(shape, dtype=bool)
    group_valid = np.zeros(shape, dtype=bool)
    if len(pose.mouse_ids) > 2 and pose.scale_valid:
        centers = pose.center
        center_valid = pose.center_mask
        valid_counts = center_valid.sum(axis=1)
        centroid = np.divide(
            np.where(center_valid[..., None], centers, 0.0).sum(axis=1),
            valid_counts[:, None],
            out=np.zeros((frames, 2), dtype=np.float32),
            where=valid_counts[:, None] > 0,
        ).astype(np.float32)
        squared_from_centroid = np.sum((centers - centroid[:, None, :]) ** 2, axis=-1)
        group_dispersion = np.sqrt(
            np.divide(
                np.where(center_valid, squared_from_centroid, 0.0).sum(axis=1),
                valid_counts,
                out=np.zeros(frames, dtype=np.float32),
                where=valid_counts > 0,
            )
        ).astype(np.float32)
        unordered = tuple(
            (left, right)
            for left in range(len(pose.mouse_ids))
            for right in range(left + 1, len(pose.mouse_ids))
        )
        unordered_distances = np.stack(
            [safe_norm(centers[:, left] - centers[:, right]) for left, right in unordered], axis=1
        )
        unordered_valid = np.stack(
            [center_valid[:, left] & center_valid[:, right] for left, right in unordered], axis=1
        )
        unordered_lookup = {pair: index for index, pair in enumerate(unordered)}
        for pair_position, (agent, target) in enumerate(pair_indices):
            thirds = tuple(
                mouse for mouse in range(len(pose.mouse_ids)) if mouse not in (agent, target)
            )
            pair_valid = center_valid[:, agent] & center_valid[:, target]
            third_valid = center_valid[:, thirds]
            usable = pair_valid & third_valid.any(axis=1) & (valid_counts >= 3)
            agent_center = centers[:, agent]
            target_center = centers[:, target]
            midpoint = ((agent_center + target_center) * np.float32(0.5)).astype(np.float32)
            third_centers = centers[:, thirds]
            to_agent = safe_norm(third_centers - agent_center[:, None, :])
            to_target = safe_norm(third_centers - target_center[:, None, :])
            to_midpoint = safe_norm(third_centers - midpoint[:, None, :])
            for destination, distances in (
                (closest_agent, to_agent),
                (closest_target, to_target),
                (closest_midpoint, to_midpoint),
            ):
                minimum = np.min(np.where(third_valid, distances, np.inf), axis=1)
                destination[usable, pair_position] = minimum[usable]
            nearby_count[:, pair_position] = np.sum(
                third_valid
                & (
                    (to_agent <= config.nearby_mouse_distance_cm)
                    | (to_target <= config.nearby_mouse_distance_cm)
                ),
                axis=1,
            )
            pair_vector = target_center - agent_center
            squared_length = np.sum(pair_vector**2, axis=1)
            relative = third_centers - agent_center[:, None, :]
            projection = np.divide(
                np.sum(relative * pair_vector[:, None, :], axis=2),
                squared_length[:, None],
                out=np.zeros_like(to_agent),
                where=squared_length[:, None] > 1e-8,
            )
            projected = agent_center[:, None, :] + projection[..., None] * pair_vector[:, None, :]
            perpendicular = safe_norm(third_centers - projected)
            between_count[:, pair_position] = np.sum(
                third_valid
                & (projection >= 0)
                & (projection <= 1)
                & (perpendicular <= config.between_mouse_distance_cm),
                axis=1,
            )
            unordered_index = unordered_lookup[(min(agent, target), max(agent, target))]
            current_distance = unordered_distances[:, unordered_index]
            rank = 1 + np.sum(
                unordered_valid & (unordered_distances < current_distance[:, None]), axis=1
            )
            valid_pair_counts = unordered_valid.sum(axis=1)
            distance_rank[:, pair_position] = np.divide(
                rank,
                valid_pair_counts,
                out=np.zeros(frames, dtype=np.float32),
                where=pair_valid & (valid_pair_counts > 0),
            ).astype(np.float32)
            forward, lateral, heading_valid = egocentric_coordinates(
                (centroid - midpoint)[:, None, :].astype(np.float32),
                pose.body_axis[:, agent][:, None, :],
                (usable & pose.body_axis_mask[:, agent])[:, None],
            )
            centroid_forward[:, pair_position] = forward[:, 0]
            centroid_lateral[:, pair_position] = lateral[:, 0]
            dispersion[:, pair_position] = np.where(usable, group_dispersion, 0.0)
            third_available[:, pair_position] = usable
            group_valid[:, pair_position] = heading_valid[:, 0]
    features = (
        ("closest_third_to_agent_cm", closest_agent),
        ("closest_third_to_target_cm", closest_target),
        ("closest_third_to_midpoint_cm", closest_midpoint),
        ("nearby_third_count", nearby_count),
        ("between_pair_third_count", between_count),
        ("pair_distance_rank", distance_rank),
        ("centroid_forward_from_pair_midpoint_cm", centroid_forward),
        ("centroid_lateral_from_pair_midpoint_cm", centroid_lateral),
        ("dispersion_cm", dispersion),
        ("third_mouse_available", third_available),
        ("egocentric_centroid_valid", group_valid),
    )
    for name, array in features:
        add_feature(values, groups, f"group__{name}", array, "group")
