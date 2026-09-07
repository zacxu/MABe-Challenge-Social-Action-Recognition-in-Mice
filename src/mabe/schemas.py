"""Canonical column orders and Polars schemas used across the pipeline."""

from __future__ import annotations

import polars as pl

MOUSE_SLOTS = ("mouse1", "mouse2", "mouse3", "mouse4")
MOUSE_METADATA_FIELDS = ("strain", "color", "sex", "id", "age", "condition")

METADATA_COLUMNS = (
    "lab_id",
    "video_id",
    *(f"{mouse}_{field}" for mouse in MOUSE_SLOTS for field in MOUSE_METADATA_FIELDS),
    "frames_per_second",
    "video_duration_sec",
    "pix_per_cm_approx",
    "video_width_pix",
    "video_height_pix",
    "arena_width_cm",
    "arena_height_cm",
    "arena_shape",
    "arena_type",
    "body_parts_tracked",
    "behaviors_labeled",
    "tracking_method",
)

METADATA_SCHEMA = pl.Schema(
    {
        "lab_id": pl.String,
        "video_id": pl.Int64,
        **{
            f"{mouse}_{field}": pl.String
            for mouse in MOUSE_SLOTS
            for field in MOUSE_METADATA_FIELDS
        },
        "frames_per_second": pl.Float64,
        "video_duration_sec": pl.Float64,
        "pix_per_cm_approx": pl.Float64,
        "video_width_pix": pl.Int64,
        "video_height_pix": pl.Int64,
        "arena_width_cm": pl.Float64,
        "arena_height_cm": pl.Float64,
        "arena_shape": pl.String,
        "arena_type": pl.String,
        "body_parts_tracked": pl.String,
        "behaviors_labeled": pl.String,
        "tracking_method": pl.String,
    }
)

TRACKING_SOURCE_COLUMNS = ("video_frame", "mouse_id", "bodypart", "x", "y")
TRACKING_COLUMNS = TRACKING_SOURCE_COLUMNS
TRACKING_SCHEMA = pl.Schema(
    {
        "video_frame": pl.Int64,
        "mouse_id": pl.String,
        "bodypart": pl.String,
        "x": pl.Float32,
        "y": pl.Float32,
    }
)

ANNOTATION_SOURCE_COLUMNS = ("agent_id", "target_id", "action", "start_frame", "stop_frame")
ANNOTATION_COLUMNS = ANNOTATION_SOURCE_COLUMNS
ANNOTATION_SCHEMA = pl.Schema(
    {
        "agent_id": pl.String,
        "target_id": pl.String,
        "action": pl.String,
        "start_frame": pl.Int64,
        "stop_frame": pl.Int64,
    }
)

CANDIDATE_TRIPLET_SCHEMA = pl.Schema(
    {
        "video_id": pl.Int64,
        "agent_id": pl.String,
        "target_id": pl.String,
        "action": pl.String,
        "is_self": pl.Boolean,
        "is_allowed": pl.Boolean,
    }
)

PREDICTION_SCHEMA = pl.Schema(
    {
        "video_id": pl.Int64,
        "video_frame": pl.Int64,
        "agent_id": pl.String,
        "target_id": pl.String,
        "action": pl.String,
        "probability": pl.Float32,
    }
)

OOF_COLUMNS = (
    "lab_id",
    "video_id",
    "video_frame",
    "agent_id",
    "target_id",
    "action",
    "fold",
    "loss_mask",
    "target",
    "probability",
    "model_id",
    "feature_version",
)
OOF_SCHEMA = pl.Schema(
    {
        "lab_id": pl.String,
        "video_id": pl.Int64,
        "video_frame": pl.Int64,
        "agent_id": pl.String,
        "target_id": pl.String,
        "action": pl.String,
        "fold": pl.Int8,
        "loss_mask": pl.UInt8,
        "target": pl.UInt8,
        "probability": pl.Float32,
        "model_id": pl.String,
        "feature_version": pl.String,
    }
)

FOLD_COLUMNS = (
    "lab_id",
    "video_id",
    "fold",
    "duplicate_group_id",
    "label_state",
    "supervised_eligible",
    "source_role",
    "source_group",
    "is_primary_validation",
    "positive_frame_count",
    "positive_action_count",
)

SUBMISSION_COLUMNS = (
    "row_id",
    "video_id",
    "agent_id",
    "target_id",
    "action",
    "start_frame",
    "stop_frame",
)
SUBMISSION_SCHEMA = pl.Schema(
    {
        "row_id": pl.Int64,
        "video_id": pl.Int64,
        "agent_id": pl.String,
        "target_id": pl.String,
        "action": pl.String,
        "start_frame": pl.Int64,
        "stop_frame": pl.Int64,
    }
)

VIDEO_MANIFEST_COLUMNS = (
    "split",
    "lab_id",
    "video_id",
    "frames_per_second",
    "video_duration_sec",
    "declared_frame_count",
    "observed_start_frame",
    "observed_stop_frame",
    "observed_frame_count",
    "tracking_row_count",
    "video_width_pix",
    "video_height_pix",
    "arena_width_cm",
    "arena_height_cm",
    "arena_shape",
    "arena_type",
    "pix_per_cm_approx",
    "tracking_method",
    "body_parts_tracked_raw",
    "body_parts_declared",
    "body_parts_observed",
    "skeleton_id",
    "mouse_ids",
    "mouse_count",
    "mouse_metadata_json",
    "tracking_path",
    "tracking_exists",
    "annotation_path",
    "annotation_exists",
    "behaviors_labeled_raw",
    "has_behavior_mask",
    "parsed_triplet_count",
    "annotation_count",
    "positive_frame_count",
    "label_state",
    "supervised_eligible",
    "quarantined_annotation_count",
    "missing_frame_rate",
    "missing_coordinate_rate",
    "non_finite_coordinate_rate",
    "extreme_jump_rate",
    "tracking_order_violation_count",
)

TRIPLET_MANIFEST_COLUMNS = (
    "split",
    "lab_id",
    "video_id",
    "entry_index",
    "raw_mask",
    "canonical_mask",
    "raw_entry",
    "raw_agent_id",
    "raw_target_id",
    "raw_action",
    "agent_id",
    "target_id",
    "action",
    "canonical_entry",
    "is_self",
    "was_repaired",
    "repair_reasons",
    "is_duplicate",
)
