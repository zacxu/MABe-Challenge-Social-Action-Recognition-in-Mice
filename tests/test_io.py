from __future__ import annotations

from pathlib import Path

import polars as pl

from mabe.io import scan_annotation, scan_metadata, scan_submission, scan_tracking
from mabe.schemas import (
    ANNOTATION_COLUMNS,
    ANNOTATION_SCHEMA,
    METADATA_COLUMNS,
    METADATA_SCHEMA,
    SUBMISSION_COLUMNS,
    SUBMISSION_SCHEMA,
    TRACKING_COLUMNS,
    TRACKING_SCHEMA,
)

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "MABe-mouse-behavior-detection"


def test_metadata_reader_has_stable_explicit_schema() -> None:
    row = scan_metadata(DATASET / "train.csv").head(1).collect()
    assert row.columns == list(METADATA_COLUMNS)
    assert row.schema == METADATA_SCHEMA


def test_tracking_reader_normalizes_integer_mouse_ids(tmp_path: Path) -> None:
    path = tmp_path / "tracking.parquet"
    pl.DataFrame(
        {
            "video_frame": pl.Series([0], dtype=pl.Int16),
            "mouse_id": pl.Series([2], dtype=pl.Int8),
            "bodypart": ["nose"],
            "x": pl.Series([1.5], dtype=pl.Float32),
            "y": pl.Series([2.5], dtype=pl.Float32),
        }
    ).write_parquet(path)

    tracking = scan_tracking(path).collect()
    assert tracking.columns == list(TRACKING_COLUMNS)
    assert tracking.schema == TRACKING_SCHEMA
    assert tracking["mouse_id"].to_list() == ["mouse2"]


def test_annotation_reader_uses_agent_equals_target_for_self_fixture(tmp_path: Path) -> None:
    path = tmp_path / "annotation.parquet"
    pl.DataFrame(
        {
            "agent_id": pl.Series([1], dtype=pl.Int8),
            "target_id": pl.Series([1], dtype=pl.Int8),
            "action": ["selfgroom"],
            "start_frame": pl.Series([3], dtype=pl.Int16),
            "stop_frame": pl.Series([7], dtype=pl.Int16),
        }
    ).write_parquet(path)

    annotation = scan_annotation(path).collect()
    assert annotation.columns == list(ANNOTATION_COLUMNS)
    assert annotation.schema == ANNOTATION_SCHEMA
    assert annotation.row(0, named=True) == {
        "agent_id": "mouse1",
        "target_id": "mouse1",
        "action": "selfgroom",
        "start_frame": 3,
        "stop_frame": 7,
    }


def test_submission_reader_matches_official_order_and_types() -> None:
    submission = scan_submission(DATASET / "sample_submission.csv").collect()
    assert submission.columns == list(SUBMISSION_COLUMNS)
    assert submission.schema == SUBMISSION_SCHEMA
    assert submission["agent_id"].to_list() == ["mouse1"]
