from __future__ import annotations

import json
from pathlib import Path

from mabe.audit import _overlap_rows, audit_annotation_rows
from mabe.masks import parse_behaviors_labeled


def annotation(
    *,
    row_id: int,
    action: str = "selfgroom",
    start: int = 0,
    stop: int = 10,
    target: str = "mouse1",
) -> dict[str, object]:
    return {
        "annotation_row_id": row_id,
        "source_path": "Lab/1.parquet",
        "lab_id": "Lab",
        "video_id": 1,
        "agent_id": "mouse1",
        "target_id": target,
        "action": action,
        "start_frame": start,
        "stop_frame": stop,
    }


def test_annotation_audit_maps_agent_equals_target_to_mask_self_and_quarantines_invalid() -> None:
    mask = parse_behaviors_labeled(json.dumps(["mouse1,self,selfgroom"]))
    tracking = {
        ("Lab", 1): {
            "observed_start_frame": 0,
            "observed_stop_frame": 20,
            "mouse_ids": ["mouse1"],
        }
    }
    rows = [annotation(row_id=0), annotation(row_id=1, start=5, stop=5)]

    audited, anomalies, counts, quarantined, positives = audit_annotation_rows(
        rows,
        tracking_by_video=tracking,
        parsed_masks={("train", "Lab", 1): mask},
    )

    assert audited[0]["mask_target_id"] == "self"
    assert not audited[0]["triplet_absent_from_mask"]
    assert audited[1]["invalid_interval"]
    assert len(anomalies) == 1
    assert counts[("Lab", 1)] == 2
    assert quarantined[("Lab", 1)] == 1
    assert positives[("Lab", 1)] == 10


def test_positive_counts_union_overlapping_same_action_intervals() -> None:
    mask = parse_behaviors_labeled(json.dumps(["mouse1,mouse2,attack"]))
    tracking = {
        ("Lab", 1): {
            "observed_start_frame": 0,
            "observed_stop_frame": 20,
            "mouse_ids": ["mouse1", "mouse2"],
        }
    }
    rows = [
        annotation(row_id=0, action="attack", target="mouse2", start=0, stop=8),
        annotation(row_id=1, action="attack", target="mouse2", start=5, stop=10),
    ]

    _, _, _, _, positives = audit_annotation_rows(
        rows,
        tracking_by_video=tracking,
        parsed_masks={("train", "Lab", 1): mask},
    )
    assert positives[("Lab", 1)] == 10


def test_overlap_report_distinguishes_same_and_different_actions() -> None:
    rows = [
        annotation(row_id=0, action="attack", target="mouse2", start=0, stop=10),
        annotation(row_id=1, action="attack", target="mouse2", start=5, stop=12),
        annotation(row_id=2, action="chase", target="mouse2", start=8, stop=15),
    ]
    overlaps = _overlap_rows(rows)

    assert len(overlaps) == 3
    assert sum(bool(row["same_action"]) for row in overlaps) == 1
    assert sum(not bool(row["same_action"]) for row in overlaps) == 2
    assert {int(row["overlap_frame_count"]) for row in overlaps} == {2, 4, 5}


def test_real_source_file_counts_and_ambiguous_video_are_reproducible() -> None:
    root = Path(__file__).resolve().parents[1] / "MABe-mouse-behavior-detection"
    annotation_keys = {
        (path.parent.name, int(path.stem))
        for path in (root / "train_annotation").rglob("*.parquet")
    }
    assert len(list((root / "train_tracking").rglob("*.parquet"))) == 8_789
    assert len(annotation_keys) == 847
    assert ("PleasantMeerkat", 1_375_833_299) not in annotation_keys
