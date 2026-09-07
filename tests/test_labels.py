from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mabe.intervals import frames_to_intervals, intervals_to_frames
from mabe.labels import (
    FrameLabelState,
    LabelStore,
    build_action_vocabulary,
    build_label_artifacts,
    build_metric_solution,
)
from mabe.masks import CanonicalTriplet
from mabe.metric import mouse_fbeta

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "MABe-mouse-behavior-detection"
VIDEOS_MANIFEST = ROOT / "artifacts" / "manifests" / "videos.parquet"
TRIPLETS_MANIFEST = ROOT / "artifacts" / "manifests" / "triplets.parquet"


def _annotation_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    if rows:
        return pl.DataFrame(rows).cast(
            {
                "agent_id": pl.Int64,
                "target_id": pl.Int64,
                "action": pl.String,
                "start_frame": pl.Int64,
                "stop_frame": pl.Int64,
            }
        )
    return pl.DataFrame(
        schema={
            "agent_id": pl.Int64,
            "target_id": pl.Int64,
            "action": pl.String,
            "start_frame": pl.Int64,
            "stop_frame": pl.Int64,
        }
    )


def _store(
    tmp_path: Path,
    *,
    allowed: list[CanonicalTriplet],
    annotations: list[dict[str, object]],
    frame_ids: list[int],
    label_state: str = "labeled",
) -> LabelStore:
    tracking_path = tmp_path / "tracking.parquet"
    annotation_path = tmp_path / "annotation.parquet"
    pl.DataFrame({"video_frame": frame_ids}).write_parquet(tracking_path)
    _annotation_frame(annotations).write_parquet(annotation_path)
    return LabelStore(
        lab_id="Lab",
        video_id=1,
        video_label_state=label_state,
        allowed_triplets=allowed,
        annotation_path=annotation_path if label_state == "labeled" else None,
        tracking_path=tracking_path,
        observed_start_frame=min(frame_ids),
        observed_stop_frame=max(frame_ids) + 1,
        mouse_ids=("mouse1", "mouse2", "mouse3"),
    )


def test_sparse_targets_are_half_open_and_disallowed_cells_are_never_negative(
    tmp_path: Path,
) -> None:
    attack = CanonicalTriplet("mouse1", "mouse2", "attack")
    disallowed = CanonicalTriplet("mouse1", "mouse2", "chase")
    store = _store(
        tmp_path,
        allowed=[attack],
        annotations=[
            {
                "agent_id": 1,
                "target_id": 2,
                "action": "attack",
                "start_frame": 11,
                "stop_frame": 15,
            }
        ],
        frame_ids=[10, 11, 13, 15, 18],
    )

    assert not store.annotations_loaded
    labels = store.targets(triplets=[attack, disallowed], start_frame=10, stop_frame=18)

    assert store.annotations_loaded
    assert labels.frame_ids.tolist() == [10, 11, 13, 15]
    assert labels.targets.tolist() == [[0, 0], [1, 0], [1, 0], [0, 0]]
    assert labels.loss_mask.tolist() == [[1, 0], [1, 0], [1, 0], [1, 0]]
    assert labels.states[:, 0].tolist() == [
        FrameLabelState.VALID_NEGATIVE,
        FrameLabelState.POSITIVE,
        FrameLabelState.POSITIVE,
        FrameLabelState.VALID_NEGATIVE,
    ]
    assert np.all(labels.states[:, 1] == FrameLabelState.UNKNOWN_MASKED)
    assert not np.any(labels.valid_negative_mask[:, 1])
    assert labels.targets.dtype == np.uint8
    assert labels.loss_mask.dtype == np.uint8


def test_allowed_triplet_with_no_event_is_an_all_zero_valid_window(tmp_path: Path) -> None:
    attack = CanonicalTriplet("mouse1", "mouse2", "attack")
    store = _store(
        tmp_path,
        allowed=[attack],
        annotations=[],
        frame_ids=[20, 21, 25],
    )

    labels = store.targets(frame_ids=[20, 25])

    assert labels.targets.tolist() == [[0], [0]]
    assert labels.loss_mask.tolist() == [[1], [1]]
    assert np.all(labels.valid_negative_mask)


@pytest.mark.parametrize(
    "label_state",
    ["ambiguous_missing_annotation", "unlabeled_mabe22", "unlabeled_missing_mask", "test"],
)
def test_excluded_video_states_never_enter_supervised_loss(
    tmp_path: Path, label_state: str
) -> None:
    triplet = CanonicalTriplet("mouse1", "mouse2", "attack")
    store = _store(
        tmp_path,
        allowed=[triplet],
        annotations=[],
        frame_ids=[0, 2],
        label_state=label_state,
    )

    labels = store.targets(frame_ids=[0, 2])

    assert not store.annotations_loaded
    assert not labels.targets.any()
    assert not labels.loss_mask.any()
    assert np.all(labels.states == FrameLabelState.EXCLUDED_VIDEO)
    assert not labels.valid_negative_mask.any()


def test_self_candidates_multiple_targets_and_overlaps_are_preserved(tmp_path: Path) -> None:
    groom = CanonicalTriplet("mouse1", "self", "selfgroom")
    attack = CanonicalTriplet("mouse1", "mouse2", "attack")
    chase = CanonicalTriplet("mouse1", "mouse2", "chase")
    avoid = CanonicalTriplet("mouse1", "mouse3", "avoid")
    store = _store(
        tmp_path,
        allowed=[groom, attack, chase, avoid],
        annotations=[
            {"agent_id": 1, "target_id": 1, "action": "selfgroom", "start_frame": 1, "stop_frame": 4},
            {"agent_id": 1, "target_id": 2, "action": "attack", "start_frame": 2, "stop_frame": 5},
            {"agent_id": 1, "target_id": 2, "action": "chase", "start_frame": 3, "stop_frame": 6},
            {"agent_id": 1, "target_id": 3, "action": "avoid", "start_frame": 3, "stop_frame": 4},
            {"agent_id": 1, "target_id": 2, "action": "attack", "start_frame": 4, "stop_frame": 7},
        ],
        frame_ids=list(range(8)),
    )

    labels = store.targets(frame_ids=[3, 4], triplets=[groom, attack, chase, avoid])

    assert store.solo_candidates == (groom,)
    assert store.pair_candidates == (attack, chase, avoid)
    assert labels.targets.tolist() == [[1, 1, 1, 1], [0, 1, 1, 0]]
    assert store.positive_intervals(attack) == ((2, 5), (4, 7))
    assert store.positive_frames(groom).tolist() == [1, 2, 3]


def test_invalid_zero_length_annotation_is_quarantined(tmp_path: Path) -> None:
    attack = CanonicalTriplet("mouse1", "mouse2", "attack")
    store = _store(
        tmp_path,
        allowed=[attack],
        annotations=[
            {"agent_id": 1, "target_id": 2, "action": "attack", "start_frame": 3, "stop_frame": 3},
            {"agent_id": 1, "target_id": 2, "action": "attack", "start_frame": 4, "stop_frame": 6},
        ],
        frame_ids=list(range(8)),
    )

    assert store.positive_intervals(attack) == ((4, 6),)
    assert store.quarantined_annotation_count == 1


def test_frame_requests_must_be_strict_and_unambiguous(tmp_path: Path) -> None:
    attack = CanonicalTriplet("mouse1", "mouse2", "attack")
    store = _store(tmp_path, allowed=[attack], annotations=[], frame_ids=[0, 1, 2])

    with pytest.raises(ValueError, match="strictly increasing"):
        store.targets(frame_ids=[0, 0])
    with pytest.raises(ValueError, match="not both"):
        store.targets(frame_ids=[0], start_frame=0, stop_frame=1)
    with pytest.raises(ValueError, match="start_frame < stop_frame"):
        store.targets(start_frame=2, stop_frame=2)


def test_action_vocabulary_and_artifacts_have_stable_inverse_maps(tmp_path: Path) -> None:
    triplets = pl.DataFrame(
        {
            "split": ["train", "train", "train"],
            "lab_id": ["Lab", "Lab", "Lab"],
            "video_id": [1, 1, 1],
            "agent_id": ["mouse1", "mouse1", "mouse1"],
            "target_id": ["mouse2", "self", "mouse2"],
            "action": ["attack", "groom", "attack"],
            "canonical_entry": [
                "mouse1,mouse2,attack",
                "mouse1,self,groom",
                "mouse1,mouse2,attack",
            ],
            "is_self": [False, True, False],
            "is_duplicate": [False, False, True],
        }
    )
    source = tmp_path / "triplets.parquet"
    triplets.write_parquet(source)

    vocabulary = build_action_vocabulary(triplets)
    outputs = build_label_artifacts(
        triplets_path=source,
        output_root=tmp_path / "labels",
        source_fingerprint="source-fingerprint",
    )

    assert vocabulary.actions == ("attack", "groom")
    assert vocabulary.solo_actions == ("groom",)
    assert vocabulary.pair_actions == ("attack",)
    assert vocabulary.index_to_action == {0: "attack", 1: "groom"}
    assert outputs.vocabulary.version == vocabulary.version
    payload = json.loads(outputs.vocabulary_path.read_text(encoding="utf-8"))
    assert payload["action_to_index"] == {"attack": 0, "groom": 1}
    assert payload["index_to_action"] == {"0": "attack", "1": "groom"}
    mappings = pl.read_parquet(outputs.triplet_mappings_path)
    assert mappings.height == 2
    assert mappings.filter(pl.col("is_self"))["solo_action_index"].to_list() == [0]
    assert mappings.filter(~pl.col("is_self"))["pair_action_index"].to_list() == [0]
    for artifact in (outputs.vocabulary_path, outputs.triplet_mappings_path):
        sidecar = json.loads(Path(f"{artifact}.meta.json").read_text(encoding="utf-8"))
        assert sidecar["source_data_fingerprint"] == "source-fingerprint"
        assert sidecar["configuration"]["vocabulary_version"] == vocabulary.version
        assert sidecar["schema_sha256"]


def test_metric_solution_joins_raw_masks_filters_subset_and_normalizes_self() -> None:
    raw_self_mask = json.dumps(["mouse1,self,selfgroom"])
    annotations = pl.DataFrame(
        {
            "lab_id": ["LabA", "LabB", "LabA", "LabA"],
            "video_id": [1, 2, 1, 1],
            "agent_id": [1, 1, 1, 1],
            "target_id": [1, 2, 1, 1],
            "action": ["selfgroom", "attack", "selfgroom", "selfgroom"],
            "start_frame": [3, 0, 8, 7],
            "stop_frame": [7, 4, 8, 9],
        }
    )
    videos = pl.DataFrame(
        {
            "lab_id": ["LabA", "LabB", "LabC"],
            "video_id": [1, 2, 3],
            "behaviors_labeled_raw": [
                raw_self_mask,
                json.dumps(["mouse1,mouse2,attack"]),
                None,
            ],
            "label_state": ["labeled", "labeled", "ambiguous_missing_annotation"],
            "observed_start_frame": [0, 0, 0],
            "observed_stop_frame": [8, 5, 1],
            "mouse_ids": [["mouse1"], ["mouse1", "mouse2"], ["mouse1"]],
        }
    )

    solution = build_metric_solution(
        annotations,
        videos,
        validation_videos=(key for key in [("LabA", 1)]),
    )

    assert isinstance(solution, pd.DataFrame)
    assert solution.to_dict("records") == [
        {
            "video_id": 1,
            "agent_id": "mouse1",
            "target_id": "self",
            "action": "selfgroom",
            "start_frame": 3,
            "stop_frame": 7,
            "lab_id": "LabA",
            "behaviors_labeled": raw_self_mask,
        }
    ]
    submission = solution.drop(columns=["lab_id", "behaviors_labeled"])
    assert mouse_fbeta(solution, submission) == 1.0


@pytest.mark.integration
def test_real_video_round_trips_sparse_intervals_without_off_by_one() -> None:
    store = LabelStore.from_manifests(
        lab_id="CalMS21_task1",
        video_id=1_578_732_722,
        dataset_root=DATASET,
        videos_path=VIDEOS_MANIFEST,
        triplets_path=TRIPLETS_MANIFEST,
    )

    assert not store.annotations_loaded
    labels = store.targets()
    assert labels.frame_ids.tolist() == list(range(576))
    assert labels.targets.shape == (576, 3)
    assert int(labels.targets.sum()) == 429
    assert store.quarantined_annotation_count == 0
    for index, triplet in enumerate(labels.triplets):
        expected = intervals_to_frames(store.positive_intervals(triplet))
        actual = set(labels.frame_ids[labels.targets[:, index].astype(bool)].tolist())
        assert actual == expected
        assert intervals_to_frames(frames_to_intervals(actual)) == expected
