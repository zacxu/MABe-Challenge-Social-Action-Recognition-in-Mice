from __future__ import annotations

import json
from pathlib import Path

import pytest

from mabe.io import scan_metadata
from mabe.masks import BehaviorMaskError, CanonicalTriplet, parse_behaviors_labeled

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "MABe-mouse-behavior-detection"


def test_parser_repairs_components_deduplicates_and_retains_raw_entries() -> None:
    raw = json.dumps(
        [
            " mouse1 , mouse2 , 'Attack' ",
            "mouse1,mouse2,attack",
            "mouse3,mouse3,selfgroom",
        ]
    )

    parsed = parse_behaviors_labeled(raw)

    assert parsed.raw_entries[0] == " mouse1 , mouse2 , 'Attack' "
    assert parsed.triplets == (
        CanonicalTriplet("mouse1", "mouse2", "attack"),
        CanonicalTriplet("mouse3", "self", "selfgroom"),
    )
    assert parsed.entries[0].was_repaired
    assert parsed.entries[1].is_duplicate
    assert "normalized_self_target" in parsed.entries[2].repair_reasons
    assert json.loads(parsed.canonical_json) == [
        "mouse1,mouse2,attack",
        "mouse3,self,selfgroom",
    ]


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("not-json", "invalid behaviors_labeled JSON"),
        ('{"x": 1}', "must decode to a list"),
        ('["mouse1,mouse2"]', "exactly agent,target,action"),
        ('["rat1,mouse2,attack"]', "invalid agent ID"),
        ('["mouse1,mouse9,attack"]', "invalid target ID"),
        ('["mouse1,mouse2,not an action"]', "invalid action"),
    ],
)
def test_parser_fails_loudly_when_repair_would_require_guessing(raw: str, message: str) -> None:
    with pytest.raises(BehaviorMaskError, match=message):
        parse_behaviors_labeled(raw)


def test_real_metadata_reproduces_action_vocabulary_and_unlabeled_mabe22_count() -> None:
    metadata = scan_metadata(DATASET / "train.csv").collect()
    masks = [
        parse_behaviors_labeled(raw)
        for raw in metadata["behaviors_labeled"].drop_nulls().to_list()
    ]
    actions = {triplet.action for mask in masks for triplet in mask.triplets}
    self_actions = {
        triplet.action for mask in masks for triplet in mask.triplets if triplet.is_self
    }
    unlabeled_mabe22 = metadata.filter(
        metadata["lab_id"].str.starts_with("MABe22")
        & metadata["behaviors_labeled"].is_null()
    ).height

    assert len(actions) == 37
    assert len(self_actions) == 11
    assert len(actions - self_actions) == 26
    assert unlabeled_mabe22 == 7_926
    assert any(entry.was_repaired for mask in masks for entry in mask.entries)
