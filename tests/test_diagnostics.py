from __future__ import annotations

import json

import pandas as pd
import pytest

from mabe.diagnostics import metric_breakdown


def test_breakdown_reconstructs_official_score_and_exposes_counts() -> None:
    active = json.dumps(["1,2,attack", "1,2,mount"])
    solution = pd.DataFrame(
        [
            {
                "video_id": 1,
                "agent_id": 1,
                "target_id": 2,
                "action": "attack",
                "start_frame": 0,
                "stop_frame": 10,
                "lab_id": 7,
                "behaviors_labeled": active,
            }
        ]
    )
    submission = pd.DataFrame(
        [
            {
                "video_id": 1,
                "agent_id": 1,
                "target_id": 2,
                "action": "attack",
                "start_frame": 0,
                "stop_frame": 5,
            },
            {
                "video_id": 1,
                "agent_id": 1,
                "target_id": 2,
                "action": "mount",
                "start_frame": 20,
                "stop_frame": 24,
            },
        ]
    )

    breakdown = metric_breakdown(solution, submission)
    actions = breakdown.by_lab_action.set_index("action")

    assert breakdown.overall == pytest.approx(1 / 3)
    assert breakdown.by_lab.loc[0, "fbeta"] == pytest.approx(breakdown.overall)
    assert actions.loc["attack", ["tp", "fp", "fn"]].tolist() == [5, 0, 5]
    assert actions.loc["attack", "fbeta"] == pytest.approx(2 / 3)
    assert actions.loc["mount", ["tp", "fp", "fn"]].tolist() == [0, 4, 0]
    assert actions.loc["mount", "fbeta"] == 0.0


def test_breakdown_macro_averages_labs_equally() -> None:
    columns = [
        "video_id",
        "agent_id",
        "target_id",
        "action",
        "start_frame",
        "stop_frame",
        "lab_id",
        "behaviors_labeled",
    ]
    rows = [
        [1, 1, 2, "attack", 0, 10, 1, '["1,2,attack"]'],
        [2, 1, 2, "attack", 0, 10, 2, '["1,2,attack"]'],
    ]
    solution = pd.DataFrame(rows, columns=columns)
    submission = pd.DataFrame(
        [[1, 1, 2, "attack", 0, 10]],
        columns=columns[:6],
    )

    breakdown = metric_breakdown(solution, submission)
    assert sorted(breakdown.by_lab["fbeta"].tolist()) == [0.0, 1.0]
    assert breakdown.overall == 0.5
