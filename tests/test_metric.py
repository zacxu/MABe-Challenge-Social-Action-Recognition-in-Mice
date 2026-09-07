from __future__ import annotations

import json

import pandas as pd
import pytest

from mabe.metric import HostVisibleError, mouse_fbeta, score


def truth(
    *,
    video: int = 1,
    agent: int = 1,
    target: int = 2,
    action: str = "attack",
    start: int = 0,
    stop: int = 10,
    lab: int = 1,
    active: list[str] | None = None,
) -> dict[str, object]:
    if active is None:
        active = [f"{agent},{target},{action}"]
    return {
        "video_id": video,
        "agent_id": agent,
        "target_id": target,
        "action": action,
        "start_frame": start,
        "stop_frame": stop,
        "lab_id": lab,
        "behaviors_labeled": json.dumps(active),
    }


def prediction(
    *,
    video: int = 1,
    agent: int = 1,
    target: int = 2,
    action: str = "attack",
    start: int = 0,
    stop: int = 10,
) -> dict[str, object]:
    return {
        "video_id": video,
        "agent_id": agent,
        "target_id": target,
        "action": action,
        "start_frame": start,
        "stop_frame": stop,
    }


def evaluate(
    solution_rows: list[dict[str, object]],
    prediction_rows: list[dict[str, object]],
    beta: float = 1,
) -> float:
    return mouse_fbeta(pd.DataFrame(solution_rows), pd.DataFrame(prediction_rows), beta=beta)


def test_perfect_prediction_scores_one() -> None:
    assert evaluate([truth()], [prediction()]) == 1.0


def test_wrong_action_scores_zero() -> None:
    assert evaluate([truth()], [prediction(action="mount")]) == 0.0


def test_half_open_single_frame_boundary_does_not_overlap() -> None:
    assert evaluate([truth(start=0, stop=1)], [prediction(start=1, stop=2)]) == 0.0


def test_missing_and_surplus_frames_have_expected_counts() -> None:
    solution = [truth(start=0, stop=10)]
    assert evaluate(solution, [prediction(start=0, stop=5)]) == pytest.approx(2 / 3)
    assert evaluate(solution, [prediction(start=0, stop=15)]) == pytest.approx(0.8)


def test_beta_parameter_changes_false_negative_weight() -> None:
    assert evaluate([truth()], [prediction(stop=5)], beta=2) == pytest.approx(5 / 9)


def test_adjacent_same_key_rows_equal_one_merged_interval() -> None:
    solution = [truth(start=0, stop=5), truth(start=5, stop=10)]
    assert evaluate(solution, [prediction(start=0, stop=10)]) == 1.0


def test_labs_are_macro_averaged_equally() -> None:
    solution = [truth(video=1, lab=10), truth(video=2, lab=20)]
    assert evaluate(solution, [prediction(video=1)]) == 0.5


def test_actions_are_macro_averaged_not_weighted_by_duration() -> None:
    active = ["1,2,attack", "1,2,mount"]
    solution = [
        truth(action="attack", start=0, stop=100, active=active),
        truth(action="mount", start=200, stop=201, active=active),
    ]
    assert evaluate(solution, [prediction(action="attack", start=0, stop=100)]) == 0.5


def test_allowed_predicted_only_action_enters_macro_average_at_zero() -> None:
    active = ["1,2,attack", "1,2,mount"]
    solution = [truth(action="attack", active=active)]
    submission = [prediction(action="attack"), prediction(action="mount", start=20, stop=30)]

    assert evaluate(solution, submission) == 0.5


def test_prediction_not_in_behaviors_labeled_is_ignored() -> None:
    solution = [truth()]
    submission = [prediction(), prediction(action="mount", start=20, stop=30)]
    assert evaluate(solution, submission) == 1.0


def test_different_actions_may_not_overlap_for_same_agent_target_pair() -> None:
    active = ["1,2,attack", "1,2,mount"]
    solution = [
        truth(action="attack", start=0, stop=10, active=active),
        truth(action="mount", start=10, stop=20, active=active),
    ]
    submission = [
        prediction(action="attack", start=0, stop=10),
        prediction(action="mount", start=5, stop=20),
    ]

    with pytest.raises(HostVisibleError, match="same frame"):
        evaluate(solution, submission)


def test_same_agent_may_predict_different_targets_on_same_frame() -> None:
    active = ["1,2,attack", "1,3,avoid"]
    solution = [
        truth(target=2, action="attack", active=active),
        truth(target=3, action="avoid", active=active),
    ]
    submission = [prediction(target=2, action="attack"), prediction(target=3, action="avoid")]
    assert evaluate(solution, submission) == 1.0


def test_redundant_overlaps_for_same_prediction_key_are_deduplicated() -> None:
    solution = [truth()]
    submission = [prediction(start=0, stop=6), prediction(start=4, stop=10)]
    assert evaluate(solution, submission) == 1.0


@pytest.mark.parametrize("which", ["solution", "submission"])
def test_empty_input_is_rejected(which: str) -> None:
    solution = pd.DataFrame([truth()])
    submission = pd.DataFrame([prediction()])
    if which == "solution":
        solution = solution.iloc[0:0]
    else:
        submission = submission.iloc[0:0]

    with pytest.raises(ValueError, match="Missing solution or submission data"):
        mouse_fbeta(solution, submission)


@pytest.mark.parametrize("which", ["solution", "submission"])
def test_missing_required_column_is_rejected(which: str) -> None:
    solution = pd.DataFrame([truth()])
    submission = pd.DataFrame([prediction()])
    if which == "solution":
        solution = solution.drop(columns="target_id")
    else:
        submission = submission.drop(columns="target_id")

    with pytest.raises(ValueError, match=f"{which.capitalize()} is missing column target_id"):
        mouse_fbeta(solution, submission)


def test_row_id_wrapper_ignores_row_ids() -> None:
    solution = pd.DataFrame([{**truth(), "row_id": 100}])
    submission = pd.DataFrame([{**prediction(), "row_id": 999}])
    assert score(solution, submission, row_id_column_name="row_id") == 1.0
