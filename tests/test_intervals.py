from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from mabe.intervals import frames_to_intervals, intervals_to_frames, validate_interval


def test_frames_to_intervals_uses_maximal_half_open_runs() -> None:
    assert frames_to_intervals([7, 2, 3, 3, 4, 9, 10]) == [(2, 5), (7, 8), (9, 11)]


def test_intervals_to_frames_uses_half_open_boundaries_and_unions_overlaps() -> None:
    assert intervals_to_frames([(0, 1), (1, 3), (2, 5)]) == {0, 1, 2, 3, 4}


def test_numpy_integer_frames_are_supported() -> None:
    assert validate_interval(np.int64(2), np.int32(5)) == (2, 5)


@pytest.mark.parametrize("interval", [(0, 0), (2, 1), (-1, 2)])
def test_invalid_intervals_are_rejected(interval: tuple[int, int]) -> None:
    with pytest.raises(ValueError):
        validate_interval(*interval)


@pytest.mark.parametrize("value", [True, 1.5, "1", None])
def test_non_integer_frames_are_rejected(value: object) -> None:
    with pytest.raises(TypeError):
        frames_to_intervals([value])  # type: ignore[list-item]


@given(st.sets(st.integers(min_value=0, max_value=1_000), max_size=200))
def test_frame_interval_round_trip(frames: set[int]) -> None:
    assert intervals_to_frames(frames_to_intervals(frames)) == frames


@given(
    st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=500),
            st.integers(min_value=1, max_value=50),
        ),
        max_size=100,
    )
)
def test_interval_frame_normalization_round_trip(starts_and_lengths: list[tuple[int, int]]) -> None:
    intervals = [(start, start + length) for start, length in starts_and_lengths]
    expected_frames = intervals_to_frames(intervals)
    normalized = frames_to_intervals(expected_frames)

    assert intervals_to_frames(normalized) == expected_frames
    assert all(
        left_stop < right_start
        for (_, left_stop), (right_start, _) in zip(normalized, normalized[1:], strict=False)
    )
