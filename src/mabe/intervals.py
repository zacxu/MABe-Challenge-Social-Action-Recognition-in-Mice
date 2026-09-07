"""Strict half-open frame interval conversion utilities."""

from collections.abc import Iterable
from numbers import Integral

FrameInterval = tuple[int, int]


def _frame_number(value: Integral, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer frame number, got {type(value).__name__}")
    frame = int(value)
    if frame < 0:
        raise ValueError(f"{name} must be non-negative, got {frame}")
    return frame


def validate_interval(start_frame: Integral, stop_frame: Integral) -> FrameInterval:
    """Validate and return a non-empty half-open frame interval."""
    start = _frame_number(start_frame, name="start_frame")
    stop = _frame_number(stop_frame, name="stop_frame")
    if start >= stop:
        raise ValueError(f"interval must satisfy start_frame < stop_frame, got [{start}, {stop})")
    return start, stop


def frames_to_intervals(frames: Iterable[Integral]) -> list[FrameInterval]:
    """Convert frame IDs to sorted maximal half-open contiguous runs."""
    ordered_frames = sorted({_frame_number(frame, name="frame") for frame in frames})
    if not ordered_frames:
        return []

    intervals: list[FrameInterval] = []
    start = previous = ordered_frames[0]
    for frame in ordered_frames[1:]:
        if frame != previous + 1:
            intervals.append((start, previous + 1))
            start = frame
        previous = frame
    intervals.append((start, previous + 1))
    return intervals


def intervals_to_frames(intervals: Iterable[tuple[Integral, Integral]]) -> set[int]:
    """Expand valid half-open intervals into their union of integer frame IDs."""
    frames: set[int] = set()
    for start_frame, stop_frame in intervals:
        start, stop = validate_interval(start_frame, stop_frame)
        frames.update(range(start, stop))
    return frames

