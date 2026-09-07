"""Label-aligned temporal pose augmentations for Phase 9."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from mabe.phase9_config import TemporalAugmentationConfig
from mabe.temporal_data import TemporalWindow


def _copy_window(window: TemporalWindow) -> TemporalWindow:
    return replace(
        window,
        coordinates=window.coordinates.copy(),
        landmark_mask=window.landmark_mask.copy(),
        center=window.center.copy(),
        body_axis=window.body_axis.copy(),
        body_length=window.body_length.copy(),
        confidence=window.confidence.copy(),
        mouse_valid=window.mouse_valid.copy(),
        solo_targets=window.solo_targets.copy(),
        pair_targets=window.pair_targets.copy(),
        solo_action_valid=window.solo_action_valid.copy(),
        pair_action_valid=window.pair_action_valid.copy(),
        temporal_valid=window.temporal_valid.copy(),
    )


def permute_mice(
    window: TemporalWindow,
    permutation: NDArray[np.int64],
    agent_index: int,
    target_index: int,
) -> tuple[TemporalWindow, int, int]:
    """Permute every mouse-indexed input, mask, and target axis together."""
    mice = window.coordinates.shape[1]
    if sorted(permutation.tolist()) != list(range(mice)):
        raise ValueError("mouse permutation must contain every index exactly once")
    inverse = np.empty(mice, dtype=np.int64)
    inverse[permutation] = np.arange(mice)
    result = _copy_window(window)
    for name in (
        "coordinates",
        "landmark_mask",
        "center",
        "body_axis",
        "body_length",
        "confidence",
        "mouse_valid",
        "solo_targets",
    ):
        setattr(result, name, getattr(result, name)[:, permutation])
    result.pair_targets = result.pair_targets[:, permutation][:, :, permutation]
    result.solo_action_valid = result.solo_action_valid[permutation]
    result.pair_action_valid = result.pair_action_valid[permutation][:, permutation]
    return result, int(inverse[agent_index]), int(inverse[target_index])


def swap_left_right(window: TemporalWindow) -> TemporalWindow:
    """Swap canonical ear-left/ear-right channels without changing labels."""
    if window.coordinates.shape[2] < 3:
        raise ValueError("left/right swap requires nose plus two ear channels")
    result = _copy_window(window)
    result.coordinates[:, :, [1, 2]] = result.coordinates[:, :, [2, 1]]
    result.landmark_mask[:, :, [1, 2]] = result.landmark_mask[:, :, [2, 1]]
    return result


def _resample_continuous(value: NDArray[np.float32], positions: NDArray[np.float64]) -> NDArray[np.float32]:
    source = np.arange(value.shape[0], dtype=np.float64)
    flattened = value.reshape(value.shape[0], -1)
    output = np.empty_like(flattened)
    for column in range(flattened.shape[1]):
        output[:, column] = np.interp(positions, source, flattened[:, column])
    return output.reshape(value.shape)


def time_stretch_window(window: TemporalWindow, factor: float) -> TemporalWindow:
    """Stretch/compress time while mapping labels and masks by nearest frame."""
    if factor <= 0:
        raise ValueError("time-stretch factor must be positive")
    length = window.coordinates.shape[0]
    center = (length - 1) / 2
    positions = np.clip((np.arange(length) - center) / factor + center, 0, length - 1)
    nearest = np.rint(positions).astype(np.int64)
    result = _copy_window(window)
    for name in ("coordinates", "center", "body_axis", "body_length", "confidence"):
        setattr(result, name, _resample_continuous(getattr(result, name), positions))
    for name in (
        "landmark_mask",
        "mouse_valid",
        "solo_targets",
        "pair_targets",
        "temporal_valid",
    ):
        setattr(result, name, getattr(result, name)[nearest])
    result.coordinates[~result.landmark_mask] = 0
    result.center[~result.mouse_valid] = 0
    return result


def geometric_transform(
    window: TemporalWindow, *, angle_radians: float, reflect_x: bool
) -> TemporalWindow:
    """Apply one rigid arena transform to every mouse/keypoint and body axis."""
    cosine = float(np.cos(angle_radians))
    sine = float(np.sin(angle_radians))
    rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float32)
    if reflect_x:
        rotation = rotation @ np.asarray([[-1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    result = _copy_window(window)
    result.coordinates = result.coordinates @ rotation.T
    result.center = result.center @ rotation.T
    result.body_axis = result.body_axis @ rotation.T
    return result


def scale_window(window: TemporalWindow, factor: float) -> TemporalWindow:
    """Apply isotropic scale jitter while preserving headings and supervision."""
    if factor <= 0:
        raise ValueError("spatial scale factor must be positive")
    result = _copy_window(window)
    result.coordinates *= np.float32(factor)
    result.center *= np.float32(factor)
    result.body_length *= np.float32(factor)
    return result


def shift_input_frames(window: TemporalWindow, offset: int) -> TemporalWindow:
    """Shift only pose inputs; labels stay fixed to regularize fuzzy boundaries."""
    if offset == 0:
        return _copy_window(window)
    length = window.coordinates.shape[0]
    indices = np.clip(np.arange(length, dtype=np.int64) + offset, 0, length - 1)
    result = _copy_window(window)
    for name in (
        "coordinates",
        "landmark_mask",
        "center",
        "body_axis",
        "body_length",
        "confidence",
        "mouse_valid",
    ):
        setattr(result, name, getattr(result, name)[indices].copy())
    return result


class TemporalAugmenter:
    def __init__(self, config: TemporalAugmentationConfig, seed: int) -> None:
        self.config = config
        self.generator = np.random.default_rng(seed)

    def __call__(
        self, window: TemporalWindow, agent_index: int, target_index: int
    ) -> tuple[TemporalWindow, int, int]:
        result = _copy_window(window)
        if self.config.time_stretch_min != 1 or self.config.time_stretch_max != 1:
            factor = self.generator.uniform(
                self.config.time_stretch_min, self.config.time_stretch_max
            )
            result = time_stretch_window(result, float(factor))
        if (
            self.config.maximum_frame_shift > 0
            and self.generator.random() < self.config.frame_shift_probability
        ):
            offset = int(
                self.generator.integers(
                    -self.config.maximum_frame_shift,
                    self.config.maximum_frame_shift + 1,
                )
            )
            result = shift_input_frames(result, offset)
        if self.config.coordinate_noise_std > 0:
            noise = self.generator.normal(
                0, self.config.coordinate_noise_std, result.coordinates.shape
            ).astype(np.float32)
            result.coordinates[result.landmark_mask] += noise[result.landmark_mask]
        probability = self.config.keypoint_dropout_probability
        if probability > 0:
            dropped = self.generator.random(result.landmark_mask.shape) < probability
            result.landmark_mask[dropped] = False
            result.coordinates[~result.landmark_mask] = 0
            result.mouse_valid = result.landmark_mask.any(axis=2) & result.mouse_valid
            result.center[~result.mouse_valid] = 0
            result.body_axis[~result.mouse_valid] = 0
            result.body_length[~result.mouse_valid] = 0
            result.confidence[~result.mouse_valid] = 0
        if self.generator.random() < self.config.left_right_swap_probability:
            result = swap_left_right(result)
        if self.generator.random() < self.config.mouse_permutation_probability:
            permutation = self.generator.permutation(result.coordinates.shape[1]).astype(np.int64)
            result, agent_index, target_index = permute_mice(
                result, permutation, agent_index, target_index
            )
        if self.generator.random() < self.config.geometric_transform_probability:
            maximum = np.deg2rad(self.config.maximum_rotation_degrees)
            angle = float(self.generator.uniform(-maximum, maximum))
            reflect = bool(self.generator.random() < self.config.reflection_probability)
            result = geometric_transform(result, angle_radians=angle, reflect_x=reflect)
        if self.generator.random() < self.config.scale_probability:
            factor = float(
                self.generator.uniform(self.config.scale_min, self.config.scale_max)
            )
            result = scale_window(result, factor)
        return result, agent_index, target_index
