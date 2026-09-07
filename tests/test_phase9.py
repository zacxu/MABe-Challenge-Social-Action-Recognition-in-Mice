from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from mabe.phase9_config import load_phase9_config
from mabe.phase9_pipeline import _per_lab_action_comparison, _window_starts
from mabe.temporal_augmentation import (
    geometric_transform,
    permute_mice,
    scale_window,
    shift_input_frames,
    swap_left_right,
    time_stretch_window,
)
from mabe.temporal_data import PAIR_FEATURE_NAMES, SOLO_FEATURE_NAMES, TemporalWindow
from mabe.temporal_model import masked_bce_with_logits, overlap_weights
from mabe.temporal_training import build_temporal_model


def _window(*, frames: int = 9, mice: int = 3) -> TemporalWindow:
    coordinates = np.zeros((frames, mice, 5, 2), dtype=np.float32)
    for frame in range(frames):
        for mouse in range(mice):
            coordinates[frame, mouse, :, 0] = 100 * mouse + frame
            coordinates[frame, mouse, :, 1] = np.arange(5)
    center = coordinates[:, :, 3].copy()
    solo = np.zeros((frames, mice, 2), dtype=np.uint8)
    pair = np.zeros((frames, mice, mice, 3), dtype=np.uint8)
    solo[3, 0, 1] = 1
    pair[4, 0, 2, 2] = 1
    return TemporalWindow(
        coordinates=coordinates,
        landmark_mask=np.ones((frames, mice, 5), dtype=bool),
        center=center,
        body_axis=np.tile(
            np.asarray([1.0, 0.0], dtype=np.float32), (frames, mice, 1)
        ),
        body_length=np.ones((frames, mice), dtype=np.float32),
        confidence=np.ones((frames, mice), dtype=np.float32),
        mouse_valid=np.ones((frames, mice), dtype=bool),
        solo_targets=solo,
        pair_targets=pair,
        solo_action_valid=np.ones((mice, 2), dtype=bool),
        pair_action_valid=np.ones((mice, mice, 3), dtype=bool),
        temporal_valid=np.ones(frames, dtype=bool),
    )


def test_mouse_permutation_preserves_agent_target_and_labels() -> None:
    source = _window()
    permutation = np.asarray([2, 0, 1], dtype=np.int64)

    transformed, agent, target = permute_mice(source, permutation, 0, 2)

    assert (agent, target) == (1, 0)
    assert transformed.solo_targets[3, agent, 1] == 1
    assert transformed.pair_targets[4, agent, target, 2] == 1
    np.testing.assert_array_equal(
        transformed.coordinates[:, agent], source.coordinates[:, 0]
    )
    np.testing.assert_array_equal(
        transformed.pair_action_valid[agent, target],
        source.pair_action_valid[0, 2],
    )


def test_left_right_swap_changes_only_ear_channels() -> None:
    source = _window()
    transformed = swap_left_right(source)

    np.testing.assert_array_equal(transformed.coordinates[:, :, 1], source.coordinates[:, :, 2])
    np.testing.assert_array_equal(transformed.coordinates[:, :, 2], source.coordinates[:, :, 1])
    np.testing.assert_array_equal(transformed.coordinates[:, :, 0], source.coordinates[:, :, 0])
    np.testing.assert_array_equal(transformed.solo_targets, source.solo_targets)
    np.testing.assert_array_equal(transformed.pair_targets, source.pair_targets)


def test_time_stretch_uses_same_nearest_mapping_for_all_labels() -> None:
    source = _window()
    factor = 0.75
    center = (source.coordinates.shape[0] - 1) / 2
    positions = np.clip(
        (np.arange(source.coordinates.shape[0]) - center) / factor + center,
        0,
        source.coordinates.shape[0] - 1,
    )
    nearest = np.rint(positions).astype(np.int64)

    transformed = time_stretch_window(source, factor)

    np.testing.assert_array_equal(transformed.solo_targets, source.solo_targets[nearest])
    np.testing.assert_array_equal(transformed.pair_targets, source.pair_targets[nearest])
    np.testing.assert_array_equal(transformed.mouse_valid, source.mouse_valid[nearest])


def test_rigid_geometry_preserves_pairwise_distances_and_labels() -> None:
    source = _window()
    transformed = geometric_transform(source, angle_radians=0.7, reflect_x=True)
    before = np.linalg.norm(source.center[:, 2] - source.center[:, 0], axis=1)
    after = np.linalg.norm(transformed.center[:, 2] - transformed.center[:, 0], axis=1)

    np.testing.assert_allclose(after, before, rtol=1e-6, atol=1e-5)
    np.testing.assert_array_equal(transformed.solo_targets, source.solo_targets)
    np.testing.assert_array_equal(transformed.pair_targets, source.pair_targets)


def test_scale_window_scales_geometry_without_changing_labels_or_heading() -> None:
    source = _window()
    transformed = scale_window(source, 1.25)

    np.testing.assert_allclose(transformed.coordinates, source.coordinates * 1.25)
    np.testing.assert_allclose(transformed.body_length, source.body_length * 1.25)
    np.testing.assert_array_equal(transformed.body_axis, source.body_axis)
    np.testing.assert_array_equal(transformed.solo_targets, source.solo_targets)


def test_frame_shift_moves_inputs_but_keeps_supervision_fixed() -> None:
    source = _window()
    transformed = shift_input_frames(source, 2)

    np.testing.assert_array_equal(transformed.coordinates[0], source.coordinates[2])
    np.testing.assert_array_equal(transformed.coordinates[-1], source.coordinates[-1])
    np.testing.assert_array_equal(transformed.solo_targets, source.solo_targets)
    np.testing.assert_array_equal(transformed.temporal_valid, source.temporal_valid)


def test_model_shapes_action_mask_and_weighted_masked_loss() -> None:
    config = load_phase9_config("configs/temporal.yaml")
    model = build_temporal_model(
        config=config, lab_count=3, solo_actions=11, pair_actions=26
    )
    features = torch.randn(2, 17, len(SOLO_FEATURE_NAMES))
    metadata = torch.randn(2, 3)
    labs = torch.tensor([0, 2], dtype=torch.long)
    validity = torch.ones((2, 11), dtype=torch.bool)
    validity[:, 4] = False

    logits = model("solo", features, labs, metadata, validity)

    assert logits.shape == (2, 17, 11)
    assert len(PAIR_FEATURE_NAMES) > len(SOLO_FEATURE_NAMES)
    assert torch.all(logits[:, :, 4] == -20)
    loss = masked_bce_with_logits(
        torch.zeros((2, 1, 1)),
        torch.tensor([[[1.0]], [[0.0]]]),
        torch.tensor([[[1.0]], [[0.0]]]),
        positive_weight=1.0,
        sample_weights=torch.tensor([2.0, 100.0]),
    )
    assert float(loss) == pytest.approx(float(np.log(2)), rel=1e-6)


def test_sliding_windows_and_overlap_weights_cover_every_frame() -> None:
    frames, length, stride, edge = 1_337, 512, 128, 32
    starts = _window_starts(frames, length, stride)
    total = np.zeros(frames, dtype=np.float32)
    for start in starts:
        stop = min(start + length, frames)
        weights = overlap_weights(
            length,
            edge,
            left_boundary=start == 0,
            right_boundary=stop == frames,
        ).numpy()[: stop - start]
        total[start:stop] += weights

    assert starts[0] == 0
    assert starts[-1] == frames - length
    assert np.all(total > 0)


def test_per_lab_action_comparison_is_outer_and_reports_winner(tmp_path: Path) -> None:
    columns = ["lab_id", "action", "fbeta", "precision", "recall", "tp", "fp", "fn"]
    neural = pd.DataFrame(
        [["lab-a", "attack", 0.4, 0.5, 0.4, 4, 4, 6]], columns=columns
    )
    gbdt = pd.DataFrame(
        [
            ["lab-a", "attack", 0.6, 0.7, 0.6, 6, 2, 4],
            ["lab-b", "groom", 0.2, 0.2, 0.2, 2, 8, 8],
        ],
        columns=columns,
    )
    path = tmp_path / "gbdt.csv"
    gbdt.to_csv(path, index=False)

    result = _per_lab_action_comparison(neural, path)

    assert len(result) == 2
    assert result.loc[0, "stronger_model"] == "gbdt"
    assert result.loc[1, "neural_fbeta"] == 0
