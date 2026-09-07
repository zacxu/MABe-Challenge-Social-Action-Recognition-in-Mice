"""Memory-mapped temporal sequence artifacts, features, and balanced window plans."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
from numpy.typing import NDArray

from mabe.fingerprints import sha256_file
from mabe.labels import LabelStore
from mabe.phase9_config import Phase9Config
from mabe.pose import build_time_map, load_clean_pose_from_manifest, load_pose_config

Route = Literal["solo", "pair"]
def solo_feature_names(landmark_count: int) -> tuple[str, ...]:
    return (
        *(
            f"landmark_{landmark}_{axis}"
            for landmark in range(landmark_count)
            for axis in ("x", "y")
        ),
        *(f"landmark_{landmark}_valid" for landmark in range(landmark_count)),
        "center_velocity_x",
        "center_velocity_y",
        "speed",
        "body_axis_x",
        "body_axis_y",
        "body_length",
        "confidence",
        "mouse_valid",
    )


def pair_feature_names(landmark_count: int) -> tuple[str, ...]:
    solo = solo_feature_names(landmark_count)
    return (
        *(f"agent_{name}" for name in solo),
        *(f"target_{name}" for name in solo),
        "relative_center_x",
        "relative_center_y",
        "center_distance",
        "relative_velocity_x",
        "relative_velocity_y",
        "heading_dot",
        "heading_cross",
        "pair_valid",
    )


SOLO_FEATURE_NAMES = solo_feature_names(5)
PAIR_FEATURE_NAMES = pair_feature_names(5)


@dataclass(frozen=True, slots=True)
class SequenceManifestRow:
    lab_id: str
    video_id: int
    fold: int
    artifact_path: Path
    frame_count: int
    mouse_count: int
    fps: float


@dataclass(frozen=True, slots=True)
class WindowRef:
    lab_id: str
    video_id: int
    fold: int
    route: Route
    agent_index: int
    target_index: int
    center_index: int
    sampling_kind: str
    action_index: int


@dataclass(slots=True)
class TemporalWindow:
    coordinates: NDArray[np.float32]
    landmark_mask: NDArray[np.bool_]
    center: NDArray[np.float32]
    body_axis: NDArray[np.float32]
    body_length: NDArray[np.float32]
    confidence: NDArray[np.float32]
    mouse_valid: NDArray[np.bool_]
    solo_targets: NDArray[np.uint8]
    pair_targets: NDArray[np.uint8]
    solo_action_valid: NDArray[np.bool_]
    pair_action_valid: NDArray[np.bool_]
    temporal_valid: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class MaterializedWindow:
    features: NDArray[np.float32]
    targets: NDArray[np.float32]
    loss_mask: NDArray[np.float32]
    metadata: NDArray[np.float32]
    lab_index: int
    route: Route
    ref: WindowRef


def _write_array(path: Path, value: NDArray[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, value, allow_pickle=False)


def _set_packed_action(
    packed: NDArray[np.uint8], active: NDArray[np.bool_], index: tuple[int, ...], action: int
) -> None:
    byte = action // 8
    bit = np.uint8(1 << (action % 8))
    if len(index) == 1:
        values = packed[:, index[0], byte]
    elif len(index) == 2:
        values = packed[:, index[0], index[1], byte]
    else:
        raise ValueError("packed action index must contain one or two entity axes")
    values[active] = np.bitwise_or(values[active], bit)


def _read_vocabulary(path: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return (
        tuple(str(value) for value in payload["solo_actions"]),
        tuple(str(value) for value in payload["pair_actions"]),
    )


def _positive_centers(active: NDArray[np.bool_], maximum: int) -> NDArray[np.int32]:
    indices = np.flatnonzero(active)
    if indices.size > maximum:
        selected = np.linspace(0, indices.size - 1, maximum).round().astype(np.int64)
        indices = indices[selected]
    return indices.astype(np.int32, copy=False)


def build_sequence_artifact(
    row: dict[str, Any],
    *,
    fold: int,
    lab_index: int,
    dataset_root: Path,
    videos_path: Path,
    triplets_path: Path,
    pose_config_path: Path,
    vocabulary_path: Path,
    output_root: Path,
    config: Phase9Config,
    configuration_sha256: str,
    split: str = "train",
) -> SequenceManifestRow:
    """Build one resampled, compact, mmap-compatible sequence directory."""
    lab = str(row["lab_id"])
    video_id = int(row["video_id"])
    artifact = output_root / lab / str(video_id)
    metadata_path = artifact / "metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("configuration_sha256") == configuration_sha256:
            return SequenceManifestRow(
                lab_id=lab,
                video_id=video_id,
                fold=int(metadata["fold"]),
                artifact_path=artifact,
                frame_count=int(metadata["frame_count"]),
                mouse_count=int(metadata["mouse_count"]),
                fps=float(metadata["target_fps"]),
            )
    pose = load_clean_pose_from_manifest(
        row, dataset_root=dataset_root, config=load_pose_config(pose_config_path)
    )
    landmark_lookup = {name: index for index, name in enumerate(pose.landmarks)}
    missing = set(config.data.landmarks) - set(landmark_lookup)
    if missing:
        raise ValueError(f"sequence landmarks unavailable for {lab}/{video_id}: {sorted(missing)}")
    landmark_indices = [landmark_lookup[name] for name in config.data.landmarks]
    time_map = build_time_map(pose, target_fps=config.data.target_fps)
    nearest = time_map.resampled_to_original
    coordinates = pose.coordinates[nearest][:, :, landmark_indices].astype(np.float32)
    landmark_mask = pose.availability_mask[nearest][:, :, landmark_indices].astype(bool)
    center = pose.center[nearest].astype(np.float32)
    body_axis = pose.body_axis[nearest].astype(np.float32)
    body_length = pose.body_length[nearest].astype(np.float32)
    confidence = pose.confidence[nearest].astype(np.float32)
    mouse_valid = landmark_mask.any(axis=2) & pose.center_mask[nearest]
    valid_lengths = body_length[(body_length > 0) & np.isfinite(body_length)]
    scale = float(np.median(valid_lengths)) if valid_lengths.size else 1.0
    scale = max(scale, 1e-6)
    valid_centers = center[mouse_valid]
    origin = np.median(valid_centers, axis=0) if valid_centers.size else np.zeros(2)
    coordinates = ((coordinates - origin) / scale).astype(np.float32)
    center = ((center - origin) / scale).astype(np.float32)
    body_length /= scale
    coordinates[~landmark_mask] = 0
    center[~mouse_valid] = 0
    body_axis[~mouse_valid] = 0
    body_length[~mouse_valid] = 0
    confidence[~mouse_valid] = 0
    solo_actions, pair_actions = _read_vocabulary(vocabulary_path)
    solo_lookup = {action: index for index, action in enumerate(solo_actions)}
    pair_lookup = {action: index for index, action in enumerate(pair_actions)}
    mice = pose.mouse_ids
    mouse_lookup = {mouse: index for index, mouse in enumerate(mice)}
    frames = nearest.size
    solo_packed = np.zeros((frames, len(mice), (len(solo_actions) + 7) // 8), dtype=np.uint8)
    pair_packed = np.zeros(
        (frames, len(mice), len(mice), (len(pair_actions) + 7) // 8), dtype=np.uint8
    )
    solo_valid = np.zeros((len(mice), len(solo_actions)), dtype=bool)
    pair_valid = np.zeros((len(mice), len(mice), len(pair_actions)), dtype=bool)
    positive_records: list[tuple[int, int, int, int, int]] = []
    store = LabelStore.from_manifests(
        lab_id=lab,
        video_id=video_id,
        dataset_root=dataset_root,
        videos_path=videos_path,
        triplets_path=triplets_path,
        split=split,
    )
    intervals = store.positive_intervals()
    assert isinstance(intervals, dict)
    resampled_frames = pose.frame_ids[nearest]
    for triplet in store.allowed_triplets:
        agent = mouse_lookup.get(triplet.agent_id)
        target = agent if triplet.target_id == "self" else mouse_lookup.get(triplet.target_id)
        if agent is None or target is None:
            continue
        active = np.zeros(frames, dtype=bool)
        for start, stop in intervals.get(triplet, ()):
            active |= (resampled_frames >= start) & (resampled_frames < stop)
        if triplet.is_self and triplet.action in solo_lookup:
            action = solo_lookup[triplet.action]
            solo_valid[agent, action] = True
            _set_packed_action(solo_packed, active, (agent,), action)
            for center_index in _positive_centers(
                active, config.data.maximum_positive_centers_per_action_entity
            ):
                positive_records.append((0, agent, agent, action, int(center_index)))
        elif not triplet.is_self and triplet.action in pair_lookup:
            action = pair_lookup[triplet.action]
            pair_valid[agent, target, action] = True
            _set_packed_action(pair_packed, active, (agent, target), action)
            for center_index in _positive_centers(
                active, config.data.maximum_positive_centers_per_action_entity
            ):
                positive_records.append((1, agent, target, action, int(center_index)))
    arrays: dict[str, NDArray[Any]] = {
        "coordinates": coordinates.astype(np.float16),
        "landmark_mask": landmark_mask,
        "center": center.astype(np.float16),
        "body_axis": body_axis.astype(np.float16),
        "body_length": body_length.astype(np.float16),
        "confidence": confidence.astype(np.float16),
        "mouse_valid": mouse_valid,
        "original_frame_ids": pose.frame_ids,
        "original_to_resampled": time_map.original_to_resampled.astype(np.int32),
        "resampled_original_frames": resampled_frames,
        "solo_targets_packed": solo_packed,
        "pair_targets_packed": pair_packed,
        "solo_action_valid": solo_valid,
        "pair_action_valid": pair_valid,
        "positive_centers": np.asarray(positive_records, dtype=np.int32).reshape(-1, 5),
    }
    for name, value in arrays.items():
        _write_array(artifact / f"{name}.npy", value)
    array_hashes = {
        f"{name}.npy": sha256_file(artifact / f"{name}.npy") for name in arrays
    }
    metadata = {
        "schema_version": 1,
        "temporal_version": config.temporal_version,
        "configuration_sha256": configuration_sha256,
        "created_at": datetime.now(UTC).isoformat(),
        "lab_id": lab,
        "lab_index": lab_index,
        "video_id": video_id,
        "fold": fold,
        "source_fps": pose.fps,
        "target_fps": config.data.target_fps,
        "frame_count": frames,
        "original_frame_count": pose.frame_ids.size,
        "mouse_count": len(mice),
        "mouse_ids": list(mice),
        "landmarks": list(config.data.landmarks),
        "scale_valid": pose.scale_valid,
        "normalization_scale": scale,
        "solo_action_count": len(solo_actions),
        "pair_action_count": len(pair_actions),
        "solo_feature_names": list(solo_feature_names(len(config.data.landmarks))),
        "pair_feature_names": list(pair_feature_names(len(config.data.landmarks))),
        "array_sha256": array_hashes,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return SequenceManifestRow(
        lab_id=lab,
        video_id=video_id,
        fold=fold,
        artifact_path=artifact,
        frame_count=frames,
        mouse_count=len(mice),
        fps=config.data.target_fps,
    )


class SequenceArtifact:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.metadata = json.loads((self.path / "metadata.json").read_text(encoding="utf-8"))
        self._arrays: dict[str, NDArray[Any]] = {}

    def array(self, name: str) -> NDArray[Any]:
        if name not in self._arrays:
            self._arrays[name] = np.load(
                self.path / f"{name}.npy", mmap_mode="r", allow_pickle=False
            )
        return self._arrays[name]

    @property
    def frame_count(self) -> int:
        return int(self.metadata["frame_count"])

    @property
    def mouse_count(self) -> int:
        return int(self.metadata["mouse_count"])

    @property
    def fold(self) -> int:
        return int(self.metadata["fold"])

    @property
    def lab_id(self) -> str:
        return str(self.metadata["lab_id"])

    @property
    def video_id(self) -> int:
        return int(self.metadata["video_id"])

    def _slice(self, name: str, start: int, stop: int, length: int) -> NDArray[Any]:
        source = self.array(name)
        left = max(start, 0)
        right = min(stop, self.frame_count)
        output = np.zeros((length, *source.shape[1:]), dtype=source.dtype)
        output[left - start : right - start] = source[left:right]
        return output

    def temporal_window(self, center_index: int, length: int) -> TemporalWindow:
        start = center_index - length // 2
        stop = start + length
        left = max(start, 0)
        right = min(stop, self.frame_count)
        temporal_valid = np.zeros(length, dtype=bool)
        temporal_valid[left - start : right - start] = True
        solo_packed = self._slice("solo_targets_packed", start, stop, length)
        pair_packed = self._slice("pair_targets_packed", start, stop, length)
        solo_count = int(self.metadata["solo_action_count"])
        pair_count = int(self.metadata["pair_action_count"])
        return TemporalWindow(
            coordinates=self._slice("coordinates", start, stop, length).astype(np.float32),
            landmark_mask=self._slice("landmark_mask", start, stop, length).astype(bool),
            center=self._slice("center", start, stop, length).astype(np.float32),
            body_axis=self._slice("body_axis", start, stop, length).astype(np.float32),
            body_length=self._slice("body_length", start, stop, length).astype(np.float32),
            confidence=self._slice("confidence", start, stop, length).astype(np.float32),
            mouse_valid=self._slice("mouse_valid", start, stop, length).astype(bool),
            solo_targets=np.unpackbits(solo_packed, axis=-1, bitorder="little")[..., :solo_count],
            pair_targets=np.unpackbits(pair_packed, axis=-1, bitorder="little")[..., :pair_count],
            solo_action_valid=np.asarray(self.array("solo_action_valid"), dtype=bool).copy(),
            pair_action_valid=np.asarray(self.array("pair_action_valid"), dtype=bool).copy(),
            temporal_valid=temporal_valid,
        )


def _velocity(center: NDArray[np.float32], valid: NDArray[np.bool_]) -> NDArray[np.float32]:
    result = np.zeros_like(center, dtype=np.float32)
    both = valid[1:] & valid[:-1]
    result[1:][both] = center[1:][both] - center[:-1][both]
    return result


def solo_features(window: TemporalWindow, mouse: int) -> NDArray[np.float32]:
    coordinates = window.coordinates[:, mouse]
    mask = window.landmark_mask[:, mouse]
    center = window.center[:, mouse]
    valid = window.mouse_valid[:, mouse] & window.temporal_valid
    relative = coordinates - center[:, None, :]
    relative[~mask] = 0
    velocity = _velocity(center, valid)
    speed = np.linalg.norm(velocity, axis=1, keepdims=True).astype(np.float32)
    return np.concatenate(
        [
            relative.reshape(relative.shape[0], -1),
            mask.astype(np.float32),
            velocity,
            speed,
            window.body_axis[:, mouse],
            window.body_length[:, mouse, None],
            window.confidence[:, mouse, None],
            valid[:, None].astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)


def pair_features(window: TemporalWindow, agent: int, target: int) -> NDArray[np.float32]:
    agent_features = solo_features(window, agent)
    target_features = solo_features(window, target)
    valid = (
        window.mouse_valid[:, agent]
        & window.mouse_valid[:, target]
        & window.temporal_valid
    )
    relative = window.center[:, target] - window.center[:, agent]
    distance = np.linalg.norm(relative, axis=1, keepdims=True).astype(np.float32)
    agent_velocity = _velocity(window.center[:, agent], window.mouse_valid[:, agent])
    target_velocity = _velocity(window.center[:, target], window.mouse_valid[:, target])
    relative_velocity = target_velocity - agent_velocity
    agent_axis = window.body_axis[:, agent]
    target_axis = window.body_axis[:, target]
    heading_dot = np.sum(agent_axis * target_axis, axis=1, keepdims=True)
    heading_cross = (
        agent_axis[:, 0] * target_axis[:, 1]
        - agent_axis[:, 1] * target_axis[:, 0]
    )[:, None]
    geometric = np.concatenate(
        [relative, distance, relative_velocity, heading_dot, heading_cross], axis=1
    )
    geometric[~valid] = 0
    return np.concatenate(
        [agent_features, target_features, geometric, valid[:, None].astype(np.float32)],
        axis=1,
    ).astype(np.float32)


def materialize_window(
    artifact: SequenceArtifact,
    ref: WindowRef,
    *,
    length: int,
    window: TemporalWindow | None = None,
) -> MaterializedWindow:
    value = artifact.temporal_window(ref.center_index, length) if window is None else window
    if ref.route == "solo":
        features = solo_features(value, ref.agent_index)
        targets = value.solo_targets[:, ref.agent_index].astype(np.float32)
        valid = value.solo_action_valid[ref.agent_index]
        entity_valid = value.mouse_valid[:, ref.agent_index]
    else:
        features = pair_features(value, ref.agent_index, ref.target_index)
        targets = value.pair_targets[:, ref.agent_index, ref.target_index].astype(np.float32)
        valid = value.pair_action_valid[ref.agent_index, ref.target_index]
        entity_valid = (
            value.mouse_valid[:, ref.agent_index]
            & value.mouse_valid[:, ref.target_index]
        )
    loss_mask = (
        value.temporal_valid[:, None]
        & entity_valid[:, None]
        & valid[None, :]
    ).astype(np.float32)
    metadata = np.asarray(
        [
            float(artifact.metadata["source_fps"]) / 30.0,
            artifact.mouse_count / 4.0,
            float(bool(artifact.metadata["scale_valid"])),
        ],
        dtype=np.float32,
    )
    return MaterializedWindow(
        features=features,
        targets=targets,
        loss_mask=loss_mask,
        metadata=metadata,
        lab_index=int(artifact.metadata["lab_index"]),
        route=ref.route,
        ref=ref,
    )


class ActionRichWindowSampler:
    """Deterministic lab/action-balanced positive, background, and hard-negative sampler."""

    def __init__(
        self,
        manifest: pl.DataFrame,
        *,
        cache_root: Path,
        config: Phase9Config,
        hard_negatives: pl.DataFrame | None = None,
    ) -> None:
        self.config = config
        self.artifacts = {
            (str(row["lab_id"]), int(row["video_id"])): SequenceArtifact(
                cache_root / str(row["lab_id"]) / str(row["video_id"])
            )
            for row in manifest.iter_rows(named=True)
        }
        positives: defaultdict[Route, list[WindowRef]] = defaultdict(list)
        for artifact in self.artifacts.values():
            centers = np.asarray(artifact.array("positive_centers"), dtype=np.int32)
            for route_value, agent, target, action, center in centers.tolist():
                route: Route = "solo" if route_value == 0 else "pair"
                positives[route].append(
                    WindowRef(
                        lab_id=artifact.lab_id,
                        video_id=artifact.video_id,
                        fold=artifact.fold,
                        route=route,
                        agent_index=agent,
                        target_index=target,
                        center_index=center,
                        sampling_kind="positive_action_balanced",
                        action_index=action,
                    )
                )
        self.positives = dict(positives)
        self.hard_negatives = self._hard_refs(hard_negatives)

    def _hard_refs(self, frame: pl.DataFrame | None) -> dict[Route, list[WindowRef]]:
        output: defaultdict[Route, list[WindowRef]] = defaultdict(list)
        if frame is None or frame.is_empty():
            return {}
        for row in frame.iter_rows(named=True):
            key = (str(row["lab_id"]), int(row["video_id"]))
            artifact = self.artifacts.get(key)
            if artifact is None:
                continue
            mice = [str(value) for value in artifact.metadata["mouse_ids"]]
            agent_id = str(row["agent_id"])
            target_id = str(row["target_id"])
            if agent_id not in mice:
                continue
            agent = mice.index(agent_id)
            route: Route = "solo" if target_id == "self" else "pair"
            target = agent if route == "solo" else (mice.index(target_id) if target_id in mice else -1)
            if target < 0:
                continue
            original = np.asarray(artifact.array("original_frame_ids"), dtype=np.int64)
            position = int(np.searchsorted(original, int(row["video_frame"])))
            position = min(max(position, 0), original.size - 1)
            center = int(np.asarray(artifact.array("original_to_resampled"))[position])
            output[route].append(
                WindowRef(
                    lab_id=artifact.lab_id,
                    video_id=artifact.video_id,
                    fold=artifact.fold,
                    route=route,
                    agent_index=agent,
                    target_index=target,
                    center_index=center,
                    sampling_kind="hard_negative_prior_oof",
                    action_index=-1,
                )
            )
        return dict(output)

    def _background_ref(
        self, generator: np.random.Generator, artifacts: list[SequenceArtifact], route: Route
    ) -> WindowRef:
        eligible = []
        for artifact in artifacts:
            if route == "solo":
                valid_artifact = bool(
                    np.asarray(artifact.array("solo_action_valid")).any()
                )
            else:
                valid_artifact = bool(
                    np.asarray(artifact.array("pair_action_valid")).any()
                )
            if valid_artifact:
                eligible.append(artifact)
        if not eligible:
            raise ValueError(f"no fold-eligible artifacts support the {route} route")
        by_lab: defaultdict[str, list[SequenceArtifact]] = defaultdict(list)
        for artifact in eligible:
            by_lab[artifact.lab_id].append(artifact)
        labs = sorted(by_lab)
        while True:
            lab = labs[int(generator.integers(0, len(labs)))]
            candidates = by_lab[lab]
            artifact = candidates[int(generator.integers(0, len(candidates)))]
            agent = int(generator.integers(0, artifact.mouse_count))
            if route == "solo":
                target = agent
                entity_valid = bool(
                    np.asarray(artifact.array("solo_action_valid"))[agent].any()
                )
            else:
                choices = [value for value in range(artifact.mouse_count) if value != agent]
                target = choices[int(generator.integers(0, len(choices)))] if choices else agent
                entity_valid = bool(
                    target != agent
                    and np.asarray(artifact.array("pair_action_valid"))[
                        agent, target
                    ].any()
                )
            if entity_valid:
                return WindowRef(
                    lab_id=artifact.lab_id,
                    video_id=artifact.video_id,
                    fold=artifact.fold,
                    route=route,
                    agent_index=agent,
                    target_index=target,
                    center_index=int(generator.integers(0, artifact.frame_count)),
                    sampling_kind="background",
                    action_index=-1,
                )

    def plan(
        self,
        *,
        held_out_fold: int,
        route: Route,
        count: int,
        seed: int,
        validation: bool = False,
    ) -> list[WindowRef]:
        generator = np.random.default_rng(seed)
        allowed = [
            artifact
            for artifact in self.artifacts.values()
            if (artifact.fold == held_out_fold) == validation
        ]
        if not allowed:
            raise ValueError("window plan has no fold-eligible artifacts")
        allowed_keys = {(artifact.lab_id, artifact.video_id) for artifact in allowed}
        positives = [
            ref for ref in self.positives.get(route, []) if (ref.lab_id, ref.video_id) in allowed_keys
        ]
        hard = [
            ref
            for ref in self.hard_negatives.get(route, [])
            if (ref.lab_id, ref.video_id) in allowed_keys
        ]
        if validation:
            positive_count = count // 2
            hard_count = 0
        else:
            positive_count = round(count * self.config.data.positive_fraction)
            hard_count = round(count * self.config.data.hard_negative_fraction)
        background_count = count - positive_count - hard_count
        groups: defaultdict[tuple[str, int], list[WindowRef]] = defaultdict(list)
        for ref in positives:
            groups[(ref.lab_id, ref.action_index)].append(ref)
        selected: list[WindowRef] = []
        group_keys = sorted(groups)
        if group_keys:
            generator.shuffle(group_keys)
            for index in range(positive_count):
                values = groups[group_keys[index % len(group_keys)]]
                selected.append(values[int(generator.integers(0, len(values)))])
        else:
            background_count += positive_count
        if hard:
            selected.extend(hard[int(value)] for value in generator.integers(0, len(hard), hard_count))
        else:
            background_count += hard_count
        selected.extend(
            self._background_ref(generator, allowed, route) for _ in range(background_count)
        )
        generator.shuffle(selected)
        if any((ref.fold == held_out_fold) != validation for ref in selected):
            raise AssertionError("window sampler violated complete outer-fold isolation")
        return selected
