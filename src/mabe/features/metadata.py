"""Stable numeric metadata features with explicit missing/unknown categories."""

from __future__ import annotations

import hashlib
import re

import numpy as np

from mabe.features.common import (
    CategoryRegistry,
    FloatArray,
    MetadataContext,
    MouseMetadata,
    add_feature,
)


def _token(value: str) -> str:
    readable = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")[:28] or "empty"
    digest = hashlib.sha1(value.encode(), usedforsecurity=False).hexdigest()[:6]
    return f"{readable}_{digest}"


def _constant(shape: tuple[int, int], value: float) -> FloatArray:
    return np.full(shape, value, dtype=np.float32)


def _mouse_value(mouse: MouseMetadata | None, field: str) -> str | None:
    return getattr(mouse, field) if mouse is not None else None


def _condition_numeric(value: str | None) -> tuple[float, float, float, float]:
    """Return day, hour, light-on, and timing-known channels from condition text."""
    if value is None:
        return 0.0, 0.0, 0.0, 0.0
    lowered = value.lower()
    day_match = re.search(r"day\s*(\d+)", lowered)
    time_match = re.search(r"(\d{1,2}):(\d{2})", lowered)
    day = float(day_match.group(1)) if day_match else 0.0
    hour = (
        float(time_match.group(1)) + float(time_match.group(2)) / 60.0
        if time_match
        else 0.0
    )
    light_on = float("lights on" in lowered)
    timing_known = float(day_match is not None or time_match is not None)
    return day, hour, light_on, timing_known


def _device_indicator(mouse: MouseMetadata | None) -> float:
    condition = (mouse.condition or "").lower() if mouse is not None else ""
    return float(any(token in condition for token in ("device", "cable", "wireless")))


def _add_categorical(
    values: dict[str, FloatArray],
    groups: dict[str, str],
    *,
    prefix: str,
    field: str,
    per_entity: list[str | None],
    frames: int,
    registry: CategoryRegistry,
) -> None:
    codes = np.asarray([registry.encode(field, item) for item in per_entity], dtype=np.float32)
    matrix = np.broadcast_to(codes[None, :], (frames, len(per_entity))).copy()
    add_feature(values, groups, f"metadata__{prefix}_{field}_code", matrix, "metadata")
    for index, category in enumerate(registry.categories[field]):
        encoded = np.broadcast_to((codes == index)[None, :], matrix.shape)
        add_feature(
            values,
            groups,
            f"metadata__{prefix}_{field}_is_{_token(category)}",
            encoded,
            "metadata",
        )


def add_metadata_features(
    values: dict[str, FloatArray],
    groups: dict[str, str],
    *,
    context: MetadataContext,
    registry: CategoryRegistry,
    agent_ids: tuple[str, ...],
    target_ids: tuple[str, ...] | None,
    frames: int,
) -> None:
    """Append global, agent, and optional target metadata in row-major time order."""
    entities = len(agent_ids)
    shape = (frames, entities)
    global_values = {
        "lab": context.lab_id,
        "tracker": context.tracker,
        "skeleton": context.skeleton,
        "arena_type": context.arena_type,
        "arena_shape": context.arena_shape,
    }
    for field, item in global_values.items():
        _add_categorical(
            values,
            groups,
            prefix="video",
            field=field,
            per_entity=[item] * entities,
            frames=frames,
            registry=registry,
        )
    add_feature(values, groups, "metadata__fps", _constant(shape, context.fps), "metadata")
    add_feature(
        values,
        groups,
        "metadata__mouse_count",
        _constant(shape, context.mouse_count),
        "metadata",
    )
    for name, dimension_value in (
        ("arena_width_cm", context.arena_width_cm),
        ("arena_height_cm", context.arena_height_cm),
    ):
        add_feature(
            values,
            groups,
            f"metadata__{name}",
            _constant(shape, dimension_value or 0.0),
            "metadata",
        )
        add_feature(
            values,
            groups,
            f"metadata__{name}_missing",
            _constant(shape, float(dimension_value is None)),
            "metadata",
        )

    agent_mice = [context.mice.get(mouse_id) for mouse_id in agent_ids]
    for field in ("strain", "color", "sex", "age", "condition"):
        _add_categorical(
            values,
            groups,
            prefix="agent",
            field=field,
            per_entity=[_mouse_value(mouse, field) for mouse in agent_mice],
            frames=frames,
            registry=registry,
        )
    agent_devices = np.asarray(
        [_device_indicator(mouse) for mouse in agent_mice],
        dtype=np.float32,
    )
    add_feature(
        values,
        groups,
        "metadata__agent_device_indicator",
        np.broadcast_to(agent_devices[None, :], shape),
        "metadata",
    )
    agent_condition = [_condition_numeric(mouse.condition if mouse else None) for mouse in agent_mice]
    for position, name in enumerate(("day", "hour", "light_on", "timing_known")):
        channel = np.asarray([item[position] for item in agent_condition], dtype=np.float32)
        add_feature(
            values,
            groups,
            f"metadata__agent_condition_{name}",
            np.broadcast_to(channel[None, :], shape),
            "metadata",
        )

    if target_ids is None:
        return
    target_mice = [context.mice.get(mouse_id) for mouse_id in target_ids]
    for field in ("strain", "color", "sex", "age", "condition"):
        agent_items = [_mouse_value(mouse, field) for mouse in agent_mice]
        target_items = [_mouse_value(mouse, field) for mouse in target_mice]
        _add_categorical(
            values,
            groups,
            prefix="target",
            field=field,
            per_entity=target_items,
            frames=frames,
            registry=registry,
        )
        both_known = np.asarray(
            [left is not None and right is not None for left, right in zip(agent_items, target_items, strict=True)]
        )
        same = np.asarray(
            [left == right and left is not None for left, right in zip(agent_items, target_items, strict=True)]
        )
        add_feature(
            values,
            groups,
            f"metadata__pair_same_{field}",
            np.broadcast_to(same[None, :], shape),
            "metadata",
        )
        add_feature(
            values,
            groups,
            f"metadata__pair_{field}_known",
            np.broadcast_to(both_known[None, :], shape),
            "metadata",
        )
    target_devices = np.asarray(
        [_device_indicator(mouse) for mouse in target_mice],
        dtype=np.float32,
    )
    add_feature(
        values,
        groups,
        "metadata__target_device_indicator",
        np.broadcast_to(target_devices[None, :], shape),
        "metadata",
    )
    target_condition = [
        _condition_numeric(mouse.condition if mouse else None) for mouse in target_mice
    ]
    for position, name in enumerate(("day", "hour", "light_on", "timing_known")):
        channel = np.asarray([item[position] for item in target_condition], dtype=np.float32)
        add_feature(
            values,
            groups,
            f"metadata__target_condition_{name}",
            np.broadcast_to(channel[None, :], shape),
            "metadata",
        )
    add_feature(
        values,
        groups,
        "metadata__pair_same_device_indicator",
        np.broadcast_to((target_devices == agent_devices)[None, :], shape),
        "metadata",
    )
