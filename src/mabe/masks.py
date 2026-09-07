"""Canonical parsing for sparse ``behaviors_labeled`` masks."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

_MOUSE_PATTERN = re.compile(r"^mouse[1-4]$")
_ACTION_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


class BehaviorMaskError(ValueError):
    """Raised when a behavior mask cannot be repaired without guessing."""


@dataclass(frozen=True, slots=True)
class CanonicalTriplet:
    agent_id: str
    target_id: str
    action: str

    @property
    def canonical_entry(self) -> str:
        return f"{self.agent_id},{self.target_id},{self.action}"

    @property
    def is_self(self) -> bool:
        return self.target_id == "self"


@dataclass(frozen=True, slots=True)
class ParsedMaskEntry:
    index: int
    raw_entry: str
    raw_agent_id: str
    raw_target_id: str
    raw_action: str
    triplet: CanonicalTriplet
    repair_reasons: tuple[str, ...]
    is_duplicate: bool

    @property
    def was_repaired(self) -> bool:
        return bool(self.repair_reasons)


@dataclass(frozen=True, slots=True)
class ParsedBehaviorMask:
    raw_json: str
    raw_entries: tuple[str, ...]
    entries: tuple[ParsedMaskEntry, ...]
    triplets: tuple[CanonicalTriplet, ...]

    @property
    def canonical_json(self) -> str:
        return json.dumps([triplet.canonical_entry for triplet in self.triplets], separators=(",", ":"))


def _clean_component(value: str, *, component: str) -> tuple[str, list[str]]:
    reasons: list[str] = []
    cleaned = value.strip()
    if cleaned != value:
        reasons.append(f"trimmed_{component}_whitespace")
    without_apostrophes = cleaned.strip("'").strip()
    if without_apostrophes != cleaned:
        reasons.append(f"removed_{component}_apostrophes")
    normalized = without_apostrophes.lower()
    if normalized != without_apostrophes:
        reasons.append(f"lowercased_{component}")
    return normalized, reasons


def parse_behaviors_labeled(raw_mask: str) -> ParsedBehaviorMask:
    """Parse, repair, validate, and deduplicate one JSON behavior mask."""
    try:
        decoded: object = json.loads(raw_mask)
    except (TypeError, json.JSONDecodeError) as error:
        raise BehaviorMaskError(f"invalid behaviors_labeled JSON: {error}") from error
    if not isinstance(decoded, list):
        raise BehaviorMaskError("behaviors_labeled must decode to a list")
    if not all(isinstance(entry, str) for entry in decoded):
        raise BehaviorMaskError("every behaviors_labeled entry must be a string")

    raw_entries = tuple(decoded)
    entries: list[ParsedMaskEntry] = []
    unique_triplets: list[CanonicalTriplet] = []
    seen: set[CanonicalTriplet] = set()
    for index, raw_entry in enumerate(raw_entries):
        components = raw_entry.split(",")
        if len(components) != 3:
            raise BehaviorMaskError(
                f"entry {index} must contain exactly agent,target,action: {raw_entry!r}"
            )
        raw_agent, raw_target, raw_action = components
        agent, agent_repairs = _clean_component(raw_agent, component="agent")
        target, target_repairs = _clean_component(raw_target, component="target")
        action, action_repairs = _clean_component(raw_action, component="action")
        repairs = [*agent_repairs, *target_repairs, *action_repairs]

        if not _MOUSE_PATTERN.fullmatch(agent):
            raise BehaviorMaskError(f"entry {index} has invalid agent ID: {raw_agent!r}")
        if target == agent:
            target = "self"
            repairs.append("normalized_self_target")
        if target != "self" and not _MOUSE_PATTERN.fullmatch(target):
            raise BehaviorMaskError(f"entry {index} has invalid target ID: {raw_target!r}")
        if not _ACTION_PATTERN.fullmatch(action):
            raise BehaviorMaskError(f"entry {index} has invalid action: {raw_action!r}")

        triplet = CanonicalTriplet(agent_id=agent, target_id=target, action=action)
        is_duplicate = triplet in seen
        if not is_duplicate:
            seen.add(triplet)
            unique_triplets.append(triplet)
        entries.append(
            ParsedMaskEntry(
                index=index,
                raw_entry=raw_entry,
                raw_agent_id=raw_agent,
                raw_target_id=raw_target,
                raw_action=raw_action,
                triplet=triplet,
                repair_reasons=tuple(repairs),
                is_duplicate=is_duplicate,
            )
        )

    return ParsedBehaviorMask(
        raw_json=raw_mask,
        raw_entries=raw_entries,
        entries=tuple(entries),
        triplets=tuple(unique_triplets),
    )
