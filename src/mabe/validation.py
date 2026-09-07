"""Leakage-safe duplicate grouping, fold assignment, and split artifacts."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from mabe.fingerprints import schema_fingerprint, sha256_file
from mabe.io import scan_annotation
from mabe.masks import CanonicalTriplet
from mabe.schemas import FOLD_COLUMNS, OOF_COLUMNS, OOF_SCHEMA

PHASE4_SCHEMA_VERSION = 1
EXPECTED_COMPETITION_LAB_COUNT = 15
EXPECTED_AUXILIARY_SOURCE_COUNT = 6
N_PRIMARY_FOLDS = 5


@dataclass(frozen=True, slots=True)
class LabClassification:
    labs: pl.DataFrame
    competition_labs: tuple[str, ...]
    auxiliary_sources: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TrajectorySignature:
    strict: str
    coarse: str
    sampled_frame_count: int
    feature_count: int


@dataclass(frozen=True, slots=True)
class Phase4Outputs:
    folds_path: Path
    duplicate_groups_path: Path
    video_action_support_path: Path
    support_distribution_path: Path
    stress_splits_path: Path
    quality_report_path: Path
    oof_schema_path: Path
    competition_labs: tuple[str, ...]
    auxiliary_sources: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SplitMembership:
    train: pl.DataFrame
    validation: pl.DataFrame


class _UnionFind:
    def __init__(self, keys: Iterable[str]) -> None:
        self.parent = {key: key for key in keys}

    def find(self, key: str) -> str:
        root = key
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[key] != key:
            parent = self.parent[key]
            self.parent[key] = root
            key = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        self.parent[second] = first


def _source_class(lab_id: str) -> tuple[str, str]:
    if lab_id.startswith("CalMS21_"):
        return "auxiliary", "calms21"
    if lab_id == "CRIM13":
        return "auxiliary", "crim13"
    if lab_id.startswith("MABe22_"):
        return "auxiliary", "mabe22"
    return "competition", "competition"


def classify_labs(
    videos: pl.DataFrame,
    *,
    validate_expected: bool = True,
) -> LabClassification:
    """Classify source IDs from naming conventions rather than a lab allow-list."""
    if not {"split", "lab_id"}.issubset(videos.columns):
        raise ValueError("videos must contain split and lab_id")
    lab_ids = sorted(str(value) for value in videos.filter(pl.col("split") == "train")["lab_id"].unique())
    records = []
    for lab_id in lab_ids:
        role, group = _source_class(lab_id)
        records.append({"lab_id": lab_id, "source_role": role, "source_group": group})
    frame = pl.DataFrame(records).sort("lab_id")
    competition = tuple(frame.filter(pl.col("source_role") == "competition")["lab_id"].to_list())
    auxiliary = tuple(frame.filter(pl.col("source_role") == "auxiliary")["lab_id"].to_list())
    if validate_expected:
        if len(competition) != EXPECTED_COMPETITION_LAB_COUNT:
            raise AssertionError(
                f"expected {EXPECTED_COMPETITION_LAB_COUNT} competition labs, found {len(competition)}"
            )
        if len(auxiliary) != EXPECTED_AUXILIARY_SOURCE_COUNT:
            raise AssertionError(
                f"expected {EXPECTED_AUXILIARY_SOURCE_COUNT} auxiliary sources, found {len(auxiliary)}"
            )
    return LabClassification(frame, competition, auxiliary)


def _video_key(split: object, lab_id: object, video_id: object) -> str:
    return f"{split}:{lab_id}:{int(str(video_id))}"


def _structural_signature(row: Mapping[str, object]) -> str:
    payload = {
        "mouse_count": int(str(row["mouse_count"])),
        "observed_frame_count": int(str(row["observed_frame_count"])),
        "skeleton_id": str(row["skeleton_id"]),
        "tracking_row_count": int(str(row["tracking_row_count"])),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _signature_candidates(rows: Sequence[dict[str, Any]], *, frame_tolerance: float) -> set[str]:
    by_shape: defaultdict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if bool(row.get("supervised_eligible")) or str(row.get("split")) == "test":
            by_shape[(str(row["skeleton_id"]), int(row["mouse_count"]))].append(row)
    candidates: set[str] = set()
    for shape_rows in by_shape.values():
        ordered = sorted(shape_rows, key=lambda row: int(row["observed_frame_count"]))
        for left_index, left in enumerate(ordered):
            left_frames = int(left["observed_frame_count"])
            for right in ordered[left_index + 1 :]:
                right_frames = int(right["observed_frame_count"])
                relative = abs(right_frames - left_frames) / max(left_frames, right_frames, 1)
                if relative > frame_tolerance:
                    break
                candidates.add(_video_key(left["split"], left["lab_id"], left["video_id"]))
                candidates.add(_video_key(right["split"], right["lab_id"], right["video_id"]))
    return candidates


def normalized_trajectory_signature(
    path: str | Path,
    *,
    observed_start_frame: int,
    observed_stop_frame: int,
    mouse_count: int,
    sample_count: int = 64,
) -> TrajectorySignature:
    """Hash sampled, translation/rotation/scale/permutation-invariant trajectories."""
    if observed_start_frame >= observed_stop_frame:
        raise ValueError("observed frame bounds must satisfy start < stop")
    if mouse_count < 1:
        raise ValueError("mouse_count must be positive")
    requested = np.unique(
        np.rint(
            np.linspace(observed_start_frame, observed_stop_frame - 1, sample_count)
        ).astype(np.int64)
    )
    centers = (
        pl.scan_parquet(path)
        .select(
            pl.col("video_frame").cast(pl.Int64),
            pl.col("mouse_id").cast(pl.Int64),
            pl.col("x").cast(pl.Float64),
            pl.col("y").cast(pl.Float64),
        )
        .filter(pl.col("video_frame").is_in(requested.tolist()))
        .group_by("video_frame", "mouse_id")
        .agg(pl.col("x").mean(), pl.col("y").mean())
        .sort("video_frame", "mouse_id")
        .collect(engine="streaming")
    )
    by_frame: defaultdict[int, dict[int, tuple[float, float]]] = defaultdict(dict)
    for row in centers.iter_rows(named=True):
        by_frame[int(row["video_frame"])][int(row["mouse_id"])] = (
            float(row["x"]),
            float(row["y"]),
        )

    pair_count = mouse_count * (mouse_count - 1) // 2
    pair_features: list[list[float]] = []
    speed_features: list[list[float]] = []
    previous: dict[int, tuple[float, float]] | None = None
    scales: list[float] = []
    for frame_id in requested.tolist():
        positions = by_frame.get(int(frame_id), {})
        mouse_ids = sorted(positions)
        distances = [
            float(np.hypot(
                positions[mouse_ids[left]][0] - positions[mouse_ids[right]][0],
                positions[mouse_ids[left]][1] - positions[mouse_ids[right]][1],
            ))
            for left in range(len(mouse_ids))
            for right in range(left + 1, len(mouse_ids))
        ]
        scales.extend(value for value in distances if np.isfinite(value) and value > 1e-12)
        pair_features.append(sorted(distances)[:pair_count])
        if previous is None:
            speeds: list[float] = []
        else:
            speeds = sorted(
                float(np.hypot(position[0] - previous[mouse][0], position[1] - previous[mouse][1]))
                for mouse, position in positions.items()
                if mouse in previous
            )
        speed_features.append(speeds[:mouse_count])
        previous = positions

    scale = float(np.median(scales)) if scales else 1.0
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    feature_values: list[float] = []
    for pairs, speeds in zip(pair_features, speed_features, strict=True):
        feature_values.extend(value / scale for value in pairs)
        feature_values.extend([float("nan")] * (pair_count - len(pairs)))
        feature_values.extend(value / scale for value in speeds)
        feature_values.extend([float("nan")] * (mouse_count - len(speeds)))
    values = np.asarray(feature_values, dtype=np.float64)

    def digest(decimals: int) -> str:
        factor = 10**decimals
        quantized = np.rint(values * factor)
        quantized = np.nan_to_num(quantized, nan=-2_147_483_648).astype("<i8")
        header = np.asarray(
            [mouse_count, requested.size, pair_count, decimals], dtype="<i8"
        ).tobytes()
        return hashlib.sha256(header + quantized.tobytes()).hexdigest()

    return TrajectorySignature(
        strict=digest(4),
        coarse=digest(2),
        sampled_frame_count=int(requested.size),
        feature_count=int(values.size),
    )


def build_duplicate_groups(
    videos: pl.DataFrame,
    source_files: pl.DataFrame,
    *,
    dataset_root: str | Path,
    sample_count: int = 64,
    candidate_frame_tolerance: float = 0.005,
    max_workers: int = 4,
) -> pl.DataFrame:
    """Assign stable duplicate groups from exact and normalized trajectory hashes."""
    tracking_hashes = source_files.filter(pl.col("kind") == "tracking").select(
        "split", "lab_id", "video_id", pl.col("sha256").alias("exact_sha256")
    )
    merged = videos.join(tracking_hashes, on=["split", "lab_id", "video_id"], how="left")
    if merged["exact_sha256"].null_count():
        raise ValueError("every video must have an exact tracking-file hash")
    rows = merged.to_dicts()
    candidates = _signature_candidates(rows, frame_tolerance=candidate_frame_tolerance)
    root = Path(dataset_root)

    signature_jobs: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = _video_key(row["split"], row["lab_id"], row["video_id"])
        if key in candidates:
            signature_jobs.setdefault(str(row["exact_sha256"]), row)

    def calculate(item: tuple[str, dict[str, Any]]) -> tuple[str, TrajectorySignature]:
        exact_hash, row = item
        path = Path(str(row["tracking_path"]))
        if not path.is_absolute():
            path = root / path
        signature = normalized_trajectory_signature(
            path,
            observed_start_frame=int(row["observed_start_frame"]),
            observed_stop_frame=int(row["observed_stop_frame"]),
            mouse_count=int(row["mouse_count"]),
            sample_count=sample_count,
        )
        return exact_hash, signature

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        signatures = dict(executor.map(calculate, sorted(signature_jobs.items())))

    keys = [_video_key(row["split"], row["lab_id"], row["video_id"]) for row in rows]
    union_find = _UnionFind(keys)
    exact_members: defaultdict[str, list[str]] = defaultdict(list)
    for row, key in zip(rows, keys, strict=True):
        exact_members[str(row["exact_sha256"])].append(key)
    for members in exact_members.values():
        for other in members[1:]:
            union_find.union(members[0], other)

    strict_members: defaultdict[tuple[str, int, str], list[str]] = defaultdict(list)
    coarse_members: defaultdict[tuple[str, int, str], list[str]] = defaultdict(list)
    for row, key in zip(rows, keys, strict=True):
        signature = signatures.get(str(row["exact_sha256"]))
        if signature is None:
            continue
        shape = (str(row["skeleton_id"]), int(row["mouse_count"]))
        strict_members[(*shape, signature.strict)].append(key)
        coarse_members[(*shape, signature.coarse)].append(key)
    for groups in (strict_members, coarse_members):
        for members in groups.values():
            for other in members[1:]:
                union_find.union(members[0], other)

    components: defaultdict[str, list[str]] = defaultdict(list)
    for key in keys:
        components[union_find.find(key)].append(key)
    group_id_by_key: dict[str, str] = {}
    for members in components.values():
        stable_members = sorted(members)
        digest = hashlib.sha256("|".join(stable_members).encode()).hexdigest()[:16]
        for key in members:
            group_id_by_key[key] = f"dup-{digest}"

    exact_counts = {key: len(value) for key, value in exact_members.items()}
    strict_counts = {key: len(value) for key, value in strict_members.items()}
    coarse_counts = {key: len(value) for key, value in coarse_members.items()}
    records: list[dict[str, object]] = []
    for row, key in zip(rows, keys, strict=True):
        signature = signatures.get(str(row["exact_sha256"]))
        shape = (str(row["skeleton_id"]), int(row["mouse_count"]))
        strict_count = strict_counts.get((*shape, signature.strict), 0) if signature else 0
        coarse_count = coarse_counts.get((*shape, signature.coarse), 0) if signature else 0
        exact_count = exact_counts[str(row["exact_sha256"])]
        duplicate_type = "singleton"
        if coarse_count > 1:
            duplicate_type = "normalized_near"
        if strict_count > 1:
            duplicate_type = "normalized_exact"
        if exact_count > 1:
            duplicate_type = "exact_file"
        records.append(
            {
                "split": str(row["split"]),
                "lab_id": str(row["lab_id"]),
                "video_id": int(row["video_id"]),
                "exact_sha256": str(row["exact_sha256"]),
                "structural_signature": _structural_signature(row),
                "normalized_signature": signature.strict if signature else None,
                "normalized_coarse_signature": signature.coarse if signature else None,
                "signature_status": "computed" if signature else "structurally_unique",
                "duplicate_group_id": group_id_by_key[key],
                "duplicate_group_size": len(components[union_find.find(key)]),
                "duplicate_type": duplicate_type,
            }
        )
    return pl.DataFrame(records).sort("split", "lab_id", "video_id")


def _union_length(intervals: Iterable[tuple[int, int]]) -> int:
    ordered = sorted(intervals)
    if not ordered:
        return 0
    total = 0
    current_start, current_stop = ordered[0]
    for start, stop in ordered[1:]:
        if start > current_stop:
            total += current_stop - current_start
            current_start, current_stop = start, stop
        else:
            current_stop = max(current_stop, stop)
    return total + current_stop - current_start


def build_video_action_support(
    videos: pl.DataFrame,
    triplets: pl.DataFrame,
    *,
    dataset_root: str | Path,
) -> pl.DataFrame:
    """Compute positive-frame duration per video/action without dense frame expansion."""
    allowed_by_video: defaultdict[tuple[str, int], set[CanonicalTriplet]] = defaultdict(set)
    for row in triplets.filter(
        (pl.col("split") == "train") & ~pl.col("is_duplicate")
    ).iter_rows(named=True):
        allowed_by_video[(str(row["lab_id"]), int(row["video_id"]))].add(
            CanonicalTriplet(str(row["agent_id"]), str(row["target_id"]), str(row["action"]))
        )
    root = Path(dataset_root)
    records: list[dict[str, object]] = []
    for video in videos.filter(pl.col("supervised_eligible")).iter_rows(named=True):
        lab_id = str(video["lab_id"])
        video_id = int(video["video_id"])
        annotation_path = Path(str(video["annotation_path"]))
        if not annotation_path.is_absolute():
            annotation_path = root / annotation_path
        allowed = allowed_by_video[(lab_id, video_id)]
        mouse_ids = set(str(value) for value in (video["mouse_ids"] or ()))
        intervals: defaultdict[CanonicalTriplet, list[tuple[int, int]]] = defaultdict(list)
        interval_counts: defaultdict[CanonicalTriplet, int] = defaultdict(int)
        for row in scan_annotation(annotation_path).collect().iter_rows(named=True):
            agent = str(row["agent_id"])
            source_target = str(row["target_id"])
            target = "self" if source_target == agent else source_target
            triplet = CanonicalTriplet(agent, target, str(row["action"]))
            start = int(row["start_frame"])
            stop = int(row["stop_frame"])
            if (
                start >= stop
                or start < int(video["observed_start_frame"])
                or stop > int(video["observed_stop_frame"])
                or agent not in mouse_ids
                or source_target not in mouse_ids
                or triplet not in allowed
            ):
                continue
            intervals[triplet].append((start, stop))
            interval_counts[triplet] += 1
        by_action_frames: defaultdict[str, int] = defaultdict(int)
        by_action_triplets: defaultdict[str, int] = defaultdict(int)
        by_action_intervals: defaultdict[str, int] = defaultdict(int)
        for triplet, triplet_intervals in intervals.items():
            by_action_frames[triplet.action] += _union_length(triplet_intervals)
            by_action_triplets[triplet.action] += 1
            by_action_intervals[triplet.action] += interval_counts[triplet]
        observed_total = sum(by_action_frames.values())
        if observed_total != int(video["positive_frame_count"]):
            raise AssertionError(
                f"support mismatch for {(lab_id, video_id)}: {observed_total} != "
                f"{video['positive_frame_count']}"
            )
        for action in sorted(by_action_frames):
            records.append(
                {
                    "lab_id": lab_id,
                    "video_id": video_id,
                    "action": action,
                    "positive_frames": by_action_frames[action],
                    "positive_triplet_count": by_action_triplets[action],
                    "positive_interval_count": by_action_intervals[action],
                }
            )
    return pl.DataFrame(
        records,
        schema={
            "lab_id": pl.String,
            "video_id": pl.Int64,
            "action": pl.String,
            "positive_frames": pl.Int64,
            "positive_triplet_count": pl.Int64,
            "positive_interval_count": pl.Int64,
        },
    ).sort("lab_id", "video_id", "action")


def _assignment_objective(
    assignment: np.ndarray[Any, np.dtype[np.int64]],
    values: np.ndarray[Any, np.dtype[np.float64]],
    presence: np.ndarray[Any, np.dtype[np.bool_]],
    group_sizes: np.ndarray[Any, np.dtype[np.float64]],
    *,
    n_folds: int,
) -> float:
    loads = np.zeros((n_folds, values.shape[1]), dtype=np.float64)
    support = np.zeros((n_folds, values.shape[1]), dtype=bool)
    counts = np.zeros(n_folds, dtype=np.float64)
    for group_index, fold in enumerate(assignment.tolist()):
        if fold < 0:
            continue
        loads[fold] += values[group_index]
        support[fold] |= presence[group_index]
        counts[fold] += group_sizes[group_index]
    target = values.sum(axis=0) / n_folds
    active = target > 0
    duration_cost = float(np.mean(((loads[:, active] - target[active]) / target[active]) ** 2))
    total_count = group_sizes.sum()
    count_target = total_count / n_folds
    count_cost = float(np.mean(((counts - count_target) / max(count_target, 1.0)) ** 2))
    independent_support = presence.sum(axis=0)
    desired_coverage = np.minimum(independent_support, n_folds)
    actual_coverage = support.sum(axis=0)
    coverage_cost = float(
        np.mean(
            np.maximum(desired_coverage - actual_coverage, 0)
            / np.maximum(desired_coverage, 1)
        )
    )
    return duration_cost + 0.25 * count_cost + 3.0 * coverage_cost


def assign_grouped_folds(
    supervised_videos: pl.DataFrame,
    support: pl.DataFrame,
    *,
    n_folds: int = N_PRIMARY_FOLDS,
    seed: int = 2025,
    restarts: int = 16,
) -> dict[str, int]:
    """Balance lab/action durations while assigning each duplicate group once."""
    required = {"lab_id", "video_id", "duplicate_group_id"}
    if missing := required - set(supervised_videos.columns):
        raise ValueError(f"supervised videos are missing columns: {sorted(missing)}")
    groups = sorted(str(value) for value in supervised_videos["duplicate_group_id"].unique())
    group_index = {group: index for index, group in enumerate(groups)}
    video_to_group = {
        (str(row["lab_id"]), int(row["video_id"])): str(row["duplicate_group_id"])
        for row in supervised_videos.iter_rows(named=True)
    }
    labs = sorted(str(value) for value in supervised_videos["lab_id"].unique())
    action_keys = sorted(
        {(str(row["lab_id"]), str(row["action"])) for row in support.iter_rows(named=True)}
    )
    dimensions = [f"lab::{lab}" for lab in labs] + [
        f"action::{lab}::{action}" for lab, action in action_keys
    ]
    dimension_index = {dimension: index for index, dimension in enumerate(dimensions)}
    raw_values = np.zeros((len(groups), len(dimensions)), dtype=np.float64)
    presence = np.zeros((len(groups), len(dimensions)), dtype=bool)
    group_sizes = np.zeros(len(groups), dtype=np.float64)
    for row in supervised_videos.iter_rows(named=True):
        group = video_to_group[(str(row["lab_id"]), int(row["video_id"]))]
        group_row = group_index[group]
        dimension = dimension_index[f"lab::{row['lab_id']}"]
        raw_values[group_row, dimension] += 1.0
        presence[group_row, dimension] = True
        group_sizes[group_row] += 1.0
    for row in support.iter_rows(named=True):
        video_key = (str(row["lab_id"]), int(row["video_id"]))
        if video_key not in video_to_group:
            continue
        group_row = group_index[video_to_group[video_key]]
        dimension = dimension_index[f"action::{row['lab_id']}::{row['action']}"]
        raw_values[group_row, dimension] += float(row["positive_frames"])
        presence[group_row, dimension] = True
    totals = raw_values.sum(axis=0)
    values = np.divide(raw_values, totals, out=np.zeros_like(raw_values), where=totals > 0)
    independent_support = presence.sum(axis=0)
    rarity = (presence / np.maximum(independent_support, 1)).sum(axis=1)
    magnitude = values.max(axis=1)
    best_assignment: np.ndarray[Any, np.dtype[np.int64]] | None = None
    best_objective = float("inf")
    for restart in range(max(restarts, 1)):
        rng = np.random.default_rng(seed + restart)
        jitter = rng.random(len(groups)) * 1e-8
        order = np.lexsort((np.asarray(groups), -(rarity + magnitude + jitter)))
        assignment = np.full(len(groups), -1, dtype=np.int64)
        for group_row in order.tolist():
            fold_order = rng.permutation(n_folds)
            candidates: list[tuple[float, int]] = []
            for fold in fold_order.tolist():
                assignment[group_row] = fold
                score = _assignment_objective(
                    assignment, values, presence, group_sizes, n_folds=n_folds
                )
                candidates.append((score, fold))
            assignment[group_row] = min(candidates)[1]
        score = _assignment_objective(
            assignment, values, presence, group_sizes, n_folds=n_folds
        )
        if score < best_objective:
            best_objective = score
            best_assignment = assignment.copy()
    assert best_assignment is not None

    # Repair any feasible dimension that ended up in only one fold.
    for _ in range(3):
        changed = False
        for dimension in np.argsort(independent_support).tolist():
            supporting_groups = np.flatnonzero(presence[:, dimension])
            if supporting_groups.size < 2:
                continue
            occupied = set(best_assignment[supporting_groups].tolist())
            if len(occupied) >= 2:
                continue
            destinations = sorted(set(range(n_folds)) - occupied)
            moves: list[tuple[float, int, int]] = []
            for group_row in supporting_groups.tolist():
                for destination in destinations:
                    candidate = best_assignment.copy()
                    candidate[group_row] = destination
                    moves.append(
                        (
                            _assignment_objective(
                                candidate, values, presence, group_sizes, n_folds=n_folds
                            ),
                            group_row,
                            destination,
                        )
                    )
            if moves:
                _, group_row, destination = min(moves)
                best_assignment[group_row] = destination
                changed = True
        if not changed:
            break

    for dimension, independent in enumerate(independent_support.tolist()):
        if independent >= 2:
            occupied = set(best_assignment[np.flatnonzero(presence[:, dimension])].tolist())
            if len(occupied) < 2:
                raise AssertionError(f"feasible support dimension was not split: {dimensions[dimension]}")
    return {group: int(best_assignment[index]) for group, index in group_index.items()}


def build_fold_table(
    videos: pl.DataFrame,
    duplicate_groups: pl.DataFrame,
    support: pl.DataFrame,
    classifications: LabClassification,
    *,
    n_folds: int = N_PRIMARY_FOLDS,
    seed: int = 2025,
) -> pl.DataFrame:
    """Create one fold-manifest row for every training video."""
    train = videos.filter(pl.col("split") == "train").join(
        duplicate_groups.filter(pl.col("split") == "train").select(
            "lab_id", "video_id", "duplicate_group_id"
        ),
        on=["lab_id", "video_id"],
        how="left",
    ).join(classifications.labs, on="lab_id", how="left")
    supervised = train.filter(pl.col("supervised_eligible"))
    assignments = assign_grouped_folds(
        supervised,
        support,
        n_folds=n_folds,
        seed=seed,
    )
    action_counts = support.group_by("lab_id", "video_id").agg(
        pl.len().cast(pl.Int64).alias("positive_action_count")
    )
    folds = (
        train.with_columns(
            pl.col("duplicate_group_id")
            .replace_strict(assignments, default=None, return_dtype=pl.Int8)
            .alias("fold"),
            (
                (pl.col("source_role") == "competition") & pl.col("supervised_eligible")
            ).alias("is_primary_validation"),
        )
        .join(action_counts, on=["lab_id", "video_id"], how="left")
        .with_columns(pl.col("positive_action_count").fill_null(0))
        .select(FOLD_COLUMNS)
        .sort("lab_id", "video_id")
    )
    if folds.filter(pl.col("supervised_eligible") & pl.col("fold").is_null()).height:
        raise AssertionError("every supervised video must receive a fold")
    if folds.filter(~pl.col("supervised_eligible") & pl.col("fold").is_not_null()).height:
        raise AssertionError("excluded videos must not receive supervised folds")
    leakage = (
        folds.filter(pl.col("fold").is_not_null())
        .group_by("duplicate_group_id")
        .agg(pl.col("fold").n_unique().alias("n_folds"))
        .filter(pl.col("n_folds") > 1)
    )
    if leakage.height:
        raise AssertionError("duplicate groups cross folds")
    return folds


def support_distribution(
    folds: pl.DataFrame,
    support: pl.DataFrame,
    *,
    n_folds: int = N_PRIMARY_FOLDS,
) -> pl.DataFrame:
    """Report positive durations and independent coverage for every lab/action/fold."""
    joined = support.join(
        folds.select("lab_id", "video_id", "fold", "duplicate_group_id"),
        on=["lab_id", "video_id"],
        how="inner",
    )
    combinations = support.select("lab_id", "action").unique()
    grid = combinations.join(
        pl.DataFrame({"fold": pl.Series(range(n_folds), dtype=pl.Int8)}), how="cross"
    )
    per_fold = joined.group_by("lab_id", "action", "fold").agg(
        pl.col("positive_frames").sum(),
        pl.col("video_id").n_unique().alias("positive_video_count"),
        pl.col("duplicate_group_id").n_unique().alias("positive_group_count"),
    )
    totals = joined.group_by("lab_id", "action").agg(
        pl.col("positive_frames").sum().alias("total_positive_frames"),
        pl.col("video_id").n_unique().alias("total_positive_videos"),
        pl.col("duplicate_group_id").n_unique().alias("independent_positive_groups"),
        pl.col("fold").n_unique().alias("positive_fold_count"),
    ).with_columns(
        pl.min_horizontal(pl.col("independent_positive_groups"), pl.lit(n_folds)).alias(
            "feasible_fold_count"
        )
    )
    result = (
        grid.join(per_fold, on=["lab_id", "action", "fold"], how="left")
        .join(totals, on=["lab_id", "action"], how="left")
        .with_columns(
            pl.col("positive_frames", "positive_video_count", "positive_group_count").fill_null(0),
            pl.when(pl.col("independent_positive_groups") < 2)
            .then(pl.lit("only_one_independent_positive_group"))
            .when(pl.col("positive_fold_count") < 2)
            .then(pl.lit("assignment_coverage_shortfall"))
            .otherwise(pl.lit(None, dtype=pl.String))
            .alias("coverage_exception"),
        )
        .sort("lab_id", "action", "fold")
    )
    shortfalls = result.filter(pl.col("coverage_exception") == "assignment_coverage_shortfall")
    if shortfalls.height:
        raise AssertionError("feasible lab/action support was not spread over multiple folds")
    return result


def primary_split(
    folds: pl.DataFrame,
    *,
    held_out_fold: int,
    target_lab: str | None = None,
    include_auxiliary: bool = False,
) -> SplitMembership:
    """Return leakage-safe primary train/validation membership for one held fold."""
    if held_out_fold not in range(N_PRIMARY_FOLDS):
        raise ValueError(f"held_out_fold must be in [0, {N_PRIMARY_FOLDS})")
    primary = folds.filter(pl.col("is_primary_validation"))
    if target_lab is not None:
        primary = primary.filter(pl.col("lab_id") == target_lab)
    validation = primary.filter(pl.col("fold") == held_out_fold)
    train = primary.filter(pl.col("fold") != held_out_fold)
    if include_auxiliary:
        auxiliary = folds.filter(
            pl.col("supervised_eligible")
            & (pl.col("source_role") == "auxiliary")
            & pl.col("source_group").is_in(["calms21", "crim13"])
        )
        train = pl.concat([train, auxiliary], how="vertical")
    if train.filter(pl.col("source_group") == "mabe22").height:
        raise AssertionError("MABe22 may not enter supervised training")
    return SplitMembership(train.sort("lab_id", "video_id"), validation.sort("lab_id", "video_id"))


def stress_split(
    folds: pl.DataFrame,
    videos: pl.DataFrame,
    *,
    dimension: str,
    value: str,
) -> SplitMembership:
    """Create a diagnostic leave-one-domain-out split, separate from primary CV."""
    column = {
        "lab": "lab_id",
        "skeleton": "skeleton_id",
        "tracker": "tracking_method",
    }.get(dimension)
    if column is None:
        raise ValueError("dimension must be lab, skeleton, or tracker")
    eligible = folds.filter(pl.col("is_primary_validation")).join(
        videos.filter(pl.col("split") == "train").select(
            "lab_id", "video_id", "skeleton_id", "tracking_method"
        ),
        on=["lab_id", "video_id"],
        how="left",
    )
    validation = eligible.filter(pl.col(column) == value)
    train = eligible.filter(pl.col(column) != value)
    return SplitMembership(train, validation)


def _package_version() -> str:
    try:
        return metadata.version("mabe-behavior")
    except metadata.PackageNotFoundError:
        return "0.1.0"


def _write_sidecar(
    path: Path,
    *,
    row_count: int | None,
    schema_hash: str | None,
    source_fingerprint: str,
    configuration: dict[str, object],
) -> None:
    payload = {
        "artifact": path.name,
        "artifact_sha256": sha256_file(path),
        "code_version": _package_version(),
        "configuration": configuration,
        "created_at": datetime.now(UTC).isoformat(),
        "row_count": row_count,
        "schema_sha256": schema_hash,
        "source_data_fingerprint": source_fingerprint,
    }
    Path(f"{path}.meta.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _quality_markdown(report: dict[str, object]) -> str:
    counts = report["counts"]
    assert isinstance(counts, dict)
    return "\n".join(
        [
            "# Phase 4 fold-quality report",
            "",
            f"Source fingerprint: `{report['source_fingerprint']}`",
            "",
            "## Source classification",
            "",
            f"- Competition labs: {counts['competition_labs']}",
            f"- Auxiliary source IDs: {counts['auxiliary_sources']}",
            "",
            "## Fold and leakage checks",
            "",
            f"- Primary videos: {counts['primary_videos']}",
            f"- Supervised auxiliary videos: {counts['auxiliary_supervised_videos']}",
            f"- Exact duplicate groups: {counts['exact_duplicate_groups']}",
            f"- Normalized duplicate groups: {counts['normalized_duplicate_groups']}",
            f"- Duplicate groups crossing folds: {counts['duplicate_groups_crossing_folds']}",
            f"- Rare lab/action exceptions: {counts['rare_support_exceptions']}",
            f"- Feasible support shortfalls: {counts['feasible_support_shortfalls']}",
            "",
            "Primary validation contains only competition labs. CalMS21/CRIM13 are optional",
            "training-only auxiliary sources; MABe22 is excluded from supervised training.",
            "Stress splits are diagnostic and are not used for primary model selection.",
            "",
        ]
    )


def build_phase4_artifacts(
    *,
    videos_path: str | Path,
    triplets_path: str | Path,
    source_files_path: str | Path,
    dataset_root: str | Path,
    output_root: str | Path,
    source_fingerprint: str,
    seed: int = 2025,
    n_folds: int = N_PRIMARY_FOLDS,
) -> Phase4Outputs:
    """Build all fixed Phase 4 folds, reports, and evaluation schemas."""
    videos = pl.read_parquet(videos_path)
    triplets = pl.read_parquet(triplets_path)
    source_files = pl.read_parquet(source_files_path)
    classifications = classify_labs(videos)
    duplicates = build_duplicate_groups(
        videos,
        source_files,
        dataset_root=dataset_root,
    )
    support = build_video_action_support(videos, triplets, dataset_root=dataset_root)
    folds = build_fold_table(
        videos,
        duplicates,
        support,
        classifications,
        n_folds=n_folds,
        seed=seed,
    )
    distribution = support_distribution(folds, support, n_folds=n_folds)
    stress = (
        folds.filter(pl.col("is_primary_validation"))
        .join(
            videos.filter(pl.col("split") == "train").select(
                "lab_id", "video_id", "skeleton_id", "tracking_method"
            ),
            on=["lab_id", "video_id"],
            how="left",
        )
        .select(
            "lab_id",
            "video_id",
            pl.col("lab_id").alias("leave_one_lab_out"),
            pl.col("skeleton_id").alias("leave_one_skeleton_out"),
            pl.col("tracking_method").alias("leave_one_tracker_out"),
        )
        .sort("lab_id", "video_id")
    )
    duplicate_crossings = (
        folds.filter(pl.col("fold").is_not_null())
        .group_by("duplicate_group_id")
        .agg(pl.col("fold").n_unique().alias("n_folds"))
        .filter(pl.col("n_folds") > 1)
        .height
    )
    exact_duplicate_groups = (
        duplicates.filter(pl.col("duplicate_type") == "exact_file")["duplicate_group_id"].n_unique()
    )
    normalized_duplicate_groups = (
        duplicates.filter(pl.col("duplicate_type").is_in(["normalized_exact", "normalized_near"]))[
            "duplicate_group_id"
        ].n_unique()
    )
    exception_table = distribution.select(
        "lab_id", "action", "coverage_exception"
    ).unique()
    rare_exceptions = exception_table.filter(
        pl.col("coverage_exception") == "only_one_independent_positive_group"
    ).height
    shortfalls = exception_table.filter(
        pl.col("coverage_exception") == "assignment_coverage_shortfall"
    ).height
    fold_counts = (
        folds.filter(pl.col("is_primary_validation"))
        .group_by("fold")
        .len()
        .sort("fold")
        .to_dicts()
    )
    report: dict[str, object] = {
        "schema_version": PHASE4_SCHEMA_VERSION,
        "source_fingerprint": source_fingerprint,
        "seed": seed,
        "n_folds": n_folds,
        "competition_labs": list(classifications.competition_labs),
        "auxiliary_sources": list(classifications.auxiliary_sources),
        "auxiliary_groups": ["calms21", "crim13", "mabe22"],
        "fold_counts": fold_counts,
        "counts": {
            "competition_labs": len(classifications.competition_labs),
            "auxiliary_sources": len(classifications.auxiliary_sources),
            "primary_videos": folds.filter(pl.col("is_primary_validation")).height,
            "auxiliary_supervised_videos": folds.filter(
                pl.col("supervised_eligible") & (pl.col("source_role") == "auxiliary")
            ).height,
            "exact_duplicate_groups": exact_duplicate_groups,
            "normalized_duplicate_groups": normalized_duplicate_groups,
            "duplicate_groups_crossing_folds": duplicate_crossings,
            "rare_support_exceptions": rare_exceptions,
            "feasible_support_shortfalls": shortfalls,
        },
        "policies": {
            "primary_validation_role": "competition",
            "optional_training_auxiliary_groups": ["calms21", "crim13"],
            "never_supervised_groups": ["mabe22"],
            "decoder_calibration": "other_oof_folds_only",
            "stress_scores_are_primary": False,
        },
    }

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    folds_path = root / "folds.parquet"
    duplicates_path = root / "duplicate_groups.parquet"
    support_path = root / "video_action_support.parquet"
    distribution_path = root / "support_distribution.parquet"
    stress_path = root / "stress_splits.parquet"
    quality_path = root / "fold_quality.json"
    quality_markdown_path = root / "fold_quality.md"
    oof_schema_path = root / "oof_schema.json"
    frame_artifacts = {
        folds_path: folds,
        duplicates_path: duplicates,
        support_path: support,
        distribution_path: distribution,
        stress_path: stress,
    }
    for path, frame in frame_artifacts.items():
        frame.write_parquet(path, compression="zstd", statistics=True)
    quality_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    quality_markdown_path.write_text(_quality_markdown(report), encoding="utf-8")
    oof_payload = {
        "schema_version": PHASE4_SCHEMA_VERSION,
        "columns": list(OOF_COLUMNS),
        "schema": {name: str(dtype) for name, dtype in OOF_SCHEMA.items()},
        "unique_key": [
            "lab_id",
            "video_id",
            "video_frame",
            "agent_id",
            "target_id",
            "action",
            "model_id",
            "feature_version",
        ],
    }
    oof_schema_path.write_text(
        json.dumps(oof_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    configuration: dict[str, object] = {
        "schema_version": PHASE4_SCHEMA_VERSION,
        "seed": seed,
        "n_folds": n_folds,
        "trajectory_sample_count": 64,
        "candidate_frame_tolerance": 0.005,
    }
    for path, frame in frame_artifacts.items():
        _write_sidecar(
            path,
            row_count=frame.height,
            schema_hash=schema_fingerprint(frame),
            source_fingerprint=source_fingerprint,
            configuration=configuration,
        )
    for path in (quality_path, quality_markdown_path, oof_schema_path):
        _write_sidecar(
            path,
            row_count=None,
            schema_hash=None,
            source_fingerprint=source_fingerprint,
            configuration=configuration,
        )
    return Phase4Outputs(
        folds_path,
        duplicates_path,
        support_path,
        distribution_path,
        stress_path,
        quality_path,
        oof_schema_path,
        classifications.competition_labs,
        classifications.auxiliary_sources,
    )
