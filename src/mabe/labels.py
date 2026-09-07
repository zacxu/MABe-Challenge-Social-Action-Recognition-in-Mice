"""Sparse supervision, label vocabularies, and validation-solution construction."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import IntEnum
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from numpy.typing import NDArray

from mabe.fingerprints import schema_fingerprint, sha256_file
from mabe.io import scan_annotation
from mabe.masks import CanonicalTriplet, parse_behaviors_labeled

LABEL_ARTIFACT_SCHEMA_VERSION = 1
SUPERVISED_VIDEO_STATE = "labeled"
EXCLUDED_VIDEO_STATES = frozenset(
    {
        "ambiguous_missing_annotation",
        "test",
        "unlabeled_mabe22",
        "unlabeled_missing_mask",
    }
)

type TripletTuple = tuple[str, str, str]
type TripletLike = CanonicalTriplet | TripletTuple
type VideoKey = tuple[str, int]


class FrameLabelState(IntEnum):
    """Mutually exclusive supervision states for one frame/triplet cell."""

    UNKNOWN_MASKED = 0
    VALID_NEGATIVE = 1
    POSITIVE = 2
    EXCLUDED_VIDEO = 3


@dataclass(frozen=True, slots=True)
class FrameTargets:
    """Compact targets for only the requested frames and candidate triplets."""

    frame_ids: NDArray[np.int64]
    triplets: tuple[CanonicalTriplet, ...]
    targets: NDArray[np.uint8]
    loss_mask: NDArray[np.uint8]
    states: NDArray[np.uint8]

    def __post_init__(self) -> None:
        expected = (self.frame_ids.size, len(self.triplets))
        for name, value in (
            ("targets", self.targets),
            ("loss_mask", self.loss_mask),
            ("states", self.states),
        ):
            if value.shape != expected:
                raise ValueError(f"{name} shape {value.shape} does not match {expected}")
            if value.dtype != np.uint8:
                raise TypeError(f"{name} must use uint8 storage")
        if self.frame_ids.dtype != np.int64:
            raise TypeError("frame_ids must use int64 storage")

    @property
    def positive_mask(self) -> NDArray[np.bool_]:
        result: NDArray[np.bool_] = self.states == FrameLabelState.POSITIVE
        return result

    @property
    def valid_negative_mask(self) -> NDArray[np.bool_]:
        """Cells that are safe to use as negative supervision."""
        result: NDArray[np.bool_] = self.states == FrameLabelState.VALID_NEGATIVE
        return result


@dataclass(frozen=True, slots=True)
class ActionVocabulary:
    """Stable global and modality-specific action indices."""

    version: str
    actions: tuple[str, ...]
    solo_actions: tuple[str, ...]
    pair_actions: tuple[str, ...]

    @property
    def action_to_index(self) -> dict[str, int]:
        return {action: index for index, action in enumerate(self.actions)}

    @property
    def index_to_action(self) -> dict[int, str]:
        return dict(enumerate(self.actions))

    @property
    def solo_action_to_index(self) -> dict[str, int]:
        return {action: index for index, action in enumerate(self.solo_actions)}

    @property
    def index_to_solo_action(self) -> dict[int, str]:
        return dict(enumerate(self.solo_actions))

    @property
    def pair_action_to_index(self) -> dict[str, int]:
        return {action: index for index, action in enumerate(self.pair_actions)}

    @property
    def index_to_pair_action(self) -> dict[int, str]:
        return dict(enumerate(self.pair_actions))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": LABEL_ARTIFACT_SCHEMA_VERSION,
            "vocabulary_version": self.version,
            "action_count": len(self.actions),
            "solo_action_count": len(self.solo_actions),
            "pair_action_count": len(self.pair_actions),
            "actions": list(self.actions),
            "solo_actions": list(self.solo_actions),
            "pair_actions": list(self.pair_actions),
            "action_to_index": self.action_to_index,
            "index_to_action": {str(key): value for key, value in self.index_to_action.items()},
            "solo_action_to_index": self.solo_action_to_index,
            "index_to_solo_action": {
                str(key): value for key, value in self.index_to_solo_action.items()
            },
            "pair_action_to_index": self.pair_action_to_index,
            "index_to_pair_action": {
                str(key): value for key, value in self.index_to_pair_action.items()
            },
        }


@dataclass(frozen=True, slots=True)
class LabelArtifactOutputs:
    vocabulary_path: Path
    triplet_mappings_path: Path
    vocabulary: ActionVocabulary


def _canonical_triplet(value: TripletLike) -> CanonicalTriplet:
    if isinstance(value, CanonicalTriplet):
        return value
    if len(value) != 3:
        raise ValueError(f"triplet must contain agent, target, action: {value!r}")
    agent, target, action = (str(component) for component in value)
    if target == agent:
        target = "self"
    return CanonicalTriplet(agent_id=agent, target_id=target, action=action)


def _source_path(dataset_root: Path, raw_path: object) -> Path | None:
    if raw_path is None:
        return None
    path = Path(str(raw_path))
    return path if path.is_absolute() else dataset_root / path


def _strict_frame_ids(frame_ids: Sequence[int] | NDArray[Any]) -> NDArray[np.int64]:
    values = np.asarray(frame_ids)
    if values.ndim != 1:
        raise ValueError("frame_ids must be one-dimensional")
    if values.dtype.kind not in "iu":
        raise TypeError("frame_ids must contain integers")
    result: NDArray[np.int64] = values.astype(np.int64, copy=False)
    if result.size > 1 and np.any(result[1:] <= result[:-1]):
        raise ValueError("frame_ids must be strictly increasing and unique")
    return result


class LabelStore:
    """Lazy per-video sparse label store backed by Phase 2 manifests."""

    def __init__(
        self,
        *,
        lab_id: str,
        video_id: int,
        video_label_state: str,
        allowed_triplets: Iterable[TripletLike],
        annotation_path: Path | None,
        tracking_path: Path | None,
        observed_start_frame: int | None,
        observed_stop_frame: int | None,
        mouse_ids: Iterable[str] = (),
    ) -> None:
        self.lab_id = str(lab_id)
        self.video_id = int(video_id)
        self.video_label_state = str(video_label_state)
        known_video_states = EXCLUDED_VIDEO_STATES | {SUPERVISED_VIDEO_STATE}
        if self.video_label_state not in known_video_states:
            raise ValueError(f"unknown video label state: {self.video_label_state!r}")
        canonical = tuple(dict.fromkeys(_canonical_triplet(item) for item in allowed_triplets))
        self.allowed_triplets = canonical
        self._allowed_set = frozenset(canonical)
        self.annotation_path = annotation_path
        self.tracking_path = tracking_path
        self.observed_start_frame = observed_start_frame
        self.observed_stop_frame = observed_stop_frame
        self.mouse_ids = frozenset(str(mouse) for mouse in mouse_ids)
        self._intervals_by_triplet: dict[CanonicalTriplet, tuple[tuple[int, int], ...]] | None = None
        self._tracking_frame_ids: NDArray[np.int64] | None = None
        self._quarantined_annotation_count = 0

        if self.video_label_state == SUPERVISED_VIDEO_STATE and self.annotation_path is None:
            raise ValueError("a labeled video must have an annotation path")
        if (
            self.observed_start_frame is not None
            and self.observed_stop_frame is not None
            and self.observed_start_frame >= self.observed_stop_frame
        ):
            raise ValueError("observed frame bounds must satisfy start < stop")

    @classmethod
    def from_manifests(
        cls,
        *,
        lab_id: str,
        video_id: int,
        dataset_root: str | Path,
        videos_path: str | Path,
        triplets_path: str | Path,
        split: str = "train",
    ) -> LabelStore:
        """Create one store without loading its tracking or annotation parquet."""
        videos = (
            pl.scan_parquet(videos_path)
            .filter(
                (pl.col("split") == split)
                & (pl.col("lab_id") == lab_id)
                & (pl.col("video_id") == video_id)
            )
            .collect()
        )
        if videos.height != 1:
            raise KeyError(
                f"expected exactly one manifest row for {(split, lab_id, video_id)}, found {videos.height}"
            )
        row = videos.row(0, named=True)
        triplet_rows = (
            pl.scan_parquet(triplets_path)
            .filter(
                (pl.col("split") == split)
                & (pl.col("lab_id") == lab_id)
                & (pl.col("video_id") == video_id)
                & ~pl.col("is_duplicate")
            )
            .sort("entry_index")
            .select("agent_id", "target_id", "action")
            .collect()
        )
        triplets = (
            CanonicalTriplet(
                agent_id=str(item["agent_id"]),
                target_id=str(item["target_id"]),
                action=str(item["action"]),
            )
            for item in triplet_rows.iter_rows(named=True)
        )
        root = Path(dataset_root)
        return cls(
            lab_id=lab_id,
            video_id=video_id,
            video_label_state=str(row["label_state"]),
            allowed_triplets=triplets,
            annotation_path=_source_path(root, row.get("annotation_path")),
            tracking_path=_source_path(root, row.get("tracking_path")),
            observed_start_frame=(
                int(row["observed_start_frame"])
                if row.get("observed_start_frame") is not None
                else None
            ),
            observed_stop_frame=(
                int(row["observed_stop_frame"])
                if row.get("observed_stop_frame") is not None
                else None
            ),
            mouse_ids=row.get("mouse_ids") or (),
        )

    @property
    def is_supervised(self) -> bool:
        return self.video_label_state == SUPERVISED_VIDEO_STATE

    @property
    def annotations_loaded(self) -> bool:
        return self._intervals_by_triplet is not None

    @property
    def quarantined_annotation_count(self) -> int:
        self._load_intervals()
        return self._quarantined_annotation_count

    @property
    def solo_candidates(self) -> tuple[CanonicalTriplet, ...]:
        return tuple(item for item in self.allowed_triplets if item.is_self)

    @property
    def pair_candidates(self) -> tuple[CanonicalTriplet, ...]:
        return tuple(item for item in self.allowed_triplets if not item.is_self)

    @property
    def tracking_frame_ids(self) -> NDArray[np.int64]:
        if self._tracking_frame_ids is None:
            if self.tracking_path is None:
                raise ValueError("tracking frame IDs are unavailable; pass frame_ids explicitly")
            frame = (
                pl.scan_parquet(self.tracking_path)
                .select(pl.col("video_frame").cast(pl.Int64))
                .unique()
                .sort("video_frame")
                .collect()
            )
            self._tracking_frame_ids = _strict_frame_ids(frame["video_frame"].to_numpy())
        return self._tracking_frame_ids

    def _load_intervals(self) -> None:
        if self._intervals_by_triplet is not None:
            return
        collected: dict[CanonicalTriplet, list[tuple[int, int]]] = {
            triplet: [] for triplet in self.allowed_triplets
        }
        if not self.is_supervised:
            self._intervals_by_triplet = {key: tuple(value) for key, value in collected.items()}
            return
        if self.annotation_path is None or not self.annotation_path.is_file():
            raise FileNotFoundError(f"missing annotation parquet: {self.annotation_path}")

        annotations = scan_annotation(self.annotation_path).collect()
        for row in annotations.iter_rows(named=True):
            agent = str(row["agent_id"])
            source_target = str(row["target_id"])
            target = "self" if source_target == agent else source_target
            triplet = CanonicalTriplet(agent, target, str(row["action"]))
            start = int(row["start_frame"])
            stop = int(row["stop_frame"])
            invalid = start >= stop or triplet not in self._allowed_set
            invalid = invalid or (
                self.observed_start_frame is not None and start < self.observed_start_frame
            )
            invalid = invalid or (
                self.observed_stop_frame is not None and stop > self.observed_stop_frame
            )
            invalid = invalid or (
                bool(self.mouse_ids)
                and (agent not in self.mouse_ids or source_target not in self.mouse_ids)
            )
            if invalid:
                self._quarantined_annotation_count += 1
                continue
            collected[triplet].append((start, stop))

        self._intervals_by_triplet = {
            key: tuple(sorted(value)) for key, value in collected.items()
        }

    def positive_intervals(
        self, triplet: TripletLike | None = None
    ) -> Mapping[CanonicalTriplet, tuple[tuple[int, int], ...]] | tuple[tuple[int, int], ...]:
        """Return original half-open events without merging overlaps."""
        self._load_intervals()
        assert self._intervals_by_triplet is not None
        if triplet is None:
            return dict(self._intervals_by_triplet)
        return self._intervals_by_triplet.get(_canonical_triplet(triplet), ())

    def _requested_frame_ids(
        self,
        *,
        frame_ids: Sequence[int] | NDArray[Any] | None,
        start_frame: int | None,
        stop_frame: int | None,
    ) -> NDArray[np.int64]:
        if frame_ids is not None:
            if start_frame is not None or stop_frame is not None:
                raise ValueError("pass frame_ids or a start/stop window, not both")
            return _strict_frame_ids(frame_ids)
        all_frames = self.tracking_frame_ids
        if start_frame is None and stop_frame is None:
            return all_frames
        if start_frame is None or stop_frame is None or start_frame >= stop_frame:
            raise ValueError("a window must satisfy start_frame < stop_frame")
        return all_frames[(all_frames >= start_frame) & (all_frames < stop_frame)]

    def targets(
        self,
        *,
        triplets: Iterable[TripletLike] | None = None,
        frame_ids: Sequence[int] | NDArray[Any] | None = None,
        start_frame: int | None = None,
        stop_frame: int | None = None,
    ) -> FrameTargets:
        """Expand supervision only for the requested frame/triplet rectangle."""
        requested_frames = self._requested_frame_ids(
            frame_ids=frame_ids,
            start_frame=start_frame,
            stop_frame=stop_frame,
        )
        requested_triplets = tuple(
            _canonical_triplet(item)
            for item in (self.allowed_triplets if triplets is None else triplets)
        )
        shape = (requested_frames.size, len(requested_triplets))
        targets = np.zeros(shape, dtype=np.uint8)
        loss_mask = np.zeros(shape, dtype=np.uint8)
        if not self.is_supervised:
            states = np.full(shape, FrameLabelState.EXCLUDED_VIDEO, dtype=np.uint8)
            return FrameTargets(requested_frames, requested_triplets, targets, loss_mask, states)

        states = np.full(shape, FrameLabelState.UNKNOWN_MASKED, dtype=np.uint8)
        self._load_intervals()
        assert self._intervals_by_triplet is not None
        for column, triplet in enumerate(requested_triplets):
            if triplet not in self._allowed_set:
                continue
            loss_mask[:, column] = 1
            states[:, column] = FrameLabelState.VALID_NEGATIVE
            for start, stop in self._intervals_by_triplet[triplet]:
                active = (requested_frames >= start) & (requested_frames < stop)
                targets[active, column] = 1
                states[active, column] = FrameLabelState.POSITIVE
        return FrameTargets(requested_frames, requested_triplets, targets, loss_mask, states)

    def positive_frames(
        self,
        triplet: TripletLike,
        *,
        frame_ids: Sequence[int] | NDArray[Any] | None = None,
        start_frame: int | None = None,
        stop_frame: int | None = None,
    ) -> NDArray[np.int64]:
        expanded = self.targets(
            triplets=(triplet,),
            frame_ids=frame_ids,
            start_frame=start_frame,
            stop_frame=stop_frame,
        )
        result: NDArray[np.int64] = expanded.frame_ids[expanded.targets[:, 0].astype(bool)]
        return result

    def loss_masks(
        self,
        *,
        triplets: Iterable[TripletLike] | None = None,
        frame_ids: Sequence[int] | NDArray[Any] | None = None,
        start_frame: int | None = None,
        stop_frame: int | None = None,
    ) -> NDArray[np.uint8]:
        return self.targets(
            triplets=triplets,
            frame_ids=frame_ids,
            start_frame=start_frame,
            stop_frame=stop_frame,
        ).loss_mask


def build_action_vocabulary(triplets: pl.DataFrame | str | Path) -> ActionVocabulary:
    """Build deterministic global, solo, and pair action indices."""
    frame = pl.read_parquet(triplets) if isinstance(triplets, (str, Path)) else triplets
    required = {"action", "is_self"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"triplet data is missing columns: {sorted(missing)}")
    if "split" in frame.columns:
        frame = frame.filter(pl.col("split") == "train")
    if "is_duplicate" in frame.columns:
        frame = frame.filter(~pl.col("is_duplicate"))
    actions = tuple(sorted(str(value) for value in frame["action"].unique()))
    solo_actions = tuple(
        sorted(str(value) for value in frame.filter(pl.col("is_self"))["action"].unique())
    )
    pair_actions = tuple(
        sorted(str(value) for value in frame.filter(~pl.col("is_self"))["action"].unique())
    )
    payload = json.dumps(
        {
            "schema_version": LABEL_ARTIFACT_SCHEMA_VERSION,
            "actions": actions,
            "solo_actions": solo_actions,
            "pair_actions": pair_actions,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    version = f"phase3-v{LABEL_ARTIFACT_SCHEMA_VERSION}-{hashlib.sha256(payload).hexdigest()[:12]}"
    return ActionVocabulary(version, actions, solo_actions, pair_actions)


def _package_version() -> str:
    try:
        return metadata.version("mabe-behavior")
    except metadata.PackageNotFoundError:
        return "0.1.0"


def _write_sidecar(
    path: Path,
    *,
    row_count: int,
    schema_hash: str,
    source_fingerprint: str,
    configuration: dict[str, object],
) -> None:
    sidecar = {
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
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_label_artifacts(
    *,
    triplets_path: str | Path,
    output_root: str | Path,
    source_fingerprint: str,
) -> LabelArtifactOutputs:
    """Write versioned vocabulary and per-video candidate-index mappings."""
    source = Path(triplets_path)
    triplets = pl.read_parquet(source)
    vocabulary = build_action_vocabulary(triplets)
    action_index = vocabulary.action_to_index
    solo_index = vocabulary.solo_action_to_index
    pair_index = vocabulary.pair_action_to_index
    mappings = (
        triplets.filter(~pl.col("is_duplicate"))
        .select(
            "split",
            "lab_id",
            "video_id",
            "agent_id",
            "target_id",
            "action",
            "canonical_entry",
            "is_self",
        )
        .unique()
        .with_columns(
            pl.col("action")
            .replace_strict(action_index, return_dtype=pl.Int32)
            .alias("action_index"),
            pl.when(pl.col("is_self"))
            .then(
                pl.col("action").replace_strict(
                    solo_index, default=None, return_dtype=pl.Int32
                )
            )
            .otherwise(pl.lit(None, dtype=pl.Int32))
            .alias("solo_action_index"),
            pl.when(~pl.col("is_self"))
            .then(
                pl.col("action").replace_strict(
                    pair_index, default=None, return_dtype=pl.Int32
                )
            )
            .otherwise(pl.lit(None, dtype=pl.Int32))
            .alias("pair_action_index"),
            pl.lit(vocabulary.version).alias("vocabulary_version"),
        )
        .sort("split", "lab_id", "video_id", "agent_id", "target_id", "action")
    )

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    vocabulary_path = root / "action_vocabulary.json"
    mappings_path = root / "triplet_mappings.parquet"
    vocabulary_path.write_text(
        json.dumps(vocabulary.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    mappings.write_parquet(mappings_path, compression="zstd", statistics=True)

    configuration = {
        "schema_version": LABEL_ARTIFACT_SCHEMA_VERSION,
        "source_triplets": str(source),
        "sort_order": ["split", "lab_id", "video_id", "agent_id", "target_id", "action"],
        "vocabulary_version": vocabulary.version,
    }
    vocabulary_schema_hash = hashlib.sha256(
        json.dumps(sorted(vocabulary.to_dict()), separators=(",", ":")).encode()
    ).hexdigest()
    _write_sidecar(
        vocabulary_path,
        row_count=len(vocabulary.actions),
        schema_hash=vocabulary_schema_hash,
        source_fingerprint=source_fingerprint,
        configuration=configuration,
    )
    _write_sidecar(
        mappings_path,
        row_count=mappings.height,
        schema_hash=schema_fingerprint(mappings),
        source_fingerprint=source_fingerprint,
        configuration=configuration,
    )
    return LabelArtifactOutputs(vocabulary_path, mappings_path, vocabulary)


def _as_polars(frame: pl.DataFrame | pd.DataFrame) -> pl.DataFrame:
    return frame if isinstance(frame, pl.DataFrame) else pl.from_pandas(frame)


def _mouse_id(value: object) -> str:
    text = str(value)
    return text if text.startswith("mouse") else f"mouse{int(text)}"


def build_metric_solution(
    annotations: pl.DataFrame | pd.DataFrame,
    videos: pl.DataFrame | pd.DataFrame,
    *,
    validation_videos: Iterable[VideoKey] | None = None,
) -> pd.DataFrame:
    """Join interval labels to lab/raw-mask metadata for exact official scoring."""
    annotation_frame = _as_polars(annotations)
    video_frame = _as_polars(videos)
    mask_column = (
        "behaviors_labeled_raw"
        if "behaviors_labeled_raw" in video_frame.columns
        else "behaviors_labeled"
    )
    required_annotations = {
        "lab_id",
        "video_id",
        "agent_id",
        "target_id",
        "action",
        "start_frame",
        "stop_frame",
    }
    required_videos = {"lab_id", "video_id", mask_column}
    if missing := required_annotations - set(annotation_frame.columns):
        raise ValueError(f"annotations are missing columns: {sorted(missing)}")
    if missing := required_videos - set(video_frame.columns):
        raise ValueError(f"videos are missing columns: {sorted(missing)}")

    if validation_videos is not None:
        requested_keys = tuple(validation_videos)
        keys = pl.DataFrame(
            {
                "lab_id": [str(key[0]) for key in requested_keys],
                "video_id": [int(key[1]) for key in requested_keys],
            },
            schema={"lab_id": pl.String, "video_id": pl.Int64},
        )
        video_frame = video_frame.join(keys, on=["lab_id", "video_id"], how="semi")
    if "supervised_eligible" in video_frame.columns:
        video_frame = video_frame.filter(pl.col("supervised_eligible"))
    elif "label_state" in video_frame.columns:
        video_frame = video_frame.filter(pl.col("label_state") == SUPERVISED_VIDEO_STATE)

    optional_validation_columns = [
        column
        for column in ("observed_start_frame", "observed_stop_frame", "mouse_ids")
        if column in video_frame.columns
    ]
    metadata_rows = video_frame.select(
        "lab_id", "video_id", mask_column, *optional_validation_columns
    ).unique()
    if metadata_rows.select(pl.struct("lab_id", "video_id").n_unique()).item() != metadata_rows.height:
        raise ValueError("videos must contain one metadata row per lab/video")
    joined = annotation_frame.join(metadata_rows, on=["lab_id", "video_id"], how="inner")
    records: list[dict[str, object]] = []
    for row in joined.iter_rows(named=True):
        start = int(row["start_frame"])
        stop = int(row["stop_frame"])
        if start >= stop or bool(row.get("quarantined", False)):
            continue
        if row.get("observed_start_frame") is not None and start < int(
            row["observed_start_frame"]
        ):
            continue
        if row.get("observed_stop_frame") is not None and stop > int(
            row["observed_stop_frame"]
        ):
            continue
        agent = _mouse_id(row["agent_id"])
        source_target = _mouse_id(row["target_id"])
        observed_mice = {str(mouse) for mouse in (row.get("mouse_ids") or ())}
        if observed_mice and (agent not in observed_mice or source_target not in observed_mice):
            continue
        target = "self" if source_target == agent else source_target
        action = str(row["action"])
        parsed = parse_behaviors_labeled(str(row[mask_column]))
        allowed = set(parsed.triplets)
        if CanonicalTriplet(agent, target, action) not in allowed:
            continue
        records.append(
            {
                "video_id": int(row["video_id"]),
                "agent_id": agent,
                "target_id": target,
                "action": action,
                "start_frame": start,
                "stop_frame": stop,
                "lab_id": str(row["lab_id"]),
                "behaviors_labeled": str(row[mask_column]),
            }
        )
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
    return pd.DataFrame.from_records(records, columns=columns)


def build_validation_solution(
    *,
    videos_path: str | Path,
    dataset_root: str | Path,
    validation_videos: Iterable[VideoKey],
) -> pd.DataFrame:
    """Load only a validation subset's annotations and build the metric solution."""
    keys = tuple((str(lab), int(video)) for lab, video in validation_videos)
    videos = pl.read_parquet(videos_path)
    selected = videos.join(
        pl.DataFrame(
            {
                "lab_id": [key[0] for key in keys],
                "video_id": [key[1] for key in keys],
            },
            schema={"lab_id": pl.String, "video_id": pl.Int64},
        ),
        on=["lab_id", "video_id"],
        how="semi",
    )
    annotation_frames: list[pl.DataFrame] = []
    root = Path(dataset_root)
    for row in selected.iter_rows(named=True):
        if row.get("label_state") != SUPERVISED_VIDEO_STATE:
            continue
        path = _source_path(root, row.get("annotation_path"))
        if path is None:
            continue
        annotation_frames.append(
            scan_annotation(path)
            .with_columns(
                pl.lit(str(row["lab_id"])).alias("lab_id"),
                pl.lit(int(row["video_id"]), dtype=pl.Int64).alias("video_id"),
            )
            .collect()
        )
    annotations = (
        pl.concat(annotation_frames, how="vertical")
        if annotation_frames
        else pl.DataFrame(
            schema={
                "agent_id": pl.String,
                "target_id": pl.String,
                "action": pl.String,
                "start_frame": pl.Int64,
                "stop_frame": pl.Int64,
                "lab_id": pl.String,
                "video_id": pl.Int64,
            }
        )
    )
    return build_metric_solution(annotations, selected, validation_videos=keys)
