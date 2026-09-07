"""Phase 9 cache, training, five-fold OOF inference, and exact evaluation."""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch

from mabe.config import RunConfig
from mabe.decode import decode_probabilities
from mabe.fingerprints import schema_fingerprint, sha256_file
from mabe.labels import build_validation_solution
from mabe.phase8_config import load_phase8_config
from mabe.phase8_decoder import nested_interval_postprocessing
from mabe.phase9_config import load_phase9_config, phase9_config_fingerprint
from mabe.schemas import OOF_COLUMNS, OOF_SCHEMA, SUBMISSION_COLUMNS
from mabe.submission import validate_submission
from mabe.temporal_augmentation import geometric_transform, swap_left_right
from mabe.temporal_data import (
    Route,
    SequenceArtifact,
    WindowRef,
    build_sequence_artifact,
    materialize_window,
)
from mabe.temporal_model import TemporalBehaviorModel, overlap_weights
from mabe.temporal_training import (
    FoldTrainingResult,
    load_temporal_checkpoint,
    resolve_temporal_device,
    train_temporal_fold,
)


@dataclass(frozen=True, slots=True)
class Phase9Outputs:
    report_path: Path
    sequence_manifest_path: Path
    checkpoint_registry_path: Path
    oof_manifest_path: Path
    submission_path: Path
    decoder_parameters_path: Path
    overall_fbeta: float


@dataclass(frozen=True, slots=True)
class TemporalVideoPredictions:
    keys: tuple[tuple[str, str, str], ...]
    probabilities: np.ndarray[tuple[int, int], np.dtype[np.float16]]


def _sidecar(
    path: Path,
    *,
    configuration_sha256: str,
    source_fingerprint: str,
    row_count: int | None,
    schema_sha256: str | None = None,
) -> None:
    payload = {
        "artifact": path.name,
        "artifact_sha256": sha256_file(path),
        "configuration_sha256": configuration_sha256,
        "created_at": datetime.now(UTC).isoformat(),
        "row_count": row_count,
        "schema_sha256": schema_sha256,
        "source_data_fingerprint": source_fingerprint,
    }
    Path(f"{path}.meta.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_temporal_sequence_cache(
    *,
    run_config: RunConfig,
    temporal_config_path: Path,
    pose_config_path: Path,
    source_fingerprint: str,
) -> pl.DataFrame:
    config = load_phase9_config(temporal_config_path)
    configuration_sha = phase9_config_fingerprint(temporal_config_path)
    videos_path = run_config.paths.artifact_root / "manifests" / "videos.parquet"
    triplets_path = run_config.paths.artifact_root / "manifests" / "triplets.parquet"
    folds_path = run_config.paths.artifact_root / "folds" / "folds.parquet"
    vocabulary_path = run_config.paths.artifact_root / "labels" / "action_vocabulary.json"
    sequence_hasher = hashlib.sha256()
    sequence_hasher.update(
        json.dumps(config.data.model_dump(mode="json"), sort_keys=True).encode("utf-8")
    )
    sequence_hasher.update(sha256_file(pose_config_path).encode("ascii"))
    sequence_hasher.update(sha256_file(vocabulary_path).encode("ascii"))
    sequence_hasher.update(source_fingerprint.encode("ascii"))
    sequence_configuration_sha = sequence_hasher.hexdigest()
    videos = pl.read_parquet(videos_path)
    fold_frame = pl.read_parquet(folds_path)
    membership = fold_frame.filter(pl.col("is_primary_validation")).select(
        "lab_id", "video_id", "fold"
    )
    if config.data.include_auxiliary_training:
        auxiliary = (
            fold_frame.filter(
                (pl.col("source_role") == "auxiliary")
                & pl.col("supervised_eligible")
            )
            .select("lab_id", "video_id")
            .with_columns(pl.lit(-1, dtype=pl.Int8).alias("fold"))
        )
        membership = pl.concat((membership, auxiliary), how="vertical")
    selected = videos.join(membership, on=["lab_id", "video_id"], how="inner").sort(
        "lab_id", "video_id"
    )
    labs = sorted(str(value) for value in selected["lab_id"].unique())
    lab_lookup = {lab: index for index, lab in enumerate(labs)}
    cache_root = run_config.paths.cache_root / config.temporal_version / "sequences"
    records: list[dict[str, object]] = []
    futures = []
    executor = ProcessPoolExecutor(max_workers=config.execution.cache_workers)
    for row in selected.iter_rows(named=True):
        futures.append(
            executor.submit(
                build_sequence_artifact,
                row,
                fold=int(row["fold"]),
                lab_index=lab_lookup[str(row["lab_id"])],
                dataset_root=run_config.paths.dataset_root,
                videos_path=videos_path,
                triplets_path=triplets_path,
                pose_config_path=pose_config_path,
                vocabulary_path=vocabulary_path,
                output_root=cache_root,
                config=config,
                configuration_sha256=sequence_configuration_sha,
            )
        )
    try:
        for completed, future in enumerate(as_completed(futures), start=1):
            artifact = future.result()
            records.append(
                {
                    "lab_id": artifact.lab_id,
                    "video_id": artifact.video_id,
                    "fold": artifact.fold,
                    "artifact_path": artifact.artifact_path.as_posix(),
                    "frame_count": artifact.frame_count,
                    "mouse_count": artifact.mouse_count,
                    "target_fps": artifact.fps,
                }
            )
            if completed == 1 or completed % 25 == 0 or completed == len(futures):
                print(
                    f"Phase 9 sequence cache: {completed}/{len(futures)} videos complete",
                    flush=True,
                )
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    manifest = pl.DataFrame(records).sort("lab_id", "video_id")
    output_root = run_config.paths.artifact_root / "temporal" / config.temporal_version
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "sequence_manifest.parquet"
    manifest.write_parquet(manifest_path, compression="zstd", statistics=True)
    lab_registry_path = output_root / "lab_registry.json"
    lab_registry_path.write_text(
        json.dumps({"labs": labs, "lab_to_index": lab_lookup}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _sidecar(
        manifest_path,
        configuration_sha256=configuration_sha,
        source_fingerprint=source_fingerprint,
        row_count=manifest.height,
        schema_sha256=schema_fingerprint(manifest),
    )
    _sidecar(
        lab_registry_path,
        configuration_sha256=configuration_sha,
        source_fingerprint=source_fingerprint,
        row_count=len(labs),
    )
    return manifest


def _window_starts(frames: int, length: int, stride: int) -> list[int]:
    if frames <= length:
        return [0]
    starts = list(range(0, frames - length + 1, stride))
    final = frames - length
    if starts[-1] != final:
        starts.append(final)
    return starts


@torch.no_grad()
def _infer_entity(
    *,
    artifact: SequenceArtifact,
    route: Route,
    agent: int,
    target: int,
    model: TemporalBehaviorModel,
    config_path: Path,
    device: torch.device,
) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
    config = load_phase9_config(config_path)
    action_valid = (
        np.asarray(artifact.array("solo_action_valid"))[agent]
        if route == "solo"
        else np.asarray(artifact.array("pair_action_valid"))[agent, target]
    )
    actions = action_valid.size
    totals = np.zeros((artifact.frame_count, actions), dtype=np.float32)
    weights = np.zeros(artifact.frame_count, dtype=np.float32)
    starts = _window_starts(
        artifact.frame_count,
        config.data.window_frames,
        config.data.inference_stride_frames,
    )
    batch_size = config.execution.inference_batch_size
    for offset in range(0, len(starts), batch_size):
        batch_starts = starts[offset : offset + batch_size]
        values = []
        reflected_values = []
        for start in batch_starts:
            ref = WindowRef(
                lab_id=artifact.lab_id,
                video_id=artifact.video_id,
                fold=artifact.fold,
                route=route,
                agent_index=agent,
                target_index=target,
                center_index=start + config.data.window_frames // 2,
                sampling_kind="sliding_inference",
                action_index=-1,
            )
            raw = artifact.temporal_window(
                ref.center_index, config.data.window_frames
            )
            values.append(
                materialize_window(
                    artifact,
                    ref,
                    length=config.data.window_frames,
                    window=raw,
                )
            )
            if config.execution.reflection_tta:
                reflected = swap_left_right(
                    geometric_transform(raw, angle_radians=0.0, reflect_x=True)
                )
                reflected_values.append(
                    materialize_window(
                        artifact,
                        ref,
                        length=config.data.window_frames,
                        window=reflected,
                    )
                )
        inference_values = values + reflected_values
        features = torch.from_numpy(
            np.stack([value.features for value in inference_values])
        ).to(device)
        metadata = torch.from_numpy(
            np.stack([value.metadata for value in inference_values])
        ).to(device)
        labs = torch.tensor(
            [value.lab_index for value in inference_values],
            dtype=torch.long,
            device=device,
        )
        validity = torch.from_numpy(
            np.repeat(action_valid[None, :], len(inference_values), axis=0)
        ).to(device)
        logits = model(route, features, labs, metadata, validity)
        probability = torch.sigmoid(logits).float().cpu().numpy()
        if reflected_values:
            normal_count = len(values)
            probability = 0.5 * (
                probability[:normal_count] + probability[normal_count:]
            )
        for local, start in enumerate(batch_starts):
            stop = min(start + config.data.window_frames, artifact.frame_count)
            valid_length = stop - start
            window_weights = overlap_weights(
                config.data.window_frames,
                config.data.inference_edge_frames,
                left_boundary=start == 0,
                right_boundary=stop == artifact.frame_count,
            )[:valid_length].numpy()
            totals[start:stop] += probability[local, :valid_length] * window_weights[:, None]
            weights[start:stop] += window_weights
    return (totals / np.maximum(weights[:, None], np.float32(1e-8))).astype(
        np.float32, copy=False
    )


def infer_temporal_video(
    artifact: SequenceArtifact,
    *,
    checkpoint_path: Path,
    temporal_config_path: Path,
    vocabulary_path: Path | None = None,
) -> TemporalVideoPredictions:
    config = load_phase9_config(temporal_config_path)
    device = resolve_temporal_device(config.execution.device)
    model = load_temporal_checkpoint(checkpoint_path, config=config, device=device)
    mouse_ids = tuple(str(value) for value in artifact.metadata["mouse_ids"])
    selected_vocabulary = vocabulary_path
    if selected_vocabulary is None:
        selected_vocabulary = artifact.path.parents[4] / "labels" / "action_vocabulary.json"
        if not selected_vocabulary.is_file():
            selected_vocabulary = Path("artifacts/labels/action_vocabulary.json")
    vocabulary = json.loads(selected_vocabulary.read_text(encoding="utf-8"))
    solo_actions = tuple(str(value) for value in vocabulary["solo_actions"])
    pair_actions = tuple(str(value) for value in vocabulary["pair_actions"])
    keys: list[tuple[str, str, str]] = []
    columns: list[np.ndarray[tuple[int], np.dtype[np.float32]]] = []
    solo_valid = np.asarray(artifact.array("solo_action_valid"), dtype=bool)
    pair_valid = np.asarray(artifact.array("pair_action_valid"), dtype=bool)
    for agent, agent_id in enumerate(mouse_ids):
        if solo_valid[agent].any():
            values = _infer_entity(
                artifact=artifact,
                route="solo",
                agent=agent,
                target=agent,
                model=model,
                config_path=temporal_config_path,
                device=device,
            )
            for action in np.flatnonzero(solo_valid[agent]):
                keys.append((agent_id, "self", solo_actions[int(action)]))
                columns.append(values[:, action])
        for target, target_id in enumerate(mouse_ids):
            if target == agent or not pair_valid[agent, target].any():
                continue
            values = _infer_entity(
                artifact=artifact,
                route="pair",
                agent=agent,
                target=target,
                model=model,
                config_path=temporal_config_path,
                device=device,
            )
            for action in np.flatnonzero(pair_valid[agent, target]):
                keys.append((agent_id, target_id, pair_actions[int(action)]))
                columns.append(values[:, action])
    matrix = (
        np.stack(columns, axis=1).astype(np.float16)
        if columns
        else np.zeros((artifact.frame_count, 0), dtype=np.float16)
    )
    return TemporalVideoPredictions(tuple(keys), matrix)


def _replace_part_probability(
    source_path: Path,
    output_path: Path,
    artifact: SequenceArtifact,
    predictions: TemporalVideoPredictions,
    model_id: str,
) -> int:
    frame = pl.read_parquet(source_path).with_row_index("_row_index")
    output = np.zeros(frame.height, dtype=np.float32)
    key_lookup = {key: index for index, key in enumerate(predictions.keys)}
    original = np.asarray(artifact.array("original_frame_ids"), dtype=np.int64)
    original_to_resampled = np.asarray(
        artifact.array("original_to_resampled"), dtype=np.int32
    )
    for group in frame.partition_by(
        ["agent_id", "target_id", "action"], maintain_order=True
    ):
        key = (
            str(group["agent_id"][0]),
            str(group["target_id"][0]),
            str(group["action"][0]),
        )
        column = key_lookup.get(key)
        if column is None:
            continue
        requested = group["video_frame"].to_numpy().astype(np.int64, copy=False)
        positions = np.searchsorted(original, requested)
        positions = np.clip(positions, 0, original.size - 1)
        resampled = original_to_resampled[positions]
        indices = group["_row_index"].to_numpy().astype(np.int64, copy=False)
        output[indices] = predictions.probabilities[resampled, column].astype(np.float32)
    result = frame.drop("_row_index").with_columns(
        pl.Series("probability", output, dtype=pl.Float32),
        pl.lit(model_id).alias("model_id"),
    ).select(
        pl.col(name).cast(dtype, strict=True) for name, dtype in OOF_SCHEMA.items()
    ).select(OOF_COLUMNS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(output_path, compression="zstd", statistics=True)
    return result.height


def build_temporal_oof(
    *,
    manifest: pl.DataFrame,
    cache_root: Path,
    checkpoint_root: Path,
    temporal_config_path: Path,
    baseline_oof_manifest: Path,
    output_root: Path,
) -> tuple[list[Path], int, float]:
    started = time.perf_counter()
    source = json.loads(baseline_oof_manifest.read_text(encoding="utf-8"))
    paths_by_video: defaultdict[int, list[Path]] = defaultdict(list)
    for value in source["parts"]:
        path = Path(str(value))
        paths_by_video[int(path.parent.name)].append(path)
    fold_lookup = {
        int(row["video_id"]): int(row["fold"])
        for row in manifest.iter_rows(named=True)
    }
    lab_lookup = {
        int(row["video_id"]): str(row["lab_id"])
        for row in manifest.iter_rows(named=True)
    }
    output_paths: list[Path] = []
    row_count = 0
    model_id = load_phase9_config(temporal_config_path).temporal_version
    for video_id in sorted(paths_by_video):
        lab = lab_lookup[video_id]
        fold = fold_lookup[video_id]
        expected_outputs = [
            output_root / f"fold_{fold}" / lab / str(video_id) / path.name
            for path in paths_by_video[video_id]
        ]
        if all(path.is_file() for path in expected_outputs):
            for path in expected_outputs:
                output_paths.append(path)
                row_count += pl.scan_parquet(path).select(pl.len()).collect().item()
            continue
        artifact = SequenceArtifact(cache_root / lab / str(video_id))
        predictions = infer_temporal_video(
            artifact,
            checkpoint_path=checkpoint_root / f"fold_{fold}" / "model.pt",
            temporal_config_path=temporal_config_path,
        )
        for source_path in paths_by_video[video_id]:
            output_path = output_root / f"fold_{fold}" / lab / str(video_id) / source_path.name
            row_count += _replace_part_probability(
                source_path, output_path, artifact, predictions, model_id
            )
            output_paths.append(output_path)
    return output_paths, row_count, time.perf_counter() - started


def _threshold_counts(paths: list[Path], thresholds: tuple[float, ...]) -> pl.DataFrame:
    scan = pl.scan_parquet([str(path) for path in paths])
    expressions: list[pl.Expr] = [pl.col("target").sum().alias("positive_frames")]
    for index, threshold in enumerate(thresholds):
        predicted = pl.col("probability") >= threshold
        expressions.extend(
            [
                predicted.sum().alias(f"predicted_{index}"),
                (predicted & (pl.col("target") == 1)).sum().alias(f"tp_{index}"),
            ]
        )
    return scan.group_by("lab_id", "action", "fold").agg(expressions).collect(
        engine="streaming"
    )


def _best_threshold(frame: pl.DataFrame, thresholds: tuple[float, ...]) -> float:
    positive = int(frame["positive_frames"].sum()) if frame.height else 0
    scores: list[tuple[float, float, float]] = []
    for index, threshold in enumerate(thresholds):
        predicted = int(frame[f"predicted_{index}"].sum()) if frame.height else 0
        tp = int(frame[f"tp_{index}"].sum()) if frame.height else 0
        score = 0.0 if positive + predicted == 0 else 2 * tp / (positive + predicted)
        scores.append((score, -abs(threshold - 0.5), threshold))
    return float(max(scores)[2])


def _nested_thresholds(
    counts: pl.DataFrame,
    requested: pl.DataFrame,
    *,
    folds: tuple[int, ...],
    threshold_grid: tuple[float, ...],
    fallback: float,
) -> pl.DataFrame:
    records: list[dict[str, object]] = []
    for held_fold in folds:
        calibration = counts.filter(pl.col("fold") != held_fold)
        global_threshold = (
            _best_threshold(calibration, threshold_grid) if calibration.height else fallback
        )
        for row in requested.iter_rows(named=True):
            lab = str(row["lab_id"])
            action = str(row["action"])
            lab_action = calibration.filter(
                (pl.col("lab_id") == lab) & (pl.col("action") == action)
            )
            action_frame = calibration.filter(pl.col("action") == action)
            if lab_action.height and int(lab_action["positive_frames"].sum()) > 0:
                threshold = _best_threshold(lab_action, threshold_grid)
                source = "lab_action"
            elif action_frame.height and int(action_frame["positive_frames"].sum()) > 0:
                threshold = _best_threshold(action_frame, threshold_grid)
                source = "action"
            else:
                threshold = global_threshold
                source = "global"
            records.append(
                {
                    "held_out_fold": held_fold,
                    "lab_id": lab,
                    "action": action,
                    "threshold": threshold,
                    "threshold_source": source,
                    "calibration_folds": json.dumps(
                        [value for value in folds if value != held_fold]
                    ),
                    "used_held_out_labels": False,
                }
            )
    return pl.DataFrame(records).sort("held_out_fold", "lab_id", "action")


def _decode_oof(
    paths: list[Path], manifest: pl.DataFrame, thresholds: pl.DataFrame
) -> pd.DataFrame:
    paths_by_video: defaultdict[int, list[Path]] = defaultdict(list)
    for path in paths:
        paths_by_video[int(path.parent.name)].append(path)
    membership = {
        int(row["video_id"]): (str(row["lab_id"]), int(row["fold"]))
        for row in manifest.iter_rows(named=True)
    }
    decoded: list[pd.DataFrame] = []
    for video_id, video_paths in sorted(paths_by_video.items()):
        lab, fold = membership[video_id]
        selected = thresholds.filter(
            (pl.col("held_out_fold") == fold) & (pl.col("lab_id") == lab)
        )
        part = decode_probabilities(pl.read_parquet(video_paths), selected)
        if not part.empty:
            decoded.append(part)
    result = pd.concat(decoded, ignore_index=True)
    result["row_id"] = np.arange(len(result), dtype=np.int64)
    return result[list(SUBMISSION_COLUMNS)]


def _probability_diversity(
    neural_paths: list[Path], baseline_manifest: Path, *, maximum_per_part: int = 500
) -> dict[str, float | int]:
    baseline = json.loads(baseline_manifest.read_text(encoding="utf-8"))
    baseline_lookup = {
        "/".join(Path(str(path)).parts[-4:]): Path(str(path))
        for path in baseline["parts"]
    }
    neural_values: list[np.ndarray[tuple[int], np.dtype[np.float32]]] = []
    baseline_values: list[np.ndarray[tuple[int], np.dtype[np.float32]]] = []
    targets: list[np.ndarray[tuple[int], np.dtype[np.uint8]]] = []
    for neural_path in neural_paths:
        suffix = "/".join(neural_path.parts[-4:])
        baseline_path = baseline_lookup.get(suffix)
        if baseline_path is None:
            continue
        neural = pl.read_parquet(neural_path, columns=["probability", "target"])
        base = pl.read_parquet(baseline_path, columns=["probability"])
        count = min(neural.height, maximum_per_part)
        if count == 0:
            continue
        indices = np.linspace(0, neural.height - 1, count).round().astype(np.int64)
        neural_values.append(neural["probability"].to_numpy()[indices].astype(np.float32))
        baseline_values.append(base["probability"].to_numpy()[indices].astype(np.float32))
        targets.append(neural["target"].to_numpy()[indices].astype(np.uint8))
    if not neural_values:
        raise ValueError("neural and GBDT OOF manifests do not contain matching parts")
    neural_all = np.concatenate(neural_values)
    baseline_all = np.concatenate(baseline_values)
    target_all = np.concatenate(targets)
    correlation = float(np.corrcoef(neural_all, baseline_all)[0, 1])
    neural_error = (neural_all >= 0.5) != target_all
    baseline_error = (baseline_all >= 0.5) != target_all
    union = np.logical_or(neural_error, baseline_error).sum()
    overlap = np.logical_and(neural_error, baseline_error).sum()
    return {
        "sample_rows": int(target_all.size),
        "probability_correlation": correlation,
        "error_jaccard": 0.0 if union == 0 else float(overlap / union),
        "neural_only_error_rate": float(np.logical_and(neural_error, ~baseline_error).mean()),
        "gbdt_only_error_rate": float(np.logical_and(~neural_error, baseline_error).mean()),
    }


def _per_lab_action_comparison(
    neural: pd.DataFrame, gbdt_path: Path
) -> pd.DataFrame:
    metrics = ("fbeta", "precision", "recall", "tp", "fp", "fn")
    gbdt = pd.read_csv(gbdt_path)
    selected_neural = neural[["lab_id", "action", *metrics]].rename(
        columns={metric: f"neural_{metric}" for metric in metrics}
    )
    selected_gbdt = gbdt[["lab_id", "action", *metrics]].rename(
        columns={metric: f"gbdt_{metric}" for metric in metrics}
    )
    result = selected_neural.merge(
        selected_gbdt, on=["lab_id", "action"], how="outer", validate="one_to_one"
    ).fillna(0)
    result["fbeta_delta"] = result["neural_fbeta"] - result["gbdt_fbeta"]
    result["stronger_model"] = np.select(
        [result["fbeta_delta"] > 0, result["fbeta_delta"] < 0],
        ["neural", "gbdt"],
        default="tie",
    )
    return result.sort_values(["lab_id", "action"], kind="stable").reset_index(
        drop=True
    )


def build_phase9_artifacts(
    *,
    run_config: RunConfig,
    temporal_config_path: str | Path,
    pose_config_path: str | Path,
    phase8_config_path: str | Path,
    source_fingerprint: str,
) -> Phase9Outputs:
    started = time.perf_counter()
    temporal_path = Path(temporal_config_path)
    config = load_phase9_config(temporal_path)
    configuration_sha = phase9_config_fingerprint(temporal_path)
    output_root = run_config.paths.artifact_root / "temporal" / config.temporal_version
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = build_temporal_sequence_cache(
        run_config=run_config,
        temporal_config_path=temporal_path,
        pose_config_path=Path(pose_config_path),
        source_fingerprint=source_fingerprint,
    )
    manifest_path = output_root / "sequence_manifest.parquet"
    cache_root = run_config.paths.cache_root / config.temporal_version / "sequences"
    hard_path = (
        run_config.paths.artifact_root
        / "phase8"
        / load_phase8_config(phase8_config_path).phase8_version
        / "hard_negative_plan.parquet"
    )
    hard_negatives = pl.read_parquet(hard_path)
    results: list[FoldTrainingResult] = []
    ordered_folds = (config.training.gate_fold,) + tuple(
        fold for fold in config.training.folds if fold != config.training.gate_fold
    )
    for fold in ordered_folds:
        checkpoint = output_root / "checkpoints" / f"fold_{fold}" / "model.pt"
        card = checkpoint.parent / "model_card.json"
        payload = json.loads(card.read_text(encoding="utf-8")) if card.is_file() else {}
        if (
            checkpoint.is_file()
            and card.is_file()
            and payload.get("configuration_sha256") == configuration_sha
        ):
            result = FoldTrainingResult(
                held_out_fold=int(payload["held_out_fold"]),
                checkpoint_path=checkpoint,
                best_epoch=int(payload["best_epoch"]),
                best_validation_loss=float(payload["best_validation_loss"]),
                validation_average_precision=float(payload["validation_average_precision"]),
                training_seconds=float(payload["training_seconds"]),
                device=str(payload["device"]),
                parameter_count=int(payload["parameter_count"]),
            )
        else:
            result = train_temporal_fold(
                held_out_fold=fold,
                manifest=manifest,
                cache_root=cache_root,
                hard_negatives=hard_negatives,
                output_root=output_root / "checkpoints",
                config=config,
                configuration_sha256=configuration_sha,
            )
        results.append(result)
        if (
            fold == config.training.gate_fold
            and result.validation_average_precision
            < config.training.minimum_gate_average_precision
        ):
            raise RuntimeError("one-fold temporal gate did not justify five-fold training")
    checkpoint_registry = pl.DataFrame(
        [asdict(result) | {"checkpoint_path": result.checkpoint_path.as_posix()} for result in results]
    ).sort("held_out_fold")
    checkpoint_registry_path = output_root / "checkpoint_registry.parquet"
    checkpoint_registry.write_parquet(
        checkpoint_registry_path, compression="zstd", statistics=True
    )
    phase8 = load_phase8_config(phase8_config_path)
    phase8_root = (
        run_config.paths.artifact_root / "phase8" / phase8.phase8_version
    )
    # Phase 8 may promote a fully cross-fitted model selected by its screen.
    # Its final manifest is the single source of truth for every downstream
    # comparison; falling back to the Phase 7 stream is retained only for
    # backwards compatibility with pre-promotion Phase 8 artifacts.
    phase8_manifest = phase8_root / "final_oof_manifest.json"
    stable_baseline_manifest = (
        run_config.paths.artifact_root
        / "xgb"
        / phase8.baseline_model_version
        / "oof_manifest.json"
    )
    baseline_manifests = (
        [phase8_manifest]
        if phase8_manifest.is_file()
        else [stable_baseline_manifest]
        if stable_baseline_manifest.is_file()
        else sorted(
            (
                run_config.paths.artifact_root
                / "oof"
                / phase8.baseline_model_version
            ).glob("*/manifest.json")
        )
    )
    if len(baseline_manifests) != 1:
        raise ValueError("Phase 9 requires exactly one selected Phase 8 OOF manifest")
    oof_root = run_config.paths.artifact_root / "oof" / config.temporal_version / configuration_sha[:12]
    neural_paths, row_count, inference_seconds = build_temporal_oof(
        manifest=manifest,
        cache_root=cache_root,
        checkpoint_root=output_root / "checkpoints",
        temporal_config_path=temporal_path,
        baseline_oof_manifest=baseline_manifests[0],
        output_root=oof_root,
    )
    oof_manifest = {
        "schema_version": 1,
        "model_version": config.temporal_version,
        "configuration_sha256": configuration_sha,
        "row_count": row_count,
        "parts": [path.as_posix() for path in neural_paths],
        "columns": list(OOF_COLUMNS),
        "held_out_labels_used_for_training_or_early_stopping": False,
    }
    oof_manifest_path = oof_root / "manifest.json"
    oof_manifest_path.write_text(
        json.dumps(oof_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    stable_oof_manifest_path = output_root / "oof_manifest.json"
    stable_oof_manifest_path.write_text(
        json.dumps(oof_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    counts = _threshold_counts(neural_paths, config.decoder.threshold_grid)
    requested = pl.read_parquet(phase8_root / "final_model_registry.parquet").select(
        "lab_id", "action"
    ).unique()
    thresholds = _nested_thresholds(
        counts,
        requested,
        folds=config.training.folds,
        threshold_grid=config.decoder.threshold_grid,
        fallback=config.decoder.fallback_threshold,
    )
    decoded = _decode_oof(neural_paths, manifest, thresholds)
    videos_path = run_config.paths.artifact_root / "manifests" / "videos.parquet"
    triplets_path = run_config.paths.artifact_root / "manifests" / "triplets.parquet"
    folds_path = run_config.paths.artifact_root / "folds" / "folds.parquet"
    videos = pl.read_parquet(videos_path)
    triplets = pl.read_parquet(triplets_path)
    folds = pl.read_parquet(folds_path)
    keys = [
        (str(row["lab_id"]), int(row["video_id"]))
        for row in folds.filter(pl.col("is_primary_validation")).iter_rows(named=True)
    ]
    solution = build_validation_solution(
        videos_path=videos_path,
        dataset_root=run_config.paths.dataset_root,
        validation_videos=keys,
    )
    postprocessed = nested_interval_postprocessing(
        solution=solution,
        baseline_submission=decoded,
        videos=videos,
        folds=folds,
        baseline_parameters=thresholds,
        config=load_phase8_config(phase8_config_path).decoder,
        baseline_smoothing_seconds=config.decoder.smoothing_seconds,
    )
    validated = validate_submission(
        postprocessed.submission,
        videos=videos,
        triplets=triplets,
        allowed_splits=("train",),
    )
    submission_path = output_root / "held_out_submission.csv"
    validated.write_csv(submission_path)
    decoder_parameters_path = output_root / "decoder_parameters.parquet"
    postprocessed.parameters.write_parquet(
        decoder_parameters_path, compression="zstd", statistics=True
    )
    by_lab_action_path = output_root / "cv_by_lab_action.csv"
    postprocessed.breakdown.by_lab_action.to_csv(by_lab_action_path, index=False)
    comparison_path = output_root / "gbdt_comparison_by_lab_action.csv"
    comparison = _per_lab_action_comparison(
        postprocessed.breakdown.by_lab_action,
        phase8_root / "cv_by_lab_action.csv",
    )
    comparison.to_csv(comparison_path, index=False)
    diversity = _probability_diversity(neural_paths, baseline_manifests[0])
    phase8_report = json.loads((phase8_root / "phase8_report.json").read_text(encoding="utf-8"))
    baseline_score = float(phase8_report["final_overall_fbeta"])
    projected_200_seconds = inference_seconds / manifest.height * 200
    report = {
        "schema_version": 1,
        "temporal_version": config.temporal_version,
        "configuration_sha256": configuration_sha,
        "architecture_config": config.architecture.model_dump(mode="json"),
        "architecture_source": "src/mabe/temporal_model.py",
        "architecture_configuration": "configs/temporal.yaml#architecture",
        "device": results[0].device,
        "folds_trained": list(config.training.folds),
        "one_fold_gate": asdict(results[0])
        | {"checkpoint_path": results[0].checkpoint_path.as_posix()},
        "sequence_videos": manifest.height,
        "sequence_frames": int(manifest["frame_count"].sum()),
        "oof_rows": row_count,
        "neural_overall_fbeta": postprocessed.breakdown.overall,
        "phase8_gbdt_fbeta": baseline_score,
        "standalone_delta": postprocessed.breakdown.overall - baseline_score,
        "diversity": diversity,
        "per_lab_action_comparison": {
            "rows": len(comparison),
            "neural_wins": int((comparison["stronger_model"] == "neural").sum()),
            "gbdt_wins": int((comparison["stronger_model"] == "gbdt").sum()),
            "ties": int((comparison["stronger_model"] == "tie").sum()),
        },
        "inference_seconds": inference_seconds,
        "projected_200_video_seconds": projected_200_seconds,
        "projected_under_six_hours_with_20pct_headroom": projected_200_seconds <= 17_280,
        "decoder_used_held_out_labels": bool(
            postprocessed.parameters["used_held_out_labels"].any()
        ),
        "runtime_seconds": time.perf_counter() - started,
    }
    report_path = output_root / "phase9_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifacts: list[tuple[Path, int | None, str | None]] = [
        (manifest_path, manifest.height, schema_fingerprint(manifest)),
        (
            checkpoint_registry_path,
            checkpoint_registry.height,
            schema_fingerprint(checkpoint_registry),
        ),
        (oof_manifest_path, row_count, None),
        (stable_oof_manifest_path, row_count, None),
        (submission_path, validated.height, None),
        (
            decoder_parameters_path,
            postprocessed.parameters.height,
            schema_fingerprint(postprocessed.parameters),
        ),
        (by_lab_action_path, len(postprocessed.breakdown.by_lab_action), None),
        (comparison_path, len(comparison), None),
        (report_path, None, None),
    ]
    for path, rows, schema in artifacts:
        _sidecar(
            path,
            configuration_sha256=configuration_sha,
            source_fingerprint=source_fingerprint,
            row_count=rows,
            schema_sha256=schema,
        )
    return Phase9Outputs(
        report_path=report_path,
        sequence_manifest_path=manifest_path,
        checkpoint_registry_path=checkpoint_registry_path,
        oof_manifest_path=oof_manifest_path,
        submission_path=submission_path,
        decoder_parameters_path=decoder_parameters_path,
        overall_fbeta=postprocessed.breakdown.overall,
    )
