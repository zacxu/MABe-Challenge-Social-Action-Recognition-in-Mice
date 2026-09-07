"""Economical fixed-fold GBDT studies required before Phase 8 promotion."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, log_loss

from mabe.features.common import load_feature_config
from mabe.features.storage import load_category_registry
from mabe.phase8_config import Phase8Config
from mabe.pose import load_pose_config
from mabe.training_data import filter_feasible_triplets
from mabe.xgb_config import XGBoostBaselineConfig
from mabe.xgb_model import TrainedFoldModel, fit_fold_model, load_fold_model
from mabe.xgb_pipeline import _build_training_caches, _manifest_lookup

KEY_COLUMNS = ["lab_id", "video_id", "video_frame", "agent_id", "target_id", "action"]
NON_FEATURE_COLUMNS = {
    *KEY_COLUMNS,
    "route",
    "fold",
    "target",
    "sampling_kind",
    "priority",
}


def _load_or_fit_fold_model(
    frame: pl.DataFrame,
    feature_names: tuple[str, ...],
    *,
    held_out_fold: int,
    scope: str,
    action: str,
    output_root: Path,
    config: XGBoostBaselineConfig,
) -> TrainedFoldModel:
    """Resume a Phase 8 screen when its exact fitted model already exists."""
    model_path = output_root / f"fold_{held_out_fold}" / scope / f"{action}.json"
    metadata_path = Path(f"{model_path}.meta.json")
    configuration_sha256 = hashlib.sha256(
        json.dumps(
            config.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    training_key_hash = int(
        frame.select(*KEY_COLUMNS, "fold", "target")
        .hash_rows(seed=2025)
        .sum()
    )
    if model_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = {
            "model_version": config.model_version,
            "scope": scope,
            "action": action,
            "held_out_fold": held_out_fold,
            "feature_names": list(feature_names),
            "held_out_labels_used_for_early_stopping": False,
            "screen_model_configuration_sha256": configuration_sha256,
            "screen_training_key_hash": str(training_key_hash),
            "screen_training_rows": frame.height,
            "screen_training_positive_rows": int(frame["target"].sum()),
        }
        if all(metadata.get(key) == value for key, value in expected.items()):
            return TrainedFoldModel(
                model_path=model_path,
                metadata_path=metadata_path,
                best_iteration=int(metadata["best_iteration"]),
                training_rows=int(metadata["training_rows"]),
                positive_rows=int(metadata["positive_rows"]),
                training_videos=int(metadata["training_videos"]),
                inner_validation_fold=int(metadata["inner_validation_fold"]),
                device=str(metadata["device"]),
            )
    trained = fit_fold_model(
        frame,
        feature_names,
        held_out_fold=held_out_fold,
        scope=scope,
        action=action,
        output_root=output_root,
        config=config,
    )
    metadata = json.loads(trained.metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "screen_model_configuration_sha256": configuration_sha256,
            "screen_training_key_hash": str(training_key_hash),
            "screen_training_rows": frame.height,
            "screen_training_positive_rows": int(frame["target"].sum()),
        }
    )
    trained.metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return trained


@dataclass(frozen=True, slots=True)
class StudyOutputs:
    representatives: pl.DataFrame
    feature_ablation: pl.DataFrame
    hyperparameter_search: pl.DataFrame
    hard_negative_plan: pl.DataFrame
    hard_negative_comparison: pl.DataFrame
    gain_importance: pl.DataFrame
    group_stability: pl.DataFrame
    permutation_importance: pl.DataFrame
    route_comparison: pl.DataFrame
    route_selection: pl.DataFrame
    alternative_models: pl.DataFrame
    selected_feature_candidate: str
    selected_hyperparameter_candidate: str


def load_feature_groups(registry_root: str | Path) -> dict[str, dict[str, tuple[str, ...]]]:
    output: dict[str, dict[str, tuple[str, ...]]] = {}
    for route in ("solo", "pair"):
        payload = json.loads((Path(registry_root) / f"{route}.json").read_text(encoding="utf-8"))
        grouped: defaultdict[str, list[str]] = defaultdict(list)
        for value in payload["feature_columns"]:
            grouped[str(value["group"])].append(str(value["name"]))
        output[route] = {group: tuple(names) for group, names in grouped.items()}
    return output


def feature_names_for_groups(
    grouped: dict[str, dict[str, tuple[str, ...]]],
    route: str,
    groups: tuple[str, ...],
) -> tuple[str, ...]:
    missing = set(groups) - set(grouped[route])
    if missing:
        raise ValueError(f"unknown {route} feature groups: {sorted(missing)}")
    return tuple(name for group in groups for name in grouped[route][group])


def select_representative_routes(
    lab_frames: dict[str, pl.DataFrame],
    baseline_registry: pl.DataFrame,
    *,
    folds: tuple[int, ...],
    per_lab: int,
) -> pl.DataFrame:
    """Choose stable direct routes with positives and negatives in every screen fold."""
    records: list[dict[str, object]] = []
    direct = baseline_registry.filter(pl.col("routing") == "lab_action").select(
        "lab_id", "action", "route_kind"
    ).unique()
    for lab in sorted(lab_frames):
        candidates = direct.filter(pl.col("lab_id") == lab)
        scored: list[tuple[int, int, str, str]] = []
        for row in candidates.iter_rows(named=True):
            action = str(row["action"])
            route = str(row["route_kind"])
            frame = lab_frames[lab].filter(
                (pl.col("action") == action) & (pl.col("route") == route)
            )
            fold_support = frame.group_by("fold").agg(
                pl.col("target").sum().alias("positive_rows"),
                (pl.len() - pl.col("target").sum()).alias("negative_rows"),
            )
            support = {
                int(value["fold"]): (
                    int(value["positive_rows"]),
                    int(value["negative_rows"]),
                )
                for value in fold_support.iter_rows(named=True)
            }
            if any(fold not in support or min(support[fold]) <= 0 for fold in folds):
                continue
            minimum_positive = min(support[fold][0] for fold in folds)
            scored.append((minimum_positive, int(frame["target"].sum()), action, route))
        for minimum_positive, positive_rows, action, route in sorted(scored, reverse=True)[:per_lab]:
            records.append(
                {
                    "lab_id": lab,
                    "action": action,
                    "route": route,
                    "minimum_fold_positive_rows": minimum_positive,
                    "positive_rows": positive_rows,
                }
            )
    if not records:
        raise ValueError("no stable representative lab/action routes were found")
    return pl.DataFrame(records).sort("lab_id", "action")


def _fold_balanced_class_cap(
    frame: pl.DataFrame, *, maximum_rows: int, seed: int
) -> pl.DataFrame:
    """Deterministically cap one class while retaining every represented fold."""
    if frame.height <= maximum_rows:
        return frame
    folds = sorted(int(value) for value in frame["fold"].unique())
    if maximum_rows < len(folds):
        raise ValueError(
            f"row cap {maximum_rows} cannot retain all {len(folds)} represented folds"
        )
    counts = {
        int(row["fold"]): int(row["len"])
        for row in frame.group_by("fold").len().iter_rows(named=True)
    }
    remaining = maximum_rows - len(folds)
    capacities = {fold: counts[fold] - 1 for fold in folds}
    total_capacity = sum(capacities.values())
    raw_extra = {
        fold: remaining * capacities[fold] / total_capacity for fold in folds
    }
    extra = {fold: math.floor(raw_extra[fold]) for fold in folds}
    leftover = remaining - sum(extra.values())
    for fold in sorted(
        folds,
        key=lambda value: (raw_extra[value] - extra[value], capacities[value], -value),
        reverse=True,
    )[:leftover]:
        extra[fold] += 1
    selected: list[pl.DataFrame] = []
    for fold in folds:
        fold_cap = 1 + extra[fold]
        selected.append(
            frame.filter(pl.col("fold") == fold)
            .with_columns(
                pl.struct(*KEY_COLUMNS, "fold", "target", "sampling_kind")
                .hash(seed=seed)
                .alias("screen_priority")
            )
            .sort("screen_priority")
            .head(fold_cap)
            .drop("screen_priority")
        )
    return pl.concat(selected, how="vertical_relaxed")


def _cap_training_frame(
    frame: pl.DataFrame,
    *,
    maximum_rows: int,
    negative_to_positive_ratio: float | None,
    seed: int,
) -> pl.DataFrame:
    positives = frame.filter(pl.col("target") == 1)
    negatives = frame.filter(pl.col("target") == 0)
    if positives.is_empty() or negatives.is_empty():
        return frame.head(maximum_rows)
    represented_folds = frame["fold"].n_unique()
    if maximum_rows < 2 * represented_folds:
        raise ValueError(
            f"training row cap {maximum_rows} is too small to preserve both "
            f"classes across {represented_folds} folds"
        )
    eligible_negative_rows = negatives.height
    if negative_to_positive_ratio is not None:
        eligible_negative_rows = min(
            eligible_negative_rows,
            max(
                represented_folds,
                math.ceil(positives.height * negative_to_positive_ratio),
            ),
        )
    eligible_rows = positives.height + eligible_negative_rows
    if eligible_rows <= maximum_rows:
        positive_cap = positives.height
        negative_cap = eligible_negative_rows
    else:
        minimum_per_class = represented_folds
        distributable = maximum_rows - 2 * minimum_per_class
        positive_share = positives.height / eligible_rows
        positive_cap = minimum_per_class + math.floor(
            distributable * positive_share
        )
        negative_cap = maximum_rows - positive_cap
        if positive_cap > positives.height:
            negative_cap += positive_cap - positives.height
            positive_cap = positives.height
        if negative_cap > eligible_negative_rows:
            positive_cap += negative_cap - eligible_negative_rows
            negative_cap = eligible_negative_rows
    selected_positives = _fold_balanced_class_cap(
        positives, maximum_rows=positive_cap, seed=seed
    )
    selected_negatives = _fold_balanced_class_cap(
        negatives, maximum_rows=negative_cap, seed=seed + 1
    )
    return pl.concat(
        [selected_positives, selected_negatives], how="vertical_relaxed"
    ).sort("fold", "video_id", "video_frame", "agent_id", "target_id")


def _best_frame_f1(
    target: np.ndarray[Any, np.dtype[np.uint8]],
    probability: np.ndarray[Any, np.dtype[np.float32]],
    thresholds: tuple[float, ...],
) -> tuple[float, float]:
    scored: list[tuple[float, float]] = []
    positive = target.astype(bool)
    for threshold in thresholds:
        predicted = probability >= threshold
        tp = int(np.count_nonzero(predicted & positive))
        fp = int(np.count_nonzero(predicted & ~positive))
        fn = int(np.count_nonzero(~predicted & positive))
        denominator = 2 * tp + fp + fn
        scored.append((0.0 if denominator == 0 else 2 * tp / denominator, threshold))
    return max(scored)


def _score_probability(
    target: np.ndarray[Any, np.dtype[np.uint8]],
    probability: np.ndarray[Any, np.dtype[np.float32]],
    thresholds: tuple[float, ...],
) -> dict[str, float]:
    f1, threshold = _best_frame_f1(target, probability, thresholds)
    return {
        "average_precision": float(average_precision_score(target, probability)),
        "log_loss": float(log_loss(target, probability, labels=[0, 1])),
        "best_frame_f1": f1,
        "best_threshold": threshold,
    }


def _screen_candidate(
    *,
    candidate_kind: str,
    candidate_name: str,
    representatives: pl.DataFrame,
    lab_frames: dict[str, pl.DataFrame],
    feature_names: dict[str, tuple[str, ...]],
    held_out_folds: tuple[int, ...],
    output_root: Path,
    baseline_config: XGBoostBaselineConfig,
    phase8_config: Phase8Config,
    model_updates: dict[str, object] | None = None,
    maximum_training_rows: int | None = None,
    negative_to_positive_ratio: float | None = None,
    augmented_frames: dict[str, pl.DataFrame] | None = None,
) -> pl.DataFrame:
    updates = {} if model_updates is None else model_updates
    xgb_parameters = baseline_config.xgboost.model_copy(update=updates)
    candidate_config = baseline_config.model_copy(
        update={
            "model_version": f"{phase8_config.phase8_version}-{candidate_kind}-{candidate_name}",
            "xgboost": xgb_parameters,
        }
    )
    records: list[dict[str, object]] = []
    for route_row in representatives.iter_rows(named=True):
        lab = str(route_row["lab_id"])
        action = str(route_row["action"])
        route = str(route_row["route"])
        source = (
            augmented_frames.get(lab, lab_frames[lab])
            if augmented_frames is not None
            else lab_frames[lab]
        )
        route_frame = source.filter(
            (pl.col("action") == action) & (pl.col("route") == route)
        )
        route_frame = _cap_training_frame(
            route_frame,
            maximum_rows=(
                maximum_training_rows
                if maximum_training_rows is not None
                else phase8_config.screen.maximum_training_rows_per_route
            ),
            negative_to_positive_ratio=negative_to_positive_ratio,
            seed=phase8_config.promotion.seed,
        )
        validation_source = lab_frames[lab].filter(
            (pl.col("action") == action) & (pl.col("route") == route)
        )
        for held_fold in held_out_folds:
            validation = validation_source.filter(pl.col("fold") == held_fold)
            if validation.height > phase8_config.screen.maximum_validation_rows_per_route:
                validation = _cap_training_frame(
                    validation,
                    maximum_rows=phase8_config.screen.maximum_validation_rows_per_route,
                    negative_to_positive_ratio=None,
                    seed=phase8_config.promotion.seed + held_fold,
                )
            started = time.perf_counter()
            trained = _load_or_fit_fold_model(
                route_frame,
                feature_names[route],
                held_out_fold=held_fold,
                scope=lab,
                action=action,
                output_root=output_root / candidate_kind / candidate_name,
                config=candidate_config,
            )
            model = load_fold_model(trained.model_path)
            probability = model.predict_proba(
                validation.select(feature_names[route]).to_numpy().astype(np.float32, copy=False)
            )[:, 1].astype(np.float32)
            target = validation["target"].to_numpy().astype(np.uint8, copy=False)
            scores = _score_probability(target, probability, phase8_config.screen.threshold_grid)
            records.append(
                {
                    "candidate_kind": candidate_kind,
                    "candidate": candidate_name,
                    "lab_id": lab,
                    "action": action,
                    "route": route,
                    "held_out_fold": held_fold,
                    "feature_count": len(feature_names[route]),
                    "training_rows": trained.training_rows,
                    "validation_rows": validation.height,
                    "best_iteration": trained.best_iteration,
                    "device": trained.device,
                    "runtime_seconds": time.perf_counter() - started,
                    **scores,
                }
            )
    return pl.DataFrame(records).sort("candidate", "lab_id", "action", "held_out_fold")


def run_feature_ablation(
    *,
    representatives: pl.DataFrame,
    lab_frames: dict[str, pl.DataFrame],
    grouped_features: dict[str, dict[str, tuple[str, ...]]],
    output_root: Path,
    baseline_config: XGBoostBaselineConfig,
    phase8_config: Phase8Config,
) -> tuple[pl.DataFrame, str]:
    outputs: list[pl.DataFrame] = []
    for name, candidate in phase8_config.screen.feature_candidates.items():
        names = {
            "solo": feature_names_for_groups(grouped_features, "solo", candidate.solo),
            "pair": feature_names_for_groups(grouped_features, "pair", candidate.pair),
        }
        outputs.append(
            _screen_candidate(
                candidate_kind="features",
                candidate_name=name,
                representatives=representatives,
                lab_frames=lab_frames,
                feature_names=names,
                held_out_folds=phase8_config.screen.held_out_folds,
                output_root=output_root,
                baseline_config=baseline_config,
                phase8_config=phase8_config,
            )
        )
    result = pl.concat(outputs, how="vertical")
    summary = result.group_by("candidate").agg(
        pl.col("average_precision").mean().alias("macro_average_precision"),
        pl.col("best_frame_f1").mean().alias("macro_best_frame_f1"),
        pl.col("log_loss").mean().alias("mean_log_loss"),
        pl.col("runtime_seconds").sum().alias("runtime_seconds"),
    )
    winner = str(
        summary.sort(
            ["macro_average_precision", "macro_best_frame_f1", "candidate"],
            descending=[True, True, False],
        )["candidate"][0]
    )
    return result.join(summary, on="candidate", how="left"), winner


def run_hyperparameter_search(
    *,
    representatives: pl.DataFrame,
    lab_frames: dict[str, pl.DataFrame],
    grouped_features: dict[str, dict[str, tuple[str, ...]]],
    selected_feature_candidate: str,
    output_root: Path,
    baseline_config: XGBoostBaselineConfig,
    phase8_config: Phase8Config,
) -> tuple[pl.DataFrame, str]:
    selected = phase8_config.screen.feature_candidates[selected_feature_candidate]
    names = {
        "solo": feature_names_for_groups(grouped_features, "solo", selected.solo),
        "pair": feature_names_for_groups(grouped_features, "pair", selected.pair),
    }
    outputs: list[pl.DataFrame] = []
    for name, candidate in phase8_config.screen.hyperparameter_candidates.items():
        outputs.append(
            _screen_candidate(
                candidate_kind="hyperparameters",
                candidate_name=name,
                representatives=representatives,
                lab_frames=lab_frames,
                feature_names=names,
                held_out_folds=phase8_config.screen.held_out_folds,
                output_root=output_root,
                baseline_config=baseline_config,
                phase8_config=phase8_config,
                model_updates=candidate.model_updates(),
                maximum_training_rows=(
                    candidate.maximum_training_rows_per_route
                    or phase8_config.screen.maximum_training_rows_per_route
                ),
                negative_to_positive_ratio=candidate.negative_to_positive_ratio,
            )
        )
    result = pl.concat(outputs, how="vertical")
    summary = result.group_by("candidate").agg(
        pl.col("average_precision").mean().alias("macro_average_precision"),
        pl.col("best_frame_f1").mean().alias("macro_best_frame_f1"),
        pl.col("log_loss").mean().alias("mean_log_loss"),
        pl.col("runtime_seconds").sum().alias("runtime_seconds"),
    )
    winner = str(
        summary.sort(
            ["macro_average_precision", "macro_best_frame_f1", "candidate"],
            descending=[True, True, False],
        )["candidate"][0]
    )
    return result.join(summary, on="candidate", how="left"), winner


def compute_gain_importance(
    baseline_registry: pl.DataFrame,
    grouped_features: dict[str, dict[str, tuple[str, ...]]],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Aggregate split/gain importance and quantify group stability across folds."""
    group_by_feature = {
        name: group
        for route_values in grouped_features.values()
        for group, names in route_values.items()
        for name in names
    }
    unique_models = baseline_registry.filter(pl.col("model_path").is_not_null()).unique(
        subset=["model_path"]
    )
    records: list[dict[str, object]] = []
    for row in unique_models.iter_rows(named=True):
        model = load_fold_model(str(row["model_path"]))
        booster = model.get_booster()
        gains = booster.get_score(importance_type="gain")
        weights = booster.get_score(importance_type="weight")
        scalar_gains = {
            feature: float(sum(value) if isinstance(value, list) else value)
            for feature, value in gains.items()
        }
        total_gain = math.fsum(scalar_gains.values())
        for feature in gains:
            scalar_gain = scalar_gains[feature]
            raw_weight = weights.get(feature, 0.0)
            scalar_weight = float(
                sum(raw_weight) if isinstance(raw_weight, list) else raw_weight
            )
            records.append(
                {
                    "held_out_fold": int(row["held_out_fold"]),
                    "lab_id": str(row["lab_id"]),
                    "action": str(row["action"]),
                    "routing": str(row["routing"]),
                    "feature": feature,
                    "group": group_by_feature.get(feature, "unknown"),
                    "gain": scalar_gain,
                    "normalized_gain": 0.0 if total_gain == 0 else scalar_gain / total_gain,
                    "split_count": int(scalar_weight),
                }
            )
    gain_frame = pl.DataFrame(records).sort(
        "held_out_fold", "lab_id", "action", "normalized_gain", descending=[False, False, False, True]
    )
    fold_groups = gain_frame.group_by("held_out_fold", "group").agg(
        pl.col("normalized_gain").sum().alias("normalized_gain")
    )
    stability = fold_groups.group_by("group").agg(
        pl.col("normalized_gain").mean().alias("mean_normalized_gain"),
        pl.col("normalized_gain").std().fill_null(0.0).alias("fold_std_normalized_gain"),
        pl.col("normalized_gain").min().alias("minimum_fold_gain"),
        pl.col("normalized_gain").max().alias("maximum_fold_gain"),
    ).with_columns(
        pl.when(pl.col("mean_normalized_gain") > 0)
        .then(pl.col("fold_std_normalized_gain") / pl.col("mean_normalized_gain"))
        .otherwise(pl.lit(0.0))
        .alias("coefficient_of_variation")
    ).sort("mean_normalized_gain", descending=True)
    return gain_frame, stability


def compute_group_permutation_importance(
    *,
    representatives: pl.DataFrame,
    lab_frames: dict[str, pl.DataFrame],
    baseline_registry: pl.DataFrame,
    grouped_features: dict[str, dict[str, tuple[str, ...]]],
    phase8_config: Phase8Config,
) -> pl.DataFrame:
    """Permute complete feature families on held-out sampled video rows."""
    held_fold = phase8_config.screen.held_out_folds[0]
    generator = np.random.default_rng(phase8_config.importance.seed)
    records: list[dict[str, object]] = []
    for row in representatives.iter_rows(named=True):
        lab = str(row["lab_id"])
        action = str(row["action"])
        route = str(row["route"])
        entry = baseline_registry.filter(
            (pl.col("lab_id") == lab)
            & (pl.col("action") == action)
            & (pl.col("held_out_fold") == held_fold)
        ).row(0, named=True)
        model_path = Path(str(entry["model_path"]))
        metadata = json.loads(Path(f"{model_path}.meta.json").read_text(encoding="utf-8"))
        names = tuple(str(value) for value in metadata["feature_names"])
        validation = lab_frames[lab].filter(
            (pl.col("action") == action)
            & (pl.col("route") == route)
            & (pl.col("fold") == held_fold)
        )
        if validation.height > phase8_config.importance.maximum_validation_rows:
            validation = validation.sample(
                n=phase8_config.importance.maximum_validation_rows,
                with_replacement=False,
                shuffle=True,
                seed=phase8_config.importance.seed,
            )
        matrix = validation.select(names).to_numpy().astype(np.float32, copy=False)
        target = validation["target"].to_numpy().astype(np.uint8, copy=False)
        model = load_fold_model(model_path)
        baseline_probability = model.predict_proba(matrix)[:, 1].astype(np.float32)
        baseline_ap = float(average_precision_score(target, baseline_probability))
        for group, group_names in grouped_features[route].items():
            selected = [index for index, name in enumerate(names) if name in set(group_names)]
            if not selected:
                continue
            permuted = matrix.copy()
            order = generator.permutation(matrix.shape[0])
            permuted[:, selected] = matrix[order][:, selected]
            probability = model.predict_proba(permuted)[:, 1].astype(np.float32)
            permuted_ap = float(average_precision_score(target, probability))
            records.append(
                {
                    "lab_id": lab,
                    "action": action,
                    "route": route,
                    "held_out_fold": held_fold,
                    "group": group,
                    "feature_count": len(selected),
                    "validation_rows": validation.height,
                    "baseline_average_precision": baseline_ap,
                    "permuted_average_precision": permuted_ap,
                    "average_precision_drop": baseline_ap - permuted_ap,
                }
            )
    return pl.DataFrame(records).sort(
        "average_precision_drop", "lab_id", "action", descending=[True, False, False]
    )


def mine_hard_negatives(
    *,
    representatives: pl.DataFrame,
    lab_frames: dict[str, pl.DataFrame],
    oof_paths: tuple[str, ...],
    phase8_config: Phase8Config,
) -> pl.DataFrame:
    """Mine only cross-fitted prior-OOF false positives from valid sampled negatives."""
    key_parts: list[pl.DataFrame] = []
    for row in representatives.iter_rows(named=True):
        lab = str(row["lab_id"])
        action = str(row["action"])
        route = str(row["route"])
        key_parts.append(
            lab_frames[lab]
            .filter(
                (pl.col("action") == action)
                & (pl.col("route") == route)
                & (pl.col("target") == 0)
            )
            .select(*KEY_COLUMNS, "fold")
        )
    keys = pl.concat(key_parts, how="vertical").unique(subset=KEY_COLUMNS)
    probabilities = (
        pl.scan_parquet(list(oof_paths))
        .select(*KEY_COLUMNS, "fold", "target", "probability")
        .filter(
            (pl.col("target") == 0)
            & (pl.col("probability") >= phase8_config.hard_negative.minimum_probability)
        )
        .join(keys.lazy(), on=[*KEY_COLUMNS, "fold"], how="inner")
        .collect(engine="streaming")
    )
    if probabilities.is_empty():
        return probabilities
    parts = [
        part.sort("probability", descending=True).head(
            phase8_config.hard_negative.top_rows_per_lab_action_fold
        )
        for part in probabilities.partition_by(
            ["lab_id", "action", "fold"], maintain_order=True
        )
    ]
    return pl.concat(parts, how="vertical").sort(
        "lab_id", "action", "fold", "probability", descending=[False, False, False, True]
    )


def augment_with_hard_negatives(
    lab_frames: dict[str, pl.DataFrame],
    hard_negatives: pl.DataFrame,
    *,
    repeat_count: int,
) -> dict[str, pl.DataFrame]:
    output: dict[str, pl.DataFrame] = {}
    hard_keys = hard_negatives.select(*KEY_COLUMNS).unique()
    for lab, frame in lab_frames.items():
        selected = frame.join(
            hard_keys.filter(pl.col("lab_id") == lab),
            on=KEY_COLUMNS,
            how="semi",
        ).with_columns(pl.lit("hard_negative_oof").alias("sampling_kind"))
        output[lab] = pl.concat([frame, *([selected] * repeat_count)], how="diagonal_relaxed")
    return output


def run_hard_negative_comparison(
    *,
    representatives: pl.DataFrame,
    lab_frames: dict[str, pl.DataFrame],
    hard_negatives: pl.DataFrame,
    grouped_features: dict[str, dict[str, tuple[str, ...]]],
    selected_feature_candidate: str,
    selected_hyperparameter_candidate: str,
    output_root: Path,
    baseline_config: XGBoostBaselineConfig,
    phase8_config: Phase8Config,
) -> pl.DataFrame:
    selected_features = phase8_config.screen.feature_candidates[selected_feature_candidate]
    feature_names = {
        "solo": feature_names_for_groups(grouped_features, "solo", selected_features.solo),
        "pair": feature_names_for_groups(grouped_features, "pair", selected_features.pair),
    }
    hyperparameters = phase8_config.screen.hyperparameter_candidates[
        selected_hyperparameter_candidate
    ]
    augmented = augment_with_hard_negatives(
        lab_frames,
        hard_negatives,
        repeat_count=phase8_config.hard_negative.repeat_count,
    )
    ordinary = _screen_candidate(
        candidate_kind="hard_negative",
        candidate_name="without_hard_negatives",
        representatives=representatives,
        lab_frames=lab_frames,
        feature_names=feature_names,
        held_out_folds=phase8_config.screen.held_out_folds,
        output_root=output_root,
        baseline_config=baseline_config,
        phase8_config=phase8_config,
        model_updates=hyperparameters.model_updates(),
        maximum_training_rows=(
            hyperparameters.maximum_training_rows_per_route
            or phase8_config.screen.maximum_training_rows_per_route
        ),
        negative_to_positive_ratio=hyperparameters.negative_to_positive_ratio,
    )
    mined = _screen_candidate(
        candidate_kind="hard_negative",
        candidate_name="with_cross_fitted_hard_negatives",
        representatives=representatives,
        lab_frames=lab_frames,
        feature_names=feature_names,
        held_out_folds=phase8_config.screen.held_out_folds,
        output_root=output_root,
        baseline_config=baseline_config,
        phase8_config=phase8_config,
        model_updates=hyperparameters.model_updates(),
        maximum_training_rows=(
            hyperparameters.maximum_training_rows_per_route
            or phase8_config.screen.maximum_training_rows_per_route
        ),
        negative_to_positive_ratio=hyperparameters.negative_to_positive_ratio,
        augmented_frames=augmented,
    )
    result = pl.concat([ordinary, mined], how="vertical")
    summary = result.group_by("candidate").agg(
        pl.col("average_precision").mean().alias("macro_average_precision"),
        pl.col("best_frame_f1").mean().alias("macro_best_frame_f1"),
        pl.col("runtime_seconds").sum().alias("runtime_seconds"),
    )
    return result.join(summary, on="candidate", how="left")


def run_route_comparison(
    *,
    lab_frames: dict[str, pl.DataFrame],
    baseline_registry: pl.DataFrame,
    baseline_feature_names: dict[str, tuple[str, ...]],
    output_root: Path,
    baseline_config: XGBoostBaselineConfig,
    phase8_config: Phase8Config,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Compare direct, pooled, and direct/pooled hybrid predictions on fixed OOF folds."""
    direct_routes = baseline_registry.filter(pl.col("routing") == "lab_action").select(
        "lab_id", "action", "route_kind"
    ).unique()
    action_support = direct_routes.group_by("action", "route_kind").agg(
        pl.col("lab_id").n_unique().alias("lab_count")
    ).filter(pl.col("lab_count") >= 2)
    all_samples = pl.concat(list(lab_frames.values()), how="diagonal_relaxed")
    route_config = baseline_config.model_copy(
        update={"model_version": f"{phase8_config.phase8_version}-route-screen"}
    )
    records: list[dict[str, object]] = []
    for support_row in action_support.iter_rows(named=True):
        action = str(support_row["action"])
        route = str(support_row["route_kind"])
        pooled_frame = all_samples.filter(
            (pl.col("action") == action) & (pl.col("route") == route)
        )
        labs = direct_routes.filter(
            (pl.col("action") == action) & (pl.col("route_kind") == route)
        )["lab_id"].to_list()
        for held_fold in baseline_config.execution.folds:
            trained = _load_or_fit_fold_model(
                pooled_frame,
                baseline_feature_names[route],
                held_out_fold=held_fold,
                scope="pooled_screen",
                action=action,
                output_root=output_root / "routing",
                config=route_config,
            )
            pooled_model = load_fold_model(trained.model_path)
            for lab_value in labs:
                lab = str(lab_value)
                validation = lab_frames[lab].filter(
                    (pl.col("action") == action)
                    & (pl.col("route") == route)
                    & (pl.col("fold") == held_fold)
                )
                if validation.is_empty() or validation["target"].n_unique() != 2:
                    continue
                direct_entry = baseline_registry.filter(
                    (pl.col("lab_id") == lab)
                    & (pl.col("action") == action)
                    & (pl.col("held_out_fold") == held_fold)
                ).row(0, named=True)
                direct_model = load_fold_model(str(direct_entry["model_path"]))
                matrix = validation.select(baseline_feature_names[route]).to_numpy().astype(
                    np.float32, copy=False
                )
                target = validation["target"].to_numpy().astype(np.uint8, copy=False)
                direct_probability = direct_model.predict_proba(matrix)[:, 1].astype(np.float32)
                pooled_probability = pooled_model.predict_proba(matrix)[:, 1].astype(np.float32)
                for weight in phase8_config.routing.blend_weights:
                    probability = (
                        weight * direct_probability + (1.0 - weight) * pooled_probability
                    ).astype(np.float32)
                    scores = _score_probability(
                        target, probability, phase8_config.screen.threshold_grid
                    )
                    route_name = (
                        "pooled" if weight == 0 else "direct" if weight == 1 else "hybrid_blend"
                    )
                    records.append(
                        {
                            "lab_id": lab,
                            "action": action,
                            "route": route,
                            "held_out_fold": held_fold,
                            "candidate_route": route_name,
                            "direct_weight": weight,
                            "validation_rows": validation.height,
                            "pooled_model_path": trained.model_path.as_posix(),
                            "direct_model_path": str(direct_entry["model_path"]),
                            **scores,
                        }
                    )
    comparison = pl.DataFrame(records).sort(
        "lab_id", "action", "held_out_fold", "direct_weight"
    )
    selections: list[dict[str, object]] = []
    for route_row in direct_routes.iter_rows(named=True):
        lab = str(route_row["lab_id"])
        action = str(route_row["action"])
        route = str(route_row["route_kind"])
        available = comparison.filter(
            (pl.col("lab_id") == lab) & (pl.col("action") == action)
        )
        for held_fold in baseline_config.execution.folds:
            calibration = available.filter(pl.col("held_out_fold") != held_fold)
            if calibration.is_empty():
                selections.append(
                    {
                        "held_out_fold": held_fold,
                        "lab_id": lab,
                        "action": action,
                        "route": route,
                        "selected_route": "direct",
                        "direct_weight": 1.0,
                        "calibration_f1": None,
                        "direct_calibration_f1": None,
                        "calibration_gain": 0.0,
                        "winning_folds": 0,
                        "promoted": False,
                        "reason": "single_lab_action_or_no_pooled_comparison",
                        "used_held_out_labels": False,
                    }
                )
                continue
            means = calibration.group_by("candidate_route", "direct_weight").agg(
                pl.col("best_frame_f1").mean().alias("calibration_f1")
            ).sort(
                ["calibration_f1", "direct_weight"], descending=[True, True]
            )
            winner = means.row(0, named=True)
            direct_score = float(
                means.filter(pl.col("direct_weight") == 1.0)["calibration_f1"][0]
            )
            winner_weight = float(winner["direct_weight"])
            winner_score = float(winner["calibration_f1"])
            paired = calibration.pivot(
                on="direct_weight",
                index="held_out_fold",
                values="best_frame_f1",
                aggregate_function="first",
            )
            winner_column = str(winner_weight)
            direct_column = "1.0"
            winning_folds = (
                paired.filter(pl.col(winner_column) > pl.col(direct_column)).height
                if winner_column in paired.columns and direct_column in paired.columns
                else 0
            )
            gain = winner_score - direct_score
            promoted = (
                winner_weight < 1.0
                and gain >= phase8_config.routing.minimum_component_improvement
                and winning_folds >= phase8_config.routing.minimum_winning_folds
            )
            selections.append(
                {
                    "held_out_fold": held_fold,
                    "lab_id": lab,
                    "action": action,
                    "route": route,
                    "selected_route": str(winner["candidate_route"]) if promoted else "direct",
                    "direct_weight": winner_weight if promoted else 1.0,
                    "calibration_f1": winner_score,
                    "direct_calibration_f1": direct_score,
                    "calibration_gain": gain,
                    "winning_folds": winning_folds,
                    "promoted": promoted,
                    "reason": (
                        "nested_sampled_oof_gain"
                        if promoted
                        else "insufficient_stable_gain_over_direct"
                    ),
                    "used_held_out_labels": False,
                }
            )
    sparse = baseline_registry.filter(pl.col("routing") != "lab_action")
    for row in sparse.iter_rows(named=True):
        selections.append(
            {
                "held_out_fold": int(row["held_out_fold"]),
                "lab_id": str(row["lab_id"]),
                "action": str(row["action"]),
                "route": str(row["route_kind"]),
                "selected_route": str(row["routing"]),
                "direct_weight": 0.0,
                "calibration_f1": None,
                "direct_calibration_f1": None,
                "calibration_gain": 0.0,
                "winning_folds": 0,
                "promoted": False,
                "reason": "phase7_support_hierarchy",
                "used_held_out_labels": False,
            }
        )
    selection = pl.DataFrame(selections).sort("held_out_fold", "lab_id", "action")
    return comparison, selection


def run_alternative_model_screen(
    *,
    representatives: pl.DataFrame,
    lab_frames: dict[str, pl.DataFrame],
    baseline_registry: pl.DataFrame,
    feature_names: dict[str, tuple[str, ...]],
    phase8_config: Phase8Config,
) -> pl.DataFrame:
    """Screen LightGBM/CatBoost only after stable XGBoost, retaining useful diversity only."""
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier

    held_fold = phase8_config.screen.held_out_folds[0]
    records: list[dict[str, object]] = []
    for row in representatives.iter_rows(named=True):
        lab = str(row["lab_id"])
        action = str(row["action"])
        route = str(row["route"])
        frame = lab_frames[lab].filter(
            (pl.col("action") == action) & (pl.col("route") == route)
        )
        frame = _cap_training_frame(
            frame,
            maximum_rows=phase8_config.screen.maximum_training_rows_per_route,
            negative_to_positive_ratio=None,
            seed=phase8_config.promotion.seed,
        )
        training = frame.filter(pl.col("fold") != held_fold)
        validation = frame.filter(pl.col("fold") == held_fold)
        names = feature_names[route]
        train_x = training.select(names).to_numpy().astype(np.float32, copy=False)
        train_y = training["target"].to_numpy().astype(np.uint8, copy=False)
        valid_x = validation.select(names).to_numpy().astype(np.float32, copy=False)
        valid_y = validation["target"].to_numpy().astype(np.uint8, copy=False)
        direct_entry = baseline_registry.filter(
            (pl.col("lab_id") == lab)
            & (pl.col("action") == action)
            & (pl.col("held_out_fold") == held_fold)
        ).row(0, named=True)
        xgb_probability = load_fold_model(str(direct_entry["model_path"])).predict_proba(
            valid_x
        )[:, 1].astype(np.float32)
        xgb_scores = _score_probability(
            valid_y, xgb_probability, phase8_config.screen.threshold_grid
        )
        records.append(
            {
                "lab_id": lab,
                "action": action,
                "route": route,
                "held_out_fold": held_fold,
                "model_family": "xgboost",
                "average_precision": xgb_scores["average_precision"],
                "best_frame_f1": xgb_scores["best_frame_f1"],
                "correlation_with_xgboost": 1.0,
                "runtime_seconds": 0.0,
            }
        )
        positives = max(1, int(train_y.sum()))
        class_weight = float((train_y.size - positives) / positives)
        candidates: list[tuple[str, object]] = [
            (
                "lightgbm",
                LGBMClassifier(
                    objective="binary",
                    n_estimators=160,
                    learning_rate=0.08,
                    max_depth=5,
                    num_leaves=31,
                    min_child_samples=20,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=2.0,
                    scale_pos_weight=class_weight,
                    random_state=phase8_config.promotion.seed,
                    n_jobs=4,
                    verbosity=-1,
                ),
            ),
            (
                "catboost",
                CatBoostClassifier(
                    loss_function="Logloss",
                    iterations=160,
                    learning_rate=0.08,
                    depth=5,
                    l2_leaf_reg=3.0,
                    random_seed=phase8_config.promotion.seed,
                    thread_count=4,
                    verbose=False,
                    allow_writing_files=False,
                    scale_pos_weight=class_weight,
                ),
            ),
        ]
        for family, model in candidates:
            started = time.perf_counter()
            model.fit(train_x, train_y)  # type: ignore[attr-defined]
            probability = np.asarray(model.predict_proba(valid_x))[:, 1].astype(np.float32)  # type: ignore[attr-defined]
            scores = _score_probability(
                valid_y, probability, phase8_config.screen.threshold_grid
            )
            correlation = float(np.corrcoef(xgb_probability, probability)[0, 1])
            records.append(
                {
                    "lab_id": lab,
                    "action": action,
                    "route": route,
                    "held_out_fold": held_fold,
                    "model_family": family,
                    "average_precision": scores["average_precision"],
                    "best_frame_f1": scores["best_frame_f1"],
                    "correlation_with_xgboost": correlation,
                    "runtime_seconds": time.perf_counter() - started,
                }
            )
    result = pl.DataFrame(records)
    xgb_reference = result.filter(pl.col("model_family") == "xgboost").select(
        "lab_id", "action", pl.col("average_precision").alias("xgboost_average_precision")
    )
    result = result.join(xgb_reference, on=["lab_id", "action"], how="left").with_columns(
        (pl.col("average_precision") - pl.col("xgboost_average_precision")).alias(
            "average_precision_gain"
        ),
        (
            (pl.col("model_family") != "xgboost")
            & (
                (pl.col("average_precision") >= pl.col("xgboost_average_precision") + 0.01)
                | (
                    (pl.col("average_precision") >= pl.col("xgboost_average_precision") - 0.005)
                    & (pl.col("correlation_with_xgboost") < 0.95)
                )
            )
        ).alias("retain_for_future_ensemble"),
    )
    return result.sort("lab_id", "action", "model_family")


def build_or_load_all_family_cache(
    *,
    primary_videos: pl.DataFrame,
    training_plan: pl.DataFrame,
    dataset_root: Path,
    cache_root: Path,
    pose_config_path: Path,
    feature_config_path: Path,
    category_registry_path: Path,
    baseline_config: XGBoostBaselineConfig,
    grouped_features: dict[str, dict[str, tuple[str, ...]]],
) -> tuple[dict[str, Path], dict[str, tuple[str, ...]]]:
    """Build a separate raw-inclusive cache once; never overwrite Phase 7 caches."""
    cache_root.mkdir(parents=True, exist_ok=True)
    labs = sorted(str(value) for value in primary_videos["lab_id"].unique())
    paths = {lab: cache_root / f"{lab}.parquet" for lab in labs}
    raw_probe = grouped_features["solo"]["raw"][0]
    if all(path.is_file() for path in paths.values()):
        probe_schema = pl.read_parquet_schema(paths[labs[0]])
        if raw_probe in probe_schema:
            return paths, {
                route: tuple(
                    name
                    for names in grouped_features[route].values()
                    for name in names
                )
                for route in ("solo", "pair")
            }
    all_groups = {
        route: tuple(grouped_features[route]) for route in ("solo", "pair")
    }
    all_feature_config = baseline_config.features.model_copy(
        update={
            "solo_groups": all_groups["solo"],
            "pair_groups": all_groups["pair"],
        }
    )
    all_config = baseline_config.model_copy(update={"features": all_feature_config})
    return _build_training_caches(
        primary_videos,
        training_plan,
        manifests=_manifest_lookup(primary_videos),
        cache_root=cache_root,
        dataset_root=dataset_root,
        pose_config=load_pose_config(pose_config_path),
        feature_config=load_feature_config(feature_config_path),
        baseline_config=all_config,
        categories=load_category_registry(category_registry_path),
    )


def run_phase8_studies(
    *,
    videos_path: Path,
    triplets_path: Path,
    folds_path: Path,
    dataset_root: Path,
    cache_root: Path,
    artifact_root: Path,
    pose_config_path: Path,
    feature_config_path: Path,
    category_registry_path: Path,
    baseline_config: XGBoostBaselineConfig,
    phase8_config: Phase8Config,
    output_root: Path,
) -> StudyOutputs:
    videos = pl.read_parquet(videos_path)
    triplets = pl.read_parquet(triplets_path)
    folds = pl.read_parquet(folds_path)
    primary_membership = folds.filter(pl.col("is_primary_validation")).select(
        "lab_id", "video_id", "fold"
    )
    primary_videos = videos.join(
        primary_membership, on=["lab_id", "video_id"], how="inner"
    )
    _ = filter_feasible_triplets(
        triplets.filter((pl.col("split") == "train") & ~pl.col("is_duplicate")).join(
            primary_membership.select("lab_id", "video_id"),
            on=["lab_id", "video_id"],
            how="inner",
        ),
        primary_videos,
    )
    baseline_root = artifact_root / "xgb" / phase8_config.baseline_model_version
    plan_paths = sorted(baseline_root.glob("training_sampling_plan_*.parquet"))
    if len(plan_paths) != 1:
        raise ValueError(f"expected exactly one Phase 7 sampling plan, found {len(plan_paths)}")
    training_plan = pl.read_parquet(plan_paths[0])
    baseline_registry = pl.read_parquet(baseline_root / "model_registry.parquet")
    baseline_features_raw = json.loads(
        (baseline_root / "model_features.json").read_text(encoding="utf-8")
    )
    baseline_feature_names = {
        route: tuple(str(value) for value in values)
        for route, values in baseline_features_raw.items()
    }
    grouped_features = load_feature_groups(category_registry_path.parent)
    lab_paths, _ = build_or_load_all_family_cache(
        primary_videos=primary_videos,
        training_plan=training_plan,
        dataset_root=dataset_root,
        cache_root=cache_root / "phase8_all_family_training_samples",
        pose_config_path=pose_config_path,
        feature_config_path=feature_config_path,
        category_registry_path=category_registry_path,
        baseline_config=baseline_config,
        grouped_features=grouped_features,
    )
    lab_frames = {lab: pl.read_parquet(path) for lab, path in lab_paths.items()}
    representatives = select_representative_routes(
        lab_frames,
        baseline_registry,
        folds=baseline_config.execution.folds,
        per_lab=phase8_config.screen.representative_routes_per_lab,
    )
    experiment_root = output_root / "screen_models"
    feature_ablation, selected_features = run_feature_ablation(
        representatives=representatives,
        lab_frames=lab_frames,
        grouped_features=grouped_features,
        output_root=experiment_root,
        baseline_config=baseline_config,
        phase8_config=phase8_config,
    )
    hyperparameter_search, selected_hyperparameters = run_hyperparameter_search(
        representatives=representatives,
        lab_frames=lab_frames,
        grouped_features=grouped_features,
        selected_feature_candidate=selected_features,
        output_root=experiment_root,
        baseline_config=baseline_config,
        phase8_config=phase8_config,
    )
    oof_manifest_paths = sorted(
        (artifact_root / "oof" / phase8_config.baseline_model_version).glob("*/manifest.json")
    )
    if len(oof_manifest_paths) != 1:
        raise ValueError(f"expected exactly one baseline OOF manifest, found {len(oof_manifest_paths)}")
    oof_manifest = json.loads(oof_manifest_paths[0].read_text(encoding="utf-8"))
    hard_negatives = mine_hard_negatives(
        representatives=representatives,
        lab_frames=lab_frames,
        oof_paths=tuple(str(value) for value in oof_manifest["parts"]),
        phase8_config=phase8_config,
    )
    hard_negative_comparison = run_hard_negative_comparison(
        representatives=representatives,
        lab_frames=lab_frames,
        hard_negatives=hard_negatives,
        grouped_features=grouped_features,
        selected_feature_candidate=selected_features,
        selected_hyperparameter_candidate=selected_hyperparameters,
        output_root=experiment_root,
        baseline_config=baseline_config,
        phase8_config=phase8_config,
    )
    gain_importance, group_stability = compute_gain_importance(
        baseline_registry, grouped_features
    )
    permutation_importance = compute_group_permutation_importance(
        representatives=representatives,
        lab_frames=lab_frames,
        baseline_registry=baseline_registry,
        grouped_features=grouped_features,
        phase8_config=phase8_config,
    )
    route_comparison, route_selection = run_route_comparison(
        lab_frames=lab_frames,
        baseline_registry=baseline_registry,
        baseline_feature_names=baseline_feature_names,
        output_root=experiment_root,
        baseline_config=baseline_config,
        phase8_config=phase8_config,
    )
    alternative_models = run_alternative_model_screen(
        representatives=representatives,
        lab_frames=lab_frames,
        baseline_registry=baseline_registry,
        feature_names=baseline_feature_names,
        phase8_config=phase8_config,
    )
    return StudyOutputs(
        representatives=representatives,
        feature_ablation=feature_ablation,
        hyperparameter_search=hyperparameter_search,
        hard_negative_plan=hard_negatives,
        hard_negative_comparison=hard_negative_comparison,
        gain_importance=gain_importance,
        group_stability=group_stability,
        permutation_importance=permutation_importance,
        route_comparison=route_comparison,
        route_selection=route_selection,
        alternative_models=alternative_models,
        selected_feature_candidate=selected_features,
        selected_hyperparameter_candidate=selected_hyperparameters,
    )
