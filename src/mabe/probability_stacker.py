"""Leakage-safe nonlinear stacking over aligned GBDT and temporal OOF streams."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import xgboost as xgb

from mabe.phase11_config import StackerConfig
from mabe.schemas import OOF_COLUMNS
from mabe.xgb_model import resolve_xgboost_device

STACK_FEATURES = (
    "gbdt_probability",
    "temporal_probability",
    "gbdt_logit",
    "temporal_logit",
    "probability_product",
    "probability_absolute_difference",
)
STACK_KEYS = (
    "lab_id",
    "video_id",
    "video_frame",
    "agent_id",
    "target_id",
    "action",
    "fold",
)


@dataclass(frozen=True, slots=True)
class ProbabilityStackerArtifacts:
    manifest_path: Path
    registry_path: Path
    sample_rows: int
    model_count: int


def stacker_feature_matrix(
    gbdt_probability: np.ndarray[Any, np.dtype[np.float32]],
    temporal_probability: np.ndarray[Any, np.dtype[np.float32]],
) -> np.ndarray[Any, np.dtype[np.float32]]:
    epsilon = np.float32(1e-5)
    gbdt = np.clip(gbdt_probability, epsilon, 1 - epsilon)
    temporal = np.clip(temporal_probability, epsilon, 1 - epsilon)
    return np.column_stack(
        (
            gbdt,
            temporal,
            np.log(gbdt / (1 - gbdt)),
            np.log(temporal / (1 - temporal)),
            gbdt * temporal,
            np.abs(gbdt - temporal),
        )
    ).astype(np.float32, copy=False)


def _aligned_part(
    gbdt_path: Path, temporal_path: Path
) -> tuple[pl.DataFrame, np.ndarray[Any, np.dtype[np.float32]]]:
    base = pl.read_parquet(gbdt_path)
    temporal = pl.read_parquet(temporal_path)
    if base.height != temporal.height or not base.select(STACK_KEYS).equals(
        temporal.select(STACK_KEYS)
    ):
        raise ValueError(f"stacker OOF rows are not aligned: {gbdt_path}")
    features = stacker_feature_matrix(
        base["probability"].to_numpy().astype(np.float32, copy=False),
        temporal["probability"].to_numpy().astype(np.float32, copy=False),
    )
    return base, features


def _build_samples(
    gbdt_parts: dict[str, Path],
    temporal_parts: dict[str, Path],
    *,
    config: StackerConfig,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    samples: list[pl.DataFrame] = []
    counts: list[pl.DataFrame] = []
    for suffix in sorted(gbdt_parts):
        base, matrix = _aligned_part(gbdt_parts[suffix], temporal_parts[suffix])
        frame = base.select(*STACK_KEYS, "target", "loss_mask").with_columns(
            *(pl.Series(name, matrix[:, index]) for index, name in enumerate(STACK_FEATURES)),
            pl.struct(STACK_KEYS).hash(seed=2025).alias("sample_priority"),
        ).filter(pl.col("loss_mask").cast(pl.Boolean))
        counts.append(
            frame.group_by("action", "fold", "target").len().rename({"len": "population_rows"})
        )
        selected = frame.filter(
            (pl.col("target") == 1)
            | (
                pl.col("sample_priority") % 1000
                < config.negative_hash_keep_per_thousand
            )
        )
        if not selected.is_empty():
            samples.append(selected)
    if not samples:
        raise ValueError("no aligned rows were available for probability stacking")
    combined = pl.concat(samples, how="vertical_relaxed")
    capped: list[pl.DataFrame] = []
    fold_count = max(1, combined["fold"].n_unique())
    positive_cap_per_fold = int(
        np.ceil(config.maximum_positive_rows_per_action / fold_count)
    )
    negative_cap_per_fold = int(
        np.ceil(config.maximum_negative_rows_per_action / fold_count)
    )
    # Cap independently within every fold. A global per-action cap would let
    # held-out labels change which training-fold rows survive the cap, creating
    # a small but real cross-fold dependency.
    for action_frame in combined.partition_by(
        ["action", "fold"], maintain_order=True
    ):
        positives = (
            action_frame.filter(pl.col("target") == 1)
            .sort("sample_priority")
            .head(positive_cap_per_fold)
        )
        negatives = (
            action_frame.filter(pl.col("target") == 0)
            .sort("sample_priority")
            .head(negative_cap_per_fold)
        )
        capped.extend((positives, negatives))
    return (
        pl.concat(capped, how="vertical_relaxed").sort(
            "action", "fold", "sample_priority"
        ),
        pl.concat(counts, how="vertical_relaxed")
        .group_by("action", "fold", "target")
        .agg(pl.col("population_rows").sum()),
    )


def _safe_action(action: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", action)


def _sample_weights(
    frame: pl.DataFrame, counts: pl.DataFrame, training_folds: tuple[int, ...]
) -> np.ndarray[Any, np.dtype[np.float32]]:
    selected_counts = {
        int(row["target"]): int(row["len"])
        for row in frame.group_by("target").len().iter_rows(named=True)
    }
    population_counts = {
        int(row["target"]): int(row["population_rows"])
        for row in counts.filter(pl.col("fold").is_in(training_folds))
        .group_by("target")
        .agg(pl.col("population_rows").sum())
        .iter_rows(named=True)
    }
    return np.asarray(
        [
            population_counts.get(int(target), 0)
            / max(1, selected_counts.get(int(target), 0))
            for target in frame["target"]
        ],
        dtype=np.float32,
    )


def _fit_model(
    frame: pl.DataFrame,
    counts: pl.DataFrame,
    *,
    action: str,
    training_folds: tuple[int, ...],
    output_path: Path,
    config: StackerConfig,
) -> tuple[Path | None, str, int, int]:
    training = frame.filter(
        (pl.col("action") == action) & pl.col("fold").is_in(training_folds)
    )
    positives = int(training["target"].sum()) if training.height else 0
    if training.is_empty() or training["target"].n_unique() < 2:
        return None, "insufficient_two_class_support", training.height, positives
    device = resolve_xgboost_device(config.device)
    parameters: dict[str, object] = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "device": device,
        "n_estimators": config.n_estimators,
        "learning_rate": config.learning_rate,
        "max_depth": config.max_depth,
        "min_child_weight": config.min_child_weight,
        "subsample": config.subsample,
        "colsample_bytree": config.colsample_bytree,
        "reg_lambda": config.reg_lambda,
        "n_jobs": 4,
        "random_state": 2025,
        "verbosity": 0,
    }
    model = xgb.XGBClassifier(**parameters)
    matrix = training.select(STACK_FEATURES).to_numpy().astype(np.float32, copy=False)
    target = training["target"].to_numpy().astype(np.uint8, copy=False)
    weights = _sample_weights(
        training,
        counts.filter(pl.col("action") == action),
        training_folds,
    )
    try:
        model.fit(matrix, target, sample_weight=weights, verbose=False)
    except xgb.core.XGBoostError:
        if device != "cuda":
            raise
        parameters["device"] = "cpu"
        model = xgb.XGBClassifier(**parameters)
        model.fit(matrix, target, sample_weight=weights, verbose=False)
        device = "cpu"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(output_path)
    return output_path, device, training.height, positives


def build_probability_stacker(
    *,
    gbdt_parts: dict[str, Path],
    temporal_parts: dict[str, Path],
    folds: tuple[int, ...],
    output_root: Path,
    oof_root: Path,
    configuration_sha256: str,
    source_names: tuple[str, str],
    config: StackerConfig,
) -> ProbabilityStackerArtifacts:
    """Fit fold-isolated stackers, deployment stackers, and a full OOF stream."""
    samples, counts = _build_samples(gbdt_parts, temporal_parts, config=config)
    output_root.mkdir(parents=True, exist_ok=True)
    samples.write_parquet(
        output_root / "training_sample.parquet", compression="zstd", statistics=True
    )
    actions = tuple(sorted(str(value) for value in counts["action"].unique()))
    records: list[dict[str, object]] = []
    models: dict[tuple[int, str], xgb.XGBClassifier | None] = {}
    for held_fold in (*folds, -1):
        training_folds = (
            tuple(fold for fold in folds if fold != held_fold)
            if held_fold >= 0
            else folds
        )
        scope = f"fold_{held_fold}" if held_fold >= 0 else "deployment"
        for action in actions:
            path, device, training_rows, positive_rows = _fit_model(
                samples,
                counts,
                action=action,
                training_folds=training_folds,
                output_path=output_root / "models" / scope / f"{_safe_action(action)}.json",
                config=config,
            )
            model: xgb.XGBClassifier | None = None
            if path is not None:
                model = xgb.XGBClassifier()
                model.load_model(path)
            models[(held_fold, action)] = model
            records.append(
                {
                    "held_out_fold": held_fold,
                    "action": action,
                    "model_path": None if path is None else path.as_posix(),
                    "routing": "stacker" if path is not None else "equal_blend_fallback",
                    "training_rows": training_rows,
                    "positive_rows": positive_rows,
                    "device": device,
                    "used_held_out_labels": False,
                }
            )
    registry = pl.DataFrame(records).sort("held_out_fold", "action")
    registry_path = output_root / "model_registry.parquet"
    registry.write_parquet(registry_path, compression="zstd", statistics=True)
    output_paths: list[Path] = []
    row_count = 0
    for index, suffix in enumerate(sorted(gbdt_parts), start=1):
        base, matrix = _aligned_part(gbdt_parts[suffix], temporal_parts[suffix])
        fold = int(base["fold"][0])
        probability = np.zeros(base.height, dtype=np.float32)
        actions_array = base["action"].to_numpy()
        base_probability = base["probability"].to_numpy().astype(np.float32, copy=False)
        temporal_probability = pl.read_parquet(
            temporal_parts[suffix], columns=["probability"]
        )["probability"].to_numpy().astype(np.float32, copy=False)
        for action in np.unique(actions_array):
            selected = np.flatnonzero(actions_array == action)
            model = models[(fold, str(action))]
            probability[selected] = (
                0.5 * (base_probability[selected] + temporal_probability[selected])
                if model is None
                else model.predict_proba(matrix[selected])[:, 1].astype(np.float32)
            )
        output = base.with_columns(
            pl.Series("probability", probability, dtype=pl.Float32),
            pl.lit("phase11-crossfit-stacker").alias("model_id"),
            pl.lit("phase11-crossfit-stacker").alias("feature_version"),
        ).select(OOF_COLUMNS)
        path = oof_root / suffix
        path.parent.mkdir(parents=True, exist_ok=True)
        output.write_parquet(path, compression="zstd", statistics=True)
        output_paths.append(path)
        row_count += output.height
        if index == 1 or index % 100 == 0 or index == len(gbdt_parts):
            print(
                f"Phase 11 probability stacker OOF: {index}/{len(gbdt_parts)} parts",
                flush=True,
            )
    manifest_path = oof_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_version": "phase11-crossfit-stacker",
                "configuration_sha256": configuration_sha256,
                "row_count": row_count,
                "parts": [path.as_posix() for path in output_paths],
                "probability_lineage": {
                    "type": "cross_fitted_probability_xgboost_stacker",
                    "sources": list(source_names),
                    "deployment_models": registry_path.as_posix(),
                    "used_held_out_labels": False,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return ProbabilityStackerArtifacts(
        manifest_path=manifest_path,
        registry_path=registry_path,
        sample_rows=samples.height,
        model_count=registry.filter(pl.col("model_path").is_not_null()).height,
    )
