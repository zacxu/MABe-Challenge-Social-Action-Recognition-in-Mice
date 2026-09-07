"""Leakage-safe early-stopped XGBoost fitting and model provenance."""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

from mabe.xgb_config import SupportConfig, XGBoostBaselineConfig


@dataclass(frozen=True, slots=True)
class RouteDecision:
    route: str
    reason: str


@dataclass(frozen=True, slots=True)
class TrainedFoldModel:
    model_path: Path
    metadata_path: Path
    best_iteration: int
    training_rows: int
    positive_rows: int
    training_videos: int
    inner_validation_fold: int
    device: str


def choose_model_route(
    *,
    direct_positive_rows: int,
    direct_positive_videos: int,
    pooled_positive_rows: int,
    auxiliary_positive_rows: int,
    action: str,
    support: SupportConfig,
) -> RouteDecision:
    """Apply the configured direct, pooled, reviewed-auxiliary, no-prediction hierarchy."""
    if (
        direct_positive_rows >= support.minimum_positive_rows
        and direct_positive_videos >= support.minimum_positive_videos
    ):
        return RouteDecision("lab_action", "sufficient_direct_support")
    if (
        pooled_positive_rows >= support.pooled_minimum_positive_rows
        and (support.allow_pooled_without_direct_positives or direct_positive_rows > 0)
    ):
        return RouteDecision("pooled_action", "insufficient_direct_support")
    reviewed_labs = support.auxiliary_labs_by_action.get(action, ())
    if reviewed_labs and auxiliary_positive_rows >= support.minimum_positive_rows:
        return RouteDecision("auxiliary_action", "reviewed_auxiliary_semantics")
    return RouteDecision("no_prediction", "no_reliable_supervision")


def resolve_xgboost_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        return "cuda"
    return "cuda" if shutil.which("nvidia-smi") else "cpu"


def _seed(base: int, held_out_fold: int, scope: str, action: str) -> int:
    digest = sum((index + 1) * ord(character) for index, character in enumerate(scope + action))
    return int((base + 1009 * held_out_fold + digest) % (2**31 - 1))


def _positive_weight(
    target: np.ndarray[tuple[int], np.dtype[np.uint8]], configured: str | float
) -> float:
    if configured not in {"auto", "sqrt_auto"}:
        return float(configured)
    positives = int(target.sum())
    negatives = int(target.size - positives)
    ratio = float(negatives / positives) if positives else 1.0
    return math.sqrt(ratio) if configured == "sqrt_auto" else ratio


def _parameters(
    config: XGBoostBaselineConfig,
    *,
    seed: int,
    n_estimators: int,
    device: str,
    early_stopping_rounds: int | None,
    positive_weight: float,
) -> dict[str, object]:
    source = config.xgboost
    result: dict[str, object] = {
        "objective": source.objective,
        "eval_metric": source.eval_metric,
        "tree_method": source.tree_method,
        "device": device,
        "n_estimators": n_estimators,
        "learning_rate": source.learning_rate,
        "max_depth": source.max_depth,
        "min_child_weight": source.min_child_weight,
        "gamma": source.gamma,
        "subsample": source.subsample,
        "colsample_bytree": source.colsample_bytree,
        "reg_alpha": source.reg_alpha,
        "reg_lambda": source.reg_lambda,
        "max_bin": source.max_bin,
        "n_jobs": source.n_jobs,
        "random_state": seed,
        "scale_pos_weight": positive_weight,
        "verbosity": 0,
    }
    if early_stopping_rounds is not None:
        result["early_stopping_rounds"] = early_stopping_rounds
    return result


def _matrix(frame: pl.DataFrame, feature_names: tuple[str, ...]) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
    matrix = frame.select(feature_names).to_numpy().astype(np.float32, copy=False)
    if not np.isfinite(matrix).all():
        raise ValueError("non-finite values entered XGBoost training")
    return matrix


def _inner_fold(frame: pl.DataFrame, held_out_fold: int, folds: tuple[int, ...]) -> int:
    candidates = [fold for fold in folds if fold != held_out_fold]
    candidates.sort(key=lambda fold: ((fold - held_out_fold) % len(folds), fold))
    for fold in candidates:
        evaluation = frame.filter(pl.col("fold") == fold)
        training = frame.filter((pl.col("fold") != held_out_fold) & (pl.col("fold") != fold))
        if evaluation["target"].n_unique() == 2 and training["target"].n_unique() == 2:
            return fold
    raise ValueError("no inner fold contains both classes for leakage-safe early stopping")


def fit_fold_model(
    frame: pl.DataFrame,
    feature_names: tuple[str, ...],
    *,
    held_out_fold: int,
    scope: str,
    action: str,
    output_root: str | Path,
    config: XGBoostBaselineConfig,
) -> TrainedFoldModel:
    """Early-stop inside outer training membership, then refit all outer-training rows."""
    outer_training = frame.filter(pl.col("fold") != held_out_fold)
    if outer_training.is_empty() or outer_training["target"].n_unique() != 2:
        raise ValueError(f"outer training data lacks both classes for {(scope, action, held_out_fold)}")
    inner_fold = _inner_fold(frame, held_out_fold, config.execution.folds)
    inner_training = frame.filter(
        (pl.col("fold") != held_out_fold) & (pl.col("fold") != inner_fold)
    )
    inner_validation = frame.filter(pl.col("fold") == inner_fold)
    device = resolve_xgboost_device(config.xgboost.device)
    random_seed = _seed(config.execution.seed, held_out_fold, scope, action)
    inner_target = inner_training["target"].to_numpy().astype(np.uint8, copy=False)
    positive_weight = _positive_weight(inner_target, config.xgboost.positive_class_weight)
    early_model = xgb.XGBClassifier(
        **_parameters(
            config,
            seed=random_seed,
            n_estimators=config.xgboost.n_estimators,
            device=device,
            early_stopping_rounds=config.xgboost.early_stopping_rounds,
            positive_weight=positive_weight,
        )
    )
    try:
        early_model.fit(
            _matrix(inner_training, feature_names),
            inner_target,
            eval_set=[
                (
                    _matrix(inner_validation, feature_names),
                    inner_validation["target"].to_numpy().astype(np.uint8, copy=False),
                )
            ],
            verbose=False,
        )
    except xgb.core.XGBoostError:
        if device != "cuda":
            raise
        device = "cpu"
        early_model = xgb.XGBClassifier(
            **_parameters(
                config,
                seed=random_seed,
                n_estimators=config.xgboost.n_estimators,
                device=device,
                early_stopping_rounds=config.xgboost.early_stopping_rounds,
                positive_weight=positive_weight,
            )
        )
        early_model.fit(
            _matrix(inner_training, feature_names),
            inner_target,
            eval_set=[
                (
                    _matrix(inner_validation, feature_names),
                    inner_validation["target"].to_numpy().astype(np.uint8, copy=False),
                )
            ],
            verbose=False,
        )
    best_iteration = int(early_model.best_iteration) + 1
    outer_target = outer_training["target"].to_numpy().astype(np.uint8, copy=False)
    final_weight = _positive_weight(outer_target, config.xgboost.positive_class_weight)
    final_model = xgb.XGBClassifier(
        **_parameters(
            config,
            seed=random_seed,
            n_estimators=best_iteration,
            device=device,
            early_stopping_rounds=None,
            positive_weight=final_weight,
        )
    )
    final_model.fit(_matrix(outer_training, feature_names), outer_target, verbose=False)
    model_path = Path(output_root) / f"fold_{held_out_fold}" / scope / f"{action}.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    final_model.save_model(model_path)
    metadata = {
        "schema_version": 1,
        "model_version": config.model_version,
        "scope": scope,
        "action": action,
        "held_out_fold": held_out_fold,
        "inner_validation_fold": inner_fold,
        "held_out_labels_used_for_early_stopping": False,
        "best_iteration": best_iteration,
        "device": device,
        "feature_names": list(feature_names),
        "training_rows": outer_training.height,
        "positive_rows": int(outer_target.sum()),
        "training_videos": outer_training["video_id"].n_unique(),
        "positive_class_weight": final_weight,
    }
    metadata_path = Path(f"{model_path}.meta.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return TrainedFoldModel(
        model_path=model_path,
        metadata_path=metadata_path,
        best_iteration=best_iteration,
        training_rows=outer_training.height,
        positive_rows=int(outer_target.sum()),
        training_videos=outer_training["video_id"].n_unique(),
        inner_validation_fold=inner_fold,
        device=device,
    )


def load_fold_model(path: str | Path) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier()
    model.load_model(path)
    return model
