"""Bounded fixed-fold screens and retention decisions for Phase 10."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import polars as pl
import yaml

from mabe.config import RunConfig
from mabe.fingerprints import schema_fingerprint, sha256_file
from mabe.labels import build_validation_solution
from mabe.phase8_config import load_phase8_config
from mabe.phase8_decoder import nested_interval_postprocessing
from mabe.phase9_config import Phase9Config, load_phase9_config
from mabe.phase9_pipeline import (
    _decode_oof,
    _nested_thresholds,
    _per_lab_action_comparison,
    _probability_diversity,
    _sidecar,
    _threshold_counts,
    build_temporal_oof,
    build_temporal_sequence_cache,
)
from mabe.phase10_config import (
    Phase10Config,
    Phase10ExperimentConfig,
    load_phase10_config,
    phase10_config_fingerprint,
)
from mabe.submission import validate_submission
from mabe.temporal_training import FoldTrainingResult, train_temporal_fold


@dataclass(frozen=True, slots=True)
class Phase10Outputs:
    report_path: Path
    results_path: Path
    retained_registry_path: Path
    retained_models: int


def _experiment_phase9_config(
    base: Phase9Config,
    phase10: Phase10Config,
    experiment: Phase10ExperimentConfig,
) -> Phase9Config:
    payload = base.model_dump(mode="json")
    payload["temporal_version"] = f"{phase10.phase10_version}-{experiment.name}"
    payload["data"].update(experiment.data_overrides)
    payload["architecture"].update(experiment.architecture_overrides)
    payload["training"].update(
        {
            "epochs": phase10.screen.epochs,
            "steps_per_epoch": phase10.screen.steps_per_epoch,
            "validation_windows_per_route": phase10.screen.validation_windows_per_route,
            "batch_size": phase10.screen.batch_size,
            "warmup_steps": phase10.screen.warmup_steps,
            "early_stopping_patience": phase10.screen.early_stopping_patience,
            "gate_fold": phase10.screen.held_out_fold,
        }
    )
    payload["training"].update(experiment.training_overrides)
    payload["augmentation"].update(experiment.augmentation_overrides)
    return Phase9Config.model_validate(payload)


def _write_generated_config(config: Phase9Config, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    return sha256_file(path)


def _load_training_result(card_path: Path, checkpoint_path: Path) -> FoldTrainingResult:
    payload = json.loads(card_path.read_text(encoding="utf-8"))
    return FoldTrainingResult(
        held_out_fold=int(payload["held_out_fold"]),
        checkpoint_path=checkpoint_path,
        best_epoch=int(payload["best_epoch"]),
        best_validation_loss=float(payload["best_validation_loss"]),
        validation_average_precision=float(payload["validation_average_precision"]),
        training_seconds=float(payload["training_seconds"]),
        device=str(payload["device"]),
        parameter_count=int(payload["parameter_count"]),
        peak_memory_mb=float(payload.get("peak_memory_mb", 0.0)),
    )


def _manifest_and_cache(
    *,
    config: Phase9Config,
    config_path: Path,
    base: Phase9Config,
    run_config: RunConfig,
    pose_config_path: Path,
    source_fingerprint: str,
) -> tuple[pl.DataFrame, Path]:
    if config.data.landmarks == base.data.landmarks:
        manifest_path = (
            run_config.paths.artifact_root
            / "temporal"
            / base.temporal_version
            / "sequence_manifest.parquet"
        )
        return (
            pl.read_parquet(manifest_path),
            run_config.paths.cache_root / base.temporal_version / "sequences",
        )
    manifest = build_temporal_sequence_cache(
        run_config=run_config,
        temporal_config_path=config_path,
        pose_config_path=pose_config_path,
        source_fingerprint=source_fingerprint,
    )
    return (
        manifest,
        run_config.paths.cache_root / config.temporal_version / "sequences",
    )


def _result_record(
    *,
    experiment: Phase10ExperimentConfig,
    config: Phase9Config,
    config_path: Path,
    result: FoldTrainingResult | None,
    status: str,
    control_ap: float | None,
    reason: str,
) -> dict[str, Any]:
    average_precision = result.validation_average_precision if result else None
    return {
        "experiment": experiment.name,
        "family": experiment.family,
        "status": status,
        "reason": reason,
        "held_out_fold": result.held_out_fold if result else None,
        "landmark_count": len(config.data.landmarks),
        "landmarks_json": json.dumps(config.data.landmarks),
        "window_frames": config.data.window_frames,
        "architecture_json": json.dumps(
            config.architecture.model_dump(mode="json"), sort_keys=True
        ),
        "augmentation_json": json.dumps(
            config.augmentation.model_dump(mode="json"), sort_keys=True
        ),
        "augmentation_delta_vs_no_augmentation": None,
        "soft_f1_weight": config.training.soft_f1_weight,
        "validation_average_precision": average_precision,
        "control_average_precision": control_ap,
        "average_precision_delta": (
            average_precision - control_ap
            if average_precision is not None and control_ap is not None
            else None
        ),
        "best_validation_loss": result.best_validation_loss if result else None,
        "best_epoch": result.best_epoch if result else None,
        "training_seconds": result.training_seconds if result else None,
        "peak_memory_mb": result.peak_memory_mb if result else None,
        "parameter_count": result.parameter_count if result else None,
        "device": result.device if result else None,
        "configuration_path": config_path.as_posix(),
        "checkpoint_path": result.checkpoint_path.as_posix() if result else None,
        "exact_oof_fbeta": None,
        "probability_correlation_with_gbdt": None,
        "error_jaccard_with_gbdt": None,
        "retained": False,
    }


def _run_full_oof_gate(
    *,
    experiment: Phase10ExperimentConfig,
    config: Phase9Config,
    config_path: Path,
    configuration_sha256: str,
    manifest: pl.DataFrame,
    cache_root: Path,
    checkpoint_root: Path,
    hard_negatives: pl.DataFrame,
    run_config: RunConfig,
    phase10_root: Path,
    phase10: Phase10Config,
    source_fingerprint: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    fold_results: list[FoldTrainingResult] = []
    for fold in config.training.folds:
        checkpoint = checkpoint_root / f"fold_{fold}" / "model.pt"
        card = checkpoint.parent / "model_card.json"
        existing = json.loads(card.read_text(encoding="utf-8")) if card.is_file() else {}
        if (
            checkpoint.is_file()
            and card.is_file()
            and existing.get("configuration_sha256") == configuration_sha256
        ):
            result = _load_training_result(card, checkpoint)
        else:
            print(
                f"Phase 10 full OOF candidate {experiment.name}: training fold {fold}",
                flush=True,
            )
            result = train_temporal_fold(
                held_out_fold=fold,
                manifest=manifest,
                cache_root=cache_root,
                hard_negatives=hard_negatives,
                output_root=checkpoint_root,
                config=config,
                configuration_sha256=configuration_sha256,
            )
        fold_results.append(result)
    phase8_root = run_config.paths.artifact_root / "phase8" / "gbdt-phase8-v1"
    baseline_manifest = phase8_root / "final_oof_manifest.json"
    if not baseline_manifest.is_file():
        raise FileNotFoundError("Phase 10 requires the frozen Phase 8 final OOF manifest")
    oof_root = (
        run_config.paths.artifact_root
        / "oof"
        / config.temporal_version
        / configuration_sha256[:12]
    )
    print(
        f"Phase 10 full OOF candidate {experiment.name}: sliding inference",
        flush=True,
    )
    neural_paths, row_count, inference_seconds = build_temporal_oof(
        manifest=manifest,
        cache_root=cache_root,
        checkpoint_root=checkpoint_root,
        temporal_config_path=config_path,
        baseline_oof_manifest=baseline_manifest,
        output_root=oof_root,
    )
    oof_manifest = {
        "schema_version": 1,
        "model_version": config.temporal_version,
        "configuration_sha256": configuration_sha256,
        "row_count": row_count,
        "parts": [path.as_posix() for path in neural_paths],
        "held_out_labels_used_for_training_or_early_stopping": False,
    }
    oof_manifest_path = oof_root / "manifest.json"
    oof_manifest_path.write_text(
        json.dumps(oof_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
    validation_keys = [
        (str(row["lab_id"]), int(row["video_id"]))
        for row in folds.filter(pl.col("is_primary_validation")).iter_rows(named=True)
    ]
    solution = build_validation_solution(
        videos_path=videos_path,
        dataset_root=run_config.paths.dataset_root,
        validation_videos=validation_keys,
    )
    phase8_config_path = run_config.config_file.parent / "phase8.yaml"
    phase8_config = load_phase8_config(phase8_config_path)
    postprocessed = nested_interval_postprocessing(
        solution=solution,
        baseline_submission=decoded,
        videos=videos,
        folds=folds,
        baseline_parameters=thresholds,
        config=phase8_config.decoder,
        baseline_smoothing_seconds=config.decoder.smoothing_seconds,
    )
    validated = validate_submission(
        postprocessed.submission,
        videos=videos,
        triplets=triplets,
        allowed_splits=("train",),
    )
    candidate_root = phase10_root / "full_oof" / experiment.name
    candidate_root.mkdir(parents=True, exist_ok=True)
    submission_path = candidate_root / "held_out_submission.csv"
    validated.write_csv(submission_path)
    decoder_path = candidate_root / "decoder_parameters.parquet"
    postprocessed.parameters.write_parquet(
        decoder_path, compression="zstd", statistics=True
    )
    by_lab_action_path = candidate_root / "cv_by_lab_action.csv"
    postprocessed.breakdown.by_lab_action.to_csv(by_lab_action_path, index=False)
    comparison = _per_lab_action_comparison(
        postprocessed.breakdown.by_lab_action,
        phase8_root / "cv_by_lab_action.csv",
    )
    comparison_path = candidate_root / "gbdt_comparison_by_lab_action.csv"
    comparison.to_csv(comparison_path, index=False)
    diversity = _probability_diversity(neural_paths, baseline_manifest)
    phase9_report = json.loads(
        (
            run_config.paths.artifact_root
            / "temporal"
            / "temporal-phase9-v1"
            / "phase9_report.json"
        ).read_text(encoding="utf-8")
    )
    phase9_fbeta = float(phase9_report["neural_overall_fbeta"])
    exact_fbeta = float(postprocessed.breakdown.overall)
    retained = (
        exact_fbeta - phase9_fbeta
        >= phase10.screen.minimum_exact_fbeta_gain_over_phase9
    )
    report = {
        "schema_version": 1,
        "experiment": experiment.name,
        "configuration_sha256": configuration_sha256,
        "architecture_config": config.architecture.model_dump(mode="json"),
        "landmarks": list(config.data.landmarks),
        "window_frames": config.data.window_frames,
        "augmentation_config": config.augmentation.model_dump(mode="json"),
        "soft_f1_weight": config.training.soft_f1_weight,
        "folds": [asdict(result) | {"checkpoint_path": result.checkpoint_path.as_posix()} for result in fold_results],
        "oof_manifest": oof_manifest_path.as_posix(),
        "oof_rows": row_count,
        "exact_oof_fbeta": exact_fbeta,
        "phase9_exact_oof_fbeta": phase9_fbeta,
        "exact_fbeta_delta_over_phase9": exact_fbeta - phase9_fbeta,
        "retained": retained,
        "retention_reason": (
            "cleared exact OOF improvement gate"
            if retained
            else "did not clear exact OOF improvement gate"
        ),
        "diversity": diversity,
        "inference_seconds": inference_seconds,
        "projected_200_video_seconds": inference_seconds / manifest.height * 200,
        "decoder_used_held_out_labels": bool(
            postprocessed.parameters["used_held_out_labels"].any()
        ),
        "runtime_seconds": time.perf_counter() - started,
    }
    report_path = candidate_root / "model_card.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for path, rows, schema in (
        (oof_manifest_path, row_count, None),
        (submission_path, validated.height, None),
        (
            decoder_path,
            postprocessed.parameters.height,
            schema_fingerprint(postprocessed.parameters),
        ),
        (by_lab_action_path, len(postprocessed.breakdown.by_lab_action), None),
        (comparison_path, len(comparison), None),
        (report_path, None, None),
    ):
        _sidecar(
            path,
            configuration_sha256=configuration_sha256,
            source_fingerprint=source_fingerprint,
            row_count=rows,
            schema_sha256=schema,
        )
    return {
        "experiment": experiment.name,
        "exact_oof_fbeta": exact_fbeta,
        "probability_correlation_with_gbdt": diversity["probability_correlation"],
        "error_jaccard_with_gbdt": diversity["error_jaccard"],
        "retained": retained,
        "retention_reason": report["retention_reason"],
        "model_card_path": report_path.as_posix(),
        "oof_manifest_path": oof_manifest_path.as_posix(),
    }


def run_phase10_screens(
    *,
    run_config: RunConfig,
    phase10_config_path: str | Path,
    pose_config_path: str | Path,
    source_fingerprint: str,
) -> Phase10Outputs:
    started = time.perf_counter()
    config_path = Path(phase10_config_path)
    config = load_phase10_config(config_path)
    configuration_sha = phase10_config_fingerprint(config_path)
    base_path = config_path.parent / config.base_temporal_config
    base = load_phase9_config(base_path)
    output_root = run_config.paths.artifact_root / "phase10" / config.phase10_version
    output_root.mkdir(parents=True, exist_ok=True)
    hard_negatives = pl.read_parquet(
        run_config.paths.artifact_root
        / "phase8"
        / "gbdt-phase8-v1"
        / "hard_negative_plan.parquet"
    )
    records: list[dict[str, Any]] = []
    control_ap: float | None = None
    failures: dict[str, int] = {}
    eligible: list[tuple[float, str]] = []
    for index, experiment in enumerate(config.experiments, start=1):
        experiment_config = _experiment_phase9_config(base, config, experiment)
        generated_path = output_root / "generated_configs" / f"{experiment.name}.yaml"
        experiment_sha = _write_generated_config(experiment_config, generated_path)
        failure_count = failures.get(experiment.family, 0)
        if (
            experiment.family != "control"
            and failure_count >= config.screen.failures_before_family_stop
        ):
            records.append(
                _result_record(
                    experiment=experiment,
                    config=experiment_config,
                    config_path=generated_path,
                    result=None,
                    status="skipped_family_stop",
                    control_ap=control_ap,
                    reason=(
                        f"stopped after {failure_count} fixed-fold configurations failed "
                        "the retention screen"
                    ),
                )
            )
            continue
        print(
            f"Phase 10 screen {index}/{len(config.experiments)}: {experiment.name}",
            flush=True,
        )
        manifest, cache_root = _manifest_and_cache(
            config=experiment_config,
            config_path=generated_path,
            base=base,
            run_config=run_config,
            pose_config_path=Path(pose_config_path),
            source_fingerprint=source_fingerprint,
        )
        checkpoint_root = output_root / "experiments" / experiment.name / "checkpoints"
        checkpoint = checkpoint_root / f"fold_{config.screen.held_out_fold}" / "model.pt"
        card = checkpoint.parent / "model_card.json"
        existing = json.loads(card.read_text(encoding="utf-8")) if card.is_file() else {}
        if (
            checkpoint.is_file()
            and card.is_file()
            and existing.get("configuration_sha256") == experiment_sha
        ):
            result = _load_training_result(card, checkpoint)
        else:
            result = train_temporal_fold(
                held_out_fold=config.screen.held_out_fold,
                manifest=manifest,
                cache_root=cache_root,
                hard_negatives=hard_negatives,
                output_root=checkpoint_root,
                config=experiment_config,
                configuration_sha256=experiment_sha,
            )
        if experiment.family == "control":
            control_ap = result.validation_average_precision
            status = "control"
            reason = "shared fixed-fold reference"
        else:
            assert control_ap is not None
            gain = result.validation_average_precision - control_ap
            if gain >= config.screen.minimum_average_precision_gain:
                status = "screen_pass"
                reason = f"average-precision gain {gain:.8f} cleared the screen"
                eligible.append((result.validation_average_precision, experiment.name))
            else:
                status = "screen_fail"
                reason = f"average-precision gain {gain:.8f} did not clear the screen"
                failures[experiment.family] = failure_count + 1
        records.append(
            _result_record(
                experiment=experiment,
                config=experiment_config,
                config_path=generated_path,
                result=result,
                status=status,
                control_ap=control_ap,
                reason=reason,
            )
        )
    eligible = sorted(eligible, reverse=True)[
        : config.screen.maximum_full_oof_candidates
    ]
    experiment_lookup = {value.name: value for value in config.experiments}
    full_evaluations: list[dict[str, Any]] = []
    for _, name in eligible:
        experiment = experiment_lookup[name]
        experiment_config = _experiment_phase9_config(base, config, experiment)
        generated_path = output_root / "generated_configs" / f"{name}.yaml"
        experiment_sha = _write_generated_config(experiment_config, generated_path)
        manifest, cache_root = _manifest_and_cache(
            config=experiment_config,
            config_path=generated_path,
            base=base,
            run_config=run_config,
            pose_config_path=Path(pose_config_path),
            source_fingerprint=source_fingerprint,
        )
        evaluation = _run_full_oof_gate(
            experiment=experiment,
            config=experiment_config,
            config_path=generated_path,
            configuration_sha256=experiment_sha,
            manifest=manifest,
            cache_root=cache_root,
            checkpoint_root=output_root / "experiments" / name / "checkpoints",
            hard_negatives=hard_negatives,
            run_config=run_config,
            phase10_root=output_root,
            phase10=config,
            source_fingerprint=source_fingerprint,
        )
        full_evaluations.append(evaluation)
        for record in records:
            if record["experiment"] == name:
                record.update(
                    {
                        "exact_oof_fbeta": evaluation["exact_oof_fbeta"],
                        "probability_correlation_with_gbdt": evaluation[
                            "probability_correlation_with_gbdt"
                        ],
                        "error_jaccard_with_gbdt": evaluation[
                            "error_jaccard_with_gbdt"
                        ],
                        "retained": evaluation["retained"],
                    }
                )
                break
    augmentation_reference = next(
        (
            float(record["validation_average_precision"])
            for record in records
            if record["experiment"] == "no_augmentation"
            and record["validation_average_precision"] is not None
        ),
        None,
    )
    if augmentation_reference is not None:
        for record in records:
            if (
                record["family"] == "augmentation"
                and record["validation_average_precision"] is not None
            ):
                record["augmentation_delta_vs_no_augmentation"] = (
                    float(record["validation_average_precision"])
                    - augmentation_reference
                )
    results = pl.DataFrame(records).sort("experiment")
    results_path = output_root / "experiment_results.parquet"
    results.write_parquet(results_path, compression="zstd", statistics=True)
    screen_scores = {name: score for score, name in eligible}
    retained_registry = pl.DataFrame(
        [
            {
                "experiment": str(value["experiment"]),
                "screen_average_precision": screen_scores[str(value["experiment"])],
                "exact_oof_fbeta": float(value["exact_oof_fbeta"]),
                "retained": bool(value["retained"]),
                "retention_reason": str(value["retention_reason"]),
                "model_card_path": str(value["model_card_path"]),
                "oof_manifest_path": str(value["oof_manifest_path"]),
            }
            for value in full_evaluations
        ],
        schema={
            "experiment": pl.String,
            "screen_average_precision": pl.Float64,
            "exact_oof_fbeta": pl.Float64,
            "retained": pl.Boolean,
            "retention_reason": pl.String,
            "model_card_path": pl.String,
            "oof_manifest_path": pl.String,
        },
    )
    retained_registry_path = output_root / "retained_model_registry.parquet"
    retained_registry.write_parquet(
        retained_registry_path, compression="zstd", statistics=True
    )
    family_summary = (
        results.group_by("family", "status").len().sort("family", "status").to_dicts()
    )
    augmentation = results.filter(pl.col("family") == "augmentation")
    no_augmentation = augmentation.filter(pl.col("experiment") == "no_augmentation")
    other_augmentations = augmentation.filter(
        pl.col("experiment") != "no_augmentation"
    )
    no_augmentation_ap = (
        float(no_augmentation["validation_average_precision"][0])
        if no_augmentation.height
        else None
    )
    best_augmentation_ap = (
        float(cast(float, other_augmentations["validation_average_precision"].max()))
        if other_augmentations.height
        else None
    )
    tta_reason = (
        "not run: no individual augmentation improved validation over no augmentation"
        if no_augmentation_ap is not None
        and best_augmentation_ap is not None
        and best_augmentation_ap <= no_augmentation_ap
        else "not run: no qualifying inference-symmetric augmentation"
    )
    retained_count = int(retained_registry["retained"].sum()) if retained_registry.height else 0
    report = {
        "schema_version": 1,
        "phase10_version": config.phase10_version,
        "configuration_sha256": configuration_sha,
        "created_at": datetime.now(UTC).isoformat(),
        "control_average_precision": control_ap,
        "no_augmentation_average_precision": no_augmentation_ap,
        "best_individual_augmentation_average_precision": best_augmentation_ap,
        "screen_fold": config.screen.held_out_fold,
        "experiments_configured": len(config.experiments),
        "experiments_trained": int(
            results.filter(pl.col("status") != "skipped_family_stop").height
        ),
        "family_summary": family_summary,
        "eligible_for_full_oof": [name for _, name in eligible],
        "full_oof_evaluations": full_evaluations,
        "retained_models": retained_count,
        "conditional_experiments": {
            "st_gcn": "not run: CPU-only host and Phase 9 temporal baseline was far below GBDT",
            "vanilla_transformer": "not run: deliberately low priority after prior negative evidence",
            "mabe22_pretraining": "not run: deliberately low priority and no pretrained lineage was supplied",
            "tta": tta_reason,
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    report_path = output_root / "phase10_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for path, rows, schema in (
        (results_path, results.height, schema_fingerprint(results)),
        (
            retained_registry_path,
            retained_registry.height,
            schema_fingerprint(retained_registry),
        ),
        (report_path, None, None),
    ):
        _sidecar(
            path,
            configuration_sha256=configuration_sha,
            source_fingerprint=source_fingerprint,
            row_count=rows,
            schema_sha256=schema,
        )
    return Phase10Outputs(
        report_path=report_path,
        results_path=results_path,
        retained_registry_path=retained_registry_path,
        retained_models=retained_count,
    )
