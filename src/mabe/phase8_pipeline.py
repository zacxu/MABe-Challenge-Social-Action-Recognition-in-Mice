"""Phase 8 study orchestration, exact nested promotion, and artifact lineage."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import polars as pl
import yaml

from mabe.config import RunConfig
from mabe.fingerprints import schema_fingerprint, sha256_file
from mabe.labels import build_validation_solution
from mabe.phase8_config import load_phase8_config, phase8_config_fingerprint
from mabe.phase8_decoder import nested_interval_postprocessing
from mabe.phase8_probability_decoder import screen_probability_decoder
from mabe.phase8_study import StudyOutputs, run_phase8_studies
from mabe.submission import validate_submission
from mabe.xgb_config import load_xgboost_config, xgboost_config_fingerprint
from mabe.xgb_pipeline import Phase7Outputs, build_phase7_artifacts


@dataclass(frozen=True, slots=True)
class Phase8Outputs:
    report_path: Path
    final_submission_path: Path
    final_registry_path: Path
    final_oof_manifest_path: Path
    decoder_parameters_path: Path
    overall_fbeta: float
    score_gain: float


def _write_sidecar(
    path: Path,
    *,
    source_fingerprint: str,
    phase8_sha256: str,
    row_count: int | None,
    schema_hash: str | None,
) -> None:
    payload = {
        "artifact": path.name,
        "artifact_sha256": sha256_file(path),
        "configuration_sha256": phase8_sha256,
        "created_at": datetime.now(UTC).isoformat(),
        "row_count": row_count,
        "schema_sha256": schema_hash,
        "source_data_fingerprint": source_fingerprint,
    }
    Path(f"{path}.meta.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_study_artifacts(output_root: Path, study: StudyOutputs) -> list[tuple[Path, pl.DataFrame]]:
    frames = [
        (output_root / "representative_routes.parquet", study.representatives),
        (output_root / "feature_ablation.parquet", study.feature_ablation),
        (output_root / "hyperparameter_search.parquet", study.hyperparameter_search),
        (output_root / "hard_negative_plan.parquet", study.hard_negative_plan),
        (output_root / "hard_negative_comparison.parquet", study.hard_negative_comparison),
        (output_root / "feature_gain_importance.parquet", study.gain_importance),
        (output_root / "feature_group_stability.parquet", study.group_stability),
        (output_root / "permutation_importance.parquet", study.permutation_importance),
        (output_root / "route_comparison.parquet", study.route_comparison),
        (output_root / "nested_route_selection.parquet", study.route_selection),
        (output_root / "alternative_gbdt_screen.parquet", study.alternative_models),
    ]
    for path, frame in frames:
        frame.write_parquet(path, compression="zstd", statistics=True)
    return frames


def _candidate_summary(frame: pl.DataFrame) -> list[dict[str, object]]:
    if frame.is_empty():
        return []
    runtime_column = (
        "runtime_seconds_right" if "runtime_seconds_right" in frame.columns else "runtime_seconds"
    )
    return (
        frame.group_by("candidate")
        .agg(
            pl.col("macro_average_precision").first(),
            pl.col("macro_best_frame_f1").first(),
            pl.col(runtime_column).first().alias("runtime_seconds"),
        )
        .sort("macro_average_precision", descending=True)
        .to_dicts()
    )


def _cached_study(output_root: Path, phase8_sha: str) -> StudyOutputs | None:
    """Load completed study tables when their configuration provenance matches."""
    names = {
        "representatives": "representative_routes.parquet",
        "feature_ablation": "feature_ablation.parquet",
        "hyperparameter_search": "hyperparameter_search.parquet",
        "hard_negative_plan": "hard_negative_plan.parquet",
        "hard_negative_comparison": "hard_negative_comparison.parquet",
        "gain_importance": "feature_gain_importance.parquet",
        "group_stability": "feature_group_stability.parquet",
        "permutation_importance": "permutation_importance.parquet",
        "route_comparison": "route_comparison.parquet",
        "route_selection": "nested_route_selection.parquet",
        "alternative_models": "alternative_gbdt_screen.parquet",
    }
    paths = {key: output_root / name for key, name in names.items()}
    for path in paths.values():
        sidecar = Path(f"{path}.meta.json")
        if not path.is_file() or not sidecar.is_file():
            return None
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        if metadata.get("configuration_sha256") != phase8_sha:
            return None
        if metadata.get("artifact_sha256") != sha256_file(path):
            return None
    frames = {key: pl.read_parquet(path) for key, path in paths.items()}
    feature_winner = str(
        frames["feature_ablation"]
        .select("candidate", "macro_average_precision", "macro_best_frame_f1")
        .unique()
        .sort(
            ["macro_average_precision", "macro_best_frame_f1", "candidate"],
            descending=[True, True, False],
        )["candidate"][0]
    )
    hyperparameter_winner = str(
        frames["hyperparameter_search"]
        .select("candidate", "macro_average_precision", "macro_best_frame_f1")
        .unique()
        .sort(
            ["macro_average_precision", "macro_best_frame_f1", "candidate"],
            descending=[True, True, False],
        )["candidate"][0]
    )
    return StudyOutputs(
        **frames,
        selected_feature_candidate=feature_winner,
        selected_hyperparameter_candidate=hyperparameter_winner,
    )


def _train_selected_full_oof_candidate(
    *,
    study: StudyOutputs,
    phase8_config_path: Path,
    xgboost_config_path: Path,
    pose_config_path: Path,
    feature_config_path: Path,
    run_config: RunConfig,
    source_fingerprint: str,
    output_root: Path,
) -> Phase7Outputs | None:
    phase8 = load_phase8_config(phase8_config_path)
    promoted_version = phase8.promotion.promoted_model_version
    if not phase8.promotion.retrain_selected_model or promoted_version is None:
        return None
    baseline = load_xgboost_config(xgboost_config_path)
    hyperparameter = phase8.screen.hyperparameter_candidates[
        study.selected_hyperparameter_candidate
    ]
    feature = phase8.screen.feature_candidates[study.selected_feature_candidate]
    xgboost = baseline.xgboost.model_copy(update=hyperparameter.model_updates())
    sampler_updates: dict[str, object] = {}
    if hyperparameter.maximum_training_rows_per_route is not None:
        sampler_updates["max_rows_per_lab_action"] = (
            hyperparameter.maximum_training_rows_per_route
        )
    if hyperparameter.negative_to_positive_ratio is not None:
        sampler_updates["negative_to_positive_ratio"] = (
            hyperparameter.negative_to_positive_ratio
        )
    sampler = baseline.sampler.model_copy(update=sampler_updates)
    features = baseline.features.model_copy(
        update={"solo_groups": feature.solo, "pair_groups": feature.pair}
    )
    candidate = baseline.model_copy(
        update={
            "model_version": promoted_version,
            "xgboost": xgboost,
            "sampler": sampler,
            "features": features,
        }
    )
    if (
        xgboost == baseline.xgboost
        and sampler == baseline.sampler
        and features == baseline.features
    ):
        return None
    candidate_path = output_root / "selected_full_oof_xgboost.yaml"
    candidate_path.write_text(
        yaml.safe_dump(
            candidate.model_dump(mode="json"),
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return build_phase7_artifacts(
        videos_path=run_config.paths.artifact_root / "manifests" / "videos.parquet",
        triplets_path=run_config.paths.artifact_root / "manifests" / "triplets.parquet",
        folds_path=run_config.paths.artifact_root / "folds" / "folds.parquet",
        dataset_root=run_config.paths.dataset_root,
        pose_config_path=pose_config_path,
        feature_config_path=feature_config_path,
        category_registry_path=(
            run_config.paths.artifact_root / "features" / "registry" / "categories.json"
        ),
        xgboost_config_path=candidate_path,
        artifact_root=run_config.paths.artifact_root,
        cache_root=run_config.paths.cache_root,
        model_root=run_config.paths.model_root,
        source_fingerprint=source_fingerprint,
    )


def refresh_completed_phase8_report(output_root: str | Path) -> Path:
    """Normalize summaries and provenance for an already completed Phase 8 run."""
    root = Path(output_root)
    report_path = root / "phase8_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    feature = pl.read_parquet(root / "feature_ablation.parquet")
    hyperparameters = pl.read_parquet(root / "hyperparameter_search.parquet")
    hard_negatives = pl.read_parquet(root / "hard_negative_comparison.parquet")
    probability_search = pl.read_parquet(root / "probability_decoder_search.parquet")
    interval_search = pl.read_parquet(root / "decoder_search_summary.parquet")
    baseline_submission = pd.read_csv(
        root.parents[1]
        / "xgb"
        / str(report["baseline_model_version"])
        / "held_out_submission.csv"
    )
    final_submission = pd.read_csv(root / "held_out_submission.csv")
    report["feature_ablation_summary"] = _candidate_summary(feature)
    report["hyperparameter_summary"] = _candidate_summary(hyperparameters)
    report["hard_negative_summary"] = (
        hard_negatives.group_by("candidate")
        .agg(
            pl.col("macro_average_precision").first(),
            pl.col("macro_best_frame_f1").first(),
            pl.col("runtime_seconds_right").first().alias("runtime_seconds"),
        )
        .sort("macro_average_precision", descending=True)
        .to_dicts()
    )
    report["resource_deltas"] = {
        "additional_deployment_models": 0,
        "oof_probability_rows_before": int(report["oof_row_count"]),
        "oof_probability_rows_after": int(report["oof_row_count"]),
        "baseline_submission_intervals": len(baseline_submission),
        "final_submission_intervals": len(final_submission),
        "submission_interval_delta": len(final_submission) - len(baseline_submission),
        "probability_decoder_policies_evaluated": probability_search[
            "candidate"
        ].n_unique(),
        "exact_candidate_fold_evaluations": probability_search.height,
        "interval_candidates_per_component": (
            interval_search["gap_fill_seconds"].n_unique()
            * interval_search["minimum_run_seconds"].n_unique()
        ),
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sidecar_path = Path(f"{report_path}.meta.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    _write_sidecar(
        report_path,
        source_fingerprint=str(sidecar["source_data_fingerprint"]),
        phase8_sha256=str(sidecar["configuration_sha256"]),
        row_count=None,
        schema_hash=None,
    )
    return report_path


def build_phase8_artifacts(
    *,
    run_config: RunConfig,
    phase8_config_path: str | Path,
    xgboost_config_path: str | Path,
    pose_config_path: str | Path,
    feature_config_path: str | Path,
    source_fingerprint: str,
) -> Phase8Outputs:
    started = time.perf_counter()
    phase8_path = Path(phase8_config_path)
    phase8_config = load_phase8_config(phase8_path)
    baseline_config = load_xgboost_config(xgboost_config_path)
    phase8_sha = phase8_config_fingerprint(phase8_path)
    output_root = run_config.paths.artifact_root / "phase8" / phase8_config.phase8_version
    output_root.mkdir(parents=True, exist_ok=True)
    videos_path = run_config.paths.artifact_root / "manifests" / "videos.parquet"
    triplets_path = run_config.paths.artifact_root / "manifests" / "triplets.parquet"
    folds_path = run_config.paths.artifact_root / "folds" / "folds.parquet"
    category_path = run_config.paths.artifact_root / "features" / "registry" / "categories.json"
    baseline_root = run_config.paths.artifact_root / "xgb" / phase8_config.baseline_model_version
    study = _cached_study(output_root, phase8_sha)
    if study is None:
        study = run_phase8_studies(
            videos_path=videos_path,
            triplets_path=triplets_path,
            folds_path=folds_path,
            dataset_root=run_config.paths.dataset_root,
            cache_root=run_config.paths.cache_root,
            artifact_root=run_config.paths.artifact_root,
            pose_config_path=Path(pose_config_path),
            feature_config_path=Path(feature_config_path),
            category_registry_path=category_path,
            baseline_config=baseline_config,
            phase8_config=phase8_config,
            output_root=output_root,
        )
    study_artifacts = _write_study_artifacts(output_root, study)
    original_baseline_report = json.loads(
        (baseline_root / "cv_report.json").read_text(encoding="utf-8")
    )
    original_baseline_score = float(original_baseline_report["overall_fbeta"])
    full_candidate = _train_selected_full_oof_candidate(
        study=study,
        phase8_config_path=phase8_path,
        xgboost_config_path=Path(xgboost_config_path),
        pose_config_path=Path(pose_config_path),
        feature_config_path=Path(feature_config_path),
        run_config=run_config,
        source_fingerprint=source_fingerprint,
        output_root=output_root,
    )
    model_probability_promoted = bool(
        full_candidate is not None
        and full_candidate.overall_fbeta
        >= original_baseline_score
        + phase8_config.promotion.minimum_exact_model_gain
    )
    probability_root = (
        full_candidate.cv_report_path.parent
        if model_probability_promoted and full_candidate is not None
        else baseline_root
    )
    selected_model_version = (
        phase8_config.promotion.promoted_model_version
        if model_probability_promoted
        else phase8_config.baseline_model_version
    )
    selected_config_source = (
        output_root / "selected_full_oof_xgboost.yaml"
        if model_probability_promoted
        else Path(xgboost_config_path)
    )
    selected_config_path = output_root / "selected_xgboost.yaml"
    selected_config_path.write_bytes(selected_config_source.read_bytes())
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
    baseline_submission = pd.read_csv(probability_root / "held_out_submission.csv")
    baseline_parameters = pl.read_parquet(probability_root / "decoder_parameters.parquet")
    stable_baseline_manifest = probability_root / "oof_manifest.json"
    baseline_oof_manifests = (
        [stable_baseline_manifest]
        if stable_baseline_manifest.is_file()
        else sorted(
            (
                run_config.paths.artifact_root
                / "oof"
                / phase8_config.baseline_model_version
            ).glob("*/manifest.json")
        )
    )
    if len(baseline_oof_manifests) != 1:
        raise ValueError("Phase 8 requires exactly one baseline OOF manifest")
    source_oof = json.loads(baseline_oof_manifests[0].read_text(encoding="utf-8"))
    probability_screen = screen_probability_decoder(
        solution=solution,
        oof_paths=tuple(str(value) for value in source_oof["parts"]),
        videos=videos,
        folds=folds,
        baseline_parameters=baseline_parameters,
        config=phase8_config.decoder,
    )
    baseline_decoder_result = nested_interval_postprocessing(
        solution=solution,
        baseline_submission=baseline_submission,
        videos=videos,
        folds=folds,
        baseline_parameters=baseline_parameters,
        config=phase8_config.decoder,
        baseline_smoothing_seconds=baseline_config.decoder.smoothing_seconds,
    )
    candidate_decoder_result = nested_interval_postprocessing(
        solution=solution,
        baseline_submission=probability_screen.submission,
        videos=videos,
        folds=folds,
        baseline_parameters=probability_screen.route_parameters,
        config=phase8_config.decoder,
        baseline_smoothing_seconds=baseline_config.decoder.smoothing_seconds,
    )
    probability_changed = bool(
        probability_screen.nested_selection.filter(
            (pl.col("threshold_policy") != "nested_lab_action")
            | (pl.col("additional_smoothing_seconds") > 0)
            | (pl.col("hysteresis_delta") > 0)
        ).height
    )
    probability_decoder_promoted = (
        probability_changed
        and candidate_decoder_result.breakdown.overall
        > baseline_decoder_result.breakdown.overall
    )
    probability_dimensions_promoted: list[str] = []
    if probability_decoder_promoted:
        selected_probability = probability_screen.nested_selection
        if selected_probability.filter(
            pl.col("threshold_policy") != "nested_lab_action"
        ).height:
            probability_dimensions_promoted.append("threshold")
        if selected_probability.filter(
            pl.col("additional_smoothing_seconds") > 0
        ).height:
            probability_dimensions_promoted.append("centered_smoothing")
        if selected_probability.filter(pl.col("hysteresis_delta") > 0).height:
            probability_dimensions_promoted.append("hysteresis")
    decoder_result = (
        candidate_decoder_result if probability_decoder_promoted else baseline_decoder_result
    )
    decoder_parameters = decoder_result.parameters
    if probability_decoder_promoted:
        probability_values = probability_screen.nested_selection.select(
            "held_out_fold",
            "additional_smoothing_seconds",
            "hysteresis_delta",
            pl.col("candidate").alias("probability_decoder_candidate"),
        )
        decoder_parameters = (
            decoder_parameters.drop(
                "additional_smoothing_seconds", "hysteresis_delta"
            )
            .join(probability_values, on="held_out_fold", how="inner")
            .sort("held_out_fold", "lab_id", "action")
        )
    final_submission = validate_submission(
        decoder_result.submission,
        videos=videos,
        triplets=triplets,
        allowed_splits=("train",),
    )
    final_submission_path = output_root / "held_out_submission.csv"
    final_submission.write_csv(final_submission_path)
    decoder_parameters_path = output_root / "decoder_parameters.parquet"
    decoder_parameters.write_parquet(
        decoder_parameters_path, compression="zstd", statistics=True
    )
    decoder_search_path = output_root / "decoder_search_summary.parquet"
    decoder_result.search_summary.write_parquet(
        decoder_search_path, compression="zstd", statistics=True
    )
    probability_search_path = output_root / "probability_decoder_search.parquet"
    probability_screen.search_summary.write_parquet(
        probability_search_path, compression="zstd", statistics=True
    )
    probability_selection_path = output_root / "probability_decoder_selection.parquet"
    probability_screen.nested_selection.write_parquet(
        probability_selection_path, compression="zstd", statistics=True
    )
    by_fold_path = output_root / "cv_by_fold.csv"
    decoder_result.by_fold.to_csv(by_fold_path, index=False)
    by_lab_path = output_root / "cv_by_lab.csv"
    by_action_path = output_root / "cv_by_lab_action.csv"
    decoder_result.breakdown.by_lab.to_csv(by_lab_path, index=False)
    decoder_result.breakdown.by_lab_action.to_csv(by_action_path, index=False)
    selected_model_report = json.loads(
        (probability_root / "cv_report.json").read_text(encoding="utf-8")
    )
    selected_model_score = float(selected_model_report["overall_fbeta"])
    score_gain = decoder_result.breakdown.overall - original_baseline_score
    if score_gain < phase8_config.promotion.minimum_exact_fbeta_gain:
        promotion_status = "documented_plateau"
    else:
        promotion_status = "decoder_promoted"
    baseline_registry = pl.read_parquet(probability_root / "model_registry.parquet")
    final_registry = baseline_registry.join(
        study.route_selection.select(
            "held_out_fold",
            "lab_id",
            "action",
            pl.col("selected_route").alias("phase8_screen_route"),
            pl.col("promoted").alias("phase8_screen_route_passed"),
            pl.col("used_held_out_labels").alias("route_used_held_out_labels"),
        ),
        on=["held_out_fold", "lab_id", "action"],
        how="left",
    ).with_columns(
        pl.lit(phase8_config.phase8_version).alias("phase8_version"),
        pl.lit(model_probability_promoted).alias(
            "model_probability_change_promoted"
        ),
        pl.lit(
            "full_cross_fitted_selected_screen_candidate_promoted"
            if model_probability_promoted
            else "baseline_probability_stream_retained_after_exact_gate"
        ).alias("model_promotion_reason"),
    )
    final_registry_path = output_root / "final_model_registry.parquet"
    final_registry.write_parquet(final_registry_path, compression="zstd", statistics=True)
    final_oof_manifest = {
        **source_oof,
        "phase8_version": phase8_config.phase8_version,
        "source_model_version": selected_model_version,
        "probability_model_promoted": model_probability_promoted,
        "probability_lineage": (
            "Fully cross-fitted selected screen candidate"
            if model_probability_promoted
            else "Baseline routed OOF retained after exact promotion gate"
        ),
        "source_manifest": baseline_oof_manifests[0].as_posix(),
    }
    final_oof_manifest_path = output_root / "final_oof_manifest.json"
    final_oof_manifest_path.write_text(
        json.dumps(final_oof_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    alternative_summary = study.alternative_models.group_by("model_family").agg(
        pl.col("average_precision").mean().alias("macro_average_precision"),
        pl.col("best_frame_f1").mean().alias("macro_best_frame_f1"),
        pl.col("correlation_with_xgboost").mean().alias("mean_correlation_with_xgboost"),
        pl.col("runtime_seconds").sum().alias("runtime_seconds"),
        pl.col("retain_for_future_ensemble").sum().alias("retained_route_count"),
    ).sort("macro_average_precision", descending=True)
    hard_negative_summary = (
        study.hard_negative_comparison.group_by("candidate")
        .agg(
            pl.col("macro_average_precision").first(),
            pl.col("macro_best_frame_f1").first(),
            pl.col("runtime_seconds_right").first().alias("runtime_seconds"),
        )
        .sort("macro_average_precision", descending=True)
    )
    report = {
        "schema_version": 1,
        "phase8_version": phase8_config.phase8_version,
        "phase8_configuration_sha256": phase8_sha,
        "baseline_xgboost_configuration_sha256": xgboost_config_fingerprint(
            xgboost_config_path
        ),
        "baseline_model_version": phase8_config.baseline_model_version,
        "baseline_overall_fbeta": original_baseline_score,
        "selected_probability_model_fbeta": selected_model_score,
        "selected_xgboost_config": selected_config_path.as_posix(),
        "final_overall_fbeta": decoder_result.breakdown.overall,
        "exact_fbeta_gain": score_gain,
        "promotion_status": promotion_status,
        "selected_feature_screen_candidate": study.selected_feature_candidate,
        "selected_hyperparameter_screen_candidate": study.selected_hyperparameter_candidate,
        "model_probability_change_promoted": model_probability_promoted,
        "model_probability_promotion_reason": (
            "The selected screen configuration was fully cross-fitted and cleared the exact F-beta gate."
            if model_probability_promoted
            else "The selected screen resolved to the baseline configuration, so no redundant retraining was needed."
            if full_candidate is None
            else "The fully cross-fitted selected screen configuration did not clear the exact F-beta gate."
        ),
        "feature_ablation_summary": _candidate_summary(study.feature_ablation),
        "hyperparameter_summary": _candidate_summary(study.hyperparameter_search),
        "hard_negative_summary": hard_negative_summary.to_dicts(),
        "hard_negative_rows": study.hard_negative_plan.height,
        "route_screen_promotions": study.route_selection.filter(pl.col("promoted")).height,
        "route_selection_used_held_out_labels": bool(
            study.route_selection["used_held_out_labels"].any()
        ),
        "alternative_gbdt_summary": alternative_summary.to_dicts(),
        "decoder_calibration": "other_oof_folds_only_exact_official_components",
        "decoder_used_held_out_labels": bool(
            decoder_parameters["used_held_out_labels"].any()
            or probability_screen.nested_selection["used_held_out_labels"].any()
        ),
        "decoder_dimensions_evaluated": [
            "threshold",
            "centered_smoothing",
            "hysteresis",
            "short_gap_fill",
            "minimum_run_duration",
        ],
        "decoder_dimensions_promoted": [
            *probability_dimensions_promoted,
            "short_gap_fill",
            "minimum_run_duration",
        ],
        "probability_decoder_nested_exact_fbeta": probability_screen.overall_fbeta,
        "probability_decoder_promoted": probability_decoder_promoted,
        "probability_decoder_selection": probability_screen.nested_selection.to_dicts(),
        "resource_deltas": {
            "additional_deployment_models": 0,
            "oof_probability_rows_before": int(source_oof["row_count"]),
            "oof_probability_rows_after": int(source_oof["row_count"]),
            "baseline_submission_intervals": len(baseline_submission),
            "final_submission_intervals": final_submission.height,
            "submission_interval_delta": final_submission.height - len(baseline_submission),
            "probability_decoder_policies_evaluated": (
                probability_screen.search_summary["candidate"].n_unique()
            ),
            "exact_candidate_fold_evaluations": probability_screen.search_summary.height,
            "interval_candidates_per_component": (
                len(phase8_config.decoder.gap_fill_seconds)
                * len(phase8_config.decoder.minimum_run_seconds)
            ),
        },
        "runtime_seconds": time.perf_counter() - started,
        "final_submission_rows": final_submission.height,
        "oof_row_count": int(source_oof["row_count"]),
    }
    report_path = output_root / "phase8_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifact_frames = [
        *study_artifacts,
        (decoder_parameters_path, decoder_parameters),
        (decoder_search_path, decoder_result.search_summary),
        (probability_search_path, probability_screen.search_summary),
        (probability_selection_path, probability_screen.nested_selection),
        (final_registry_path, final_registry),
    ]
    for path, frame in artifact_frames:
        _write_sidecar(
            path,
            source_fingerprint=source_fingerprint,
            phase8_sha256=phase8_sha,
            row_count=frame.height,
            schema_hash=schema_fingerprint(frame),
        )
    for path, row_count in (
        (final_submission_path, final_submission.height),
        (by_fold_path, len(decoder_result.by_fold)),
        (by_lab_path, len(decoder_result.breakdown.by_lab)),
        (by_action_path, len(decoder_result.breakdown.by_lab_action)),
        (final_oof_manifest_path, int(source_oof["row_count"])),
        (report_path, None),
    ):
        _write_sidecar(
            path,
            source_fingerprint=source_fingerprint,
            phase8_sha256=phase8_sha,
            row_count=row_count,
            schema_hash=None,
        )
    return Phase8Outputs(
        report_path=report_path,
        final_submission_path=final_submission_path,
        final_registry_path=final_registry_path,
        final_oof_manifest_path=final_oof_manifest_path,
        decoder_parameters_path=decoder_parameters_path,
        overall_fbeta=decoder_result.breakdown.overall,
        score_gain=score_gain,
    )
