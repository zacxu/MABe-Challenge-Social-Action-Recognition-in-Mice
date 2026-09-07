"""Project command-line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from mabe.config import load_config
from mabe.environment import environment_report

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)


@app.command()
def smoke(
    config_path: Annotated[Path | None, typer.Option("--config", help="Run configuration YAML.")] = None,
    check_imports: Annotated[
        bool,
        typer.Option("--check-imports/--skip-import-check", help="Import every required dependency."),
    ] = True,
) -> None:
    """Validate configuration, dataset location, and the runtime environment."""
    config = load_config(config_path)
    train_csv = config.require_train_csv()
    report = {
        "dataset_root": str(config.paths.dataset_root),
        "artifact_root": str(config.paths.artifact_root),
        "cache_root": str(config.paths.cache_root),
        "model_root": str(config.paths.model_root),
        "train_csv": str(train_csv),
        "seed": config.seed,
        "dependencies": environment_report(name="current", validate_imports=check_imports),
    }
    typer.echo(json.dumps(report, indent=2, sort_keys=True))


@app.command("environment")
def capture_environment(
    output: Annotated[Path, typer.Option("--output", help="Destination JSON file.")],
    name: Annotated[str, typer.Option("--name", help="Environment name.")] = "local",
    check_imports: Annotated[
        bool,
        typer.Option("--check-imports/--skip-import-check", help="Import every required dependency."),
    ] = True,
) -> None:
    """Record an auditable dependency snapshot for local or Kaggle execution."""
    report = environment_report(name=name, validate_imports=check_imports)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    typer.echo(str(output.resolve()))


@app.command("audit-data")
def audit_data(
    config_path: Annotated[Path | None, typer.Option("--config", help="Run configuration YAML.")] = None,
    extreme_jump_pixels_per_frame: Annotated[
        float,
        typer.Option("--extreme-jump-pixels", min=0.0, help="Extreme displacement threshold per frame."),
    ] = 100.0,
) -> None:
    """Build the complete Phase 2 inventory and anomaly report."""
    from mabe.audit import run_data_audit

    config = load_config(config_path)
    outputs = run_data_audit(
        config,
        extreme_jump_pixels_per_frame=extreme_jump_pixels_per_frame,
    )
    typer.echo(
        json.dumps(
            {
                "videos": str(outputs.videos_path),
                "triplets": str(outputs.triplets_path),
                "report": str(outputs.report_json_path),
                "source_fingerprint": outputs.source_fingerprint,
                "facts": outputs.facts,
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.command("build-labels")
def build_labels(
    config_path: Annotated[Path | None, typer.Option("--config", help="Run configuration YAML.")] = None,
) -> None:
    """Build the versioned Phase 3 vocabulary and triplet-index artifacts."""
    from mabe.labels import build_label_artifacts

    config = load_config(config_path)
    manifests_root = config.paths.artifact_root / "manifests"
    report_path = config.paths.artifact_root / "reports" / "data_audit.json"
    if not report_path.is_file():
        raise FileNotFoundError("Phase 2 data_audit.json is required before building labels")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    outputs = build_label_artifacts(
        triplets_path=manifests_root / "triplets.parquet",
        output_root=config.paths.artifact_root / "labels",
        source_fingerprint=str(report["source_fingerprint"]),
    )
    typer.echo(
        json.dumps(
            {
                "action_count": len(outputs.vocabulary.actions),
                "pair_action_count": len(outputs.vocabulary.pair_actions),
                "solo_action_count": len(outputs.vocabulary.solo_actions),
                "triplet_mappings": str(outputs.triplet_mappings_path),
                "vocabulary": str(outputs.vocabulary_path),
                "vocabulary_version": outputs.vocabulary.version,
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.command("build-folds")
def build_folds(
    config_path: Annotated[Path | None, typer.Option("--config", help="Run configuration YAML.")] = None,
) -> None:
    """Build fixed leakage-safe Phase 4 folds and validation artifacts."""
    from mabe.validation import build_phase4_artifacts

    config = load_config(config_path)
    manifests_root = config.paths.artifact_root / "manifests"
    report_path = config.paths.artifact_root / "reports" / "data_audit.json"
    if not report_path.is_file():
        raise FileNotFoundError("Phase 2 data_audit.json is required before building folds")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    outputs = build_phase4_artifacts(
        videos_path=manifests_root / "videos.parquet",
        triplets_path=manifests_root / "triplets.parquet",
        source_files_path=manifests_root / "source_files.parquet",
        dataset_root=config.paths.dataset_root,
        output_root=config.paths.artifact_root / "folds",
        source_fingerprint=str(report["source_fingerprint"]),
        seed=config.seed,
    )
    typer.echo(
        json.dumps(
            {
                "auxiliary_sources": list(outputs.auxiliary_sources),
                "competition_labs": list(outputs.competition_labs),
                "duplicate_groups": str(outputs.duplicate_groups_path),
                "fold_quality": str(outputs.quality_report_path),
                "folds": str(outputs.folds_path),
                "support_distribution": str(outputs.support_distribution_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.command("build-poses")
def build_poses(
    config_path: Annotated[Path | None, typer.Option("--config", help="Run configuration YAML.")] = None,
    pose_config_path: Annotated[
        Path | None,
        typer.Option("--pose-config", help="Canonical pose configuration YAML."),
    ] = None,
) -> None:
    """Build Phase 5 pose-quality, cache, timing, and normalization artifacts."""
    from mabe.pose_audit import build_phase5_artifacts

    config = load_config(config_path)
    selected_pose_config = pose_config_path or config.config_file.parent / "pose.yaml"
    manifests_root = config.paths.artifact_root / "manifests"
    report_path = config.paths.artifact_root / "reports" / "data_audit.json"
    folds_path = config.paths.artifact_root / "folds" / "folds.parquet"
    if not report_path.is_file() or not folds_path.is_file():
        raise FileNotFoundError("Phase 2 audit and Phase 4 folds are required before building poses")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    outputs = build_phase5_artifacts(
        videos_path=manifests_root / "videos.parquet",
        source_files_path=manifests_root / "source_files.parquet",
        folds_path=folds_path,
        dataset_root=config.paths.dataset_root,
        pose_config_path=selected_pose_config,
        artifact_root=config.paths.artifact_root,
        cache_root=config.paths.cache_root,
        source_fingerprint=str(report["source_fingerprint"]),
    )
    typer.echo(
        json.dumps(
            {
                "adaptable_snail_audit": str(outputs.adaptable_snail_path),
                "family_count": outputs.family_count,
                "normalization_root": str(outputs.normalization_root),
                "pose_audit": str(outputs.pose_audit_path),
                "sample_cache_root": str(outputs.sample_cache_root),
                "sample_count": outputs.sample_count,
                "tracking_quality": str(outputs.tracking_quality_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.command("build-features")
def build_features(
    config_path: Annotated[Path | None, typer.Option("--config", help="Run configuration YAML.")] = None,
    pose_config_path: Annotated[
        Path | None,
        typer.Option("--pose-config", help="Canonical pose configuration YAML."),
    ] = None,
    feature_config_path: Annotated[
        Path | None,
        typer.Option("--feature-config", help="Feature configuration YAML."),
    ] = None,
) -> None:
    """Build Phase 6 feature registries, real-data caches, audit, and benchmark."""
    from mabe.feature_audit import build_phase6_artifacts

    config = load_config(config_path)
    selected_pose_config = pose_config_path or config.config_file.parent / "pose.yaml"
    selected_feature_config = feature_config_path or config.config_file.parent / "features.yaml"
    report_path = config.paths.artifact_root / "reports" / "data_audit.json"
    pose_quality_path = config.paths.artifact_root / "pose" / "tracking_quality.parquet"
    if not report_path.is_file() or not pose_quality_path.is_file():
        raise FileNotFoundError("Phase 2 audit and Phase 5 pose artifacts are required")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    outputs = build_phase6_artifacts(
        videos_path=config.paths.artifact_root / "manifests" / "videos.parquet",
        pose_quality_path=pose_quality_path,
        dataset_root=config.paths.dataset_root,
        pose_config_path=selected_pose_config,
        feature_config_path=selected_feature_config,
        artifact_root=config.paths.artifact_root,
        cache_root=config.paths.cache_root,
        source_fingerprint=str(report["source_fingerprint"]),
    )
    typer.echo(
        json.dumps(
            {
                "benchmark": str(outputs.benchmark_path),
                "family_count": outputs.family_count,
                "pair_feature_count": outputs.pair_feature_count,
                "registry_root": str(outputs.registry_root),
                "report": str(outputs.report_path),
                "sample_cache_root": str(outputs.sample_cache_root),
                "sample_count": outputs.sample_count,
                "solo_feature_count": outputs.solo_feature_count,
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.command("train-xgb-baseline")
def train_xgb_baseline(
    config_path: Annotated[Path | None, typer.Option("--config", help="Run configuration YAML.")] = None,
    pose_config_path: Annotated[
        Path | None,
        typer.Option("--pose-config", help="Canonical pose configuration YAML."),
    ] = None,
    feature_config_path: Annotated[
        Path | None,
        typer.Option("--feature-config", help="Feature configuration YAML."),
    ] = None,
    xgboost_config_path: Annotated[
        Path | None,
        typer.Option("--xgboost-config", help="Editable XGBoost baseline configuration YAML."),
    ] = None,
) -> None:
    """Train, infer, decode, and score the complete Phase 7 XGBoost baseline."""
    from mabe.xgb_pipeline import build_phase7_artifacts

    config = load_config(config_path)
    selected_pose_config = pose_config_path or config.config_file.parent / "pose.yaml"
    selected_feature_config = feature_config_path or config.config_file.parent / "features.yaml"
    selected_xgboost_config = xgboost_config_path or config.config_file.parent / "xgboost.yaml"
    report_path = config.paths.artifact_root / "reports" / "data_audit.json"
    category_path = config.paths.artifact_root / "features" / "registry" / "categories.json"
    required = (
        report_path,
        category_path,
        config.paths.artifact_root / "folds" / "folds.parquet",
    )
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("Phase 2, Phase 4, and Phase 6 artifacts are required")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    outputs = build_phase7_artifacts(
        videos_path=config.paths.artifact_root / "manifests" / "videos.parquet",
        triplets_path=config.paths.artifact_root / "manifests" / "triplets.parquet",
        folds_path=config.paths.artifact_root / "folds" / "folds.parquet",
        dataset_root=config.paths.dataset_root,
        pose_config_path=selected_pose_config,
        feature_config_path=selected_feature_config,
        category_registry_path=category_path,
        xgboost_config_path=selected_xgboost_config,
        artifact_root=config.paths.artifact_root,
        cache_root=config.paths.cache_root,
        model_root=config.paths.model_root,
        source_fingerprint=str(report["source_fingerprint"]),
    )
    typer.echo(
        json.dumps(
            {
                "cv_report": str(outputs.cv_report_path),
                "fake_submission": str(outputs.fake_submission_path),
                "model_count": outputs.model_count,
                "model_registry": str(outputs.model_registry_path),
                "oof_manifest": str(outputs.oof_manifest_path),
                "oof_row_count": outputs.oof_row_count,
                "overall_fbeta": outputs.overall_fbeta,
                "submission": str(outputs.submission_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.command("strengthen-gbdt")
def strengthen_gbdt(
    config_path: Annotated[Path | None, typer.Option("--config", help="Run configuration YAML.")] = None,
    phase8_config_path: Annotated[
        Path | None,
        typer.Option("--phase8-config", help="Phase 8 experiment/calibration YAML."),
    ] = None,
    pose_config_path: Annotated[
        Path | None,
        typer.Option("--pose-config", help="Canonical pose configuration YAML."),
    ] = None,
    feature_config_path: Annotated[
        Path | None,
        typer.Option("--feature-config", help="Feature configuration YAML."),
    ] = None,
    xgboost_config_path: Annotated[
        Path | None,
        typer.Option("--xgboost-config", help="Frozen Phase 7 XGBoost configuration YAML."),
    ] = None,
) -> None:
    """Run Phase 8 GBDT studies and promote a leakage-safe exact decoder."""
    from mabe.phase8_config import load_phase8_config
    from mabe.phase8_pipeline import build_phase8_artifacts

    config = load_config(config_path)
    selected_phase8 = phase8_config_path or config.config_file.parent / "phase8.yaml"
    selected_pose = pose_config_path or config.config_file.parent / "pose.yaml"
    selected_features = feature_config_path or config.config_file.parent / "features.yaml"
    selected_xgboost = xgboost_config_path or config.config_file.parent / "xgboost.yaml"
    phase8 = load_phase8_config(selected_phase8)
    report_path = config.paths.artifact_root / "reports" / "data_audit.json"
    required = (
        report_path,
        config.paths.artifact_root
        / "xgb"
        / phase8.baseline_model_version
        / "cv_report.json",
        config.paths.artifact_root / "features" / "registry" / "categories.json",
    )
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("completed Phase 2, Phase 6, and Phase 7 artifacts are required")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    outputs = build_phase8_artifacts(
        run_config=config,
        phase8_config_path=selected_phase8,
        xgboost_config_path=selected_xgboost,
        pose_config_path=selected_pose,
        feature_config_path=selected_features,
        source_fingerprint=str(report["source_fingerprint"]),
    )
    typer.echo(
        json.dumps(
            {
                "decoder_parameters": str(outputs.decoder_parameters_path),
                "final_model_registry": str(outputs.final_registry_path),
                "final_oof_manifest": str(outputs.final_oof_manifest_path),
                "final_submission": str(outputs.final_submission_path),
                "overall_fbeta": outputs.overall_fbeta,
                "phase8_report": str(outputs.report_path),
                "score_gain": outputs.score_gain,
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.command("train-temporal")
def train_temporal(
    config_path: Annotated[Path | None, typer.Option("--config", help="Run configuration YAML.")] = None,
    temporal_config_path: Annotated[
        Path | None,
        typer.Option("--temporal-config", help="Phase 9 data/model/training YAML."),
    ] = None,
    pose_config_path: Annotated[
        Path | None,
        typer.Option("--pose-config", help="Canonical pose configuration YAML."),
    ] = None,
    phase8_config_path: Annotated[
        Path | None,
        typer.Option("--phase8-config", help="Phase 8 decoder configuration YAML."),
    ] = None,
) -> None:
    """Build, train, infer, and evaluate the Phase 9 temporal model."""
    from mabe.phase8_config import load_phase8_config
    from mabe.phase9_pipeline import build_phase9_artifacts

    config = load_config(config_path)
    selected_temporal = temporal_config_path or config.config_file.parent / "temporal.yaml"
    selected_pose = pose_config_path or config.config_file.parent / "pose.yaml"
    selected_phase8 = phase8_config_path or config.config_file.parent / "phase8.yaml"
    phase8 = load_phase8_config(selected_phase8)
    report_path = config.paths.artifact_root / "reports" / "data_audit.json"
    required = (
        report_path,
        config.paths.artifact_root
        / "phase8"
        / phase8.phase8_version
        / "phase8_report.json",
        config.paths.artifact_root / "labels" / "action_vocabulary.json",
    )
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("completed Phase 2, Phase 3, and Phase 8 artifacts are required")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    outputs = build_phase9_artifacts(
        run_config=config,
        temporal_config_path=selected_temporal,
        pose_config_path=selected_pose,
        phase8_config_path=selected_phase8,
        source_fingerprint=str(report["source_fingerprint"]),
    )
    typer.echo(
        json.dumps(
            {
                "checkpoint_registry": str(outputs.checkpoint_registry_path),
                "decoder_parameters": str(outputs.decoder_parameters_path),
                "held_out_submission": str(outputs.submission_path),
                "oof_manifest": str(outputs.oof_manifest_path),
                "overall_fbeta": outputs.overall_fbeta,
                "phase9_report": str(outputs.report_path),
                "sequence_manifest": str(outputs.sequence_manifest_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.command("experiment-temporal")
def experiment_temporal(
    config_path: Annotated[Path | None, typer.Option("--config", help="Run configuration YAML.")] = None,
    phase10_config_path: Annotated[
        Path | None,
        typer.Option("--phase10-config", help="Phase 10 experiment configuration YAML."),
    ] = None,
    pose_config_path: Annotated[
        Path | None,
        typer.Option("--pose-config", help="Canonical pose configuration YAML."),
    ] = None,
) -> None:
    """Run bounded Phase 10 advanced temporal experiments."""
    from mabe.phase10_pipeline import run_phase10_screens

    config = load_config(config_path)
    selected_phase10 = phase10_config_path or config.config_file.parent / "advanced_temporal.yaml"
    selected_pose = pose_config_path or config.config_file.parent / "pose.yaml"
    audit_path = config.paths.artifact_root / "reports" / "data_audit.json"
    phase9_path = config.paths.artifact_root / "temporal" / "temporal-phase9-v1" / "phase9_report.json"
    if not audit_path.is_file() or not phase9_path.is_file():
        raise FileNotFoundError("completed Phase 2 and Phase 9 artifacts are required")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    outputs = run_phase10_screens(
        run_config=config,
        phase10_config_path=selected_phase10,
        pose_config_path=selected_pose,
        source_fingerprint=str(audit["source_fingerprint"]),
    )
    typer.echo(
        json.dumps(
            {
                "experiment_results": str(outputs.results_path),
                "phase10_report": str(outputs.report_path),
                "retained_model_registry": str(outputs.retained_registry_path),
                "retained_models": outputs.retained_models,
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.command("ensemble-oof")
def ensemble_oof(
    config_path: Annotated[Path | None, typer.Option("--config", help="Run configuration YAML.")] = None,
    phase11_config_path: Annotated[
        Path | None,
        typer.Option("--phase11-config", help="Phase 11 ensemble configuration YAML."),
    ] = None,
) -> None:
    """Align OOF streams, fit cross-fitted blends, and freeze the Phase 11 ensemble."""
    from mabe.phase11_pipeline import run_phase11_ensemble

    config = load_config(config_path)
    selected_phase11 = phase11_config_path or config.config_file.parent / "ensemble.yaml"
    audit_path = config.paths.artifact_root / "reports" / "data_audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError("completed Phase 2 artifacts are required")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    outputs = run_phase11_ensemble(
        run_config=config,
        phase11_config_path=selected_phase11,
        source_fingerprint=str(audit["source_fingerprint"]),
    )
    typer.echo(
        json.dumps(
            {
                "alignment_audit": str(outputs.alignment_audit_path),
                "blend_weight_registry": str(outputs.blend_weight_registry_path),
                "final_model_registry": str(outputs.final_registry_path),
                "final_oof_manifest": str(outputs.final_oof_manifest_path),
                "final_submission": str(outputs.final_submission_path),
                "overall_fbeta": outputs.overall_fbeta,
                "phase11_report": str(outputs.report_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


def main() -> None:
    app()
