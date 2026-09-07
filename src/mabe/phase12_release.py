"""Phase 12 release bundling, integrity verification, and held-out replay."""

from __future__ import annotations

import importlib.metadata
import json
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import polars as pl

from mabe.decode import decode_probabilities_deployment
from mabe.fingerprints import sha256_file
from mabe.labels import build_validation_solution
from mabe.metric import mouse_fbeta
from mabe.phase12_config import Phase12Config
from mabe.phase12_inference import (
    InferenceManifests,
    _model_path,
    _predict_gbdt_video,
    _predict_temporal_video,
    assemble_resilient_submission,
    blend_probabilities,
    build_test_manifests_from_sources,
    compose_crossfit_decoder_parameters,
    freeze_deployment_parameters,
    predict_decode_video_resilient,
)
from mabe.schemas import SUBMISSION_COLUMNS, SUBMISSION_SCHEMA
from mabe.submission import validate_submission


def _copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"missing release asset: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_project_file(project_root: Path, bundle_root: Path, relative: str) -> None:
    _copy(project_root / relative, bundle_root / relative)


def _relative_artifact_path(path: Path, project_root: Path) -> Path:
    try:
        return path.resolve().relative_to(project_root.resolve())
    except ValueError:
        lowered = [part.lower() for part in path.parts]
        if "artifacts" not in lowered:
            raise ValueError(f"model path is outside the project and has no artifacts route: {path}") from None
        return Path(*path.parts[lowered.index("artifacts") :])


def prepare_phase12_bundle(config: Phase12Config, bundle_root: Path) -> Path:
    """Build a self-contained, no-install Kaggle input with relative model routes."""
    project_root = config.project_root
    freeze_deployment_parameters(config)
    bundle_root.mkdir(parents=True, exist_ok=True)
    # The notebook always loads this canonical route, regardless of which
    # versioned inference profile created the bundle.
    _copy(config.config_path, bundle_root / "configs/inference.yaml")
    asset_routes = (
        config.assets.pose_config,
        config.assets.feature_config,
        config.assets.xgboost_config,
        config.assets.temporal_config,
        config.assets.category_registry,
        config.assets.vocabulary,
        config.assets.temporal_lab_registry,
        config.assets.blend_weights,
        config.assets.decoder_parameters,
        config.assets.probability_decoder_selection,
        config.assets.deployment_weights,
        config.assets.deployment_decoder_parameters,
    )
    for raw in asset_routes:
        source = config.asset_path(raw)
        relative = Path(raw) if not Path(raw).is_absolute() else _relative_artifact_path(source, project_root)
        _copy(source, bundle_root / relative)
    if config.assets.final_model_registry is not None:
        source = config.asset_path(config.assets.final_model_registry)
        relative = (
            Path(config.assets.final_model_registry)
            if not Path(config.assets.final_model_registry).is_absolute()
            else _relative_artifact_path(source, project_root)
        )
        _copy(source, bundle_root / relative)
    inferred_assets = (
        config.asset_path(config.assets.category_registry).parent / "solo.json",
        config.asset_path(config.assets.category_registry).parent / "pair.json",
        config.asset_path(config.assets.temporal_lab_registry).parent
        / "checkpoint_registry.parquet",
    )
    for source in inferred_assets:
        _copy(source, bundle_root / _relative_artifact_path(source, project_root))
    for documentation_relative in (
        "docs/phase12_runbook.md",
        "notebooks/phase12_kaggle_inference.ipynb",
        "environments/kaggle-v170-gpu.json",
    ):
        source = project_root / documentation_relative
        if source.is_file():
            _copy(source, bundle_root / documentation_relative)
    for fold in config.ensemble.folds:
        checkpoint = (
            config.asset_path(config.assets.temporal_checkpoint_root)
            / f"fold_{fold}"
            / "model.pt"
        )
        _copy(checkpoint, bundle_root / _relative_artifact_path(checkpoint, project_root))
    normalization_root = project_root / "artifacts/pose/normalization"
    for path in sorted(normalization_root.glob("fold_*.json")):
        _copy(path, bundle_root / path.relative_to(project_root))

    source_registry_path = config.asset_path(config.assets.xgboost_registry)
    source_registry = pl.read_parquet(source_registry_path)
    # Model feature inventories live with the Phase 7/8 XGBoost run rather
    # than necessarily beside a promoted Phase 8 registry. Preserve every
    # represented model version for auditability without making inference
    # depend on a hard-coded registry layout.
    represented_versions: set[str] = set()
    for raw_model_path in source_registry["model_path"].drop_nulls().unique():
        model_path = _model_path(str(raw_model_path), project_root)
        if model_path is None:
            continue
        try:
            relative_model = model_path.resolve().relative_to(
                (project_root / "artifacts" / "models").resolve()
            )
        except ValueError:
            continue
        if relative_model.parts:
            represented_versions.add(relative_model.parts[0])
    for model_version in sorted(represented_versions):
        model_features = (
            project_root / "artifacts" / "xgb" / model_version / "model_features.json"
        )
        if model_features.is_file():
            _copy(
                model_features,
                bundle_root / _relative_artifact_path(model_features, project_root),
            )
    rewritten_rows: list[dict[str, object]] = []
    copied_models: set[Path] = set()
    for row in source_registry.iter_rows(named=True):
        source_model = _model_path(row.get("model_path"), project_root)
        if source_model is not None:
            model_relative = _relative_artifact_path(source_model, project_root)
            if model_relative not in copied_models:
                _copy(source_model, bundle_root / model_relative)
                metadata = Path(f"{source_model}.meta.json")
                if metadata.is_file():
                    _copy(metadata, Path(f"{bundle_root / model_relative}.meta.json"))
                copied_models.add(model_relative)
            row["model_path"] = model_relative.as_posix()
        rewritten_rows.append(row)
    registry_destination = bundle_root / config.assets.xgboost_registry
    registry_destination.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rewritten_rows, infer_schema_length=None).write_parquet(
        registry_destination, compression="zstd", statistics=True
    )

    if config.assets.stacker_registry is not None:
        source_stacker_path = config.asset_path(config.assets.stacker_registry)
        source_stacker = pl.read_parquet(source_stacker_path)
        stacker_rows: list[dict[str, object]] = []
        copied_stackers: set[Path] = set()
        for row in source_stacker.iter_rows(named=True):
            source_model = _model_path(row.get("model_path"), project_root)
            if source_model is not None:
                model_relative = _relative_artifact_path(source_model, project_root)
                if model_relative not in copied_stackers:
                    _copy(source_model, bundle_root / model_relative)
                    copied_stackers.add(model_relative)
                row["model_path"] = model_relative.as_posix()
            stacker_rows.append(row)
        stacker_destination = bundle_root / config.assets.stacker_registry
        stacker_destination.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(stacker_rows, infer_schema_length=None).write_parquet(
            stacker_destination, compression="zstd", statistics=True
        )

    source_package = project_root / "src/mabe"
    destination_package = bundle_root / "src/mabe"
    if destination_package.is_dir():
        resolved_bundle = bundle_root.resolve()
        resolved_destination = destination_package.resolve()
        if resolved_bundle not in resolved_destination.parents:
            raise ValueError(f"refusing to refresh package outside bundle: {resolved_destination}")
        shutil.rmtree(destination_package)
    shutil.copytree(
        source_package,
        destination_package,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for project_relative in (
        "scripts/run_inference.py",
        "pyproject.toml",
        "requirements.txt",
    ):
        _copy_project_file(project_root, bundle_root, project_relative)

    dependencies: dict[str, str] = {}
    for name in ("numpy", "pandas", "polars", "pyarrow", "pydantic", "PyYAML", "torch", "xgboost"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = "missing"
    environment_path = bundle_root / "environment.json"
    environment_path.write_text(
        json.dumps(
            {
                "python": sys.version,
                "dependencies": dependencies,
                "runtime_installation_required": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    readme = bundle_root / "README.md"
    readme.write_text(
        "# MABe Phase 12 Kaggle bundle\n\n"
        "This directory is an inference-only Kaggle input. It contains the five-fold XGBoost "
        "and temporal models, frozen deployment parameters, feature/category registries, pose "
        "configuration, vocabulary, and source code. The notebook adds `src` to `sys.path`; it "
        "does not use the internet or install packages.\n",
        encoding="utf-8",
    )
    manifest_path = bundle_root / "bundle_manifest.json"
    for cache_path in sorted(bundle_root.rglob("__pycache__"), reverse=True):
        resolved_cache = cache_path.resolve()
        if bundle_root.resolve() not in resolved_cache.parents:
            raise ValueError(f"refusing to remove cache outside bundle: {resolved_cache}")
        shutil.rmtree(cache_path)
    files = sorted(
        path
        for path in bundle_root.rglob("*")
        if path.is_file() and path != manifest_path and path.suffix != ".pyc"
    )
    records = [
        {
            "path": path.relative_to(bundle_root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "inference_version": config.inference_version,
                "created_at": datetime.now(UTC).isoformat(),
                "file_count": len(records),
                "total_size_bytes": sum(path.stat().st_size for path in files),
                "runtime_installation_required": False,
                "files": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    verify_phase12_bundle(bundle_root)
    return manifest_path


def verify_phase12_bundle(bundle_root: Path) -> dict[str, Any]:
    """Fail before inference if a packaged file is absent or corrupted."""
    manifest_path = bundle_root / "bundle_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing bundle manifest: {manifest_path}")
    payload = cast(dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8")))
    files = cast(list[dict[str, object]], payload["files"])
    for record in files:
        path = bundle_root / str(record["path"])
        if not path.is_file():
            raise FileNotFoundError(f"bundle file is missing: {path}")
        if path.stat().st_size != int(str(record["size_bytes"])):
            raise ValueError(f"bundle file size changed: {path}")
        if sha256_file(path) != str(record["sha256"]):
            raise ValueError(f"bundle file hash changed: {path}")
    return payload


def run_heldout_replay_simulation(project_root: Path, output_root: Path) -> Path:
    """Revalidate and rescore the exact Phase 11 held-out interval output."""
    started = time.perf_counter()
    registry_path = project_root / "artifacts/phase11/ensemble-phase11-v1/final_model_registry.json"
    registry = cast(dict[str, Any], json.loads(registry_path.read_text(encoding="utf-8")))
    submission_path = Path(str(registry["held_out_submission"]["path"]))
    if not submission_path.is_file():
        submission_path = (
            project_root
            / "artifacts/phase11/ensemble-phase11-v1/candidates/phase11-crossfit-action/held_out_submission.csv"
        )
    videos_path = project_root / "artifacts/manifests/videos.parquet"
    triplets_path = project_root / "artifacts/manifests/triplets.parquet"
    folds_path = project_root / "artifacts/folds/folds.parquet"
    videos = pl.read_parquet(videos_path)
    triplets = pl.read_parquet(triplets_path)
    folds = pl.read_parquet(folds_path).filter(pl.col("is_primary_validation"))
    submission = pl.read_csv(submission_path)
    validate_submission(submission, videos=videos, triplets=triplets, allowed_splits=("train",))
    keys = [(str(row["lab_id"]), int(row["video_id"])) for row in folds.iter_rows(named=True)]
    dataset_root = project_root / "MABe-mouse-behavior-detection"
    solution = build_validation_solution(
        videos_path=videos_path,
        dataset_root=dataset_root,
        validation_videos=keys,
    )
    exact_score = float(mouse_fbeta(solution, submission.to_pandas(), beta=1.0))
    expected = float(registry["selected_exact_oof_fbeta"])
    tolerance = 1e-12
    if abs(exact_score - expected) > tolerance:
        raise AssertionError(f"held-out replay score changed: expected={expected}, actual={exact_score}")
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "heldout_replay_report.json"
    output_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "simulation_kind": "exact_heldout_interval_replay",
                "annotations_hidden_from_model_fitting": True,
                "raw_model_inference_recomputed": False,
                "video_count": len(keys),
                "interval_count": submission.height,
                "validation_error_count": 0,
                "expected_exact_fbeta": expected,
                "actual_exact_fbeta": exact_score,
                "absolute_difference": abs(exact_score - expected),
                "tolerance": tolerance,
                "passed": True,
                "runtime_seconds": time.perf_counter() - started,
                "submission_sha256": sha256_file(submission_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_path


def run_raw_heldout_simulation(config: Phase12Config, output_root: Path) -> Path:
    """Mask annotations and reproduce held-fold inference for 2/3/4-mouse videos."""
    started = time.perf_counter()
    project_root = config.project_root
    dataset_root = project_root / "MABe-mouse-behavior-detection"
    videos_path = project_root / "artifacts/manifests/videos.parquet"
    triplets_path = project_root / "artifacts/manifests/triplets.parquet"
    folds_path = project_root / "artifacts/folds/folds.parquet"
    videos = pl.read_parquet(videos_path)
    triplets = pl.read_parquet(triplets_path)
    folds = pl.read_parquet(folds_path).filter(pl.col("is_primary_validation"))
    primary = videos.join(
        folds.select("lab_id", "video_id", "fold"),
        on=["lab_id", "video_id"],
        how="inner",
    )
    selected_parts: list[pl.DataFrame] = []
    for mouse_count in (2, 3, 4):
        group = primary.filter(pl.col("mouse_count") == mouse_count)
        if group.is_empty():
            raise ValueError(f"no primary {mouse_count}-mouse video exists for simulation")
        median = float(cast(float, group["observed_frame_count"].median()))
        selected_parts.append(
            group.with_columns((pl.col("observed_frame_count") - median).abs().alias("_distance"))
            .sort("_distance", "lab_id", "video_id")
            .head(1)
            .drop("_distance")
        )
    selected = pl.concat(selected_parts, how="vertical_relaxed").sort("mouse_count")
    selected_keys = selected.select("lab_id", "video_id")
    masked_videos = (
        selected.drop("fold")
        .with_columns(
            pl.lit("test").alias("split"),
            pl.lit(None, dtype=pl.String).alias("annotation_path"),
            pl.lit(False).alias("annotation_exists"),
            pl.lit(0).alias("annotation_count"),
            pl.lit(0).alias("positive_frame_count"),
            pl.lit("test").alias("label_state"),
            pl.lit(False).alias("supervised_eligible"),
            pl.lit(0).alias("quarantined_annotation_count"),
        )
        .select(videos.columns)
    )
    masked_triplets = (
        triplets.join(selected_keys, on=["lab_id", "video_id"], how="semi")
        .with_columns(pl.lit("test").alias("split"))
        .select(triplets.columns)
    )
    output_root.mkdir(parents=True, exist_ok=True)
    masked_videos_path = output_root / "masked_videos.parquet"
    masked_triplets_path = output_root / "masked_triplets.parquet"
    masked_videos.write_parquet(masked_videos_path, compression="zstd", statistics=True)
    masked_triplets.write_parquet(masked_triplets_path, compression="zstd", statistics=True)
    manifests = InferenceManifests(
        masked_videos_path,
        masked_triplets_path,
        masked_videos,
        masked_triplets,
    )
    registry = pl.read_parquet(config.asset_path(config.assets.xgboost_registry))
    source_weights = pl.read_parquet(config.asset_path(config.assets.blend_weights)).filter(
        pl.col("strategy") == config.ensemble.weight_strategy
    )
    source_decoder = compose_crossfit_decoder_parameters(config)
    decoded_parts: list[pl.DataFrame] = []
    video_reports: list[dict[str, object]] = []
    for position, selected_row in enumerate(selected.to_dicts(), start=1):
        lab = str(selected_row["lab_id"])
        video_id = int(selected_row["video_id"])
        fold = int(selected_row["fold"])
        masked_row = masked_videos.filter(pl.col("video_id") == video_id).row(0, named=True)
        candidates = (
            masked_triplets.filter(
                (pl.col("lab_id") == lab) & (pl.col("video_id") == video_id) & ~pl.col("is_duplicate")
            )
            .select("agent_id", "target_id", "action", "is_self")
            .unique()
        )
        video_started = time.perf_counter()
        gbdt = _predict_gbdt_video(
            masked_row,
            candidates,
            registry,
            config=config,
            dataset_root=dataset_root,
            folds=(fold,),
        )
        temporal = _predict_temporal_video(
            masked_row,
            config=config,
            dataset_root=dataset_root,
            manifests=manifests,
            cache_root=output_root / "cache",
            folds=(fold,),
        )
        weights = source_weights.filter(pl.col("held_out_fold") == fold)
        decoder = source_decoder.filter(pl.col("held_out_fold") == fold)
        probability = blend_probabilities(gbdt, temporal, weights)
        decoded_pd = decode_probabilities_deployment(
            probability,
            decoder,
            fps_by_video={video_id: float(masked_row["frames_per_second"])},
        )
        decoded = (
            pl.DataFrame(decoded_pd).select(SUBMISSION_COLUMNS)
            if not decoded_pd.empty
            else pl.DataFrame(schema=SUBMISSION_SCHEMA).select(SUBMISSION_COLUMNS)
        )
        decoded_parts.append(decoded)
        video_reports.append(
            {
                "lab_id": lab,
                "video_id": video_id,
                "fold": fold,
                "mouse_count": int(selected_row["mouse_count"]),
                "frames": int(selected_row["observed_frame_count"]),
                "skeleton_id": str(selected_row["skeleton_id"]),
                "runtime_seconds": time.perf_counter() - video_started,
                "interval_count": decoded.height,
            }
        )
        print(
            f"Phase 12 raw simulation: {position}/{selected.height} completed {lab}/{video_id}",
            flush=True,
        )
    predicted_nonempty = [part for part in decoded_parts if not part.is_empty()]
    if not predicted_nonempty:
        raise ValueError("raw held-out simulation decoded no intervals")
    submission = (
        pl.concat(predicted_nonempty, how="vertical_relaxed")
        .drop("row_id")
        .with_row_index("row_id")
        .select(SUBMISSION_COLUMNS)
    )
    validate_submission(
        submission,
        videos=masked_videos,
        triplets=masked_triplets,
        allowed_splits=("test",),
    )
    submission_path = output_root / "raw_heldout_submission.csv"
    submission.write_csv(submission_path)
    selected_video_ids = selected["video_id"].to_list()
    final_registry = cast(
        dict[str, Any],
        json.loads(
            (project_root / "artifacts/phase11/ensemble-phase11-v1/final_model_registry.json").read_text(
                encoding="utf-8"
            )
        ),
    )
    expected_path = Path(str(final_registry["held_out_submission"]["path"]))
    expected = (
        pl.read_csv(expected_path)
        .filter(pl.col("video_id").is_in(selected_video_ids))
        .drop("row_id")
        .with_row_index("row_id")
        .select(SUBMISSION_COLUMNS)
    )
    interval_columns = [column for column in SUBMISSION_COLUMNS if column != "row_id"]
    interval_parity = submission.select(interval_columns).sort(interval_columns).equals(
        expected.select(interval_columns).sort(interval_columns)
    )
    solution = build_validation_solution(
        videos_path=videos_path,
        dataset_root=dataset_root,
        validation_videos=[(str(row["lab_id"]), int(row["video_id"])) for row in selected.iter_rows(named=True)],
    )
    actual_score = float(mouse_fbeta(solution, submission.to_pandas(), beta=1.0))
    expected_score = float(mouse_fbeta(solution, expected.to_pandas(), beta=1.0))
    if not interval_parity or abs(actual_score - expected_score) > 1e-12:
        raise AssertionError(
            "raw held-out inference does not reproduce the Phase 11 development route: "
            f"interval_parity={interval_parity}, actual={actual_score}, expected={expected_score}"
        )
    report_path = output_root / "raw_heldout_report.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "simulation_kind": "raw_tracking_annotation_masked_held_fold_inference",
                "annotations_available_to_inference": False,
                "raw_model_inference_recomputed": True,
                "video_count": selected.height,
                "mouse_counts": selected["mouse_count"].to_list(),
                "skeleton_count": selected["skeleton_id"].n_unique(),
                "interval_count": submission.height,
                "interval_parity_with_phase11": interval_parity,
                "expected_exact_fbeta": expected_score,
                "actual_exact_fbeta": actual_score,
                "absolute_difference": abs(actual_score - expected_score),
                "validation_error_count": 0,
                "passed": True,
                "runtime_seconds": time.perf_counter() - started,
                "submission_sha256": sha256_file(submission_path),
                "videos": video_reports,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return report_path


def run_dynamic_manifest_stress(config: Phase12Config, output_root: Path) -> Path:
    """Exercise hidden-test manifest construction at greater than hidden scale."""
    started = time.perf_counter()
    dataset_root = config.project_root / "MABe-mouse-behavior-detection"
    registry = pl.read_parquet(config.asset_path(config.assets.xgboost_registry))
    expected_labs = {str(value) for value in registry["lab_id"].unique().to_list()}
    manifests = build_test_manifests_from_sources(
        metadata_path=dataset_root / "train.csv",
        tracking_root=dataset_root / "train_tracking",
        dataset_root=dataset_root,
        output_root=output_root,
        allowed_labs=expected_labs,
    )
    manifest_report = cast(
        dict[str, Any],
        json.loads((output_root / "manifest_build_report.json").read_text(encoding="utf-8")),
    )
    actual_labs = {str(value) for value in manifests.videos["lab_id"].unique().to_list()}
    quarantine = cast(dict[str, int], manifest_report["quarantine_counts"])
    passed = (
        actual_labs == expected_labs
        and manifests.videos.height >= config.runtime.hidden_video_count
        and not manifests.triplets.filter(~pl.col("is_duplicate")).is_empty()
        and quarantine["failed_tracking_statistics"] == 0
        and quarantine["missing_tracking_rows"] == 0
    )
    report_path = output_root / "dynamic_manifest_stress_report.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stress_kind": "dynamic_test_manifest_from_full_competition_training_collection",
                "inference_version": config.inference_version,
                "hidden_video_target": config.runtime.hidden_video_count,
                "video_count": manifests.videos.height,
                "triplet_count": manifests.triplets.height,
                "lab_count": len(actual_labs),
                "expected_labs": sorted(expected_labs),
                "actual_labs": sorted(actual_labs),
                "quarantine_counts": quarantine,
                "videos_sha256": sha256_file(manifests.videos_path),
                "triplets_sha256": sha256_file(manifests.triplets_path),
                "runtime_seconds": time.perf_counter() - started,
                "passed": passed,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if not passed:
        raise RuntimeError(f"dynamic manifest stress failed; inspect {report_path}")
    return report_path


def _select_all_lab_stress_videos(config: Phase12Config) -> pl.DataFrame:
    """Select the highest-burden primary-validation video from every trained lab."""
    project_root = config.project_root
    videos = pl.read_parquet(project_root / "artifacts/manifests/videos.parquet")
    folds = pl.read_parquet(project_root / "artifacts/folds/folds.parquet").filter(
        pl.col("is_primary_validation")
    )
    primary = videos.join(
        folds.select("lab_id", "video_id", "fold"),
        on=["lab_id", "video_id"],
        how="inner",
    )
    selected = (
        primary.with_columns(
            (
                pl.col("observed_frame_count").fill_null(0).cast(pl.Int64)
                * pl.col("mouse_count").fill_null(0).cast(pl.Int64).clip(lower_bound=1)
                * pl.col("parsed_triplet_count").fill_null(0).cast(pl.Int64).clip(lower_bound=1)
            ).alias("_stress_rows")
        )
        .sort(
            "lab_id",
            "_stress_rows",
            "observed_frame_count",
            "video_id",
            descending=[False, True, True, False],
        )
        .unique(subset=["lab_id"], keep="first", maintain_order=True)
        .drop("_stress_rows")
        .sort("lab_id")
    )
    registry = pl.read_parquet(config.asset_path(config.assets.xgboost_registry))
    expected_labs = set(registry["lab_id"].unique().to_list())
    selected_labs = set(selected["lab_id"].to_list())
    if selected_labs != expected_labs:
        raise ValueError(
            "all-lab simulation selection mismatch: "
            f"missing={sorted(expected_labs - selected_labs)}, unexpected={sorted(selected_labs - expected_labs)}"
        )
    return selected


def _masked_all_lab_manifests(
    config: Phase12Config,
    selected: pl.DataFrame,
    output_root: Path,
) -> InferenceManifests:
    project_root = config.project_root
    videos = pl.read_parquet(project_root / "artifacts/manifests/videos.parquet")
    triplets = pl.read_parquet(project_root / "artifacts/manifests/triplets.parquet")
    selected_keys = selected.select("lab_id", "video_id")
    masked_videos = (
        selected.drop("fold")
        .with_columns(
            pl.lit("test").alias("split"),
            pl.lit(None, dtype=pl.String).alias("annotation_path"),
            pl.lit(False).alias("annotation_exists"),
            pl.lit(0).alias("annotation_count"),
            pl.lit(0).alias("positive_frame_count"),
            pl.lit("test").alias("label_state"),
            pl.lit(False).alias("supervised_eligible"),
            pl.lit(0).alias("quarantined_annotation_count"),
        )
        .select(videos.columns)
    )
    masked_triplets = (
        triplets.join(selected_keys, on=["lab_id", "video_id"], how="semi")
        .with_columns(pl.lit("test").alias("split"))
        .select(triplets.columns)
    )
    output_root.mkdir(parents=True, exist_ok=True)
    videos_path = output_root / "masked_videos.parquet"
    triplets_path = output_root / "masked_triplets.parquet"
    masked_videos.write_parquet(videos_path, compression="zstd", statistics=True)
    masked_triplets.write_parquet(triplets_path, compression="zstd", statistics=True)
    return InferenceManifests(videos_path, triplets_path, masked_videos, masked_triplets)


def run_all_lab_raw_simulation(
    config: Phase12Config,
    output_root: Path,
    *,
    deployment: bool,
) -> Path:
    """Exercise every trained lab with annotations hidden from both model families."""
    started = time.perf_counter()
    project_root = config.project_root
    dataset_root = project_root / "MABe-mouse-behavior-detection"
    selected = _select_all_lab_stress_videos(config)
    manifests = _masked_all_lab_manifests(config, selected, output_root)
    registry = pl.read_parquet(config.asset_path(config.assets.xgboost_registry))
    if deployment:
        deployment_weights = pl.read_parquet(config.asset_path(config.assets.deployment_weights))
        deployment_decoder = pl.read_parquet(config.asset_path(config.assets.deployment_decoder_parameters))
        source_weights = pl.DataFrame()
        source_decoder = pl.DataFrame()
    else:
        deployment_weights = pl.DataFrame()
        deployment_decoder = pl.DataFrame()
        source_weights = pl.read_parquet(config.asset_path(config.assets.blend_weights)).filter(
            pl.col("strategy") == config.ensemble.weight_strategy
        )
        source_decoder = compose_crossfit_decoder_parameters(config)
    interval_parts: list[pl.DataFrame] = []
    fallback_parts: list[pl.DataFrame] = []
    video_reports: list[dict[str, object]] = []
    for position, selected_row in enumerate(selected.to_dicts(), start=1):
        lab = str(selected_row["lab_id"])
        video_id = int(selected_row["video_id"])
        fold = int(selected_row["fold"])
        masked_row = manifests.videos.filter(
            (pl.col("lab_id") == lab) & (pl.col("video_id") == video_id)
        ).row(0, named=True)
        candidates = (
            manifests.triplets.filter(
                (pl.col("lab_id") == lab) & (pl.col("video_id") == video_id) & ~pl.col("is_duplicate")
            )
            .select("agent_id", "target_id", "action", "is_self")
            .unique()
        )
        weights = deployment_weights if deployment else source_weights.filter(pl.col("held_out_fold") == fold)
        decoder = deployment_decoder if deployment else source_decoder.filter(pl.col("held_out_fold") == fold)
        selected_folds = config.ensemble.folds if deployment else (fold,)
        video_started = time.perf_counter()
        result = predict_decode_video_resilient(
            masked_row,
            candidates,
            registry,
            weights,
            decoder,
            config=config,
            dataset_root=dataset_root,
            manifests=manifests,
            cache_root=output_root / "cache",
            folds=selected_folds,
        )
        interval_parts.append(result.decoded)
        fallback_parts.append(result.fallback)
        video_reports.append(
            {
                "lab_id": lab,
                "video_id": video_id,
                "fold": fold,
                "folds_executed": list(selected_folds),
                "mouse_count": int(selected_row["mouse_count"]),
                "frames": int(selected_row["observed_frame_count"]),
                "allowed_triplets": int(selected_row["parsed_triplet_count"]),
                "skeleton_id": str(selected_row["skeleton_id"]),
                "runtime_seconds": time.perf_counter() - video_started,
                "interval_count": result.decoded.height,
                "inference_route": result.inference_route,
                "failure_count": len(result.failures),
                "failures": list(result.failures),
                "stage_seconds": result.stage_seconds,
            }
        )
        print(
            f"Phase 12 all-lab {'deployment' if deployment else 'held-fold'} simulation: "
            f"{position}/{selected.height} completed {lab}/{video_id} via {result.inference_route}",
            flush=True,
        )
    submission, used_fallback = assemble_resilient_submission(interval_parts, fallback_parts, manifests)
    checked = validate_submission(
        submission,
        videos=manifests.videos,
        triplets=manifests.triplets,
        allowed_splits=("test",),
    )
    mode = "deployment" if deployment else "held-fold"
    submission_path = output_root / f"all_lab_{mode}_submission.csv"
    checked.write_csv(submission_path)
    solution = build_validation_solution(
        videos_path=project_root / "artifacts/manifests/videos.parquet",
        dataset_root=dataset_root,
        validation_videos=[
            (str(row["lab_id"]), int(row["video_id"])) for row in selected.iter_rows(named=True)
        ],
    )
    exact_score = float(mouse_fbeta(solution, checked.to_pandas(), beta=1.0))
    interval_parity: bool | None = None
    expected_score: float | None = None
    if not deployment:
        final_registry = cast(
            dict[str, Any],
            json.loads(
                (
                    project_root / "artifacts/phase11/ensemble-phase11-v1/final_model_registry.json"
                ).read_text(encoding="utf-8")
            ),
        )
        expected = (
            pl.read_csv(Path(str(final_registry["held_out_submission"]["path"])))
            .filter(pl.col("video_id").is_in(selected["video_id"]))
            .drop("row_id")
            .with_row_index("row_id")
            .select(SUBMISSION_COLUMNS)
        )
        interval_columns = [column for column in SUBMISSION_COLUMNS if column != "row_id"]
        interval_parity = checked.select(interval_columns).sort(interval_columns).equals(
            expected.select(interval_columns).sort(interval_columns)
        )
        expected_score = float(mouse_fbeta(solution, expected.to_pandas(), beta=1.0))
    full_blend_count = sum(report["inference_route"] == "blend" for report in video_reports)
    failure_count = sum(int(cast(int, report["failure_count"])) for report in video_reports)
    passed = (
        full_blend_count == selected.height
        and failure_count == 0
        and not used_fallback
        and (deployment or (interval_parity is True and abs(exact_score - cast(float, expected_score)) <= 1e-12))
    )
    report_path = output_root / f"all_lab_{mode}_report.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "simulation_kind": f"all_lab_{mode}_annotation_masked_raw_inference",
                "inference_version": config.inference_version,
                "annotations_available_to_inference": False,
                "deployment_five_fold_route": deployment,
                "video_count": selected.height,
                "lab_count": selected["lab_id"].n_unique(),
                "full_blend_count": full_blend_count,
                "degraded_or_skipped_count": selected.height - full_blend_count,
                "failure_count": failure_count,
                "used_global_fallback": used_fallback,
                "interval_count": checked.height,
                "interval_parity_with_phase11": interval_parity,
                "expected_exact_fbeta": expected_score,
                "actual_exact_fbeta": exact_score,
                "validation_error_count": 0,
                "passed": passed,
                "runtime_seconds": time.perf_counter() - started,
                "submission_sha256": sha256_file(submission_path),
                "videos": video_reports,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if not passed:
        raise RuntimeError(f"Phase 12 all-lab {mode} simulation failed; inspect {report_path}")
    return report_path


def build_runtime_benchmark(
    project_root: Path,
    inference_report_path: Path,
    output_root: Path,
    *,
    hidden_video_count: int,
    kaggle_limit_seconds: int,
    required_headroom_fraction: float,
) -> Path:
    """Extrapolate measured CPU stages across stratified primary-video geometry."""
    measured = cast(dict[str, Any], json.loads(inference_report_path.read_text(encoding="utf-8")))
    videos = pl.read_parquet(project_root / "artifacts/manifests/videos.parquet")
    folds = pl.read_parquet(project_root / "artifacts/folds/folds.parquet").filter(pl.col("is_primary_validation"))
    primary = videos.join(folds.select("lab_id", "video_id"), on=["lab_id", "video_id"], how="inner").with_columns(
        (pl.col("observed_frame_count") * pl.col("mouse_count").pow(2)).alias("geometry_units"),
        (pl.col("observed_frame_count") * pl.col("parsed_triplet_count")).alias("prediction_units"),
    )
    measured_video = cast(list[dict[str, object]], measured["videos"])[0]
    source_frames = int(str(measured_video["frames"]))
    source_mice = int(str(measured_video["mouse_count"]))
    source_video_id = int(str(measured_video["video_id"]))
    source_manifest = videos.filter(pl.col("video_id") == source_video_id).row(0, named=True)
    source_triplets = int(source_manifest["parsed_triplet_count"])
    source_geometry = source_frames * source_mice**2
    source_prediction = source_frames * source_triplets
    mean_geometry = float(cast(float, primary["geometry_units"].mean()))
    mean_prediction = float(cast(float, primary["prediction_units"].mean()))
    geometry_ratio = mean_geometry / source_geometry
    prediction_ratio = mean_prediction / source_prediction
    stages = cast(dict[str, float], measured["stage_seconds"])
    measured_runtime = float(measured["runtime_seconds"])
    fixed_seconds = max(0.0, measured_runtime - sum(float(value) for value in stages.values()))
    average_seconds = (
        float(stages["gbdt"]) * max(geometry_ratio, prediction_ratio)
        + float(stages["temporal"]) * geometry_ratio
        + float(stages["blend_decode"]) * prediction_ratio
        + fixed_seconds
    )
    distribution_projection = average_seconds * hidden_video_count
    contingency_projection = distribution_projection * 1.20
    naive_dense_projection = measured_runtime * hidden_video_count
    effective_budget = kaggle_limit_seconds * (1.0 - required_headroom_fraction)
    feature_benchmark = pl.read_parquet(project_root / "artifacts/features/benchmark.parquet")
    strata = (
        primary.group_by("mouse_count")
        .agg(
            pl.len().alias("video_count"),
            pl.col("observed_frame_count").mean().alias("mean_frames"),
            pl.col("parsed_triplet_count").mean().alias("mean_triplets"),
            pl.col("skeleton_id").n_unique().alias("skeleton_count"),
        )
        .sort("mouse_count")
    )
    output_root.mkdir(parents=True, exist_ok=True)
    strata_path = output_root / "runtime_strata.csv"
    strata.write_csv(strata_path)
    phase9_report = cast(
        dict[str, Any],
        json.loads(
            (project_root / "artifacts/temporal/temporal-phase9-v1/phase9_report.json").read_text(encoding="utf-8")
        ),
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "benchmark_device": str(measured["device"]),
        "cpu_available_and_measured": True,
        "gpu_available_and_measured": False,
        "gpu_note": (
            "No CUDA device is available on this host. The auto-device branch falls back "
            "to CPU; the measured CPU route alone passes the runtime gate."
        ),
        "raw_test_measurement": {
            "video_id": source_video_id,
            "frames": source_frames,
            "mouse_count": source_mice,
            "allowed_triplets": source_triplets,
            "runtime_seconds": measured_runtime,
            "peak_observed_rss_mb": float(str(measured_video["peak_observed_rss_mb"])),
            "stage_seconds": stages,
        },
        "stratification": {
            "reference_video_count": primary.height,
            "mouse_counts": sorted(int(value) for value in primary["mouse_count"].unique()),
            "skeleton_count": primary["skeleton_id"].n_unique(),
            "lab_count": primary["lab_id"].n_unique(),
            "minimum_frames": int(cast(int, primary["observed_frame_count"].min())),
            "median_frames": float(cast(float, primary["observed_frame_count"].median())),
            "mean_frames": float(cast(float, primary["observed_frame_count"].mean())),
            "maximum_frames": int(cast(int, primary["observed_frame_count"].max())),
            "feature_benchmark_video_count": feature_benchmark.height,
            "feature_benchmark_mouse_counts": sorted(int(value) for value in feature_benchmark["mouse_count"].unique()),
            "feature_benchmark_skeleton_count": feature_benchmark["skeleton_id"].n_unique(),
            "runtime_strata_csv": str(strata_path),
        },
        "scaling": {
            "mean_geometry_ratio_to_raw_test": geometry_ratio,
            "mean_prediction_ratio_to_raw_test": prediction_ratio,
            "estimated_mean_seconds_per_video": average_seconds,
            "distribution_projected_seconds": distribution_projection,
            "distribution_plus_20_percent_contingency_seconds": contingency_projection,
            "naive_200_dense_public_copies_seconds": naive_dense_projection,
            "historical_phase9_one_fold_inference_seconds_352_videos": float(phase9_report["inference_seconds"]),
            "historical_phase9_five_fold_projected_seconds_200_videos": float(phase9_report["inference_seconds"])
            / int(phase9_report["sequence_videos"])
            * 5
            * hidden_video_count,
        },
        "runtime_gate": {
            "hidden_video_count": hidden_video_count,
            "kaggle_limit_seconds": kaggle_limit_seconds,
            "required_headroom_fraction": required_headroom_fraction,
            "effective_budget_seconds": effective_budget,
            "gated_projection_seconds": contingency_projection,
            "projected_headroom_fraction": 1.0 - contingency_projection / kaggle_limit_seconds,
            "passed": contingency_projection <= effective_budget,
        },
    }
    if not bool(cast(dict[str, object], report["runtime_gate"])["passed"]):
        raise RuntimeError("Phase 12 stratified runtime benchmark failed its headroom gate")
    output_path = output_root / "runtime_benchmark.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path
