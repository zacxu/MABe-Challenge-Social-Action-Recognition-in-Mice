"""Final reproducibility audit, diagnostics, provenance, and release archive."""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
import shutil
import tempfile
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd
import polars as pl
import yaml

from mabe.diagnostics import metric_breakdown
from mabe.fingerprints import sha256_file
from mabe.labels import build_validation_solution
from mabe.metric import mouse_fbeta
from mabe.phase12_release import verify_phase12_bundle
from mabe.schemas import SUBMISSION_COLUMNS
from mabe.submission import validate_submission

CORE_DEPENDENCIES = (
    "numpy",
    "pandas",
    "polars",
    "pyarrow",
    "scipy",
    "scikit-learn",
    "xgboost",
    "torch",
    "pydantic",
    "PyYAML",
    "matplotlib",
    "psutil",
)


def _read_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return cast(dict[str, Any], payload)


def _resolve(path: str | Path, project_root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_file() or candidate.is_dir():
        return candidate.resolve()
    lowered = [part.lower() for part in candidate.parts]
    if "artifacts" in lowered:
        return project_root.joinpath(*candidate.parts[lowered.index("artifacts") :]).resolve()
    return (project_root / candidate).resolve()


def _relative(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _canonical_interval_content(frame: pl.DataFrame) -> pl.DataFrame:
    columns = [column for column in SUBMISSION_COLUMNS if column != "row_id"]
    return frame.select(columns).sort(columns)


def _aggregate_action_errors(by_lab_action: pd.DataFrame) -> pl.DataFrame:
    frame = pl.DataFrame(by_lab_action)
    result = (
        frame.group_by("action")
        .agg(
            pl.col("tp").sum(),
            pl.col("fp").sum(),
            pl.col("fn").sum(),
            pl.col("ground_truth_frames").sum(),
            pl.col("predicted_frames").sum(),
            pl.col("fbeta").mean().alias("macro_lab_fbeta"),
            pl.col("lab_id").n_unique().alias("lab_count"),
        )
        .with_columns(
            pl.when(pl.col("tp") + pl.col("fp") > 0)
            .then(pl.col("tp") / (pl.col("tp") + pl.col("fp")))
            .otherwise(0.0)
            .alias("micro_precision"),
            pl.when(pl.col("tp") + pl.col("fn") > 0)
            .then(pl.col("tp") / (pl.col("tp") + pl.col("fn")))
            .otherwise(0.0)
            .alias("micro_recall"),
            (pl.col("fp") + pl.col("fn")).alias("error_frames"),
        )
        .with_columns(
            pl.when(2 * pl.col("tp") + pl.col("fp") + pl.col("fn") > 0)
            .then(2 * pl.col("tp") / (2 * pl.col("tp") + pl.col("fp") + pl.col("fn")))
            .otherwise(0.0)
            .alias("micro_fbeta")
        )
        .sort("error_frames", descending=True)
    )
    total_errors = int(result["error_frames"].sum())
    return result.with_columns(
        (
            pl.col("error_frames") / total_errors
            if total_errors
            else pl.lit(0.0)
        ).alias("error_share")
    )


def _hash_record(path: Path, project_root: Path, *, role: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"missing release input: {path}")
    return {
        "path": _relative(path, project_root),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "role": role,
    }


def _write_plots(
    by_lab: pl.DataFrame,
    action_errors: pl.DataFrame,
    confidence_bins: pl.DataFrame,
    output_root: Path,
) -> None:
    matplotlib_cache = tempfile.mkdtemp(prefix="mabe-phase13-matplotlib-")
    previous_matplotlib_cache = os.environ.get("MPLCONFIGDIR")
    os.environ["MPLCONFIGDIR"] = matplotlib_cache
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_root.mkdir(parents=True, exist_ok=True)
    lab = by_lab.sort("fbeta")
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.barh(lab["lab_id"].to_list(), lab["fbeta"].to_list(), color="#31688e")
    axis.set(xlabel="Exact F-beta", title="Final cross-fitted OOF score by lab", xlim=(0, 1))
    figure.tight_layout()
    figure.savefig(output_root / "per_lab_fbeta.png", dpi=160)
    plt.close(figure)

    prevalence = action_errors.sort("ground_truth_frames", descending=True)
    positions = list(range(prevalence.height))
    figure, axis = plt.subplots(figsize=(13, 7))
    axis.bar(
        [position - 0.2 for position in positions],
        prevalence["ground_truth_frames"].to_list(),
        width=0.4,
        label="ground truth",
        color="#35b779",
    )
    axis.bar(
        [position + 0.2 for position in positions],
        prevalence["predicted_frames"].to_list(),
        width=0.4,
        label="predicted",
        color="#440154",
    )
    axis.set_yscale("symlog", linthresh=1)
    axis.set_xticks(positions, prevalence["action"].to_list(), rotation=70, ha="right")
    axis.set(ylabel="Frames (symlog)", title="Action prevalence: truth versus prediction")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_root / "action_prevalence.png", dpi=160)
    plt.close(figure)

    bins = confidence_bins.sort("bin_index")
    centers = ((bins["bin_lower"] + bins["bin_upper"]) / 2).to_list()
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="ideal")
    axis.plot(
        centers,
        bins["observed_prevalence"].to_list(),
        marker="o",
        color="#31688e",
        label="sampled OOF",
    )
    axis.set(
        xlabel="Predicted-probability bin midpoint",
        ylabel="Observed positive prevalence",
        title="Deterministic OOF confidence diagnostic",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_root / "confidence_calibration.png", dpi=160)
    plt.close(figure)
    if previous_matplotlib_cache is None:
        del os.environ["MPLCONFIGDIR"]
    else:
        os.environ["MPLCONFIGDIR"] = previous_matplotlib_cache
    shutil.rmtree(matplotlib_cache, ignore_errors=True)


def _sample_oof_confidence(
    parts: Iterable[Path],
    *,
    rows_per_part: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    samples: list[pl.DataFrame] = []
    for path in parts:
        sample = pl.read_parquet(
            path,
            columns=["action", "loss_mask", "target", "probability"],
            n_rows=rows_per_part,
        ).filter(pl.col("loss_mask") == 1)
        if not sample.is_empty():
            samples.append(sample)
    if not samples:
        raise ValueError("OOF confidence sample is empty")
    frame = pl.concat(samples, how="vertical_relaxed").with_columns(
        (pl.col("probability").clip(0.0, 1.0) * 10)
        .floor()
        .cast(pl.Int8)
        .clip(0, 9)
        .alias("bin_index")
    )
    bins = (
        frame.group_by("bin_index")
        .agg(
            pl.len().alias("sample_rows"),
            pl.col("probability").mean().alias("mean_probability"),
            pl.col("target").sum().alias("positive_rows"),
            pl.col("target").mean().alias("observed_prevalence"),
        )
        .with_columns(
            (pl.col("bin_index") / 10).alias("bin_lower"),
            ((pl.col("bin_index") + 1) / 10).alias("bin_upper"),
        )
        .sort("bin_index")
    )
    by_action = (
        frame.group_by("action")
        .agg(
            pl.len().alias("sample_rows"),
            pl.col("target").sum().alias("positive_rows"),
            pl.col("target").mean().alias("observed_prevalence"),
            pl.col("probability").mean().alias("mean_probability"),
            pl.col("probability").quantile(0.5).alias("median_probability"),
            pl.col("probability").quantile(0.9).alias("p90_probability"),
        )
        .sort("action")
    )
    return bins, by_action


def _dependency_records() -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for name in CORE_DEPENDENCIES:
        metadata = importlib.metadata.metadata(name)
        license_value = (
            metadata.get("License-Expression")
            or metadata.get("License")
            or "not declared in package metadata"
        )
        project_urls = metadata.get_all("Project-URL") or []
        records.append(
            {
                "name": name,
                "version": importlib.metadata.version(name),
                "license": license_value.strip(),
                "home_page": metadata.get("Home-page") or "; ".join(project_urls),
                "license_metadata_present": str(
                    license_value != "not declared in package metadata"
                ).lower(),
            }
        )
    return records


def _all_dependency_versions() -> list[dict[str, str]]:
    records = [
        {"name": distribution.metadata["Name"], "version": distribution.version}
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    ]
    return sorted(records, key=lambda row: row["name"].lower())


def _experimental_inventory(project_root: Path) -> pl.DataFrame:
    routes = (
        (project_root / "artifacts/phase10", "phase10_screen", "not retained by Phase 10 gate"),
        (
            project_root / "artifacts/temporal/temporal-advanced-phase10-v1-keypoints_4",
            "phase10_checkpoint",
            "not retained by exact OOF gate",
        ),
        (
            project_root / "artifacts/temporal/temporal-advanced-phase10-v1-keypoints_7",
            "phase10_checkpoint",
            "not retained by exact OOF gate",
        ),
    )
    rows: list[dict[str, object]] = []
    for path, artifact_class, reason in routes:
        files = [candidate for candidate in path.rglob("*") if candidate.is_file()] if path.is_dir() else []
        rows.append(
            {
                "path": _relative(path, project_root),
                "artifact_class": artifact_class,
                "reason_excluded_from_release": reason,
                "file_count": len(files),
                "size_bytes": sum(file.stat().st_size for file in files),
            }
        )
    candidate_root = project_root / "artifacts/phase11/ensemble-phase11-v1/candidates"
    for path in sorted(candidate_root.iterdir()):
        if not path.is_dir() or path.name == "phase11-crossfit-action":
            continue
        files = [candidate for candidate in path.rglob("*") if candidate.is_file()]
        rows.append(
            {
                "path": _relative(path, project_root),
                "artifact_class": "phase11_nonselected_candidate",
                "reason_excluded_from_release": "not selected by exact cross-fitted OOF",
                "file_count": len(files),
                "size_bytes": sum(file.stat().st_size for file in files),
            }
        )
    return pl.DataFrame(rows).sort("artifact_class", "path")


def _write_notebook_log(notebook_path: Path, output_path: Path) -> None:
    notebook = cast(dict[str, Any], json.loads(notebook_path.read_text(encoding="utf-8")))
    lines: list[str] = []
    for cell in cast(list[dict[str, Any]], notebook.get("cells", [])):
        for output in cast(list[dict[str, Any]], cell.get("outputs", [])):
            if output.get("output_type") == "stream":
                raw = output.get("text", [])
                lines.append("".join(raw) if isinstance(raw, list) else str(raw))
            if output.get("output_type") == "error":
                raise ValueError(f"executed notebook contains an error output: {output.get('ename')}")
    output_path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def build_phase13_release(
    project_root: Path,
    *,
    config_path: Path,
    output_root: Path,
) -> Path:
    """Build the final audit release entirely from frozen Phase 0-12 artifacts."""
    started_at = datetime.now(UTC)
    project_root = project_root.resolve()
    config_path = config_path.resolve()
    config = _read_mapping(config_path)
    paths = cast(dict[str, str], config["paths"])
    release_version = str(config["release_version"])
    expected_score = float(config["expected_exact_oof_fbeta"])
    tolerance = float(config["score_tolerance"])
    rows_per_part = int(config["confidence_rows_per_oof_part"])
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    reports_root = output_root / "reports"
    plots_root = output_root / "plots"
    archive_root = output_root / "archive"
    essential_root = output_root / "essential"
    for path in (reports_root, plots_root, archive_root, essential_root):
        path.mkdir(parents=True, exist_ok=True)

    final_registry_path = _resolve(paths["final_registry"], project_root)
    final_registry = cast(
        dict[str, Any], json.loads(final_registry_path.read_text(encoding="utf-8"))
    )
    submission_path = _resolve(paths["final_submission"], project_root)
    oof_manifest_path = _resolve(paths["final_oof_manifest"], project_root)
    if sha256_file(submission_path) != str(final_registry["held_out_submission"]["sha256"]):
        raise ValueError("frozen held-out submission hash differs from the final model registry")
    if sha256_file(oof_manifest_path) != str(final_registry["oof_manifest"]["sha256"]):
        raise ValueError("frozen OOF manifest hash differs from the final model registry")
    oof_manifest = cast(dict[str, Any], json.loads(oof_manifest_path.read_text(encoding="utf-8")))
    if bool(oof_manifest["held_out_labels_used_for_training_or_early_stopping"]):
        raise ValueError("final OOF manifest reports held-out-label leakage")
    part_paths = [_resolve(str(path), project_root) for path in cast(list[str], oof_manifest["parts"])]
    part_records = [_hash_record(path, project_root, role="final_oof_probability_part") for path in part_paths]
    pl.DataFrame(part_records).write_parquet(
        reports_root / "oof_part_inventory.parquet", compression="zstd", statistics=True
    )

    videos_path = _resolve(paths["videos"], project_root)
    triplets_path = _resolve(paths["triplets"], project_root)
    folds_path = _resolve(paths["folds"], project_root)
    videos = pl.read_parquet(videos_path)
    triplets = pl.read_parquet(triplets_path)
    primary = pl.read_parquet(folds_path).filter(pl.col("is_primary_validation"))
    submission = pl.read_csv(submission_path)
    validate_submission(submission, videos=videos, triplets=triplets, allowed_splits=("train",))
    validation_keys = [
        (str(row["lab_id"]), int(row["video_id"])) for row in primary.iter_rows(named=True)
    ]
    solution = build_validation_solution(
        videos_path=videos_path,
        dataset_root=_resolve(paths["dataset_root"], project_root),
        validation_videos=validation_keys,
    )
    fold_rows: list[dict[str, object]] = []
    fold_parts: list[pl.DataFrame] = []
    for fold in sorted(int(value) for value in primary["fold"].unique()):
        video_ids = primary.filter(pl.col("fold") == fold)["video_id"].to_list()
        fold_solution = solution[solution["video_id"].isin(video_ids)].reset_index(drop=True)
        fold_submission = submission.filter(pl.col("video_id").is_in(video_ids))
        fold_score = float(mouse_fbeta(fold_solution, fold_submission.to_pandas(), beta=1.0))
        fold_rows.append(
            {
                "fold": fold,
                "video_count": len(video_ids),
                "solution_intervals": len(fold_solution),
                "prediction_intervals": fold_submission.height,
                "exact_fbeta": fold_score,
            }
        )
        fold_parts.append(fold_submission)
    assembled = pl.concat(fold_parts, how="vertical_relaxed")
    assembly_parity = _canonical_interval_content(assembled).equals(
        _canonical_interval_content(submission)
    )
    if not assembly_parity:
        raise AssertionError("five-fold interval assembly differs from the frozen final submission")
    exact_score = float(mouse_fbeta(solution, assembled.to_pandas(), beta=1.0))
    if not math.isclose(exact_score, expected_score, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"final OOF score changed: expected={expected_score}, actual={exact_score}")
    breakdown = metric_breakdown(solution, assembled.to_pandas(), beta=1.0)
    pl.DataFrame(fold_rows).write_csv(reports_root / "oof_by_fold.csv")
    by_lab = pl.DataFrame(breakdown.by_lab).sort("lab_id")
    by_lab_action = pl.DataFrame(breakdown.by_lab_action).sort("lab_id", "action")
    action_errors = _aggregate_action_errors(breakdown.by_lab_action)
    by_lab.write_csv(reports_root / "oof_by_lab.csv")
    by_lab_action.write_csv(reports_root / "oof_by_lab_action.csv")
    action_errors.write_csv(reports_root / "action_confusion_error_summary.csv")
    action_errors.head(20).write_csv(reports_root / "largest_action_errors.csv")
    error_summary = {
        "schema_version": 1,
        "scope": "one-vs-rest frame intersections within each lab/action scoring component",
        "true_negative_policy": (
            "not reported because sparse behavior masks do not define a universal negative frame pool"
        ),
        "true_positive_frames": int(action_errors["tp"].sum()),
        "false_positive_frames": int(action_errors["fp"].sum()),
        "false_negative_frames": int(action_errors["fn"].sum()),
        "error_frames": int(action_errors["error_frames"].sum()),
        "action_count": action_errors.height,
    }
    (reports_root / "error_summary.json").write_text(
        json.dumps(error_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    canonical = _canonical_interval_content(assembled).with_row_index("row_id").select(SUBMISSION_COLUMNS)
    canonical.write_csv(reports_root / "final_oof_submission.csv")

    confidence_bins, confidence_actions = _sample_oof_confidence(
        part_paths, rows_per_part=rows_per_part
    )
    confidence_bins.write_csv(reports_root / "confidence_bins.csv")
    confidence_actions.write_csv(reports_root / "sampled_action_probability_summary.csv")
    _write_plots(by_lab, action_errors, confidence_bins, plots_root)

    data_audit_path = _resolve(paths["data_audit"], project_root)
    data_audit = cast(dict[str, Any], json.loads(data_audit_path.read_text(encoding="utf-8")))
    bundle_root = _resolve(paths["phase12_bundle"], project_root)
    bundle = verify_phase12_bundle(bundle_root)
    executed_notebook = _resolve(paths["executed_notebook"], project_root)
    notebook_submission = _resolve(paths["notebook_submission"], project_root)
    notebook_report_path = _resolve(paths["notebook_inference_report"], project_root)
    notebook_report = cast(
        dict[str, Any], json.loads(notebook_report_path.read_text(encoding="utf-8"))
    )
    if notebook_submission.read_bytes() != (project_root / "artifacts/phase12/submission.csv").read_bytes():
        raise AssertionError("clean notebook submission is not byte-identical to the frozen Phase 12 output")
    if int(notebook_report["validation_error_count"]) != 0:
        raise AssertionError("clean notebook inference report contains validation errors")

    dependency_versions = _all_dependency_versions()
    (reports_root / "dependency_lock.json").write_text(
        json.dumps(dependency_versions, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (reports_root / "requirements-lock.txt").write_text(
        "".join(f"{row['name']}=={row['version']}\n" for row in dependency_versions),
        encoding="utf-8",
    )
    pl.DataFrame(_dependency_records()).write_csv(reports_root / "dependency_licenses.csv")

    provenance = {
        "schema_version": 1,
        "release_version": release_version,
        "created_at": started_at.isoformat(),
        "source_data_fingerprint": data_audit["source_fingerprint"],
        "folds": _hash_record(folds_path, project_root, role="fixed_five_fold_assignment"),
        "fold_version": "five-fold-seed-2025",
        "features": {
            "version": "feature-v1",
            "solo_registry": _hash_record(
                project_root / "artifacts/features/registry/solo.json",
                project_root,
                role="solo_feature_registry",
            ),
            "pair_registry": _hash_record(
                project_root / "artifacts/features/registry/pair.json",
                project_root,
                role="pair_feature_registry",
            ),
            "category_registry": _hash_record(
                project_root / "artifacts/features/registry/categories.json",
                project_root,
                role="category_registry",
            ),
        },
        "models": {
            "selected_model_id": final_registry["selected_model_id"],
            "phase11_version": final_registry["phase11_version"],
            "gbdt_model_version": "xgb-baseline-v1",
            "gbdt_route_count": 510,
            "temporal_model_version": "temporal-phase9-v1",
            "temporal_fold_checkpoints": 5,
            "temporal_parameter_count_per_fold": 10708,
        },
        "decoder_registry": {
            "crossfit": _hash_record(
                _resolve(paths["final_decoder"], project_root),
                project_root,
                role="crossfit_interval_decoder",
            ),
            "probability_selection": _hash_record(
                _resolve(paths["probability_decoder_selection"], project_root),
                project_root,
                role="crossfit_probability_decoder_selection",
            ),
            "deployment": _hash_record(
                _resolve(paths["deployment_decoder"], project_root),
                project_root,
                role="deployment_decoder",
            ),
        },
        "blend_registry": _hash_record(
            _resolve(paths["blend_registry"], project_root), project_root, role="blend_registry"
        ),
        "notebook": {
            "source": _hash_record(
                project_root / "notebooks/phase12_kaggle_inference.ipynb",
                project_root,
                role="inference_notebook_source",
            ),
            "executed": _hash_record(
                executed_notebook, project_root, role="clean_executed_inference_notebook"
            ),
        },
        "phase12_bundle": {
            "manifest": _hash_record(
                bundle_root / "bundle_manifest.json", project_root, role="kaggle_bundle_manifest"
            ),
            "file_count": bundle["file_count"],
            "total_size_bytes": bundle["total_size_bytes"],
        },
        "release_configuration": _hash_record(
            config_path, project_root, role="release_configuration"
        ),
    }
    (reports_root / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    shutil.copy2(executed_notebook, archive_root / "phase12_kaggle_inference.executed.ipynb")
    shutil.copy2(notebook_submission, archive_root / "submission.csv")
    shutil.copy2(notebook_report_path, archive_root / "inference_report.json")
    _write_notebook_log(executed_notebook, archive_root / "notebook_stdout.log")
    score_metadata = {
        "schema_version": 1,
        "release_version": release_version,
        "exact_cross_fitted_oof_fbeta": exact_score,
        "kaggle_public_score": None,
        "kaggle_private_score": None,
        "leaderboard_status": "not submitted or not recorded locally",
        "submission_rows": pl.read_csv(notebook_submission).height,
        "submission_sha256": sha256_file(notebook_submission),
        "validation_error_count": notebook_report["validation_error_count"],
    }
    (archive_root / "score_metadata.json").write_text(
        json.dumps(score_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    essential_sources: tuple[tuple[str, Path], ...] = (
        ("final_model_registry", final_registry_path),
        ("final_validation_submission", submission_path),
        ("final_oof_manifest", oof_manifest_path),
        ("video_manifest", videos_path),
        ("triplet_manifest", triplets_path),
        ("fixed_folds", folds_path),
        ("data_audit", data_audit_path),
        ("action_vocabulary", project_root / "artifacts/labels/action_vocabulary.json"),
        ("blend_registry", _resolve(paths["blend_registry"], project_root)),
        ("crossfit_decoder", _resolve(paths["final_decoder"], project_root)),
        (
            "probability_decoder_selection",
            _resolve(paths["probability_decoder_selection"], project_root),
        ),
        ("deployment_weights", _resolve(paths["deployment_weights"], project_root)),
        ("deployment_decoder", _resolve(paths["deployment_decoder"], project_root)),
        ("kaggle_bundle_manifest", bundle_root / "bundle_manifest.json"),
        ("release_configuration", config_path),
        ("project_readme", project_root / "README.md"),
        ("special_cases", project_root / "docs/special_cases.md"),
        ("third_party_notices", project_root / "THIRD_PARTY_NOTICES.md"),
        ("release_script", project_root / "scripts/release_project.py"),
        ("release_source", project_root / "src/mabe/phase13_release.py"),
        ("dependency_specification", project_root / "requirements.txt"),
    )
    copied_essential_records = [
        _hash_record(path, project_root, role=role) for role, path in essential_sources
    ]
    essential_records = copied_essential_records + part_records
    pl.DataFrame(essential_records).write_csv(output_root / "essential_artifacts.csv")
    for role, source in essential_sources:
        suffix = "".join(source.suffixes)
        shutil.copy2(source, essential_root / f"{role}{suffix}")
    _experimental_inventory(project_root).write_csv(output_root / "experimental_artifacts.csv")

    release_report = {
        "schema_version": 1,
        "release_version": release_version,
        "created_at": datetime.now(UTC).isoformat(),
        "acceptance_gate_passed": True,
        "exact_oof": {
            "expected_fbeta": expected_score,
            "actual_fbeta": exact_score,
            "absolute_difference": abs(exact_score - expected_score),
            "tolerance": tolerance,
            "fold_count": len(fold_rows),
            "interval_assembly_parity": assembly_parity,
            "prediction_intervals": submission.height,
            "probability_manifest_parts": len(part_paths),
            "probability_manifest_rows": oof_manifest["row_count"],
            "probability_parts_missing": 0,
        },
        "diagnostics": {
            "lab_rows": by_lab.height,
            "lab_action_rows": by_lab_action.height,
            "action_rows": action_errors.height,
            "confidence_sample_rows": int(confidence_bins["sample_rows"].sum()),
            "confidence_sampling_method": "first N rows of every ordered OOF parquet part",
            "confidence_rows_per_oof_part": rows_per_part,
            "plots": 3,
        },
        "clean_inference": {
            "bundle_verified": True,
            "bundle_files": bundle["file_count"],
            "notebook_submission_byte_parity": True,
            "validation_error_count": notebook_report["validation_error_count"],
            "submission_sha256": sha256_file(notebook_submission),
        },
        "special_cases_documented": True,
        "attribution_audited": True,
        "experimental_artifacts_separated_by_inventory": True,
    }
    report_path = output_root / "release_report.json"
    report_path.write_text(
        json.dumps(release_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    manifest_path = output_root / "release_manifest.json"
    local_files = sorted(
        path for path in output_root.rglob("*") if path.is_file() and path != manifest_path
    )
    manifest = {
        "schema_version": 1,
        "release_version": release_version,
        "created_at": datetime.now(UTC).isoformat(),
        "project_root": project_root.as_posix(),
        "phase12_bundle_root": _relative(bundle_root, project_root),
        "local_file_count": len(local_files),
        "local_total_size_bytes": sum(path.stat().st_size for path in local_files),
        "local_files": [
            {
                "path": path.relative_to(output_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in local_files
        ],
        "external_essential_files": essential_records,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    verify_phase13_release(manifest_path, project_root=project_root)
    return manifest_path


def verify_phase13_release(manifest_path: Path, *, project_root: Path | None = None) -> dict[str, Any]:
    """Verify every local release file, essential external artifact, and bundle hash."""
    manifest_path = manifest_path.resolve()
    output_root = manifest_path.parent
    payload = cast(dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8")))
    root = (
        project_root.resolve()
        if project_root is not None
        else Path(str(payload["project_root"])).resolve()
    )
    for record in cast(list[dict[str, object]], payload["local_files"]):
        path = output_root / str(record["path"])
        if not path.is_file() or path.stat().st_size != int(str(record["size_bytes"])):
            raise ValueError(f"release file is missing or changed size: {path}")
        if sha256_file(path) != str(record["sha256"]):
            raise ValueError(f"release file hash changed: {path}")
    for record in cast(list[dict[str, object]], payload["external_essential_files"]):
        path = _resolve(str(record["path"]), root)
        if not path.is_file() or path.stat().st_size != int(str(record["size_bytes"])):
            raise ValueError(f"essential external file is missing or changed size: {path}")
        if sha256_file(path) != str(record["sha256"]):
            raise ValueError(f"essential external file hash changed: {path}")
    verify_phase12_bundle(_resolve(str(payload["phase12_bundle_root"]), root))
    report = cast(
        dict[str, Any], json.loads((output_root / "release_report.json").read_text(encoding="utf-8"))
    )
    if not bool(report["acceptance_gate_passed"]):
        raise ValueError("Phase 13 release report did not pass its acceptance gate")
    return payload
