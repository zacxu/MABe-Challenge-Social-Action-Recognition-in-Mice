"""CLI for packaging, validating, simulating, and running Phase 12 inference."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from mabe.phase12_config import load_phase12_config  # noqa: E402
from mabe.phase12_inference import (  # noqa: E402
    discover_dataset_root,
    freeze_deployment_parameters,
    run_offline_inference,
)
from mabe.phase12_release import (  # noqa: E402
    build_runtime_benchmark,
    prepare_phase12_bundle,
    run_all_lab_raw_simulation,
    run_dynamic_manifest_stress,
    run_heldout_replay_simulation,
    run_raw_heldout_simulation,
    verify_phase12_bundle,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "freeze",
            "bundle",
            "archive",
            "verify",
            "simulate",
            "raw-simulate",
            "raw-simulate-all-labs",
            "deployment-simulate-all-labs",
            "manifest-stress",
            "benchmark",
            "run",
        ),
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/inference.yaml")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path("/kaggle/working/mabe_phase12")
        if Path("/kaggle/working").is_dir()
        else PROJECT_ROOT / "artifacts/phase12/runtime",
    )
    parser.add_argument(
        "--submission",
        type=Path,
        default=Path("/kaggle/working/submission.csv")
        if Path("/kaggle/working").is_dir()
        else PROJECT_ROOT / "artifacts/phase12/submission.csv",
    )
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts/phase12/kaggle-bundle-v1",
    )
    parser.add_argument(
        "--archive-path",
        type=Path,
        help="Destination .zip for the archive command; defaults beside --bundle-root.",
    )
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    config = load_phase12_config(arguments.config)
    if arguments.command == "freeze":
        weights, decoder = freeze_deployment_parameters(config)
        print(json.dumps({"weight_rows": weights.height, "decoder_rows": decoder.height}))
        return
    if arguments.command == "bundle":
        manifest = prepare_phase12_bundle(config, arguments.bundle_root.resolve())
        print(manifest)
        return
    if arguments.command == "archive":
        bundle_root = arguments.bundle_root.resolve()
        prepare_phase12_bundle(config, bundle_root)
        payload = verify_phase12_bundle(bundle_root)
        requested = (
            arguments.archive_path.resolve()
            if arguments.archive_path is not None
            else bundle_root.with_suffix(".zip")
        )
        archive_path = requested.with_suffix(".zip")
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        generated = Path(
            shutil.make_archive(
                str(archive_path.with_suffix("")),
                "zip",
                root_dir=bundle_root.parent,
                base_dir=bundle_root.name,
            )
        )
        print(
            json.dumps(
                {
                    "archive": str(generated),
                    "bundle_root_in_archive": bundle_root.name,
                    "file_count": payload["file_count"],
                    "status": "valid",
                },
                indent=2,
            )
        )
        return
    if arguments.command == "verify":
        payload = verify_phase12_bundle(arguments.bundle_root.resolve())
        print(json.dumps({"file_count": payload["file_count"], "status": "valid"}))
        return
    if arguments.command == "simulate":
        report = run_heldout_replay_simulation(config.project_root, arguments.work_root.resolve())
        print(report)
        return
    if arguments.command == "raw-simulate":
        report = run_raw_heldout_simulation(config, arguments.work_root.resolve())
        print(report)
        return
    if arguments.command == "raw-simulate-all-labs":
        report = run_all_lab_raw_simulation(config, arguments.work_root.resolve(), deployment=False)
        print(report)
        return
    if arguments.command == "deployment-simulate-all-labs":
        report = run_all_lab_raw_simulation(config, arguments.work_root.resolve(), deployment=True)
        print(report)
        return
    if arguments.command == "manifest-stress":
        report = run_dynamic_manifest_stress(config, arguments.work_root.resolve())
        print(report)
        return
    if arguments.command == "benchmark":
        report = build_runtime_benchmark(
            config.project_root,
            arguments.work_root.resolve() / "inference_report.json",
            arguments.work_root.resolve(),
            hidden_video_count=config.runtime.hidden_video_count,
            kaggle_limit_seconds=config.runtime.kaggle_limit_seconds,
            required_headroom_fraction=config.runtime.required_headroom_fraction,
        )
        print(report)
        return
    deployment_weights = config.asset_path(config.assets.deployment_weights)
    deployment_decoder = config.asset_path(config.assets.deployment_decoder_parameters)
    if not deployment_weights.is_file() or not deployment_decoder.is_file():
        freeze_deployment_parameters(config)
    manifest = config.project_root / "bundle_manifest.json"
    if manifest.is_file():
        verify_phase12_bundle(config.project_root)
    dataset_root = discover_dataset_root(arguments.dataset_root)
    outputs = run_offline_inference(
        config,
        dataset_root=dataset_root,
        work_root=arguments.work_root.resolve(),
        submission_path=arguments.submission.resolve(),
    )
    print(
        json.dumps(
            {
                "submission": str(outputs.submission_path),
                "report": str(outputs.report_path),
                "video_count": outputs.video_count,
                "interval_count": outputs.interval_count,
                "runtime_seconds": outputs.runtime_seconds,
                "projected_seconds_200_videos": outputs.projected_seconds_200_videos,
                "headroom_fraction": outputs.headroom_fraction,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
