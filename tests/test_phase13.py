from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import polars as pl

from mabe.fingerprints import sha256_file
from mabe.phase13_release import (
    _aggregate_action_errors,
    _canonical_interval_content,
    verify_phase13_release,
)


def test_canonical_interval_content_ignores_row_ids_and_order() -> None:
    left = pl.DataFrame(
        {
            "row_id": [0, 1],
            "video_id": [2, 1],
            "agent_id": ["mouse1", "mouse1"],
            "target_id": ["self", "self"],
            "action": ["rear", "rest"],
            "start_frame": [8, 2],
            "stop_frame": [9, 4],
        }
    )
    right = left.reverse().with_columns(pl.Series("row_id", [40, 41]))
    assert _canonical_interval_content(left).equals(_canonical_interval_content(right))


def test_action_error_aggregation_reports_frame_confusion() -> None:
    source = pd.DataFrame(
        [
            {
                "lab_id": "lab1",
                "action": "rear",
                "tp": 3,
                "fp": 1,
                "fn": 2,
                "ground_truth_frames": 5,
                "predicted_frames": 4,
                "fbeta": 2 / 3,
            },
            {
                "lab_id": "lab2",
                "action": "rear",
                "tp": 2,
                "fp": 2,
                "fn": 0,
                "ground_truth_frames": 2,
                "predicted_frames": 4,
                "fbeta": 2 / 3,
            },
        ]
    )
    result = _aggregate_action_errors(source).row(0, named=True)
    assert result["tp"] == 5
    assert result["fp"] == 3
    assert result["fn"] == 2
    assert result["error_frames"] == 5
    assert result["micro_fbeta"] == 2 * 5 / (2 * 5 + 3 + 2)


def test_release_verifier_checks_local_external_and_bundle_files(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    release_root = project_root / "release"
    bundle_root = project_root / "bundle"
    release_root.mkdir(parents=True)
    bundle_root.mkdir(parents=True)
    external = project_root / "external.txt"
    external.write_text("external\n", encoding="utf-8")
    report = release_root / "release_report.json"
    report.write_text('{"acceptance_gate_passed": true}\n', encoding="utf-8")
    bundle_manifest = bundle_root / "bundle_manifest.json"
    bundle_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "file_count": 0,
                "total_size_bytes": 0,
                "files": [],
            }
        ),
        encoding="utf-8",
    )
    manifest = release_root / "release_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "project_root": project_root.as_posix(),
                "phase12_bundle_root": "bundle",
                "local_files": [
                    {
                        "path": report.name,
                        "size_bytes": report.stat().st_size,
                        "sha256": sha256_file(report),
                    }
                ],
                "external_essential_files": [
                    {
                        "path": external.relative_to(project_root).as_posix(),
                        "size_bytes": external.stat().st_size,
                        "sha256": sha256_file(external),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    verify_phase13_release(manifest, project_root=project_root)

