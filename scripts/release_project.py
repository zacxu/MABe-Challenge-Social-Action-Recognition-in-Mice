"""Build or verify the final Phase 13 reproducibility release."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from mabe.phase13_release import build_phase13_release, verify_phase13_release  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "verify"))
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/release.yaml")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts/release/mabe-final-v1",
    )
    arguments = parser.parse_args()
    manifest = arguments.output_root.resolve() / "release_manifest.json"
    if arguments.command == "build":
        manifest = build_phase13_release(
            PROJECT_ROOT,
            config_path=arguments.config,
            output_root=arguments.output_root,
        )
        print(manifest)
        return
    payload = verify_phase13_release(manifest, project_root=PROJECT_ROOT)
    print(
        json.dumps(
            {
                "release_version": payload["release_version"],
                "local_file_count": payload["local_file_count"],
                "status": "valid",
            }
        )
    )


if __name__ == "__main__":
    main()

