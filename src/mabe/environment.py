"""Dependency availability and version reporting."""

from __future__ import annotations

import importlib
import platform
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata


@dataclass(frozen=True, slots=True)
class Dependency:
    distribution: str
    module: str


CORE_DEPENDENCIES = (
    Dependency("numpy", "numpy"),
    Dependency("scipy", "scipy"),
    Dependency("pandas", "pandas"),
    Dependency("polars", "polars"),
    Dependency("pyarrow", "pyarrow"),
    Dependency("scikit-learn", "sklearn"),
    Dependency("xgboost", "xgboost"),
    Dependency("torch", "torch"),
    Dependency("PyYAML", "yaml"),
    Dependency("pydantic", "pydantic"),
    Dependency("typer", "typer"),
    Dependency("rich", "rich"),
    Dependency("psutil", "psutil"),
)


def dependency_versions(*, validate_imports: bool = False) -> dict[str, str]:
    """Return core distribution versions, optionally importing every module."""
    versions: dict[str, str] = {}
    failures: list[str] = []
    for dependency in CORE_DEPENDENCIES:
        try:
            if validate_imports:
                importlib.import_module(dependency.module)
            versions[dependency.distribution] = metadata.version(dependency.distribution)
        except (ImportError, metadata.PackageNotFoundError) as error:
            failures.append(f"{dependency.distribution}: {error}")
    if failures:
        raise RuntimeError("core dependency check failed: " + "; ".join(failures))
    return versions


def environment_report(*, name: str, validate_imports: bool = False) -> dict[str, object]:
    """Build a stable JSON-serializable environment report."""
    return {
        "name": name,
        "captured_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "dependencies": dependency_versions(validate_imports=validate_imports),
    }
