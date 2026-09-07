"""Versioned feature registry and deterministic float32 parquet chunks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import polars as pl

from mabe.features.common import CategoryRegistry, FeatureTable
from mabe.fingerprints import sha256_file


@dataclass(frozen=True, slots=True)
class FeatureCache:
    root: Path
    chunks: tuple[Path, ...]
    row_count: int
    byte_count: int
    schema_hash: str


def save_category_registry(registry: CategoryRegistry, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(registry.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def load_category_registry(path: str | Path) -> CategoryRegistry:
    loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("category registry must contain a JSON object")
    return CategoryRegistry.from_dict(cast(dict[str, object], loaded))


def save_feature_registry(table: FeatureTable, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(table.registry_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def save_feature_table(
    table: FeatureTable,
    root: str | Path,
    *,
    chunk_rows: int,
    source_fingerprint: str,
    configuration_sha256: str,
) -> FeatureCache:
    """Persist stable-order chunks and sidecars; current manifest defines active parts."""
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    output_root = Path(root)
    output_root.mkdir(parents=True, exist_ok=True)
    frame = table.to_frame()
    chunks: list[Path] = []
    for part, offset in enumerate(range(0, frame.height, chunk_rows)):
        path = output_root / f"part-{part:05d}.parquet"
        chunk = frame.slice(offset, chunk_rows)
        chunk.write_parquet(path, compression="zstd", statistics=True)
        sidecar = {
            "artifact": path.name,
            "artifact_sha256": sha256_file(path),
            "configuration_sha256": configuration_sha256,
            "created_at": datetime.now(UTC).isoformat(),
            "feature_version": table.feature_version,
            "kind": table.kind,
            "row_count": chunk.height,
            "schema_sha256": table.schema_hash,
            "source_data_fingerprint": source_fingerprint,
        }
        Path(f"{path}.meta.json").write_text(
            json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        chunks.append(path)
    manifest = {
        "schema_version": 1,
        "feature_version": table.feature_version,
        "kind": table.kind,
        "schema_sha256": table.schema_hash,
        "row_count": frame.height,
        "feature_count": len(table.feature_names),
        "feature_groups": table.groups,
        "chunks": [path.name for path in chunks],
        "chunk_sha256": {path.name: sha256_file(path) for path in chunks},
        "manifest_content_sha256": hashlib.sha256(
            "|".join(sha256_file(path) for path in chunks).encode()
        ).hexdigest(),
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return FeatureCache(
        root=output_root,
        chunks=tuple(chunks),
        row_count=frame.height,
        byte_count=sum(path.stat().st_size for path in chunks),
        schema_hash=table.schema_hash,
    )


def validate_feature_cache(cache: FeatureCache, table: FeatureTable) -> None:
    """Read persisted chunks and enforce column order, dtypes, and finite values."""
    expected = table.to_frame().schema
    row_count = 0
    for path in cache.chunks:
        frame = pl.read_parquet(path)
        if frame.schema != expected:
            raise AssertionError(f"feature cache schema drift in {path}")
        feature_frame = frame.select(table.feature_names)
        checks = feature_frame.select(
            [(~pl.col(name).is_finite()).any().alias(name) for name in table.feature_names]
        ).row(0)
        if any(bool(value) for value in checks):
            raise AssertionError(f"non-finite values reached feature cache {path}")
        row_count += frame.height
    if row_count != cache.row_count:
        raise AssertionError("feature chunk rows do not reconcile with cache manifest")
