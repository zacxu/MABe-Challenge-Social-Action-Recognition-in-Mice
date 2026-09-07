"""Deterministic source and artifact fingerprints."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

import polars as pl


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory_fingerprint(records: Iterable[dict[str, object]]) -> str:
    """Hash stable file inventory fields in relative-path order."""
    stable = [
        {
            key: record[key]
            for key in ("relative_path", "size_bytes", "row_count", "schema_json", "sha256")
        }
        for record in sorted(records, key=lambda item: str(item["relative_path"]))
    ]
    payload = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def schema_fingerprint(frame: pl.DataFrame) -> str:
    payload = json.dumps(
        [(name, str(dtype)) for name, dtype in frame.schema.items()],
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()
