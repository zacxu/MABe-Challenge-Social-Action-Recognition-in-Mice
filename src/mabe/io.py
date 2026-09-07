"""Lazy, typed readers for competition source data."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import polars as pl

from mabe.schemas import (
    ANNOTATION_COLUMNS,
    ANNOTATION_SCHEMA,
    ANNOTATION_SOURCE_COLUMNS,
    METADATA_COLUMNS,
    METADATA_SCHEMA,
    OOF_COLUMNS,
    OOF_SCHEMA,
    SUBMISSION_COLUMNS,
    SUBMISSION_SCHEMA,
    TRACKING_COLUMNS,
    TRACKING_SOURCE_COLUMNS,
)


def _require_columns(frame: pl.LazyFrame, required: Iterable[str], *, source: Path) -> None:
    available = set(frame.collect_schema().names())
    missing = sorted(set(required) - available)
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def scan_metadata(path: str | Path) -> pl.LazyFrame:
    """Scan metadata with explicit types and canonical column order."""
    source = Path(path)
    frame = pl.scan_csv(
        source,
        schema_overrides=dict(METADATA_SCHEMA),
        infer_schema_length=10_000,
        null_values=[""],
    )
    _require_columns(frame, METADATA_COLUMNS, source=source)
    return frame.select(METADATA_COLUMNS)


def _mouse_slot(expression: pl.Expr) -> pl.Expr:
    return pl.concat_str(pl.lit("mouse"), expression.cast(pl.Int64).cast(pl.String))


def scan_tracking(path: str | Path) -> pl.LazyFrame:
    """Scan one tracking parquet and normalize mouse IDs to ``mouseN`` strings."""
    source = Path(path)
    frame = pl.scan_parquet(source)
    _require_columns(frame, TRACKING_SOURCE_COLUMNS, source=source)
    return frame.select(
        pl.col("video_frame").cast(pl.Int64),
        _mouse_slot(pl.col("mouse_id")).alias("mouse_id"),
        pl.col("bodypart").cast(pl.String),
        pl.col("x").cast(pl.Float32),
        pl.col("y").cast(pl.Float32),
    ).select(TRACKING_COLUMNS)


def scan_annotation(path: str | Path) -> pl.LazyFrame:
    """Scan one annotation parquet using canonical IDs and half-open frame columns."""
    source = Path(path)
    frame = pl.scan_parquet(source)
    _require_columns(frame, ANNOTATION_SOURCE_COLUMNS, source=source)
    return frame.select(
        _mouse_slot(pl.col("agent_id")).alias("agent_id"),
        _mouse_slot(pl.col("target_id")).alias("target_id"),
        pl.col("action").cast(pl.String),
        pl.col("start_frame").cast(pl.Int64),
        pl.col("stop_frame").cast(pl.Int64),
    ).select(ANNOTATION_COLUMNS)


def scan_annotation_files(paths: Iterable[str | Path]) -> pl.LazyFrame:
    """Scan multiple annotation parquets while retaining their source paths."""
    sources = [str(Path(path).resolve()) for path in paths]
    if not sources:
        return pl.DataFrame(schema={**dict(ANNOTATION_SCHEMA), "source_path": pl.String}).lazy()
    frame = pl.scan_parquet(
        sources,
        include_file_paths="source_path",
        cast_options=pl.ScanCastOptions(integer_cast="upcast"),
    )
    _require_columns(frame, ANNOTATION_SOURCE_COLUMNS, source=Path("<annotation files>"))
    return frame.select(
        _mouse_slot(pl.col("agent_id")).alias("agent_id"),
        _mouse_slot(pl.col("target_id")).alias("target_id"),
        pl.col("action").cast(pl.String),
        pl.col("start_frame").cast(pl.Int64),
        pl.col("stop_frame").cast(pl.Int64),
        pl.col("source_path").cast(pl.String),
    )


def scan_submission(path: str | Path) -> pl.LazyFrame:
    """Scan a submission CSV with the official stable column order."""
    source = Path(path)
    frame = pl.scan_csv(source, schema_overrides=dict(SUBMISSION_SCHEMA), infer_schema_length=1_000)
    _require_columns(frame, SUBMISSION_COLUMNS, source=source)
    return frame.select(SUBMISSION_COLUMNS)


def scan_oof(path: str | Path) -> pl.LazyFrame:
    """Scan a Phase 4 OOF probability parquet with its fixed schema."""
    source = Path(path)
    frame = pl.scan_parquet(source)
    _require_columns(frame, OOF_COLUMNS, source=source)
    return frame.select(
        pl.col(name).cast(dtype, strict=True) for name, dtype in OOF_SCHEMA.items()
    ).select(OOF_COLUMNS)
