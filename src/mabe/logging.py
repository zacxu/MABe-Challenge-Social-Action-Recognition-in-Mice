"""Structured JSON logging with stable run context fields."""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from collections.abc import Mapping, MutableMapping
from datetime import UTC, datetime
from typing import Any, TextIO

import psutil

_PROCESS_STARTED = time.perf_counter()
_PEAK_MEMORY_BYTES = 0
_MEMORY_LOCK = threading.Lock()
_CONTEXT_FIELDS = ("run_id", "phase", "fold", "lab", "action")


def _peak_memory_mb() -> float:
    global _PEAK_MEMORY_BYTES
    current = int(psutil.Process().memory_info().rss)
    with _MEMORY_LOCK:
        _PEAK_MEMORY_BYTES = max(_PEAK_MEMORY_BYTES, current)
        return _PEAK_MEMORY_BYTES / (1024 * 1024)


class JsonFormatter(logging.Formatter):
    """One JSON object per log record."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "elapsed_seconds": round(time.perf_counter() - _PROCESS_STARTED, 6),
            "peak_memory_mb": round(_peak_memory_mb(), 3),
        }
        for field in _CONTEXT_FIELDS:
            payload[field] = getattr(record, field, None)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, sort_keys=True, default=str)


class ContextLogger(logging.LoggerAdapter[logging.Logger]):
    """Logger adapter whose context can be extended for lab/action/fold work."""

    def process(
        self,
        msg: object,
        kwargs: MutableMapping[str, Any],
    ) -> tuple[object, MutableMapping[str, Any]]:
        supplied = kwargs.get("extra", {})
        supplied_context = supplied if isinstance(supplied, Mapping) else {}
        kwargs["extra"] = {**(self.extra or {}), **supplied_context}
        return msg, kwargs

    def bind(self, **context: object) -> ContextLogger:
        return ContextLogger(self.logger, {**(self.extra or {}), **context})


def configure_logging(
    *,
    run_id: str,
    phase: str,
    level: str = "INFO",
    stream: TextIO | None = None,
) -> ContextLogger:
    """Configure the package logger and return a context-bound adapter."""
    logger = logging.getLogger("mabe")
    logger.setLevel(level.upper())
    logger.handlers.clear()
    logger.propagate = False
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    return ContextLogger(
        logger,
        {"run_id": run_id, "phase": phase, "fold": None, "lab": None, "action": None},
    )
