from __future__ import annotations

import io
import json

from mabe.logging import configure_logging


def test_structured_logger_emits_complete_context() -> None:
    stream = io.StringIO()
    logger = configure_logging(run_id="run-42", phase="features", stream=stream)
    logger.bind(fold=3, lab="lab-a", action="attack").info("feature block complete")

    payload = json.loads(stream.getvalue())
    assert payload["run_id"] == "run-42"
    assert payload["phase"] == "features"
    assert payload["fold"] == 3
    assert payload["lab"] == "lab-a"
    assert payload["action"] == "attack"
    assert payload["message"] == "feature block complete"
    assert payload["elapsed_seconds"] >= 0
    assert payload["peak_memory_mb"] > 0
