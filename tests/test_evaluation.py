from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from mabe.evaluation import (
    decode_oof_probabilities,
    nested_decoder_evaluation,
    validate_oof,
)
from mabe.io import scan_oof
from mabe.schemas import OOF_SCHEMA


def _folds() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "lab_id": ["Lab"] * 5,
            "video_id": list(range(1, 6)),
            "fold": pl.Series(range(5), dtype=pl.Int8),
            "is_primary_validation": [True] * 5,
            "source_role": ["competition"] * 5,
        }
    )


def _oof() -> pl.DataFrame:
    records = []
    for fold, video_id in enumerate(range(1, 6)):
        for frame in range(4):
            target = int(frame in (1, 2))
            records.append(
                {
                    "lab_id": "Lab",
                    "video_id": video_id,
                    "video_frame": frame,
                    "agent_id": "mouse1",
                    "target_id": "mouse2",
                    "action": "attack",
                    "fold": fold,
                    "loss_mask": 1,
                    "target": target,
                    "probability": 0.9 if target else 0.1,
                    "model_id": "dummy",
                    "feature_version": "v1",
                }
            )
    return pl.DataFrame(records).cast(OOF_SCHEMA).select(OOF_SCHEMA.names())


def _solution() -> pd.DataFrame:
    mask = json.dumps(["mouse1,mouse2,attack"])
    return pd.DataFrame(
        [
            {
                "video_id": video_id,
                "agent_id": "mouse1",
                "target_id": "mouse2",
                "action": "attack",
                "start_frame": 1,
                "stop_frame": 3,
                "lab_id": "Lab",
                "behaviors_labeled": mask,
            }
            for video_id in range(1, 6)
        ]
    )


def test_oof_validation_rejects_fold_mismatch_and_masked_positive() -> None:
    oof = _oof()
    folds = _folds()
    mismatch = oof.with_columns(
        pl.when(pl.col("video_id") == 1).then(4).otherwise(pl.col("fold")).cast(pl.Int8).alias("fold")
    )
    with pytest.raises(ValueError, match="disagree"):
        validate_oof(mismatch, folds)
    masked = oof.with_columns(
        pl.when((pl.col("video_id") == 1) & (pl.col("target") == 1))
        .then(0)
        .otherwise(pl.col("loss_mask"))
        .cast(pl.UInt8)
        .alias("loss_mask")
    )
    with pytest.raises(ValueError, match="masked out"):
        validate_oof(masked, folds)


def test_decoder_resolves_pair_conflicts_by_calibrated_margin() -> None:
    probabilities = pl.DataFrame(
        {
            "lab_id": ["Lab", "Lab"],
            "video_id": [1, 1],
            "video_frame": [7, 7],
            "agent_id": ["mouse1", "mouse1"],
            "target_id": ["mouse2", "mouse2"],
            "action": ["attack", "chase"],
            "loss_mask": [1, 1],
            "probability": [0.7, 0.9],
        }
    )
    thresholds = pl.DataFrame(
        {
            "lab_id": ["Lab", "Lab"],
            "action": ["attack", "chase"],
            "threshold": [0.5, 0.8],
        }
    )

    decoded = decode_oof_probabilities(probabilities, thresholds)

    assert decoded[["action", "start_frame", "stop_frame"]].to_dict("records") == [
        {"action": "attack", "start_frame": 7, "stop_frame": 8}
    ]


def test_nested_decoder_uses_only_other_folds_and_scores_end_to_end(tmp_path: Path) -> None:
    oof_path = tmp_path / "dummy_oof.parquet"
    _oof().write_parquet(oof_path)
    loaded_oof = scan_oof(oof_path).collect()
    result = nested_decoder_evaluation(
        loaded_oof,
        _folds(),
        _solution(),
        threshold_grid=[0.5],
    )

    assert result.breakdown.overall == 1.0
    assert result.by_fold["fbeta"].tolist() == [1.0] * 5
    assert not result.calibration["used_held_out_labels"].any()
    for row in result.calibration.itertuples(index=False):
        assert row.held_out_fold not in json.loads(row.calibration_folds)
    assert result.submission["start_frame"].tolist() == [1] * 5
    assert result.submission["stop_frame"].tolist() == [3] * 5
