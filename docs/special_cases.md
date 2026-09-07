# MABe special-case and exclusion policy

This document freezes the rules used by the final `mabe-final-v1` release. No
rule below is inferred from hidden-test labels.

## AdaptableSnail timing

Four 25-FPS AdaptableSnail videos were explicitly audited. The pipeline does
not apply a global or per-video `30/25` output-frame correction:

- tracking cleanup retains every original frame ID;
- physical-time features and decoder durations use each video's declared FPS;
- the optional 30-Hz temporal representation carries a reversible mapping back
  to original frames;
- submission intervals are emitted in the original competition frame system;
- `configs/pose.yaml` keeps `adaptable_snail_frame_corrections: {}` because the
  plan requires leakage-safe nested OOF evidence before enabling a correction,
  and no such improvement was established locally.

The 2nd-place writeup reported a useful private-test correction for some
AdaptableSnail videos. That external result is recorded as inspiration, not
silently adopted as a label-dependent heuristic. Potential mouse-identity
swaps are flagged; identities are never automatically rewritten.

## Label-state exclusions

The audited metadata contains 8,790 train/test rows with these states:

- 847 labeled videos are eligible for supervised processing, subject to
  row-level quarantine;
- 7,926 `unlabeled_mabe22` videos cannot enter supervised training;
- 15 `unlabeled_missing_mask` videos cannot enter supervised training;
- `PleasantMeerkat/1375833299` is the sole
  `ambiguous_missing_annotation` video and is excluded;
- the public test row is inference-only.

Primary model selection uses 352 videos from the 15 competition labs.
CalMS21 and CRIM13 sources are training-only auxiliary data and never primary
validation. Exact and normalized duplicate groups are fold-contained.

## Annotation quarantine

Fifty-nine zero-length/invalid annotation rows are retained in
`artifacts/reports/annotation_anomalies.parquet` for audit and excluded from
targets. Of these, 54 belong to CRIM13, three to CalMS21 supplemental, one to
CalMS21 task 1, and one to CalMS21 task 2. No out-of-bounds, missing-mouse, or
behavior-mask violations were found. Ground-truth overlaps are reported and
retained rather than repaired.

## Mask, target, and interval conventions

- Solo behavior masks use `target_id=self`; source annotations encode the same
  event with `agent_id == target_id`.
- A video/action/triplet outside `behaviors_labeled` is unknown, not negative.
- Intervals are maximal, half-open `[start_frame, stop_frame)` runs and must
  have positive length.
- At most one action is emitted for each video/frame/directed agent-target pair.
- If every decoded frame is background, inference emits the highest-confidence
  valid one-frame candidate so the official metric never receives an empty
  submission.

The authoritative audit reports are `artifacts/reports/data_audit.json`,
`artifacts/folds/fold_quality.md`, and `artifacts/pose/pose_audit.md`.

