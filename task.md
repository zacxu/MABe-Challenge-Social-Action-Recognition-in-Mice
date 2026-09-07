# MABe implementation task breakdown

This document converts `plan.md` into an executable engineering backlog. Implement the phases in order. A phase is complete only when its deliverables and acceptance gate pass; do not begin expensive modeling while an earlier data, metric, or leakage issue remains unresolved.

## Target project layout

Create this structure gradually as the phases require it:

```text
configs/
  base.yaml
  features.yaml
  gbdt.yaml
  temporal.yaml
src/mabe/
  config.py
  schemas.py
  metric.py
  intervals.py
  io.py
  masks.py
  labels.py
  folds.py
  pose.py
  resample.py
  decode.py
  submission.py
  diagnostics.py
  features/
    solo.py
    pair.py
    group.py
    temporal.py
  models/
    gbdt.py
    temporal.py
    ensemble.py
scripts/
  audit_data.py
  make_folds.py
  build_features.py
  train_gbdt.py
  train_temporal.py
  tune_decoder.py
  build_ensemble.py
  run_inference.py
tests/
  test_metric.py
  test_intervals.py
  test_masks.py
  test_labels.py
  test_folds.py
  test_pose.py
  test_features.py
  test_decode.py
  test_submission.py
artifacts/
  manifests/
  folds/
  features/
  oof/
  models/
  reports/
  submissions/
```

Large generated artifacts should be ignored by version control. Every artifact must have a small JSON sidecar containing its configuration, code/version identifier, source-data fingerprint, creation time, row count, and schema hash.

---

# Phase 0 — Reproducible project foundation

**Status: complete (2026-08-25).**

## Goal

Create a testable Python package and configuration system that works locally and can be transferred to Kaggle without hard-coded machine paths.

## Tasks

- [x] Create `pyproject.toml` with a supported Python version and pinned major/minor dependency ranges.
- [x] Use Polars/PyArrow for parquet scanning and feature tables, NumPy/SciPy for numerical operations, pandas only at the official metric boundary, scikit-learn for splitting/utilities, XGBoost for the first model, and PyTorch for temporal models.
- [x] Keep LightGBM, CatBoost, Optuna, and graph-model dependencies optional until their phases.
- [x] Add `src/mabe/config.py` to load YAML configuration, resolve paths, validate required keys, and expose one immutable run configuration.
- [x] Put dataset, artifact, cache, and model paths in `configs/base.yaml`. Support environment-variable overrides for Kaggle paths.
- [x] Add a global seed utility covering Python, NumPy, XGBoost, and PyTorch.
- [x] Add structured logging with run ID, phase, fold, lab, action, elapsed time, and peak memory.
- [x] Add a lightweight command-line entry point or consistent `python -m` interface for every script.
- [x] Configure `pytest`, formatting, and static checks. Tests must run without loading the full dataset unless explicitly marked as integration tests.
- [x] Record the local and Kaggle package versions. Avoid relying on a package that cannot be imported in the offline Kaggle notebook.

## Deliverables

- Importable `mabe` package.
- Valid base configuration with local and Kaggle path examples.
- One smoke test that loads the configuration and locates `train.csv`.

## Acceptance gate

`pytest` passes from a clean shell, and a smoke command prints the dataset root, artifact root, seed, and dependency versions without modifying the data.

---

# Phase 1 — Freeze and test the official metric contract

**Status: complete (2026-08-25).**

## Goal

Make the exact competition scorer the unchanging oracle for every later decision.

## Tasks

- [x] Copy `single_lab_f1`, `mouse_fbeta`, `score`, and `HostVisibleError` from `mabe-f-beta.ipynb` into `src/mabe/metric.py` without changing their logic.
- [x] Keep helper validation or reporting code outside the copied functions so the official implementation remains auditable.
- [x] Implement `frames_to_intervals` and `intervals_to_frames` in `src/mabe/intervals.py` using half-open intervals.
- [x] Ensure run-length encoding emits maximal runs and rejects `start_frame >= stop_frame`.
- [x] Add metric tests for:
  - [x] perfect prediction;
  - [x] wrong action;
  - [x] half-open `[0, 1)` and `[0, 10)` behavior;
  - [x] adjacent labels being equivalent to one joined interval;
  - [x] missing frames, surplus frames, and the expected F1 values;
  - [x] equal weighting of labs of different sizes;
  - [x] macro-averaging ground-truth actions plus allowed predicted-only actions (the latter enter the official scorer's denominator at zero F-score);
  - [x] ignored predictions outside `behaviors_labeled`;
  - [x] overlapping different actions for one agent-target pair raising `HostVisibleError`;
  - [x] simultaneous predictions by one agent toward different targets being allowed;
  - [x] overlapping duplicate intervals of the same prediction key being harmless;
  - [x] an empty submission raising an error.
- [x] Add round-trip property tests: frame labels → intervals → frame labels must preserve the exact frame set.
- [x] Add a diagnostic wrapper that returns overall score plus per-lab/action TP, FP, FN, precision, recall, and F1 while confirming the overall score equals the untouched official scorer.

## Deliverables

- `src/mabe/metric.py` containing the official oracle.
- Interval conversion utilities.
- Comprehensive metric tests with small synthetic fixtures.

## Acceptance gate

All metric and interval tests pass, including exact numerical results from the notebook doctests. No later phase may change metric behavior.

---

# Phase 2 — Dataset inventory, schema, and anomaly audit

**Status: complete (2026-08-25).**

## Goal

Create a trustworthy manifest describing every metadata row, tracking file, annotation file, mouse, skeleton, and labeling state.

## Tasks

- [x] Define explicit schemas in `src/mabe/schemas.py` for metadata, tracking, annotations, candidate triplets, predictions, and submissions.
- [x] Implement lazy readers in `src/mabe/io.py`. Enforce integer video/frame IDs, string mouse/action IDs, float coordinates, and stable column ordering.
- [x] Scan `train.csv`, `test.csv`, `train_tracking`, `test_tracking`, and `train_annotation` without rewriting source files.
- [x] Build one row per video containing:
  - [x] lab, video ID, FPS, declared duration, observed frame range, and observed frame count;
  - [x] video/arena geometry and pixel-to-centimeter scale;
  - [x] tracking method and body-part list;
  - [x] present mouse IDs and metadata;
  - [x] tracking and annotation paths plus existence flags;
  - [x] behavior-mask presence, parsed-triplet count, annotation count, and positive-frame count;
  - [x] missing-frame, missing-coordinate, non-finite-coordinate, and extreme-jump rates.
- [x] Implement a canonical `behaviors_labeled` parser in `src/mabe/masks.py`:
  - [x] JSON-decode the list;
  - [x] split each item into exactly agent, target, and action;
  - [x] trim whitespace and stray apostrophes;
  - [x] deduplicate canonical triplets;
  - [x] retain raw entries and a repair log;
  - [x] fail loudly on an unrecoverable entry.
- [x] Verify the expected local facts rather than hard-coding them into modeling code: 8,789 training metadata/tracking rows, 847 annotation files, 7,926 unlabeled MABe22 rows, and 37 repaired actions split into 11 self/solo and 26 pair actions.
- [x] Identify the video with a non-empty mask but missing annotation file. Mark it `ambiguous_missing_annotation`; do not treat it as negative until the dataset convention is independently verified.
- [x] Inspect actual annotations to determine whether self-actions use `target_id="self"` or the agent ID. Store the verified rule and add a test fixture.
- [x] Audit annotation rows for:
  - [x] `start_frame < stop_frame`;
  - [x] bounds relative to tracking frames;
  - [x] missing mice or actions;
  - [x] triplets absent from the repaired mask;
  - [x] overlapping actions for the same directed pair;
  - [x] duplicate intervals.
- [x] Generate counts and duration quantiles by lab, action, video, directed pair, FPS, skeleton, tracker, and mouse count.
- [x] Produce a human-readable anomaly report rather than silently fixing suspicious records.

## Deliverables

- `artifacts/manifests/videos.parquet`.
- `artifacts/manifests/triplets.parquet` containing raw and canonical masks.
- `artifacts/reports/data_audit.json` and a readable Markdown/HTML summary.
- Explicit anomaly tables for malformed masks, missing annotations, invalid intervals, and self-target representation.

## Acceptance gate

All source rows and files reconcile, the 37-action vocabulary is reproducible, every video has exactly one known label-state classification, and unresolved anomalies are explicitly quarantined rather than entering training.

---

# Phase 3 — Sparse masks and frame-target construction

**Status: complete (2026-08-25).**

## Goal

Build correct training targets without materializing an enormous frame × pair × action table.

## Tasks

- [x] Define the four label states:
  - positive frame for an allowed triplet;
  - valid negative frame for an allowed triplet;
  - unknown/masked frame for a disallowed triplet;
  - unlabeled/ambiguous video excluded from supervised loss.
- [x] Implement a per-video `LabelStore` in `src/mabe/labels.py` that loads annotations on demand and exposes:
  - [x] allowed triplets;
  - [x] positive half-open intervals;
  - [x] positive frame arrays for requested triplets/windows;
  - [x] loss masks;
  - [x] solo and pair candidate lists.
- [x] Expand labels only for requested chunks or windows. Use compact boolean/uint8 arrays and retain original frame IDs.
- [x] When a triplet is allowed but has no event at a frame, set target 0. When it is not allowed, set mask 0 and never sample it as a negative.
- [x] Exclude videos with no mask. Exclude the ambiguous missing-annotation video until Phase 2 resolves it.
- [x] Preserve overlaps in ground truth for audit purposes even though the prediction decoder must produce one action per pair/frame.
- [x] Build a canonical action index, solo-action index, pair-action index, and inverse maps. Version them as artifacts.
- [x] Add tests for half-open expansion, disallowed-triplet masking, all-zero valid windows, self-actions, multiple targets, overlapping labels, and non-contiguous tracking frames.
- [x] Implement a function that joins annotation rows with `lab_id` and the raw `behaviors_labeled` string to create an official-metric solution DataFrame for any validation subset.

## Deliverables

- Window/chunk label API.
- Versioned vocabularies and triplet mappings.
- Validation-solution builder compatible with the official metric.

## Acceptance gate

Synthetic tests pass and a sample of real videos round-trips from annotation intervals to frame targets and back without off-by-one errors. No unknown frame can enter supervised loss as a negative.

---

# Phase 4 — Leakage-safe folds and evaluation harness

**Status: complete (2026-08-25).**

## Goal

Create fixed, test-aligned folds and a reusable OOF evaluation pipeline before feature/model iteration.

## Tasks

- [x] Classify labs dynamically as competition labs or auxiliary sources. Confirm the expected 15 competition labs and six auxiliary source IDs/groups.
- [x] Build exact-file hashes plus normalized trajectory signatures to detect byte-identical and near-duplicate tracking sequences, especially CalMS21 copies.
- [x] Assign a stable `duplicate_group_id`; all copies must stay in the same fold whenever those sources are evaluated.
- [x] Build a video-level target vector containing positive-frame duration for each `(lab, action)`.
- [x] Implement a deterministic five-fold grouped assignment that balances these target vectors while keeping complete videos/duplicate groups together.
- [x] Ensure every feasible lab/action has positive support in multiple folds. Document unavoidable rare-action exceptions.
- [x] Store `video_id`, `fold`, `lab_id`, `duplicate_group_id`, label-state, and support statistics in `artifacts/folds/folds.parquet`.
- [x] Primary CV rules:
  - [x] validate only on the 15 competition labs;
  - [x] train target-lab models using the other videos from that lab;
  - [x] optionally add labeled CalMS21/CRIM13 to the training side of every fold for shared models;
  - [x] never add MABe22 to supervised training.
- [x] Implement a secondary leave-one-lab/skeleton/tracker-out stress split. Keep its scores separate from primary selection.
- [x] Build an OOF schema containing video, original frame, agent, target, action, fold, mask, target, probability, model ID, and feature version.
- [x] Implement nested decoder evaluation: parameters used to score fold `k` must be chosen without using fold `k` labels.
- [x] Add fold leakage tests and support-distribution reports.

## Deliverables

- Fixed five-fold artifact.
- Duplicate groups and fold-quality report.
- Evaluation API that produces official overall and detailed diagnostic scores.

## Acceptance gate

No video or duplicate group crosses folds, all primary scores use only competition labs, and a dummy OOF prediction file can be decoded and scored end to end without touching held-out labels during calibration.

---

# Phase 5 — Pose cleanup, canonical skeleton, and time mapping

**Status: complete (2026-08-25).**

## Goal

Convert heterogeneous tracking files into stable, physically meaningful arrays with explicit missingness and reversible frame mapping.

## Tasks

- [x] Implement tracking pivot/loading by actual `video_frame`, mouse, body part, and coordinate.
- [x] Create a reviewed body-part synonym map in configuration. Apply a mapping only when its anatomical meaning is verified for that lab.
- [x] Construct the common core landmarks: nose, left/right ears, body center, and tail base. Add neck and lateral points as optional channels.
- [x] Derive body center from nose/head and tail base when absent; store a `derived_body_center` mask.
- [x] Convert pixels to centimeters only when the scale is finite and credible. Retain raw pixels and a scale-valid flag.
- [x] Detect and mark non-finite coordinates, implausible jumps, long missing spans, and potential mouse-identity swaps.
- [x] Interpolate short gaps using a configurable physical-time maximum. Keep long gaps missing and provide availability masks; never unconditionally fill an entire track.
- [x] Build per-mouse center, body axis, body length, and confidence arrays.
- [x] Implement fold-safe lab normalization statistics, fitted only on each training fold and serialized with the model.
- [x] Implement optional 30-Hz resampling for neural models:
  - [x] linear pose interpolation;
  - [x] nearest-neighbor/mode label mapping;
  - [x] original-to-resampled and resampled-to-original frame maps;
  - [x] no extrapolation beyond observed time.
- [x] Audit AdaptableSnail 25-FPS videos separately using tracking/annotation overlays and OOF experiments. Add a correction rule only if it is video/lab-specific, reproducible, and beneficial.
- [x] Add synthetic tests for translations, rotations, missing gaps, derived center, non-contiguous frames, resampling, and inverse frame mapping.
- [x] Cache cleaned pose arrays by video as float32 plus boolean masks and a version sidecar.

## Deliverables

- Clean pose loader and optional resampler.
- Canonical skeleton configuration.
- Per-fold normalization artifacts.
- Tracking-quality and AdaptableSnail audit reports.

## Acceptance gate

Several videos from every skeleton/tracker family load successfully; all outputs have finite or explicitly masked values; and resampled predictions map back to original frame IDs without drift or off-by-one errors.

---

# Phase 6 — Solo, directed-pair, group, and temporal features

**Status: complete (2026-08-25).**

## Goal

Build one deterministic, train/test-identical feature pipeline with strong geometric invariance and multi-timescale context.

## Tasks

- [x] Implement `features/solo.py` with:
  - [x] position in metric and arena/video-relative coordinates;
  - [x] speed, velocity vector, acceleration, jerk, and path length;
  - [x] body length, elongation, curvature, ear spread, and head/tail angles;
  - [x] body orientation, angular velocity, and turning rate;
  - [x] pause/contact ratios and boundary/center distances;
  - [x] missingness, interpolation, and confidence features.
- [x] Implement `features/pair.py` for ordered `(agent, target)` pairs with:
  - [x] center and cross-keypoint distances;
  - [x] nose-to-nose/body/lateral/tail distances;
  - [x] agent-egocentric target coordinates;
  - [x] approach/separation and radial/tangential velocities;
  - [x] relative velocity, acceleration, jerk, facing angles, heading alignment, and following geometry;
  - [x] contact proportions and past-versus-future interaction changes.
- [x] Implement `features/group.py` with closest-third-mouse distances, nearby-mouse counts, pair-distance rank, between-pair geometry, group centroid, and dispersion.
- [x] Implement FPS-aware lags/windows initially at approximately 0.1, 0.25, 0.5, 1, 2, and 5 seconds. Define exact rounding centrally.
- [x] Use centered temporal context because full videos are available at inference, while handling video edges consistently.
- [x] Include metadata features with explicit missing categories: lab, tracker, FPS, skeleton, mouse count, sex/strain/color/condition, arena type/shape, same-sex/strain, and device indicators.
- [x] Keep raw, metric-scale, and egocentric representations as separable feature groups for ablation.
- [x] Emit float32 feature chunks with stable column order, schema hash, video/frame/pair index, and feature-group tags.
- [x] Add invariant tests:
  - [x] distances unchanged under translation/rotation;
  - [x] egocentric features rotate correctly;
  - [x] `(agent, target)` differs from `(target, agent)` where expected;
  - [x] FPS-aware speed agrees across equivalent trajectories;
  - [x] third-mouse context disappears or masks correctly in dyadic videos;
  - [x] train and inference feature schemas match exactly.
- [x] Add a benchmark script measuring seconds/video, peak memory, output size, and non-finite columns.

## Deliverables

- Versioned feature builders and caches.
- Feature schema/registry.
- Feature correctness tests and runtime benchmark.

## Acceptance gate

The same function produces byte/schema-compatible training and inference features; invariant/directionality tests pass; and no unexplained NaN/Inf reaches a model.

---

# Phase 7 — End-to-end lab/action XGBoost baseline

**Status: complete (2026-08-25).**

## Goal

Produce the first trustworthy OOF score and Kaggle-shaped CSV with a strong, economical model.

## Tasks

- [x] Implement separate solo and pair training routes.
- [x] For each competition `(lab, action)` with sufficient positives, train one binary XGBoost classifier.
- [x] For sparse combinations, implement a configured fallback hierarchy:
  1. pooled action model with lab features;
  2. related auxiliary-lab data when label semantics match;
  3. conservative no-prediction behavior when no reliable model exists.
- [x] Build the initial sampler:
  - [x] keep all positives up to a documented cap;
  - [x] sample valid negatives by video and pair;
  - [x] include near-boundary/confusable negatives;
  - [x] preserve complete held-out videos for OOF;
  - [x] cap roughly one million rows per action/model only as a configurable starting point.
- [x] Use histogram tree building, deterministic seeds, early stopping, and GPU acceleration when available with a CPU fallback.
- [x] Train one fold first. Save model, feature list, training support, best iteration, and validation probabilities.
- [x] Run all five folds only after the one-fold pipeline and metric CSV pass.
- [x] Persist masked OOF probabilities in the Phase 4 schema. Never save only hard labels.
- [x] Implement baseline decoding in `decode.py`:
  - [x] centered probability smoothing;
  - [x] initial action thresholds;
  - [x] choose at most one action per directed pair/frame by calibrated margin;
  - [x] allow different targets to coexist;
  - [x] run-length encode half-open intervals.
- [x] Implement `submission.py` validators for columns, dtypes, row IDs, masks, bounds, self-target convention, and overlap errors.
- [x] Generate a fake held-out submission CSV and score it through the untouched official metric.

## Deliverables

- Five-fold XGBoost models and OOF probabilities.
- Baseline decoder parameters.
- Overall/per-lab/action CV report.
- Valid held-out submission CSV.

## Acceptance gate

The complete path from raw held-out tracking files to official score runs without annotations entering feature/model inference, and the emitted CSV passes every submission/metric check.

---

# Phase 8 — Strengthen and calibrate the GBDT system

**Status: complete (2026-08-26).**

## Goal

Maximize the tree baseline before introducing expensive deep models.

## Tasks

- [x] Run feature-family ablations on fixed folds: raw/metric, egocentric, temporal windows, metadata, boundary, and group context.
- [x] Inspect split/gain importance, permutation importance on held-out videos, and stability across folds. Remove expensive unstable features.
- [x] Add hard-negative mining using only prior OOF predictions. Retrain and verify gains with fresh cross-fitting.
- [x] Tune class/positive weights, negative ratios, depth, learning rate, regularization, row/column sampling, and maximum rows per model.
- [x] Compare per-lab models against pooled action models and hybrid fine-tuning. Select routing separately by lab/action using nested OOF evidence.
- [x] Add LightGBM or CatBoost candidates only after XGBoost is stable. Retain them only if they improve exact score or add useful error diversity.
- [x] Implement nested lab/action decoder tuning over:
  - [x] probability threshold;
  - [x] smoothing duration;
  - [x] optional hysteresis;
  - [x] short-gap fill;
  - [x] minimum-run duration.
- [x] Optimize against the exact metric, not ordinary row-level F1. Fit decoder parameters on folds other than the fold being reported.
- [x] Add hierarchical shrinkage: poorly supported lab/action parameters fall back toward action-global and then global defaults.
- [x] Record score deltas and resource deltas for every promoted change.

## Deliverables

- Final GBDT model registry and routed OOF probabilities.
- Nested decoder parameters with support/fallback metadata.
- Feature and hyperparameter ablation report.

## Acceptance gate

No promoted change depends on same-fold threshold fitting, and the final GBDT system materially improves the exact OOF score or establishes a well-documented plateau.

---

# Phase 9 — Temporal neural data pipeline and first TCN/CNN-RNN model

> **Status: complete (2026-08-26).**
>
> Five-fold `temporal-phase9-v1` completed on all 352 primary-validation videos and
> 106,642,451 masked OOF rows. Exact neural OOF F-beta is `0.06764260445409719`
> versus Phase 8 GBDT `0.30590749338251494`; this first small temporal model is not
> a standalone replacement. It is retained as a diversity candidate because its
> sampled probability correlation with GBDT is `0.015391230324394949`, error
> Jaccard is `0.1417841869716559`, and it wins 5 lab/action comparisons. Phase 11
> must prove an exact cross-fitted ensemble gain before including it in the final model.

## Goal

Train a temporal model that either beats GBDT or makes sufficiently different errors to improve an ensemble.

## Tasks

- [x] Build memory-mapped/NPY/Zarr-style per-video sequence artifacts from cleaned 30-Hz pose and selected invariant features.
- [x] Define tensors and masks explicitly, for example:
  - solo input `[time, mouse, solo_features]`, outputs `[time, mouse, 11]`;
  - pair input `[time, agent, target, pair_features]`, outputs `[time, agent, target, 26]`;
  - action-validity, mouse-validity, keypoint-validity, and supervised-loss masks.
- [x] Implement an action-rich window sampler:
  - [x] initial length around 512 frames at 30 Hz;
  - [x] balance labs and actions rather than raw frame count;
  - [x] oversample rare positives without duplicating validation;
  - [x] include background/hard-negative windows;
  - [x] preserve complete fold isolation.
- [x] Implement the first PyTorch model:
  - multi-scale 1D convolution stem;
  - residual/dilated TCN blocks;
  - optional BiGRU fusion;
  - separate solo and pair output heads;
  - lab embedding and metadata conditioning;
  - masked logits for unavailable actions.
- [x] Start with masked `BCEWithLogitsLoss`, class/lab weights, AdamW, warmup plus cosine decay, EMA, gradient clipping, and mixed precision.
- [x] Implement augmentations behind independent switches: small coordinate noise, time stretch, keypoint dropout, left/right swap, mouse permutation with all metadata/labels permuted, and biologically valid geometric transforms.
- [x] Add unit tests proving augmentations preserve label/triplet alignment.
- [x] Train one fold and verify learning curves, masked loss, window reconstruction, and runtime before five-fold training.
- [x] Implement sliding-window inference with stride around 128, edge discard/downweight around 32 frames, overlap averaging, and mapping from 30-Hz time back to output frames.
- [x] Persist full masked OOF probabilities using the same schema as GBDT.
- [x] Evaluate the neural model alone with nested decoder selection and compare per-lab/action errors to GBDT.

## Deliverables

- Temporal dataset/cache and sampler.
- First TCN/CNN-BiGRU family with five-fold checkpoints if one-fold results justify it.
- Neural OOF probabilities and diagnostic report.

## Acceptance gate

Sequence reconstruction and frame mapping tests pass; the model completes inference within a projected Kaggle budget; and it either improves standalone exact OOF or has demonstrably complementary errors to GBDT.

---

# Phase 10 — Selective advanced temporal/model experiments

> **Status: complete (2026-08-26).**
>
> Seventeen fixed-fold screens compared receptive fields, lightweight
> convolution-attention, macro-soft-F1, lab adapters, 4/5/7-keypoint inputs, and
> augmentation policies. Only `receptive_short_256` cleared the screen, so it was
> trained across all five folds and evaluated on 106,642,451 OOF rows with the
> same nested decoder. Its exact F-beta was `0.06360493243054911`, below Phase 9
> by `0.004037672023548072`; therefore no Phase 10 model is retained. ST-GCN was
> not compute-justified on this CPU-only host, and TTA was not run because no
> individual augmentation beat the no-augmentation validation reference.

## Goal

Add only high-value model diversity supported by the competition evidence and available compute.

## Tasks

- [x] Compare sequence lengths and receptive fields using the same folds and decoder protocol.
- [x] Test Squeezeformer or a lightweight convolution-attention encoder as an alternative temporal family.
- [x] Test a small macro-soft-F1 loss component mixed with BCE; do not replace BCE unless exact OOF improves reliably.
- [x] Test shared all-lab pretraining followed by per-lab fine-tuning or lab adapters.
- [x] Test 4-, 5-, and 7-keypoint model variants with keypoint dropout.
- [x] If compute permits, implement a skeleton GNN/ST-GCN mouse encoder followed by an egocentric temporal encoder.
- [x] Test TTA only for augmentations that improved validation. Do not assume horizontal flip helps.
- [x] Keep vanilla Transformer and MABe22 self/semi-supervised pretraining low priority; historical results did not consistently improve with them.
- [x] Measure prediction correlation and error overlap with existing models, not only standalone score.
- [x] Stop an experiment family when two well-run configurations fail to improve score or ensemble diversity.

## Deliverables

- Comparable model cards with architecture, features, folds, score, runtime, memory, and prediction-diversity statistics.
- OOF predictions only for models eligible for ensemble selection.

## Acceptance gate

Each retained advanced model improves the exact score or a cross-fitted ensemble. Models that merely add complexity are excluded from final inference.

---

# Phase 11 — Leakage-safe blending and stacking

> **Status: complete (2026-08-26).**
>
> All three available OOF streams were aligned exactly across 835 parts and
> 106,642,451 rows with zero missing routes, duplicate keys, mask mismatches, or
> target mismatches. Non-negative GBDT/Phase 9 weights were learned with
> held-fold exclusion at global, action, lab, and hierarchically shrunk
> lab/action scopes. The cross-fitted action blend won with exact F-beta
> `0.3079032676020558`, improving the Phase 8 GBDT member
> `0.30590749338251494` by `0.0019957742195408357`. Its projected 200-video
> inference time is `1047.12500802` seconds versus a 25,920-second budget with
> 20% headroom. The conditional stacker was not triggered because probability
> blending produced a leakage-free exact gain. The non-retained Phase 10 stream
> was alignment-audited but excluded from optimized blending.

## Goal

Combine only complementary systems and re-optimize decoding on the combined probabilities.

## Tasks

- [x] Align every base OOF stream on the exact video/frame/agent/target/action index and validity mask.
- [x] Fail on missing or duplicated keys rather than silently filling predictions.
- [x] Establish simple equal-weight blends as reference points.
- [x] Learn non-negative blend weights with greedy hill climbing or constrained optimization using cross-fitting.
- [x] Compare global, per-action, and per-lab weights. Apply shrinkage/minimum-support rules to prevent overfitting.
- [x] For every candidate blend, re-run nested threshold/smoothing/decoder selection. Never blend decoded interval lists.
- [x] Evaluate the conditional stacker gate. It did not trigger because the cross-fitted action blend improved exact OOF; therefore geometry/meta-XGBoost training and its additional inference route were not added.
- [x] Select a per-lab best model/blend/stack only through OOF evidence.
- [x] Penalize ensembles that cannot fit comfortably within Kaggle runtime/memory.
- [x] Freeze one final model registry containing model IDs, routes, weights, decoder parameters, normalization versions, and fallbacks.

## Deliverables

- Final cross-fitted ensemble OOF predictions.
- Final exact CV and detailed error report.
- Frozen model/decoder registry.

## Acceptance gate

The ensemble improves leakage-free exact OOF over its strongest member, and every required component has a feasible full-test inference path.

---

# Phase 12 — Kaggle inference and submission hardening

> **Status: complete (2026-08-27).**
>
> Hidden-rerun hardening v6 removes the Polars `empty_as_null` keyword that is
> accepted locally but absent from Kaggle's competition image. This exact
> incompatibility caused manifest construction to fail before model inference;
> v4 then silently scored its emergency row, while v5 correctly surfaced the
> traceback. A compatibility regression exercises manifest construction with
> the older `DataFrame.explode` signature, and the notebook now prints all core
> runtime package versions before inference.
>
> Hidden-rerun hardening v5 fixes dual-mount selection: when both the one-video
> public data and the private replacement are visible, inference now selects
> the candidate with the largest `test.csv` and tracking collection instead of
> the lexicographically first path. It also writes validated partial submissions
> every five videos and refuses to silently score an emergency row if all model
> routes fail. A regression fixture reproduces the public/private mount layout.
>
> Hidden-rerun hardening v4 removes the remaining collection-wide manifest
> failure boundary. Tracking statistics are scanned one parquet at a time, and
> malformed masks, missing/unreadable tracking, malformed metadata, invalid
> mouse references, and duplicate routes are quarantined per video. The exact
> dynamic hidden-manifest path passed on 368 real competition-lab videos across
> all 15 labs, with 368/368 tracking scans successful and 353 mask-eligible
> videos retained. The public output remains byte-identical at 842 rows
> (`be3e944097eeb8720705617f58bcc9450e8b4e4ce6b35f90de20d5581a66f126`).
>
> Hidden-rerun hardening v3 prewrites a schema-valid, behavior-mask-valid
> emergency submission before expensive inference and recovers completed
> per-video checkpoints after any ordinary top-level exception. This closes the
> remaining failure boundary outside the v2 model/video isolation logic.
>
> Hidden-rerun hardening v2 additionally exercised the highest-burden masked
> validation video from every one of the 15 labs. The held-fold route completed
> 15/15 full blends with zero failures, reproduced all 8,499 Phase 11 intervals,
> and matched exact F-beta `0.23215493891006256`. The exact five-fold Kaggle
> deployment route also completed 15/15 full blends with zero failures, emitted
> 7,752 valid intervals, and scored `0.31076126267931853` on the masked stress
> set. Per-model and per-video exception isolation now prevents a single hidden
> edge case from terminating the complete submission.
>
> The annotation-masked raw held-out simulation recomputed the held-fold XGBoost
> and temporal predictions for representative 2/3/4-mouse videos and reproduced
> all 2,557 Phase 11 intervals with exact F-beta `0.15665553606697935` and zero
> difference. The final restricted-network, clean-kernel notebook run verified
> the 1,082-file offline bundle, produced the same 842-row public-test CSV
> byte-for-byte (`be3e944097eeb8720705617f58bcc9450e8b4e4ce6b35f90de20d5581a66f126`),
> and reported zero validation errors. The distribution-adjusted CPU projection,
> including 20% contingency, is 5,737.99 seconds with 82.3% headroom against
> the nine-hour limit. CUDA was unavailable on this host, so GPU acceleration was
> not measured; the measured CPU route independently passes the release gate.

## Goal

Produce a deterministic, offline, inference-only notebook/script that handles the approximately 200 hidden videos within the nine-hour limit.

## Tasks

- [x] Implement `scripts/run_inference.py` independently of training code paths while reusing the same tested preprocessing/features.
- [x] Make it discover the hidden dataset root and dynamically read the full hidden `test.csv` and tracking tree.
- [x] Route each video using actual lab, skeleton, FPS, and mouse count from test metadata.
- [x] Stream one video or bounded chunks at a time; explicitly release feature/model intermediates.
- [x] Package all required model weights, feature lists, normalization statistics, vocabularies, decoder parameters, and registry files as Kaggle inputs.
- [x] Avoid internet access and avoid runtime package installation where possible. The Kaggle route uses only preinstalled packages, so no bundled wheel or installation step is required.
- [x] Add checkpointed progress and resumable per-video temporary predictions with exact dataset/configuration fingerprints.
- [x] Decode by directed pair, filter exact valid test triplets, map frames correctly, and emit maximal half-open intervals.
- [x] If all frames are background, add one valid one-frame fallback using the highest-confidence allowed candidate rather than an arbitrary invalid row.
- [x] Validate `submission.csv`:
  - [x] exact seven columns in required order;
  - [x] integer unique `row_id` enumeration;
  - [x] valid video/agent/target/action values;
  - [x] integer `start_frame < stop_frame` within the required frame system;
  - [x] no overlapping different actions for any video/agent/target;
  - [x] no invalid triplets;
  - [x] non-empty file.
- [x] Build an end-to-end simulation that substitutes held-out train metadata/tracking for hidden test, hides annotations, runs inference, emits CSV, and scores it with the official metric.
- [x] Stress-test the highest-burden held-out video from every trained lab with exact held-fold parity and the five-fold deployment route.
- [x] Stress-test dynamic manifest construction on more than 200 real videos while scanning tracking files independently and recording per-video quarantine reasons.
- [x] Isolate XGBoost, temporal, blend, decode, and per-video failures so unaffected hidden videos continue safely.
- [x] Benchmark the CPU route on stratified long/short, 2/3/4-mouse, and different-skeleton videos; record CUDA unavailability and verify that CPU alone exceeds the required 20% projected time headroom.
- [x] Run the final notebook with internet restricted and from a clean kernel.

## Deliverables

- Inference script and thin Kaggle notebook wrapper.
- Packaged model artifact bundle.
- Runtime/memory benchmark.
- Valid simulated and final `submission.csv` outputs.

## Acceptance gate

The clean offline run completes within projected limits, the simulated held-out score matches the development pipeline, and submission validation reports zero errors.

---

# Phase 13 — Final reproducibility, reporting, and release

> **Status: complete (2026-08-27).**
>
> The frozen five-fold interval assembly reproduces all 100,086 final OOF
> predictions and exact F-beta `0.3079032676020558` with zero difference from
> the Phase 11 registry. All 835 probability parts (106,642,451 rows) exist and
> are individually hashed. The `mabe-final-v1` release contains 46 verified
> local files, 856 essential artifact references, three inspected diagnostic
> plots, a complete clean-notebook archive, and an explicit inventory separating
> nine non-selected experiment groups. The Phase 12 bundle re-verifies with
> 1,082 files and zero submission-validation errors. The final suite passes 149
> tests, Ruff, and strict mypy.

## Goal

Freeze the exact solution so its score, artifacts, and decisions can be reproduced and audited.

## Tasks

- [x] Re-run the final fold/OOF assembly from frozen artifacts and verify the recorded exact score.
- [x] Save per-lab/action score tables, confusion/error summaries, and confidence/prevalence plots.
- [x] Document all special-case rules, especially AdaptableSnail frame conversion and any excluded/quarantined videos.
- [x] Record the final data fingerprint, fold version, feature version, model IDs, dependency lock, decoder registry, and notebook version.
- [x] Add a `README` runbook with commands for audit, folds, features, training, OOF evaluation, ensemble construction, and inference.
- [x] Separate essential final artifacts from experimental checkpoints.
- [x] Audit licenses/attribution for copied official metric code, external solution ideas, optional pretrained weights, and auxiliary data; explicitly record that competition rules govern data/model redistribution and that no repository-wide license is declared.
- [x] Archive the final Kaggle notebook output, logs, submission, and score metadata.

## Deliverables

- Reproducible final runbook and artifact manifest.
- Final OOF/model report.
- Final Kaggle-ready bundle.

## Acceptance gate

A clean environment can reproduce the final validation submission and run hidden-test-style inference using only the documented configuration and frozen artifacts.

---

# Implementation order summary

Implement one phase at a time in this order:

- Phase 0: Project foundation
- Phase 1: Exact metric and interval contract
- Phase 2: Data/schema audit
- Phase 3: Sparse masks and targets
- Phase 4: Leakage-safe folds/evaluation
- Phase 5: Pose cleanup and time mapping
- Phase 6: Feature system
- Phase 7: End-to-end XGBoost baseline
- Phase 8: GBDT improvement and nested calibration
- Phase 9: First temporal model
- Phase 10: Selective advanced models
- Phase 11: Ensemble/stacking
- Phase 12: Kaggle inference hardening
- Phase 13: Final reproducibility release

The first major milestone is the end of Phase 7: raw held-out tracking must produce a valid interval CSV and an exact official OOF score. The second is the end of Phase 8: the best economical GBDT system and its calibrated decoder are frozen. Deep models and ensembling begin only after those foundations are reliable.
