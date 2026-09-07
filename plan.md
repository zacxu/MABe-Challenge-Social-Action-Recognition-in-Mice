# Audited solution plan for MABe mouse behavior detection

## Executive judgment

The original plan had the right foundation: this is frame-wise temporal behavior recognition over multi-mouse pose trajectories; sparse annotation masks, grouped validation, relational features, temporal context, and probability post-processing are all essential.

However, several parts needed material correction:

- The official metric is frame-level, not event-matching. Splitting one interval into two does not incur an extra event penalty if the predicted frame set is unchanged.
- Intervals are half-open: `[start_frame, stop_frame)`. An annotation `(1000, 1040)` labels frames `1000..1039`, not frame `1040`.
- The score is not a simple mean over videos. It pools TP/FP/FN by action within each lab, macro-averages actions represented by ground-truth keys plus any allowed predicted-only action keys, and then macro-averages labs. Every scored lab therefore has equal weight regardless of its size. The predicted-only case is a subtle consequence of the official scorer's `defaultdict` access and must be preserved exactly.
- Predictions outside a video's `behaviors_labeled` mask are ignored by the scorer. They must still be filtered before submission for correctness and efficiency.
- The scorer allows one agent to act toward multiple targets in the same frame, but it raises an error if two different actions overlap for the same `(video, agent, target)` pair. Decoding must therefore choose at most one action per directed pair per frame, not one action per agent across all targets.
- Leave-one-lab-out validation is useful as a robustness diagnostic, but it is not the primary model-selection scheme. Historical competition solutions established that the hidden test used the 15 competition labs already represented in training; the CalMS21, CRIM13, and MABe22 sources were auxiliary and absent from test.
- The local metadata has 8,789 tracking videos but only 847 annotation parquet files. The 7,926 MABe22 videos are unlabeled and must never be treated as negative examples. One video has a non-empty annotation mask but no annotation file; it must be audited rather than automatically assumed to be all-negative, because the scorer cannot represent a completely empty ground-truth video without a solution row.
- After stripping malformed apostrophes and duplicates in `behaviors_labeled`, the local data contains 37 canonical actions: 11 self/solo and 26 pair actions.

The strongest practical final system is therefore a lab-aware ensemble of:

1. per-lab/per-action boosted trees on invariant multi-scale features; and
2. a shared temporal neural model with lab conditioning, separate solo/pair handling, masking, and optional per-lab fine-tuning.

The ensemble and all decoding parameters must be learned from leakage-free out-of-fold predictions.

---

## 1. Freeze the metric and data contract first

Copy the functions from `mabe-f-beta.ipynb` unchanged into the eventual project metric module. Do not rewrite the metric from memory or replace it with `sklearn.metrics.f1_score`.

Add unit tests that prove all of the following:

- `[0, 1)` represents exactly frame 0.
- `[0, 10)` represents ten frames.
- Adjacent intervals `[0, 10)` and `[10, 20)` are equivalent to `[0, 20)` for the scorer.
- Two predictions with different actions may not overlap for the same video/agent/target pair.
- The same agent may have predictions for different targets in the same frame.
- Predictions not listed in `behaviors_labeled` are ignored.
- Labs are weighted equally. Within a lab, ground-truth actions and allowed predicted-only actions created by the scorer are weighted equally; an allowed predicted-only action contributes zero F-score.
- Empty submissions fail; the inference pipeline needs a safe, valid fallback row.

Use half-open intervals everywhere: labels, temporal slicing, resampling, run-length encoding, validation, and submission. Reject zero-length intervals locally even though the scorer's assertion permits `start_frame == stop_frame`.

This is the first completion gate. Nothing should be modeled until an exact-copy scorer and the interval round-trip tests pass.

---

## 2. Audit the real dataset, including missing annotations

Build one video manifest from `train.csv`, `test.csv`, tracking paths, and annotation paths. Record at least:

- `video_id`, `lab_id`, FPS, duration, scale, video/arena geometry, tracking method;
- present mice and mouse metadata;
- body-part names and a canonical skeleton signature;
- minimum/maximum tracking frame, missing-frame rate, missing-keypoint rate, jump/outlier rate;
- whether `behaviors_labeled` is present;
- whether an annotation file is present;
- number of intervals and positive frames by action and directed pair.

Handle the four label states explicitly:

1. Triplet is in `behaviors_labeled` and overlaps an annotation interval: positive.
2. Triplet is in `behaviors_labeled` but has no interval at a frame: valid negative.
3. Video has a non-empty mask but no annotation file: ambiguous. Exclude it from primary supervised training until the dataset convention is verified; use it as all-negative only if independent evidence confirms that missing file means zero labeled events.
4. Video has no behavior mask, as with MABe22: unlabeled; neither positive nor negative.

Parse `behaviors_labeled` as JSON, then split each entry into exactly three fields. Strip whitespace and the stray apostrophes present in some local rows, deduplicate canonical triplets, and log every repair. Preserve the raw string for exact metric reproduction. Determine the self-action target representation from the actual annotations and keep it consistent end to end; do not silently convert `self` to an agent ID based only on the prose description.

Audit annotations for invalid bounds, overlaps, actions outside the mask, non-existent mice, and disagreement between duration/FPS and observed tracking frames. Detect duplicate or near-duplicate trajectories—especially the documented CalMS21 duplicates—and assign one duplicate-group ID so copies can never leak across any diagnostic split.

Produce a compact EDA report by lab/action with interval count, positive-frame count, prevalence, duration quantiles, number of videos, and number of directed pairs. Frame counts matter more than event counts to the official score, but event-duration statistics are still useful for sampling and temporal windows.

---

## 3. Establish the correct validation design

Use a fixed five-fold video-grouped split over the 15 competition labs. Optimize assignment so each fold has a similar distribution of positive duration for every `(lab, action)` combination. All frames and all directed pairs from a video—and any detected duplicate trajectory—must remain in one fold.

Treat the auxiliary sources as follows:

- CalMS21 and CRIM13: optionally include their labeled data in every training fold for shared representation learning, but never include them in the primary validation score because they do not occur in hidden test.
- MABe22: unlabeled. Exclude it from supervised training. Self-supervised pretraining is a late, low-priority experiment because a top solution reported no gain from semi-supervised use of this data.

Maintain two scores:

- Primary score: exact official metric over out-of-fold predictions for the 15 competition labs.
- Stress score: leave-one-lab-out or leave-one-skeleton/tracker-out evaluation for domain robustness. This is diagnostic, not the criterion used to select the final submission.

Thresholds and post-processing must also be validated without leakage. For a reported held-out fold, choose lab/action thresholds and smoothing parameters using only the other folds' OOF predictions. Never optimize a threshold on a fold and report the score on that same fold.

Store every OOF probability with `video_id`, frame, agent, target, action, fold, and mask. These artifacts are required for honest thresholding, stacking, ensembling, and error analysis.

---

## 4. Clean and normalize tracking robustly

Build both metric-scale and invariant representations; neither is sufficient alone.

Tracking cleanup:

- sort by actual `video_frame` and preserve its mapping;
- convert coordinates to centimeters where `pix_per_cm_approx` is valid;
- detect implausible jumps and likely identity swaps before deriving velocity;
- interpolate only defensible gaps and retain missingness/visibility masks;
- never turn a long missing track into a confident stationary mouse by unconditional forward/back filling;
- derive `body_center` from available anatomy, preferably the midpoint of nose/head and tail base, when it is absent;
- preserve a confidence flag for every derived landmark.

Canonical skeleton:

- map known synonyms such as `head -> nose`, `spine_1 -> neck`, and `hip_left/right -> lateral_left/right` only after verifying their definitions;
- use a strong common core: nose, left/right ear, body center, and tail base;
- add neck and lateral points when available;
- train 4-, 5-, and 7-keypoint variants only if their OOF errors are genuinely diverse.

Spatial representations:

- centimeters and arena/video-relative location;
- translation-invariant pairwise distances;
- an egocentric agent frame: translate to the agent center, rotate by its body axis, and optionally scale by robust body length;
- lab-standardized coordinates/statistics as an additional channel, fitted on training folds only;
- distance to arena/video boundaries, while flagging uncertainty when the arena origin is not known.

Temporal normalization:

- use physical-time lags/windows based on FPS for the tree pipeline;
- for temporal neural models, resample pose to a 30 Hz internal grid using linear interpolation and map predictions back to the required output frame system;
- use nearest-neighbor/mode resampling for labels, never linear interpolation of binary labels;
- preserve original FPS as a feature.

Add a dedicated audit for the known AdaptableSnail 25-FPS inconsistency. Historical solutions found some 25-FPS tracking sequences whose annotations/output frame system behaved like 30 FPS, plus mouse-ID swaps. Implement any `30/25` correction only behind a lab/video rule proven by plots and OOF score; do not apply it globally.

---

## 5. Engineer features that match the behaviors

For every mouse, compute multi-lag and rolling features in physical time:

- position, velocity, acceleration, jerk, speed, path length, pause ratio;
- body length, elongation/compactness, curvature, ear spread;
- orientation, angular velocity, turning rate, head and tail angles;
- arena-center and boundary distances;
- missingness and tracking-quality features.

For every directed agent-target pair, compute:

- center-center and all useful cross-keypoint distances;
- nose-to-target nose, body, lateral points, and tail base;
- relative position in the agent's body frame;
- approach/separation velocity and radial/tangential velocity;
- relative speed, acceleration, jerk, heading alignment, facing angles, following geometry;
- contact/near-contact proportions over several windows;
- past-versus-future changes that capture approach-contact-disengage patterns.

For videos with three or four mice, add group context rather than modeling a pair in isolation:

- closest third-mouse distance to agent, target, and their midpoint;
- number of nearby mice;
- whether a third mouse lies between the pair;
- group centroid, dispersion, and pair rank among all distances.

Use a compact ablation-driven feature set. More rolling statistics are not automatically better. Compare feature families on fixed OOF folds, inspect importance/stability, and remove expensive features that do not improve the exact score.

---

## 6. Build the strongest economical baseline: lab/action GBDTs

Train separate solo and pair pipelines. Within them, train one binary classifier per `(lab, action)` when the lab has enough positives; use a pooled action model with lab features or hierarchical fallback for sparse cases.

Start with XGBoost because strong competition solutions established it as a high-performing baseline. Add LightGBM and CatBoost only when their OOF predictions add ensemble diversity.

For each classifier:

- retain all positive frames or action-rich windows;
- sample negatives by video and temporal distance from positives, then add hard negatives from OOF mistakes;
- ensure all sampled negatives come only from valid annotation masks;
- weight labs/actions to reflect the macro metric rather than raw dataset size;
- early-stop on a metric correlated with downstream F1, while selecting the final configuration by nested exact-metric OOF;
- cap training rows per model for reproducibility and runtime.

This baseline should already include egocentric geometry, multi-scale temporal features, group context, lab-aware thresholds, and correct pair-wise decoding. It is the first serious benchmark, not a disposable toy model.

---

## 7. Add temporal neural models for complementary signal

Use separate solo and pair heads/models, with a shared encoder where helpful. A practical first neural family is:

`invariant features -> multi-scale 1D convolutions/TCN -> BiGRU or Squeezeformer -> frame logits`

Recommended setup:

- 30-Hz sequences of roughly 512 frames, tested against shorter/longer contexts;
- overlapping sliding windows, e.g. stride 128;
- discard or downweight unreliable window edges and average overlapping probabilities;
- include lab embedding, FPS, mouse metadata, landmark availability, and candidate-action mask;
- represent all mice jointly or include explicit third-mouse context;
- use masked BCE as the dependable loss; test a small macro-soft-F1 component only by ablation;
- sample action-rich windows and use class/lab-aware weighting;
- EMA, warmup plus cosine decay, mixed precision, and early stopping from leakage-free OOF score.

Useful augmentation candidates are small coordinate noise, time stretching, mouse-ID permutation with metadata/labels permuted identically, keypoint dropout, left/right landmark swaps, and valid rotations/reflections. Each must be ablated: horizontal flipping did not help at least one gold solution, and arena/device asymmetries can make geometric transforms invalid.

A vanilla Transformer is not the next automatic upgrade; a gold solution could not beat its TCN with one. Skeleton GNN/ST-GCN plus an egocentric temporal encoder is a later high-compute family that can add diversity, not a prerequisite for a competitive system.

---

## 8. Decode probabilities according to the scorer

Work in probability space until the final run-length encoding.

For each lab/action, tune on nested OOF predictions:

- decision threshold;
- probability smoothing width;
- optional hysteresis;
- short-gap filling and minimum positive-run length, but only when these change frames in a way that improves the official metric.

Use shrinkage/fallbacks for rare combinations: lab-action parameters should revert toward action-global or lab-global values when OOF support is too small.

At every frame and for each directed `(agent, target)` pair:

1. mask actions not listed for that exact video/agent/target;
2. retain actions above their tuned thresholds;
3. if more than one remains, choose the largest calibrated margin, such as `probability / threshold`;
4. otherwise assign background.

Do not compete different targets against each other: the official scorer explicitly allows the same agent to act toward multiple targets simultaneously.

Convert the resulting frame labels to maximal half-open runs with run-length encoding. Then validate:

- `start_frame < stop_frame`;
- integer frames and in-range bounds;
- no overlapping different actions for a given video/agent/target;
- only valid test triplets;
- exact target representation for self-actions;
- unique enumerated `row_id`.

Because the metric operates on frame sets, merging or splitting adjacent runs with no change to those sets cannot change the score. Report frame precision/recall, not invented event-boundary penalties.

---

## 9. Ensemble without leakage

First blend genuinely different OOF probability streams: per-lab XGBoost, pooled/shared GBDT, TCN/GRU/Squeezeformer, and any successful graph model. Learn non-negative ensemble weights by greedy hill climbing or constrained optimization on OOF predictions, preferably per lab or per action with regularization.

Re-run nested threshold/post-processing selection after blending. Do not average already-decoded event lists.

If simple blending saturates, use leakage-safe stacking: train a lab/action XGBoost meta-model on base geometric features plus neural OOF probabilities. For final training, each neural prediction used by the stacker must come from a model that did not train on that video's fold.

Per-lab model selection is valid and often better than one universal blend because the metric, annotations, tracking setup, and behavior definitions are lab-specific. Select it only from OOF evidence.

---

## 10. Diagnostics and experiment discipline

For every experiment record:

- exact overall OOF metric;
- per-lab and per-lab/action F1, TP, FP, FN;
- precision/recall and positive-frame prevalence;
- performance by FPS, skeleton, tracking method, mouse count, event duration, and missingness;
- model runtime, peak memory, feature-generation cost, and inference cost;
- fold/config/code/data hash.

Do not use mean per-video F1 as the main diagnostic because it does not reproduce the official aggregation. Video-level plots and error galleries remain useful for discovering tracking swaps, boundary problems, and behavior confusions.

Run ablations one family at a time on fixed folds. Promote a change only if it improves the exact nested OOF score or provides demonstrable ensemble diversity.

---

## 11. Build Kaggle inference early, not last

The scored notebook has no internet and a 9-hour CPU/GPU limit. Build an inference-only notebook as soon as the GBDT baseline exists, then keep it synchronized with training.

The pipeline must dynamically process the full hidden `test.csv` exposed at scoring time:

`metadata -> tracking load -> cleanup/resampling -> feature construction -> routed model inference -> calibrated blend -> pair-wise decoding -> half-open intervals -> submission.csv`

Requirements:

- package trained weights, thresholds, feature schema, canonical mappings, and normalization statistics as Kaggle inputs;
- fit no statistics on hidden test unless the operation is explicitly per-video and label-free;
- route by actual hidden-test `lab_id`, skeleton, FPS, and mouse count;
- stream video/chunks and avoid a giant frame-agent-target-action table;
- profile approximately 200 videos with headroom below nine hours;
- use deterministic seeds and bounded parallelism;
- validate the final file with the same schema and overlap checks as local OOF submissions;
- guarantee at least one valid fallback prediction because the official metric rejects an empty submission.

Before submission, run the complete inference notebook on held-out training videos with annotations hidden, then score its emitted CSV through the untouched official metric. This catches train/inference skew and interval bugs that frame-array validation misses.

---

## Recommended execution order and gates

1. **Metric gate:** exact scorer copied; interval, masking, aggregation, overlap, and round-trip tests pass.
2. **Data gate:** manifest complete; 37 canonical actions; positive/negative/ambiguous/unlabeled states correct; duplicate and FPS anomalies logged.
3. **Validation gate:** fixed five folds; primary test-lab OOF score; auxiliary sources handled correctly; nested calibration implemented.
4. **GBDT gate:** per-lab/action XGBoost with invariant, temporal, and group features; valid OOF probabilities and end-to-end CSV.
5. **Temporal gate:** one TCN/CNN-RNN family beats or complements GBDT on OOF.
6. **Calibration gate:** lab/action thresholding and pair-wise exclusivity optimized without leakage.
7. **Ensemble gate:** OOF-selected blend or stack improves the exact score.
8. **Deployment gate:** hidden-test-style notebook completes within resource limits and produces a validated non-empty `submission.csv`.

The resulting development path is:

`exact metric -> label/data audit -> test-aligned grouped CV -> robust pose normalization -> lab/action GBDT -> temporal neural model -> nested calibration -> OOF ensemble -> production inference`

---

## Evidence behind the revisions

- Official competition overview and runtime rules: https://www.kaggle.com/competitions/MABe-mouse-behavior-detection/overview
- Exact local metric: `mabe-f-beta.ipynb`
- 2nd place: separate solo/pair temporal models, 30-FPS resampling, five-fold lab-stratified video validation, lab/action thresholds, and AdaptableSnail correction: https://www.kaggle.com/competitions/MABe-mouse-behavior-detection/writeups/2nd-place-solution
- 3rd place: LSTM plus XGBoost, lab embeddings, lab/action thresholds, probability smoothing, and calibrated conflict resolution: https://www.kaggle.com/competitions/MABe-mouse-behavior-detection/writeups/3rd-place-solution
- 4th place: lab-specific XGBoost/neural selection, common five-keypoint skeleton, metadata/boundary features, joint lab training followed by per-lab fine-tuning: https://www.kaggle.com/competitions/MABe-mouse-behavior-detection/writeups/4th-place-solution-xgb-nn-ensemble
- 5th place: egocentric GNN plus Squeezeformer and group-aware relational features: https://www.kaggle.com/competitions/MABe-mouse-behavior-detection/writeups/5th-place-gnn-egocentric-squeezeformer
- 7th place: invariant neural features, correct 15-lab CV, and XGBoost stacking over neural OOF predictions: https://www.kaggle.com/competitions/MABe-mouse-behavior-detection/writeups/7th-place-gold-cnn-transformer-with-invariant-fe
- 9th place: GBDT plus TCN diversity, egocentric features, balanced fold assignment, and nested F1 threshold evaluation: https://www.kaggle.com/competitions/MABe-mouse-behavior-detection/writeups/9th-place-gold
- 10th place: joint pair modeling, macro-soft-F1, keypoint variants/dropout, and evidence that semi-supervised MABe22 use did not help: https://www.kaggle.com/competitions/MABe-mouse-behavior-detection/writeups/10th-place-solution-st-gcn-transformer
