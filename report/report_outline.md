# Detailed AAAI Report Outline

## Working title

**Leakage-Safe Hybrid Gradient Boosting and Convolution-Attention Modeling for Cross-Laboratory Mouse Behavior Detection**

Alternative shorter title: **A Hybrid GBDT-Temporal System for Multi-Laboratory Mouse Behavior Detection**

Author name, affiliation, and email remain to be inserted in the AAAI Word template.

## Central story of the paper

The report should present the project as a leakage-safe, end-to-end system for detecting sparse solo and social mouse behaviors from heterogeneous pose tracks. The main contribution is not a single classifier. It is the combination of:

1. an audited metric, label-mask, pose-cleaning, and grouped-validation pipeline;
2. specialized per-lab/per-action XGBoost classifiers using physical, invariant, temporal, relational, group, quality, and metadata features;
3. a complementary convolution-attention sequence model with separate solo and directed-pair encoders;
4. cross-fitted probability blending and metric-aware temporal decoding; and
5. deterministic, offline Kaggle inference with model and schema validation.

The strongest supported headline result in the current generated artifacts is a **local exact out-of-fold F-score of 0.470192** for the selected deployable action-wise blend. This improves over the strengthened GBDT score of **0.376471** by **0.093721 absolute**. This must be kept separate from the Kaggle public/private leaderboard score, which is not present in the repository.

## Evidence status before writing the full report

- Use the `performance-v2` reports as the current experimental lineage:
  - XGBoost: `artifacts/xgb/xgb-performance-v2/cv_report.json`;
  - strengthened GBDT: `artifacts/phase8/gbdt-performance-v2/phase8_report.json`;
  - temporal model: `artifacts/temporal/temporal-performance-v2/phase9_report.json`;
  - ensemble: `artifacts/phase11/ensemble-performance-v2/phase11_report.json`;
  - inference: `artifacts/phase12/runtime-performance-v2/inference_report.json`.
- Do not mix the older values in `task.md` (for example, final score 0.307903) into the main performance-v2 result table. Those values describe an earlier, much smaller model lineage.
- The repository contains no Kaggle score/rank screenshot or recorded public/private leaderboard result. These are required by `content.md` and must be supplied before the final report.
- The finished competition page reports 1,412 teams. Confirm this denominator from the leaderboard screenshot at the time the report is finalized.
- A current local test run produced **155 passed and 1 failed**. The failure is caused by the old `configs/inference.yaml` pointing to absent `ensemble-phase11-v1` artifacts. The performance-v2 inference configuration points to the current artifacts. Resolve or explicitly document this before claiming a completely clean final release.
- Phase 13's written status and `configs/release.yaml` also refer to the older v1 lineage, while the current result is performance-v2. Regenerate or update the release evidence before writing the final reproducibility claim.

## Recommended AAAI paper structure

The phase numbers should appear in the prose and in one overview table, but the paper should not use fourteen top-level phase headings. AAAI warns against excessive headings in a short paper. Group the phases into a small number of coherent methodological subsections as shown below.

### Title and author block

- Use Chicago-style title case.
- Center the title and author/affiliation block using the AAAI template styles.
- Replace all author, affiliation, and email placeholders.
- Add code and data links only if permitted by the course/submission rules. In the final Word/PDF, URLs must be black and not automatically hyperlinked.

### Abstract (150-190 words; maximum 200)

Draft this last. It should contain one sentence for each of the following:

1. **Problem:** frame-level recognition of solo and directed social behavior from heterogeneous multi-laboratory mouse pose data.
2. **Challenges:** sparse behavior masks, severe class imbalance, different skeletons/FPS/tracking systems, multiple mice and directed targets, and leakage-prone calibration.
3. **Approach:** audited data/metric contracts, physical and egocentric features, per-lab/action XGBoost, a separate solo/pair convolution-attention temporal model, and cross-fitted probability blending plus interval decoding.
4. **Result:** selected local exact OOF F-score 0.470192 versus 0.376471 for the strongest GBDT member; include the final Kaggle score only after it is known.
5. **Importance:** the hybrid system combines interpretable geometric cues with learned temporal context while remaining deployable in an offline Kaggle notebook.

Do not cite references or include ranking screenshots in the abstract.

### 1. Introduction

#### 1.1 Motivation and problem significance

- Explain why manual annotation of animal social behavior is costly and difficult to scale.
- Explain the scientific value of robust cross-laboratory automation: different labs use different cameras, arenas, trackers, skeletons, and behavior definitions.
- State that the task uses markerless pose trajectories rather than raw video.
- Establish that a useful solution must model both individual motion and directed relationships between animals over time.

#### 1.2 Competition description and challenges

- Identify the Kaggle **MABe Challenge - Social Action Recognition in Mice**.
- State the official scale: over 400 hours of recordings from more than 20 recording systems and a hidden test set of roughly 200 videos.
- Describe the inputs: per-video metadata and frame/mouse/body-part `(x,y)` tracking coordinates.
- Describe the output: half-open behavior intervals with `video_id`, `agent_id`, `target_id`, `action`, `start_frame`, and `stop_frame`.
- State the audited local vocabulary: 37 behaviors, split into 11 self/solo and 26 directed-pair actions.
- Explain the five main technical challenges:
  1. sparse annotations mean “not labeled” is unknown, not negative;
  2. behavior frequency is highly imbalanced;
  3. pose layouts, scale, FPS, arenas, and trackers vary by lab;
  4. social actions depend on direction and third-mouse context;
  5. thresholds and postprocessing can leak validation labels if fitted on the same fold.

#### 1.3 Concise summary of the proposed solution and novelty

Use a short contribution list, ideally four items:

1. **Audited supervision and evaluation:** exact scorer reproduction, canonical behavior masks, quarantined anomalies, and fixed duplicate-aware grouped folds.
2. **Two complementary representations:** a 240/321-feature GBDT representation for solo/pair routes and a compact 23/54-channel sequence representation for the temporal model.
3. **Hybrid model:** lab/action XGBoost plus a 1.916M-parameter dual-encoder convolution-attention network.
4. **Leakage-safe selection and deployment:** cross-fitted weights, nested decoder fitting, directed-pair conflict resolution, and offline streaming inference.

Make clear which elements are engineering rigor and which are modeling novelty. Do not claim that XGBoost, TCNs, or self-attention were invented in this project.

#### 1.4 Evaluation score and relative ranking

This required subsection cannot be completed from the current repository. Insert:

- Kaggle public score: **[TO INSERT]**;
- Kaggle private/final score, if available: **[TO INSERT]**;
- rank: **[TO INSERT] of 1,412 teams**;
- top fraction: `100 * rank / 1412` percent (smaller is better);
- percentile of teams outranked: `100 * (1412 - rank) / 1412` percent.

Define which relative-ranking convention is being reported. For example: “Rank r of N corresponds to the top `100r/N`% and outranks `100(N-r)/N`% of teams.”

Place two screenshots near this subsection:

- **Figure 1:** Kaggle evaluation/submission page showing the submitted notebook and score.
- **Figure 2:** leaderboard row showing team name, rank, and score.

Do not present the 0.470192 local OOF value as a Kaggle leaderboard score.

### 2. Problem Formulation, Data, and Evaluation

#### 2.1 Data audit and supervised subset (Phases 2-3)

Report the audited facts in a compact table:

| Item | Audited value |
|---|---:|
| Training metadata/tracking videos | 8,789 |
| Annotation files | 847 |
| MABe22 unlabeled videos excluded from supervised loss | 7,926 |
| Primary competition-lab validation videos | 352 |
| Supervised auxiliary videos | 495 |
| Canonical actions | 37 |
| Solo actions | 11 |
| Pair actions | 26 |
| Source tracking rows scanned | 801,534,603 |
| Annotation rows scanned | 83,620 |

Then explain:

- `behaviors_labeled` is parsed into exact `(agent,target,action)` triplets.
- The parser trims and normalizes recoverable text defects, converts self-targets to a canonical `self` mask representation, deduplicates entries, and fails on ambiguous malformed entries.
- Four target states are distinguished: positive, valid negative, unknown/masked, and unsupervised video.
- A permitted triplet with no event at a frame is a valid negative; a disallowed triplet is never sampled as a negative.
- One ambiguous missing-annotation video is excluded.
- Fifty-nine invalid/zero-length annotation rows are quarantined rather than silently repaired.
- Ground-truth overlaps are retained for auditing.

#### 2.2 Exact evaluation metric (Phase 1)

- State that the local oracle copies the official metric implementation.
- For action `a` within lab `l`, frame-level F1 is

  `F1(l,a) = 2 TP(l,a) / (2 TP(l,a) + FP(l,a) + FN(l,a))`.

- Explain that the implementation forms frame sets from half-open intervals, aggregates by action within each lab, macro-averages actions within the lab, and then averages lab scores.
- Predictions outside the exact behavior mask are ignored by the scorer.
- Only one action may be predicted for the same directed `(agent,target)` pair at a frame; the same agent may simultaneously act toward different targets.
- Empty submissions are invalid.
- Emphasize that adjacent intervals with the same frame set have the same score; scoring is frame-set based, not event-boundary based.

#### 2.3 Leakage-safe validation (Phase 4)

- Describe deterministic five-fold grouping at the video/duplicate-group level.
- State that no exact duplicate group crosses folds.
- State that validation uses only 352 videos from 15 competition labs.
- State that 495 CalMS21/CRIM13 videos are auxiliary training data only when enabled; MABe22 does not enter supervised training.
- Explain nested calibration: for held-out fold `k`, weights, thresholds, and postprocessing parameters are selected using the other four OOF folds.
- Distinguish model-selection CV from secondary leave-one-lab/skeleton/tracker-out stress diagnostics.

### 3. Methodology

Begin with a two-column-width system diagram.

**Figure 3: End-to-end pipeline**

```text
metadata + heterogeneous pose tracks
                |
      audit, masks, grouped folds                 (Phases 0-4)
                |
      clean canonical pose + frame map            (Phase 5)
                |
       +--------+-------------------+
       |                            |
 engineered frame features      30-Hz sequence tensors
 240 solo / 321 pair            23 solo / 54 pair
       |                            |
 lab/action XGBoost          dual conv-attention model
       |                            |
       +------ OOF probabilities --+
                       |
       cross-fitted action-wise convex blend       (Phase 11)
                       |
 sparse mask -> threshold -> conflict resolution
 -> gap/minimum-run processing -> half-open runs
                       |
        validated offline submission.csv           (Phases 12-13)
```

#### 3.1 Reproducible foundation and data contracts (Phases 0-4)

Cover these steps concisely because Sections 2.1-2.3 contain the details:

- immutable YAML configurations and deterministic seeds;
- typed Polars/PyArrow schemas and artifact sidecars with hashes;
- exact metric and interval unit tests;
- canonical label stores that expand only requested windows/chunks;
- fixed grouped folds, source classification, and OOF schemas.

The methodological point is that every later model consumes the same audited indices, masks, and folds.

#### 3.2 Pose preprocessing and normalization (Phase 5)

Describe the actual order of operations:

1. Pivot each tracking parquet by original `video_frame`, mouse, body part, and coordinate.
2. Map only reviewed body-part aliases to a canonical skeleton.
3. Retain the canonical landmarks nose, left/right ears, body center, tail base, neck, left/right lateral points, head, and tail midpoint; the temporal model uses a five-landmark subset.
4. Derive body center when absent and record a derivation mask.
5. Convert pixels to centimeters only when the reported scale and geometry are credible; otherwise keep scale-validity indicators.
6. Mark non-finite values, implausible jumps, isolated points, long gaps, and possible identity swaps.
7. Interpolate only short gaps up to 0.2 seconds; preserve masks for all derived/interpolated values.
8. Derive center, body axis, body length, confidence, and timestamps.
9. For temporal modeling, resample to 30 Hz with a reversible mapping back to original frames and no extrapolation.

State that the AdaptableSnail 25-FPS audit did not justify an automatic global `30/25` frame conversion in the core pose pipeline. The current performance-v2 inference profile instead uses a separately configured `drop` policy, which removes every predicted interval for affected 25-FPS AdaptableSnail videos. This is a consequential lab-specific decision: justify it with the recorded local policy comparison and discuss its risk under hidden-set distribution shift.

#### 3.3 Hand-engineered representation for GBDT (Phase 6)

Explain the two routes and their feature counts:

| Route | Selected feature count | Included groups |
|---|---:|---|
| Solo/self | 240 | metric, invariant, temporal, boundary, quality, metadata |
| Directed pair | 321 | metric, egocentric, temporal, group, quality, metadata |

Describe each family with concrete examples:

- **Metric kinematics:** position in cm, velocity, speed, acceleration, jerk, and validity.
- **Body invariants:** length, ear spread, elongation, curvature, head/tail angle, angular velocity, and turning rate.
- **Boundary context:** relative video/arena position, center distance, and boundary distance.
- **Temporal context:** centered statistics and lags at 0.1, 0.25, 0.5, 1, 2, and 5 seconds; speed mean/std, path length, pause ratio, displacement, confidence, interpolation, and missingness.
- **Directed pair geometry:** center and cross-keypoint distances, agent-egocentric forward/lateral target position, relative velocity/acceleration/jerk, radial/tangential motion, approach/separation rate, facing angles, heading alignment, and following score.
- **Contact dynamics:** 1-cm and 2-cm contact indicators/ratios plus past-versus-future distance changes.
- **Group context:** closest third mouse, nearby/between counts, pair-distance rank, group centroid in the agent frame, and dispersion.
- **Quality and metadata:** availability/interpolation/swap flags plus lab, tracker, skeleton, FPS, arena, mouse count, sex, strain, color, age, condition, and device indicators.

State explicitly that the raw pixel-coordinate group was excluded from the selected performance-v2 feature set. Raw geometry was evaluated as an ablation.

#### 3.4 GBDT branch architecture and training (Phases 7-8)

Explain that “architecture” here means a routed collection of binary boosted-tree classifiers, not one multiclass network.

**Routing design**

- Separate solo and directed-pair feature tables.
- One binary classifier for each sufficiently supported `(lab, action)` route.
- Five outer folds produce one model per held-out fold.
- The registry contains 102 lab/action candidates across 15 primary labs: 90 supported routes x 5 folds = 450 fitted XGBoost models, plus 60 fold-route entries set to conservative no-prediction behavior.
- Pooled/auxiliary fallbacks exist in code, but the performance-v2 registry used 450 direct lab/action models and no pooled routes.

**Training data and imbalance handling**

- Preserve positive frames up to a configurable route cap.
- Sample valid negatives only from the exact sparse label mask.
- Include near-boundary and background negatives.
- The frozen sampling plan contains 5,518,378 rows: 1,775,250 positive, 2,056,064 near-boundary, and 1,687,064 background samples.
- Per-route caps are 50,000 positives and 150,000 total rows, with a 5:1 negative-to-positive target ratio.

**Tree configuration**

- Binary logistic objective with AUC-PR monitoring.
- Histogram tree method; CUDA when available, CPU fallback.
- Up to 1,500 estimators, learning rate 0.03, maximum depth 6, minimum child weight 8, subsample 0.85, column subsample 0.80, L1 0.20, and L2 5.0.
- For an outer held-out fold, select the number of trees using a separate inner video fold, then refit that number of trees on all outer-training rows. Held-out labels never control early stopping.
- Best iterations ranged from 1 to 1,461, with a median of 113.

**Phase 8 strengthening**

- Screen feature families, hyperparameters, hard negatives, XGBoost/LightGBM/CatBoost alternatives, and route-level choices.
- The baseline XGBoost feature/hyperparameter setup remained the best overall screened configuration.
- The probability model was therefore retained; the improvement came from nested probability and interval decoding.
- Decoder search covers threshold, centered smoothing, hysteresis, short-gap fill, and minimum-run duration with support-aware fallback.

#### 3.5 Temporal branch architecture and training (Phases 9-10)

This should be the most detailed architecture subsection.

##### Input representation

- Resample cleaned pose sequences to 30 Hz.
- Use five landmarks: nose, left ear, right ear, body center, and tail base.
- Normalize each video by subtracting the median valid center origin and dividing spatial quantities by median valid body length.
- Preserve landmark, mouse, temporal, action-validity, and loss masks.
- Metadata conditioning has three numeric channels: source FPS divided by 30, mouse count divided by 4, and scale-valid flag.

For a 512-frame window:

- solo tensor: `[batch, 512, 23]`;
- pair tensor: `[batch, 512, 54]`;
- solo logits: `[batch, 512, 11]`;
- pair logits: `[batch, 512, 26]`.

The 23 solo channels are 10 body-centered landmark coordinates, 5 landmark-validity flags, center velocity `(x,y)`, speed, body axis `(x,y)`, normalized body length, confidence, and mouse validity. The 54 pair channels concatenate agent and target solo channels (46) plus relative center `(x,y)`, distance, relative velocity `(x,y)`, heading dot product, heading cross product, and pair validity (8).

##### Encoder block-by-block

The model has independent solo and pair encoders with the same configuration; their weights are not shared.

```text
Solo [B,T,23] or pair [B,T,54]
       |
transpose to [B,F,T]
       |
parallel Conv1d kernels 3,5,7,9 (32 channels each)
       |
concatenate -> 128 channels -> 1x1 projection
-> GroupNorm(1) -> GELU
       |
+ broadcast lab/metadata conditioning
+ broadcast learned lab adapter
       |
6 residual depthwise-separable TCN blocks
dilations 1,2,4,8,16,32; kernel 3; dropout 0.25
       |
transpose to [B,T,128]
       |
4 pre-normalized self-attention blocks
8 heads (16 dimensions/head), FFN 128->512->128
       |
linear solo head 128->11 or pair head 128->26
       |
mask unavailable actions to logit -20
```

Additional points to explain:

- The maximum stem kernel and six dilated TCN blocks give a convolutional receptive field of approximately 135 frames, or 4.5 seconds at 30 Hz, before attention connects all positions within the 512-frame window.
- TCN convolutions use centered padding, so the model is non-causal. This is valid because the full test video is available at inference.
- Each TCN block uses a depthwise temporal convolution, a pointwise 1x1 convolution, GroupNorm, GELU, dropout, and a residual connection.
- Each attention block uses pre-LayerNorm multi-head self-attention followed by a 4x feed-forward network and residual connections.
- The code supports an optional bidirectional GRU, but **the performance-v2 model sets `use_bigru: false`**. Do not describe the final model as a CNN-GRU.
- Total trainable parameters: **1,915,749**, consisting of 943,584 in the solo encoder, 967,392 in the pair encoder, 1,419 in the solo head, and 3,354 in the pair head.

##### Optimization and inference

- Train action-rich 512-frame windows with stride-independent random centers; balance labs/actions and sample positives/background/hard negatives.
- Use batch size 16, up to 30 epochs x 1,000 steps, AdamW at `3e-4`, weight decay `1e-4`, 1,000-step warmup plus cosine decay, gradient norm clipping at 1.0, EMA 0.999, and mixed precision.
- Optimize focal-weighted masked BCE (`positive_weight=1.5`, gamma 2) plus 0.20 times macro soft-F1.
- Weight labs approximately by inverse square root of their training-video frequency.
- Apply coordinate noise, time stretch, keypoint dropout, left/right swapping, mouse permutation, rotation/reflection, scale jitter, frame shifts, same-lab MixUp, and same-lab CutMix, with labels and masks transformed consistently.
- Early-stop on validation average precision. Fold best epochs were 19-27; fold-0 validation AP was 0.5236.
- At inference, use 512-frame windows with stride 128, downweight 32-frame internal edges, average overlapping probabilities, and map predictions back to original frame IDs.

##### Phase 10 advanced experiments

- Explain that the bounded experiment framework compared shorter/longer receptive fields, lightweight convolution-attention, soft-F1 weights, lab adapters, 4/5/7 keypoints, and isolated augmentation policies.
- The historical v1 phase record retained no Phase 10 model: the only full-OOF candidate, `receptive_short_256`, scored 0.063605 versus 0.067643 for that earlier Phase 9 model.
- Do **not** compare those old values numerically with performance-v2 in the headline table. Either rerun the Phase 10 screen under performance-v2 or label it clearly as preliminary architecture exploration from an earlier lineage.
- ST-GCN was not run in that phase because the host was CPU-only; TTA was skipped because no augmentation screen beat the no-augmentation reference. Present these as resource/selection decisions, not completed negative results for ST-GCN or TTA.

#### 3.6 Cross-fitted ensemble and scorer-aware decoding (Phase 11)

Explain the final probability flow precisely:

1. Align GBDT and temporal OOF rows on lab, video, frame, agent, target, action, fold, target, and mask.
2. Verify 835 parts and 106,642,451 rows with no missing routes, duplicate keys, target mismatches, or mask mismatches.
3. For held-out fold `k` and action `a`, select a temporal weight using only the other four folds.
4. Shrink poorly supported weights toward their parent/global value.
5. Blend probabilities before thresholding:

   `p_ensemble = (1 - w[a,k]) p_GBDT + w[a,k] p_temporal`.

6. Refit decoder parameters without using fold `k` labels.
7. Test global, action, lab, lab-action, equal-weight, per-lab selector, and nonlinear stacker candidates.

For the selected action-wise registry, the median temporal weight is 0.65 in every fold; individual support-shrunk weights vary roughly from 0.133 to 0.979. The nonlinear probability stacker was evaluated but not selected.

**Decoder**

- Mask actions not allowed for the exact video/agent/target.
- Apply route-specific thresholds; selected thresholds range from 0.10 to 0.75 with median 0.50.
- The final selected candidate used no additional probability smoothing or hysteresis at this stage.
- Resolve same-pair/frame conflicts by the largest calibrated margin `probability / threshold`.
- Apply learned gap fill (0.267 or 0.5 seconds) and minimum-run filtering (0 or 0.067 seconds).
- Convert active frames into maximal half-open intervals.
- Allow simultaneous actions toward different targets.

Include a sentence explaining why probability streams are blended before decoding: averaging already-decoded event lists would discard calibrated confidence and make conflicts/postprocessing inconsistent.

#### 3.7 Offline inference and release controls (Phases 12-13)

- Discover the hidden test mount dynamically and prefer the candidate containing the largest test metadata/tracking collection.
- Build masks and manifests per video; quarantine malformed videos rather than failing the entire run.
- Process one video at a time and route the five GBDT folds and five temporal checkpoints.
- Average test-fold probabilities, apply frozen blend/decoder parameters, checkpoint progress, and write partial submissions.
- Validate exact columns, integer frame bounds, triplet legality, row IDs, non-overlap, and non-empty output.
- If every route predicts background, use the highest-confidence valid one-frame fallback rather than an arbitrary row.
- Package assets with hashes and portable relative paths for offline execution.

Report the current performance-v2 local inference evidence:

- one public test video, 18,423 frames and four mice;
- 45.13 seconds total: 12.16 s GBDT, 31.29 s temporal, 1.35 s blend/decode;
- peak observed RSS approximately 2.80 GB;
- 339 output intervals, zero failures, and zero validation errors;
- projected 9,026.9 seconds for 200 videos (about 2.51 hours), leaving 58.2% headroom against the project's conservative six-hour budget and more against the official nine-hour limit.

For Phase 13, describe versioned configs, hashes, model cards, OOF manifests, runbooks, and the release bundle. Before the final report, regenerate this release record for performance-v2 so the claim matches the actual headline model.

### 4. Experimental Setup

#### 4.1 Hardware and software

- Record the actual CPU, GPU, RAM, operating system, Python, PyTorch, XGBoost, Polars, CUDA, and driver versions from the training machine.
- State that performance-v2 temporal training used CUDA and about 4.1 GB peak process memory according to its model cards.
- Separate full pipeline runtime from model-only training/inference runtime.

#### 4.2 Baselines

Use baselines that already have comparable OOF artifacts:

1. XGBoost performance-v2 before Phase 8 postprocessing: 0.371042.
2. Strengthened GBDT with nested decoding: 0.376471.
3. Temporal convolution-attention model alone: 0.421040.
4. Equal GBDT-temporal probability blend: 0.469314.
5. Optional comparisons: nonlinear stacker and alternative weighting scopes.

Do not call a random/majority classifier the main baseline unless it is actually run through the exact scorer and sparse masks.

#### 4.3 Ablation protocol

- Fixed folds for every comparison.
- No held-out fold labels in early stopping, blend weights, thresholds, or interval postprocessing.
- Exact OOF F-score is the selection criterion when a full decoded candidate exists.
- Screen-only tables must be labeled with their surrogate metric (macro average precision or best frame F1); do not label them as official exact score.
- Record runtime and memory together with accuracy.

### 5. Experimental Results and Discussion

Every table/figure must be followed by prose explaining the important pattern and plausible causes.

#### 5.1 Main model comparison

Use this as the main result table:

| System | Exact local OOF F-score | Delta vs. strengthened GBDT | Status |
|---|---:|---:|---|
| XGBoost before Phase 8 postprocessing | 0.371042 | -0.005428 | Baseline |
| Strengthened GBDT | 0.376471 | 0.000000 | Strongest tree member |
| Temporal conv-attention | 0.421040 | +0.044569 | Strong standalone member |
| Equal GBDT-temporal blend | 0.469314 | +0.092844 | Reference blend |
| Cross-fitted action-wise blend | **0.470192** | **+0.093721** | Selected deployable model |
| Cross-fitted global blend | 0.471882 | +0.095411 | Best measured diagnostic candidate, not selected for deployment |
| Probability stacker | 0.454654 | +0.078183 | Evaluated, not selected |

Important discussion points:

- The temporal model outperformed the engineered GBDT alone, showing that long sequence context adds information not captured by centered rolling statistics.
- The equal blend produced a much larger gain than either individual branch improvement, supporting genuine complementarity.
- Action-wise tuning only slightly exceeded equal blending, so much of the gain came from combining representations rather than highly complex weight fitting.
- The global blend scored 0.001690 above the selected action-wise blend, but the Phase 11 deployment whitelist only allowed the GBDT baseline, action-wise blend, and stacker to be selected. The final report must either justify this operational choice or update deployment to the higher-scoring global candidate. Call 0.470192 the “selected deployable” score, not the highest score observed.
- The stacker underperformed simple convex blending, suggesting that its extra nonlinear capacity did not generalize under cross-fitting.

#### 5.2 GBDT ablations

Use two compact tables rather than listing every candidate.

**Feature-screen table (surrogate macro AP, not exact official F-score)**

| Feature candidate | Macro AP | Macro best frame F1 | Interpretation |
|---|---:|---:|---|
| Selected full feature set | 0.747638 | 0.721625 | Best screened feature set |
| Without metadata | 0.743838 | 0.716733 | Metadata adds modest average value |
| Raw rather than metric geometry | 0.726809 | 0.710686 | Physical/invariant representation is more robust |

**Other screens**

- Cross-fitted hard negatives reduced macro AP from 0.747638 to 0.737155, so they were not promoted.
- None of the searched depth, weighting, row-cap, or regularization variants exceeded the baseline macro AP.
- Alternative-family screen: XGBoost macro AP 0.740373, LightGBM 0.736333, CatBoost 0.736157. A few individual LightGBM/CatBoost routes looked useful, but the final probability model remained XGBoost.
- Nested decoding improved exact OOF from 0.371042 to 0.376471 and reduced decoded intervals from 110,506 to 57,734.

Use group permutation importance as a figure or small table. Mean AP drops were largest for temporal (0.1720), metric (0.1482), and invariant (0.1239) features, followed by egocentric (0.0225), metadata (0.0137), boundary (0.0034), quality (0.0007), and group context (0.0002). Explain that group-level averages can hide action-specific utility and should not alone justify deleting a feature family.

#### 5.3 Temporal model results

- Report exact OOF 0.421040 and +0.044569 over the strengthened GBDT.
- Report fold best epochs (19-27), fold-0 validation AP 0.5236, and 1.916M parameters.
- Report sampled diversity versus GBDT: probability correlation 0.6189 and error Jaccard 0.2512.
- Report that temporal won 63 of 93 lab/action comparisons, GBDT won 27, and 3 tied.
- Include one learning-curve plot and, if space permits, an action timeline comparing ground truth, GBDT probability, temporal probability, blended probability, and final decoded interval.
- Discuss why the temporal branch helps: it sees a 17.1-second window, learns action-specific temporal patterns, and uses full-window attention after local multi-scale/dilated encoding.

#### 5.4 Per-lab behavior and error analysis

Use a sorted bar chart of final F-score by lab. Representative extremes are:

- ReflectiveManatee: 0.8273;
- SparklingTapir: 0.6422;
- UppityFerret: 0.5559;
- AdaptableSnail: 0.2323;
- PleasantMeerkat: 0.2886.

Do not attribute differences solely to a lab's difficulty without evidence. Compare each lab's action count, positive support, FPS, skeleton, mouse count, and tracking-quality statistics. Likely hypotheses to test include rare actions, annotation-style differences, tracker/skeleton domain shift, 25-FPS timing, identity swaps, and multi-mouse crowding.

Include a short qualitative error analysis with at least two examples:

- a success where temporal context corrects a framewise GBDT mistake;
- a failure involving a confusable interaction, short event, sparse action, or tracking artifact.

#### 5.5 Kaggle score, ranking, and relative ranking

Insert a separate table after the screenshots:

| Submission version | Local exact OOF | Kaggle public | Kaggle private/final | Rank | Teams | Top % |
|---|---:|---:|---:|---:|---:|---:|
| performance-v2 | 0.470192 | [INSERT] | [INSERT] | [INSERT] | 1,412 | `100*rank/1412` |

Discuss any OOF-to-leaderboard gap using evidence rather than assumption. Possible factors to investigate are hidden-lab distribution, public-test size, model routing for unseen metadata, calibration shift, or the AdaptableSnail policy.

#### 5.6 Runtime, deployment, and reproducibility

- Present the one-video timing breakdown and 200-video projection.
- Explain why streaming per video and folding probabilities is feasible within the offline time budget.
- State that the local public test contains only one video; therefore runtime scaling and robustness on the hidden set remain estimates until a committed Kaggle run is recorded.
- Report the current 155/156 unit-test result honestly, then update it after the stale inference config/release lineage is corrected.
- State all release limitations: no current leaderboard evidence in the repository, stale v1 release references, and old-lineage Phase 10 results.

### 6. Conclusions

Use two concise paragraphs.

Paragraph 1 should reiterate:

- the task and its cross-lab/sparse-label difficulty;
- the audited hybrid design;
- the selected local exact OOF result of 0.470192 and +0.093721 gain over the strengthened GBDT;
- the finding that representation complementarity and leakage-safe blending mattered more than increasingly complicated individual model variants.

Paragraph 2 should state limitations and future work:

- verify and report the final Kaggle score/rank;
- promote/test the global blend if deployment support is added;
- rerun Phase 10 screens against performance-v2;
- study per-lab calibration/domain adaptation and rare-action learning;
- test joint multi-mouse graph modeling only when compute and exact OOF evidence justify it.

Avoid introducing new results in the conclusion.

### Ethical Statement (optional, unnumbered)

- Positive impact: automated behavior annotation can reduce manual workload and improve reproducibility across neuroscience/ethology studies.
- Risks: model errors can bias scientific conclusions; lab-specific performance variation must be reported; predictions should assist rather than replace expert validation in high-stakes experiments.
- Animal-welfare context: this project analyzes existing tracking/annotation data and does not itself conduct animal experiments. Confirm the appropriate dataset/competition ethics language before finalizing.

### Acknowledgments (optional, unnumbered)

- Credit the competition organizers and dataset contributors.
- Acknowledge course/instructor/team support as appropriate.
- Keep this section to no more than three sentences per the template guidance.

### References

Use AAAI author-year style and verify complete bibliographic fields. At minimum, plan to cite:

1. MABe Challenge official competition page and citation.
2. Chen and Guestrin, “XGBoost: A Scalable Tree Boosting System,” KDD 2016.
3. Bai, Kolter, and Koltun, “An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling,” 2018.
4. Vaswani et al., “Attention Is All You Need,” NeurIPS 2017, for multi-head attention.
5. CalMS21 and MABe22 benchmark papers.
6. Any Kaggle solution writeups whose specific ideas influenced the design, clearly described as prior competition solutions rather than peer-reviewed papers.

Useful source links for later bibliography construction:

- https://www.kaggle.com/competitions/MABe-mouse-behavior-detection/overview
- https://arxiv.org/abs/1603.02754
- https://arxiv.org/abs/1803.01271
- https://arxiv.org/abs/1706.03762
- https://arxiv.org/abs/2104.02710
- https://proceedings.mlr.press/v202/sun23g.html

## Complete phase-to-report mapping

Use this table as a final coverage checklist. It can appear in condensed form as the system-overview table or remain an author-only checklist.

| Phase | What was done | Where it appears in the paper | Evidence/result to use |
|---:|---|---|---|
| 0 | Reproducible package, configs, seeds, logging, tests, environment capture | Methodology 3.1; Results 5.6 | `pyproject.toml`, configs, environment files, test result |
| 1 | Exact official metric and half-open interval contract | Problem 2.2 | `metric.py`, `intervals.py`, metric tests |
| 2 | Dataset/schema/anomaly audit | Problem 2.1 | 8,789 videos, 847 annotation files, 37 actions, anomaly report |
| 3 | Sparse masks and frame/window targets | Problem 2.1; Methodology 3.1 | four target states, `LabelStore`, action vocabulary |
| 4 | Duplicate-aware five-fold validation and nested calibration | Problem 2.3 | 352 primary videos, 495 auxiliary, no fold-crossing duplicate groups |
| 5 | Pose cleanup, physical scale, canonical skeleton, 30-Hz reversible time map | Methodology 3.2 | pose audit, five fold-normalization artifacts, four 25-FPS videos audited |
| 6 | Solo, pair, group, temporal, quality, and metadata features | Methodology 3.3 | 240 solo and 321 pair selected features |
| 7 | End-to-end per-lab/action XGBoost OOF baseline | Methodology 3.4; Results 5.1 | 450 fitted models, exact OOF 0.371042 before Phase 8 postprocessing |
| 8 | Feature/model screens and nested metric-aware decoder | Methodology 3.4; Results 5.2 | final GBDT 0.376471; +0.005428 decoder gain |
| 9 | Dual-encoder temporal conv-attention model | Methodology 3.5; Results 5.3 | 1.916M parameters; exact OOF 0.421040 |
| 10 | Bounded advanced temporal screens | Methodology 3.5; Results 5.3 or appendix | historical v1 screen only; no retained Phase 10 model |
| 11 | OOF alignment, cross-fitted blending, stacker test, nested decoding | Methodology 3.6; Results 5.1 | selected action blend 0.470192; global diagnostic 0.471882 |
| 12 | Offline hidden-test-style inference and submission validation | Methodology 3.7; Results 5.6 | 45.13 s public video; 2.51 h projected/200; zero validation errors |
| 13 | Artifact freeze, hashes, runbook, release audit | Methodology 3.7; Results 5.6 | must be regenerated for performance-v2 before final claim |

## Suggested figure and table budget

To fit an AAAI two-column paper, prioritize:

1. **Figure 1:** Kaggle score screenshot (required by brief).
2. **Figure 2:** Kaggle rank screenshot (required by brief).
3. **Figure 3:** end-to-end system architecture.
4. **Figure 4:** per-lab final F-score, or one temporal prediction timeline if per-lab results fit in a table.
5. **Table 1:** audited dataset and split statistics.
6. **Table 2:** GBDT and temporal architecture/configuration summary.
7. **Table 3:** main exact OOF/leaderboard results.
8. **Table 4:** compact ablation results.

If space is tight, combine the two Kaggle screenshots into one two-panel figure and combine architecture configuration with the main system figure.

## AAAI formatting checklist for the eventual Word report

- Use the latest supplied AAAI Word template and its custom styles.
- US Letter, two columns, 3.3-inch columns with a 0.375-inch gutter.
- First-page top margin 1.25 inches; later pages 0.75 inches; left/right 0.75 inches; bottom 1.25 inches.
- Main text 10-point Times New Roman with exactly 12-point line spacing.
- Abstract no more than 200 words, formatted with the template's abstract styles.
- No page numbers, headers, footers, bookmarks, password protection, colored text, or underlined/active hyperlinks.
- Embed all fonts; do not use Type 3 fonts, including in plots.
- Put figures/tables near their first discussion and keep all content inside margins.
- Give every figure/table a numbered caption and discuss it in the prose; visual material cannot replace textual interpretation.
- Use black text in plots; color may be used only in figures and should remain interpretable in grayscale.
- Use AAAI author-year citations and an unnumbered References section.
- Confirm the course/event page limit, because the generic template file does not specify it.
