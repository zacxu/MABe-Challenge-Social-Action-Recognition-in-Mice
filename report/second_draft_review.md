# Review of the SC4000 second draft

Reviewed 1 November 2026. Source: [SC4000 Second Draft downloaded.docx](<../SC4000 Second Draft downloaded.docx>).

## Overall assessment

**Overall score: 5.0/10. Recommendation: major revision before submission.**

The project contains substantial engineering and a credible empirical finding: the feature-based and temporal predictors combine effectively on the recorded development folds. The architecture description, supervision masks, tree-training procedure, and several numerical comparisons are useful foundations for a paper. However, the current manuscript is not yet journal-ready. Its main weaknesses are overstated validation independence, an unsupported hyperparameter-optimality claim, inconsistent leaderboard evidence, and an incomplete discussion of results and limitations. The reference list and figure labeling also need substantive repairs.

The ten-page constraint is achievable through restructuring. The manuscript spends too much space explaining implementation steps and repeats data and metric definitions, while the main ensemble result and limitations receive very little discussion. Preserve the important design decisions and evidence; remove instructional examples and duplicated exposition.

These scores assess the **current manuscript**, not the quality of the code, the amount of work completed, an instructor's likely grade, or a particular journal's acceptance probability. The overall score is a holistic judgment, not an arithmetic average of subsection scores.

### Review scope and verification limits

I read the draft's text, equations, seven Word tables, and drawing text; visually inspected all seven embedded raster images; and checked important claims against the current implementation, configurations, and saved reports. The original DOCX was not edited, and its SHA-256 remained unchanged.

A fresh full-document render could not be obtained in this session: the bundled renderer was unavailable, and native Word export did not complete. Consequently, I do not claim to have verified final pagination, equation rendering, or all full-page spacing. The DOCX's stored properties report **11 pages and 6,148 words**, but those are cached values, not a verified current PDF. Findings about embedded images and section structure are directly observed; whole-document layout judgments remain provisional. Locations below use section titles and short quotations rather than invented page numbers.

I assume the **ten-page limit includes references and figures**. If the instructor explicitly excludes references, reallocate that space to experimental evidence rather than longer methodology.

## Scoring rubric and section scorecard

- **9–10:** publication-ready or close, with convincing evidence and clear reporting.
- **7–8:** strong section with identifiable, limited gaps.
- **5–6:** useful material, but significant revision is needed.
- **3–4:** major scientific, evidential, or structural problems.
- **0–2:** missing, misleading, or substantially incomplete.

Scores consider correctness, evidential support, clarity, reproducibility, and relevance to the page budget.

| Main section or paper element | Score /10 | Principal assessment |
|---|---:|---|
| Title and author block | 6.0 | Identifies the task but not the study's approach or contribution. |
| Abstract | 5.0 | Communicates the pipeline and outcome, but overstates validation safety and submission provenance. |
| 1 Introduction | 4.5 | Relevant motivation and challenges; unfinished citations and weak research positioning. |
| 2 Problem formulation, data, and evaluation | 5.0 | Useful supervision/metric description, undermined by terminology and validation-independence errors. |
| 3 Methodology | 6.0 | Substantial correct technical detail, with several factual inconsistencies and unnecessary repetition. |
| 4 Experimental Setup | 5.0 | Useful configuration table, but insufficient protocol detail and substantial duplication. |
| 5 Experimental Results and Discussion | 4.5 | Real results, but uneven analysis and an almost absent limitations section. |
| References | 3.0 | Legitimate sources, but missing entries, unused entries, placeholders, and inconsistent formatting. |
| Figures, tables, and document structure | 4.0 | Several useful visuals; a scientifically incorrect legend and incomplete evidence/captioning. Full-page layout is unverified. |
| Scientific writing and consistency | 4.5 | Understandable content, but frequent proposal language, grammatical problems, and uncalibrated claims. |

## Submission-blocking issues

### 1. Qualify the claim that the pipeline is leakage-safe

The abstract, Section 1.3, and the heading of Section 2.3 make a categorical claim. Video grouping and cross-fitted calibration are good practices, but do not establish complete independence of the reported evaluation.

The temporal training loop constructs validation windows from the same held-out fold subsequently scored, evaluates them after each epoch, and uses validation AP to choose the saved checkpoint. Thus, that fold supplies no gradient updates, but its labels influence model selection. Downstream calibration also reuses ordinary OOF predictions from other folds whose base models can have trained on the evaluated fold. This is not a fully nested end-to-end outer evaluation.

Evidence: [temporal_training.py](../src/mabe/temporal_training.py), validation construction around lines 314–322 and checkpoint selection around lines 391–446; [temporal_data.py](../src/mabe/temporal_data.py), validation-fold selection around lines 628–634. The contrary manifest flag is hard-coded in [phase9_pipeline.py](../src/mabe/phase9_pipeline.py); it does not override the training logic.

Use a precise statement, for example:

> We used duplicate-aware video-level cross-validation and cross-fitted calibration. Temporal checkpoint selection used the evaluated fold, so local scores are development estimates rather than fully nested independent-test estimates.

This does not make the existing scores meaningless. It limits the strength of the generalization claim. Stronger independence requires a different evaluation protocol, not merely different wording. The general concern about selection bias is supported by [Cawley and Talbot 2010](https://www.jmlr.org/papers/v11/cawley10a.html); the project-specific finding comes from the code.

### 2. Remove or substantiate the claim that the loss coefficient is optimal

Section 3.7 says the coefficient 0.20 was “established as the optimal by a controlled ablation.” No such controlled performance-v2 comparison is presented in the paper or supported by the inspected records. The configuration establishes the value used, not its optimality. Historical screens of 0.10 and 0.25 used an older temporal configuration and cannot establish the best value for the current system.

Replace the claim with:

> We set the soft-F1 coefficient to 0.20 in the reported configuration; its individual contribution was not isolated in a controlled ablation.

If a new experiment exists outside the inspected records, supply its configuration, comparable training budget, split/selection policy, and results. Even then, “best among the tested values” is more defensible than “optimal.”

### 3. Reconcile the Kaggle evidence before reporting a final score and rank

The embedded screenshots do not establish the exact combination stated in the abstract and Table 1:

| Evidence in the draft | What is actually visible |
|---|---|
| Figure 1 submission crop | Private score **0.45969**, public score **0.47815**, and an **unchecked Selected box**. |
| Figure 2 leaderboard crop | Rank **82**, displayed team name **Mahdi Ravaghi**, and score **0.47806**. The adjacent rank-81 row shows 0.47816. |
| Abstract and Table 1 | Public score **0.47815** together with rank **82 of 1,412**. |

This discrepancy does not establish that the leaderboard row belongs to someone else: display names, different capture times, or different boards/submissions could explain it. However, the current crops do not verify the team/submission linkage, leaderboard type, capture date, or 1,412-team denominator. Obtain consistent evidence showing those details.

Do not label 0.45969 as the final private result until the selected submission is verified. If it is the final selected submission's private score, report and discuss it rather than presenting only the public score.

The relative-ranking label is also wrong. Conditional on rank 82 of 1,412 being verified:

- `82 / 1412 × 100 = 5.81%` means **approximately the top 5.8%**.
- `(1412 − 82) / 1412 × 100 = 94.19%` means **approximately 94.2% of teams outranked**, using this convention and ignoring ties.

The safest phrasing is “ranked 82 of 1,412 teams, approximately the top 5.8%.” The course brief requires score/rank screenshots, so combine consistent evidence into one readable multipanel figure rather than deleting all screenshots.

### 4. Correct the FP/FN legend in Figure 3

The embedded metric illustration reverses the definitions:

- The orange **prediction outside the ground truth** is a **false positive**, but is labeled FN.
- The pink **ground truth missed by the prediction** is a **false negative**, but is labeled FP.

This is a substantive error in an explanatory figure, even though the implemented scorer uses the correct counts. Fix the legend, or remove this introductory illustration to save space. An AI research audience generally does not require a large figure explaining ordinary TP/FP/FN.

### 5. Align the methods and outcomes with what was actually run

Several statements conflict with the implementation or with other parts of the draft:

| Draft statement | Required correction |
|---|---|
| Section 3.1: “Bayesian-based hyperparameter tuning” | The inspected study iterates predefined configurations. Say “bounded screening of predefined configurations” unless a separate Bayesian-search run can be supplied. |
| Section 3.3: “Rows totaled 240 … 321” | These are **feature dimensions/columns**, not dataset row counts. |
| Section 3.4: row subsampling 0.82 | The configuration and training table specify **0.85**. |
| Section 5.1: “Retrained” and “Baseline retrained” | Phase 8 retained baseline probabilities; its report explicitly says no redundant retraining was needed. Distinguish this from the genuine inner-early-stop refit during Phase 7. |

## Section-by-section review

### Title and author block — 6.0/10

The current title identifies the competition, but says little about the approach or empirical contribution. A more informative option is **A Feature–Temporal Ensemble for Pose-Based Mouse Behavior Recognition**. Avoid “leakage-free” or unseen-laboratory generalization in the title without independent evidence. The equal-contribution note is reasonable. If the alphabetical-order statement is retained, apply a consistent naming convention.

### Abstract — 5.0/10

The abstract identifies sparse supervision, heterogeneous recordings, the two model branches, and a numerical outcome. These are useful ingredients. Its main weaknesses are the categorical “leakage-safe” claim and the unresolved final-submission/rank linkage described above.

Prioritize the experimental comparison: approximately 0.3765 for decoded trees, 0.4210 for the temporal model, and 0.4702 for their action-wise blend on the development folds. That communicates the finding better than another preprocessing inventory. Identify online public/private scores separately once verified. Repair the fragment “The blend achieved …, ran as …”. Keep the abstract concise, approximately 160–190 words; it needs sharper selection of information, not extra detail.

### 1 Introduction — 4.5/10

#### 1.1 Motivation and problem significance — 4.0/10

The annotation burden and recording heterogeneity are appropriate motivation. However, three `[R]` placeholders remain, and the opening uses fragments and awkward phrases such as “gain insights to the brain” and “attempts to automate hence speed up.” Replace the placeholders with real citations, including the relevant Anderson–Perona and Datta references already listed.

State annotation savings as motivation, not a measured result: the experiments did not measure researcher annotation time. Condense this subsection to one connected paragraph.

#### 1.2 Competition description and challenges — 6.0/10

The task definition, action vocabulary, sparse labels, directionality, and recording differences are relevant. Cite the competition for the “400 hours,” recording-system count, and submission requirements. Distinguish recording systems, laboratories, and the 15-laboratory primary evaluation subset rather than treating them as interchangeable counts.

Replace “the goal is for us to build models” with an objective task description. The five challenges can become one compact paragraph. Do not imply that video-held-out results establish performance in entirely unseen laboratories.

#### 1.3 Proposed solution and novelty — 4.0/10

The hybrid design is clear, but “Notable innovations” overstates what has been demonstrated. Engineered geometry, annotation masks, grouped folds, attention, convex blending, and decoding are established techniques. Feature counts alone are not novelty.

Frame the contribution as an empirical system study: complementary representations, explicit supervision/decoding constraints, and measured comparisons in heterogeneous data. A useful research question is whether a compact temporal model adds information beyond engineered features, and whether a simple blend captures that complementarity. Present contributions that the results actually answer.

#### 1.4 Evaluation score and relative ranking — 3.0/10

Providing real evidence is appropriate and required by the assignment. However, the percentile label and screenshot provenance must be corrected. Move most of this material into Results; retain at most a short outcome statement in the Introduction. Combine matching score/rank screenshots into one compact figure with a caption specifying public/private board, team identifier, submission/version, and capture date.

#### Literature positioning, currently missing — 2.0/10

The paper does not meaningfully position its approach against prior behavior-classification work. A separate full-page Related Work section is unnecessary. Two short paragraphs in the Introduction can cover feature-based behavior classification, learned pose representations, and multi-agent benchmarks.

Useful primary references include [JAABA](https://pubmed.ncbi.nlm.nih.gov/23202433/), [SimBA](https://pubmed.ncbi.nlm.nih.gov/38778146/), and [MABe22](https://proceedings.mlr.press/v202/sun23g.html). Distinguish the already listed CalMS21/Sun 2021 work from MABe22/Sun 2023 and the 2025 competition. Cite these to explain context, not to imply that their systems were implemented or evaluated as baselines here.

### 2 Problem formulation, data, and evaluation — 5.0/10

#### 2.1 Data audit and supervised subset — 6.0/10

The concrete audit and the distinction between unannotated triplets and valid negatives are strengths. However, Table 3 calls 7,941 videos “Unsupervised training,” implying a training stage that is not implemented in this pipeline. The sentence “not used for unsupervised training instead” is also internally unclear.

The audit reconciles these as 847 labeled videos, 7,926 unlabeled MABe22 videos, 15 additional missing-mask videos, and one ambiguous video. Thus 7,941 is a count of unlabeled videos, not evidence of an unsupervised-learning experiment. Rename the category “unlabeled/excluded from supervised loss” and state actual usage. Under ten pages, remove Table 3 and reconcile the counts in the existing audit table or one sentence.

Keep the important mask semantics, but compress the cosmetic string-cleaning walkthrough. A research reader needs the canonicalization rule and supervision contract, not a step-by-step cleaning anecdote.

#### 2.2 Evaluation metric — 5.0/10

The frame-set F1 interpretation, half-open intervals, and action-within-laboratory averaging are broadly correct. Preserve one equation and explain that laboratories receive equal weight and actions receive equal weight **within each laboratory**; this is not necessarily equal weighting of each action globally.

Correct Figure 3's FP/FN reversal. Also separate the host scorer from the project's submission validator and interval decoder. The copied scorer does not itself enforce every property attributed to it here: for example, it filters submission rows to solution videos rather than requiring exact full-video prediction coverage. Positive interval length/nonnegative identifiers/order checks should be attributed to the relevant project validator where implemented. Maximal interval construction is a decoder operation, not another component of F1.

Using the host metric aligns the scoring definition; it does not make local and hidden-test scores numerically interchangeable. Remove informal phrasing such as “crashes the program” and “teaches the final score.”

#### 2.3 Leakage-safe validation — 3.5/10

Grouping complete videos and duplicate sequences is a strong choice. The explanation of OOF prediction and direct exclusion of scored-fold labels from calibration is useful. The heading and global independence claim nevertheless require qualification for temporal checkpoint selection and downstream OOF reuse.

Rename this subsection **Validation and model selection**. State separately how gradient-training data, checkpoint-selection data, and calibration data are chosen. Remove the classroom example listing folds 0, 1, 2, and 4; use its space for the actual methodological limitation. If leave-laboratory/skeleton/tracker-out diagnostics were executed, give their coverage and results or a traceable artifact; otherwise, do not imply that a planned diagnostic constitutes measured generalization evidence.

### 3 Methodology — 6.0/10

#### 3.1 Methodology overview — 4.0/10

Figure 4 communicates the basic flow. The accompanying prose is repetitive and written as a proposal: “we will use,” “we will first,” and “we will distinguish.” Describe completed experiments in past tense and architecture in present tense.

Remove the unsupported Bayesian-optimization claim. Clarify that a target identifies a mouse or self, whereas the permitted-triplet list is a supervision mask; “a target equal to the permitted … combinations” confuses these concepts. Keep one short overview paragraph and refer to the diagram instead of repeating its boxes.

#### 3.2 Pose preprocessing and normalization — 6.0/10

Canonicalization, finite/missingness handling, scale validation, and original-frame mapping are important and mostly well explained. Three corrections matter:

1. The text says ten landmarks but names nine; **neck** is omitted, although Figure 5 includes it.
2. The 250 cm/s jump threshold is not universal. It applies when physical scale is valid; invalid-scale data use the configured pixel-based fallback, 100 pixels per frame. Call this a tracking-quality heuristic rather than an established physiological limit unless sourced.
3. Scope the AdaptableSnail exception to the affected **25-Hz videos**, not necessarily every video from that laboratory. Separate the inherited drop heuristic from ordinary 30-Hz resampling and acknowledge that its benefit for performance-v2 is not independently established.

Linear interpolation over short gaps is an approximation, not a guarantee that the mouse follows a straight line. The one-minute tensor example, jump arithmetic, and explicit 25-to-30-FPS timestamp calculation can be removed. Keep the actual settings and frame-mapping caveat in approximately 150–200 words.

#### 3.3 Hand-engineered representation for GBDT — 6.0/10

The explanation of ordered pairs and relative motion is useful. Correct 240/321 from “rows” to feature dimensions. Temporal durations in this implementation are centered **radii**: a 5-s radius covers about 10 s away from boundaries. Position features should be described as video-boundary/video-center proxies where appropriate, rather than measured arena boundaries.

The prose reports metric-geometry AP as 0.745, whereas the saved screen and Section 5.1 table give 0.747638. Reconcile this to 0.748 or the full value unless a different experiment is intended.

Table 4 already lists most feature families; the subsequent family-by-family prose duplicates it. Keep the table plus one paragraph explaining directional/temporal design choices. Figure 6 illustrates pose geometry, not classification correctness. The speed value also requires neighboring observations, despite showing one frame. Either shorten this example substantially or replace it with an actual prediction/ground-truth example if available.

#### 3.4 GBDT branch architecture and training — 7.0/10

This is one of the strongest sections. It gives useful support thresholds, model routing, fold-specific model counts, sampled positives/negatives, and genuine inner early stopping followed by outer-training refit. It also correctly distinguishes the target sampling ratio from the realized ratio.

Correct row subsampling from 0.82 to **0.85** and describe screening as predefined configurations. Move screening-protocol details into Experimental Setup, and avoid duplicating the full configuration table in prose. Keep the distinction between initial model fitting, inner-fold refitting, and retaining the unchanged probability stream during later decoder optimization.

#### 3.5 Temporal branch architecture and training — 7.0/10

The main architecture is substantially accurate: 23/54 input channels, independent encoders, multiscale kernels, 128 hidden channels, six dilations, four eight-head attention blocks, and 11/26 output heads. The qualification that 135 frames describes convolutional support rather than the whole model's receptive field is particularly good.

Replace “2 mouses” and future-tense instructions. State that 512 frames at 30 Hz span about 17.1 s. Clarify that body-centered posture channels are constructed from the normalized cache rather than implying all final coordinates retain a video-level origin. Keep a short architectural account that supports one clear diagram; refer to Experimental Setup for training settings and disclose checkpoint selection under validation.

#### Second section numbered 3.5, Model and deployment pipeline — 4.5/10

This duplicates the numbering and largely repeats Figure 4's pipeline. Treat it as a figure with a numbered caption, not a separate methodology section. Merge the two pipeline representations or retain the clearer one. Internal development-stage descriptions do not require their own paper headings.

The preferred drawing branch was inspected separately from Word's fallback representation; duplicated extraction text alone is not being treated as proof of visible duplication.

#### 3.6 Temporal Conv-Attention (TCN) Model — 4.5/10

TCN conventionally means **temporal convolutional network**, not the entire convolution–attention model. Rename the diagram accordingly. Its drawing text contains two boxes with the same TCN-block description; clarify the intended route/module roles. It does not clearly identify the four attention blocks and laboratory conditioning described in the prose.

The final architecture diagram should make the path explicit: multiscale stem, conditioning/adapter, six residual TCN blocks, four attention blocks, and output heads, with independent solo/pair parameters. Check arrows, reading order, and captions in the final export. Prefer a single compound architecture/pipeline figure over several overlapping diagrams.

#### 3.7 Training Objective — 4.0/10

The two equations, active-action interpretation, and supervision/laboratory weighting are useful and broadly faithful to the implementation. The unsupported statement that coefficient 0.20 was established as optimal is the principal issue and materially lowers the score.

Keep the concise two-equation treatment. Distinguish the roles of positive BCE weighting and focal focusing rather than saying the exponent itself emphasizes positives. Add the original focal-loss citation where modulation is introduced; label the joint objective as the implemented design, not a theorem or a formula prescribed by that paper. Do not expand this into a tutorial under the page constraint.

#### 3.8 Sampling and Augmentation — 7.0/10

The configured sampling mixture and the caveat that random windows may contain positive frames are helpful. Input-only temporal shifts are correctly distinguished from target-aligned transformations. Add that MixUp uses **same-laboratory partners** and intersects their masks.

Reduce the optimizer/augmentation list to a compact description supported by the training table or a reproducible configuration reference. Correct agreement errors such as “windows can contains” and “CutMix to replaces.”

#### 3.9 Probability Ensembling — 6.5/10

The convex combination, weight grid, threshold-optimized frame-F1 surrogate, hierarchical shrinkage, and stacker inputs are mostly correct. State that decoder parameters are subsequently fitted for each candidate, so final comparisons are between complete systems rather than blends under one fixed decoder.

Clarify the restricted deployment candidate set and distinguish it from the diagnostic scopes; do not retrospectively justify exclusion through runtime when all candidates pass the recorded gate. A concise deployment description is missing: fold predictions are averaged, cross-fitted action weights are averaged, and decoder parameters are aggregated by median. These are different operations and can be specified in two or three sentences.

### 4 Experimental Setup — 5.0/10

#### 4.1 Data, Validation Splits, and Evaluation Metric — 5.0/10

The primary/auxiliary training distinction is a strength: it correctly acknowledges that the branch comparison does not isolate architecture. However, the counts, folds, and metric substantially repeat Section 2. Define data and metric once, retain the auxiliary-data distinction there, and use this space for the actual experimental protocol. Add the missing main-section number 4.

#### 4.2 Training Configurations — 5.0/10

The table is useful, but a hyperparameter list is not a complete experimental setup. Add, compactly:

- Base seed 2025 and the fold-dependent temporal seed policy.
- How validation AP is calculated and used for checkpoint selection.
- The tree screen's 29 routes, 58 route–fold comparisons, folds 0/1, row caps, and validation-threshold optimization.
- Calibration/selection scope and the distinction between sampled-screen AP/F1 and exact competition F1.
- Training/inference hardware if recorded, software/configuration identifiers, and a clearly scoped runtime measurement.

Do not invent a GPU model: the retained reports identify CUDA execution but not a precise accelerator specification. If using the available runtime report, identify it as **one video**, approximately 45.13 s, with a linear 200-video projection of approximately 2.51 h, not an actual hidden-test benchmark. Assign the table a numbered caption.

### 5 Experimental Results and Discussion — 4.5/10

#### 5.1 Feature and GBDT ablations — 6.5/10

This is the strongest results subsection. The feature-removal, mining, alternative-family, decoder-score, and interval-count values largely match the saved reports. The warning that fewer intervals alone do not establish improved boundary accuracy is appropriate.

Replace “Retrained” outcomes with **baseline retained** where the Phase 8 report says no redundant retraining occurred. Explain the sampled protocol before interpreting its results. Correct every “Marco/marco AP” to **macro AP**. Preserve the distinction between the separate alternative-family screen and the main feature-screen baseline.

Permutation-importance means also match the saved data, but their coverage differs: temporal, metric, metadata, and quality use 29 route comparisons; invariant and boundary use 3; egocentric and group use 26. Report this coverage and the shuffling procedure. Do not present these averages as a uniform causal ranking across all routes. The existing caution that low importance does not prove a family unnecessary is worth retaining.

Combine the small screening tables or make their captions and protocols self-contained. Keep the distinction between permutation diagnostics and retrained feature-removal ablations.

#### 5.2 Temporal Model — 4.0/10

The score 0.421040, improvement 0.044569 over decoded trees, and 63/27/3 laboratory–action comparison counts are supported. The discussion is too short relative to the architectural detail. Replace the illogical “However” before the win counts.

Explain the auxiliary-data confound and checkpoint-selection limitation. There are no presented performance-v2 ablations isolating attention, laboratory conditioning, window length, augmentation, or the loss coefficient. If new experiments are infeasible, limit the claim to the complete temporal pipeline instead of implying those components were validated individually.

Existing complementarity diagnostics could support a short discussion: probability correlation 0.618939 and error Jaccard 0.251207 on 417,500 sampled OOF rows. Give the sampling/threshold definition and do not interpret them as error independence.

#### 5.3 Global Ablations — 4.5/10

Rename this subsection **Ensemble Comparisons**: the candidates are not all isolated component ablations. The existing ensemble chart is useful, but needs a numbered caption and explicit discussion near its actual placement.

State the main result here: action-wise blend **0.470192**, compared with tree **0.376471**, temporal **0.421040**, and equal blend **0.469314**. The complete action-wise system exceeds equal weighting by only **0.000877**; most of the observed gain is already obtained by combining the branches. This difference does not isolate weight fitting, because each candidate has its own fitted decoder. The global blend **0.471882** and per-lab selector **0.470459** are numerically higher, but outside the deployment candidate set. The draft rightly discloses the restriction, but must explain its status and avoid implying the selected strategy is the highest-scoring candidate overall.

Do not claim small differences are reliable without uncertainty evidence. Keep one compact comparison table or the existing chart plus key numbers, rather than duplicating the complete dataset in both.

#### 5.4 Conclusion and Limitations — 2.0/10

This currently consists of two sentences in one bullet and contains no actual limitations. Replace “we proved” with “the development results support” or “we observed.”

Use approximately 150–200 words to state the principal comparison and its scope, then acknowledge the important limits: represented-laboratory video folds do not establish unseen-lab transfer; checkpoint/calibration selection is not fully nested; auxiliary-data eligibility confounds architecture comparisons; only one base seed is represented; and controlled neural ablations/uncertainty estimates are limited. The inherited 25-Hz drop policy and unresolved leaderboard provenance also need disclosure where relevant.

End with one or two direct next experiments, not a generic future-work paragraph. A properly qualified conclusion deserves space currently spent repeating preprocessing details.

### References — 3.0/10

The bibliography contains useful, legitimate sources, but it looks unfinished. Delete “References for first part is below,” merge the two lists, alphabetize them, and use one author–year style. Remove literal Markdown asterisks from Word entries. Several entries are assigned heading styles; apply a proper reference-paragraph style and verify the export.

Required citation repairs:

| Work or issue | Required action |
|---|---|
| Wu and He 2018; Zhang et al. 2018; Yun et al. 2019 | Cited in the text but missing bibliography entries. Add them. |
| Vaswani et al. 2017 | Cite the source of self-attention when introducing it. |
| MABe22 / Sun et al. 2023 | Cite the benchmark when discussing the named dataset; do not substitute CalMS21 or the competition citation. |
| XGBoost, LightGBM, CatBoost | Cite the relevant model papers where introduced; a bibliography entry alone is not an in-text citation. |
| Cawley and Talbot 2010 | Listed but unused; cite it for selection-bias context or remove it. It does not certify the code as leakage-free. |
| Three `[R]` placeholders | Replace with the actual sources supporting the corresponding claims. |

Primary links for missing methodological references: [Group Normalization](https://arxiv.org/abs/1803.08494), [mixup](https://arxiv.org/abs/1710.09412), [CutMix](https://arxiv.org/abs/1905.04899), and [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

## Figures tables and presentation

**Score: 4.0/10 for observed figure content and document structure. Full-page typography/pagination is not scored as verified.**

The skeleton illustration and overall pipeline figure help explain inputs and processing. The ensemble chart is an important results asset. Improve their integration rather than adding more figures.

- Correct the scientific error in Figure 3's FP/FN legend.
- Replace the inconsistent scoreboard crops with one traceable evidence figure.
- Merge Figure 4 and the later pipeline/encoder drawings into one coherent compound figure where possible.
- Give all seven Word tables, native diagrams, and the ensemble chart appropriate numbers, captions, and in-text references. Only the first four table captions are explicitly numbered in the extracted body.
- Remove internal “Phase 7/8/9/11” labels from results charts unless phases are defined in the paper; use method names a reader can interpret independently.
- Check the final export for equation fractions, figure placement, drawing arrows, and readable text at the required size. Flattened XML extraction is not evidence that an OMML fraction is visually broken.
- Do not shrink fonts or margins to meet the page limit. Respect the assignment's supplied template; the full-page render is the final authority on fit.

## A practical ten-page revision plan

### Suggested page budget

This is a planning budget, not a claim that a rewritten document has already been rendered to fit.

| Content | Target pages | Priority |
|---|---:|---|
| Title, authors, abstract | 0.4 | Problem, system, principal result, qualified scope. |
| Introduction and literature positioning | 1.0 | Motivation, relevant prior work, empirical contributions. |
| Problem, data, and evaluation | 1.0 | Supervision masks, population, metric, validation limitation. |
| Methodology, including architecture figure | 2.8 | Feature representation, two branches, objective, blend/decoder. |
| Experimental setup | 0.7 | Training table and reproducible screen/selection protocol. |
| Results and discussion | 2.3 | Main comparisons, useful ablations, verified Kaggle evidence. |
| Conclusion and limitations | 0.4 | Focused findings and important bounds on interpretation. |
| References | 1.1 | Complete, consistently formatted bibliography. |
| Float/layout buffer | 0.3 | Avoid squeezing content to an exact unrendered estimate. |
| **Total** | **10.0** | Includes figures and references. |

### Highest-value cuts

1. Remove repeated data counts and the repeated metric equation from Section 4.1; refer back to Section 2.
2. Remove the redundant video-category table and explain the unlabeled counts once.
3. Cut the one-minute tensor example, explicit resampling timestamps, and jump-speed arithmetic from preprocessing.
4. Keep the feature-family table and one explanatory paragraph; remove the repeated family-by-family inventory.
5. Merge overlapping pipeline/architecture diagrams and remove separate diagram-only methodology sections.
6. Remove the generic TP/FP/FN teaching figure if space remains tight; the standard equation is sufficient for this audience.
7. Shorten or remove the single-frame pose-geometry example unless it supports a specific methodological question. Prefer space for predictions, errors, or the main comparisons.
8. Combine the two required Kaggle evidence panels, while retaining enough context to establish provenance.

Use the recovered space for a credible validation statement, a reproducible screening protocol, stronger ensemble discussion, and actual limitations. Merely cutting one page from the cached eleven is insufficient if the missing evidence/discussion is then added without compensating cuts.

### Recommended final organization

1. Introduction, including concise related work and contributions.
2. Problem formulation, data, and evaluation.
3. Methodology: preprocessing/features; tree branch; temporal branch/objective; ensembling/decoding/deployment.
4. Experimental setup.
5. Results and discussion, including verified Kaggle outcome and limitations.
6. Conclusion.
7. References.

This organization is a recommendation, not a requirement to create seven long sections. Keep subsection depth shallow and the narrative centered on the experimental finding.

## Revision order and final acceptance checklist

**First, correct scientific truthfulness:** qualify validation independence, remove unsupported optimality/Bayesian-search/retraining claims, and fix FP/FN labels. Then reconcile the leaderboard evidence and ranking calculation. Next, consolidate repeated sections and add the missing experimental protocol and limitations. Finish with citations, captions, numbering, grammar, and an actual ten-page export.

Before submission, verify:

- [ ] Every performance number is identified as sampled screen, development OOF, public leaderboard, or private leaderboard evidence.
- [ ] Score, rank, team identity, selected submission, leaderboard type, capture date, and team-count denominator are consistent.
- [ ] No claim of global optimality, statistical significance, unseen-lab generalization, or leakage-free evaluation exceeds the evidence.
- [ ] The primary comparison and the selection restriction are explicit.
- [ ] The tree screen and temporal checkpoint-selection policies are reproducible.
- [ ] Figures use correct scientific labels, and all floats have captions and cross-references.
- [ ] Duplicate section numbers, `[R]`, raw Markdown artifacts, and the leftover bibliography note are gone.
- [ ] All cited works have entries, and every bibliography entry is used.
- [ ] The final PDF fits ten pages under the required template without compressed fonts or margins.
- [ ] Every final page has been visually checked; equations and native drawings render correctly.

## Evidence trail

The review is based on the supplied draft, not on an assumption that an older generated report is the current manuscript. Important checks used:

- [Data audit](../artifacts/reports/data_audit.json): population and scanned-row counts.
- [Tree configuration](../configs/xgboost-performance.yaml) and [tree CV report](../artifacts/xgb/xgb-performance-v2/cv_report.json): actual model settings and baseline score.
- [Phase 8 report](../artifacts/phase8/gbdt-performance-v2/phase8_report.json), [screen configuration](../configs/phase8-performance.yaml), and `artifacts/phase8/gbdt-performance-v2/permutation_importance.parquet`: screening values, unchanged probability stream, and importance coverage.
- [Temporal configuration](../configs/temporal-performance.yaml), [temporal report](../artifacts/temporal/temporal-performance-v2/phase9_report.json), [training code](../src/mabe/temporal_training.py), and [sequence code](../src/mabe/temporal_data.py): architecture/settings, result, masks, and checkpoint selection.
- [Pose configuration](../configs/pose.yaml), [pose code](../src/mabe/pose.py), [feature implementation](../src/mabe/features/solo.py), and [metric](../src/mabe/metric.py): landmark, threshold, temporal-radius, boundary, and scorer semantics.
- [Ensemble report](../artifacts/phase11/ensemble-performance-v2/phase11_report.json), candidate leaderboard, and [ensemble code](../src/mabe/phase11_pipeline.py): candidate scores and deployment selection.
- [Inference report](../artifacts/phase12/runtime-performance-v2/inference_report.json) and [inference code](../src/mabe/phase12_inference.py): scope of available runtime/deployment evidence.
- [Course content requirements](content.md): score/rank evidence, baselines, ablations, tables/figures, discussion, and conclusion requirements.

No new model training, inference, or leaderboard submission was performed. If additional controlled ablations, stress-test results, or final-submission records exist outside these files, the corresponding criticisms should be reassessed against that evidence.

Reviewed source SHA-256: `C1F93ED21D8DC9A5DF9AE9145F89CA1408F5B677863A015E4E5274EA2F68C0A1`.
