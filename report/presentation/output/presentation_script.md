# Presentation script: temporal model, ensemble and inference

This five-slide segment follows the GBDT presentation. The spoken script is **359 words**, targeting **about 2 minutes 47 seconds**, including brief slide transitions. Aim for roughly 135 words per minute and rehearse once with a timer. Timings are estimates, not an audio measurement.

The PowerPoint contains the same script in its speaker notes, together with sources and optional answers for questions. Read the paragraphs below; headings and preparation notes are not part of the timed script.

## Slide 1 — Temporal model · 0:00–0:39

The GBDT already describes motion using engineered features. Our temporal branch learns its own representation from pose sequences. Each input covers 512 frames, or about seventeen seconds. Separate encoders handle individual mice and directed pairs, using 23 and 54 input channels. The individual and pair encoders use the same structure and learn independent weights. Each encoder starts with convolutions at several scales. We add laboratory context, then use six residual temporal blocks and four attention blocks. Finally, separate output heads produce probabilities for each action at every frame.

## Slide 2 — Learning with incomplete labels · 0:39–1:07

We train only on reviewed actor, target, and action combinations. Unknown labels contribute no loss. The objective combines focal binary cross-entropy with a soft-F1 penalty. We set the soft-F1 coefficient to zero point two in this configuration. The focal term reduces the influence of easy examples, while soft F1 balances supervised actions within a batch. Pose augmentation, same-laboratory MixUp, and temporal CutMix add training variation.

## Slide 3 — Combining the two predictors · 1:07–1:43

We combine the two branches before interval decoding. The temporal weight varies by action, and estimates with limited support shrink towards a global weight. On our local evaluation, the temporal model scores zero point four two, and the selected blend reaches zero point four seven. Equal weighting is already very close. Each pipeline has its own fitted decoder. So the main finding is that the two representations complement each other, rather than a large benefit from learning action-specific weights.

## Slide 4 — Decoding behaviour intervals · 1:43–2:14

The blended probabilities still need to become valid behaviour intervals. We smooth the scores and use hysteresis to reduce fragmentation. For each directed mouse pair, we resolve competing actions, form intervals, remove very short intervals, and then merge small unblocked gaps. During offline inference, we average overlapping windows and reflected predictions, then map the probabilities back to the original video frames.

## Slide 5 — Offline deployment and interpretation · 2:14–2:47

Finally, we average predictions from five fold models and process videos individually. One public video took forty-five seconds and produced 339 validated intervals. This checks execution on one recording. Checkpoint selection uses the evaluated fold, so the reported scores are development estimates. The temporal model also receives extra labelled data, which limits a direct architecture comparison. The results support combining engineered features with learned temporal representations.

## Preparation notes — outside the timed script

- **Architecture:** two independent encoders; kernels 3/5/7/9; 128 hidden channels; laboratory conditioning and adapter; six residual temporal blocks; four attention blocks with eight heads; 11 individual and 26 pairwise outputs. The full model has 1,915,749 parameters. These output counts cover the canonical vocabulary; primary evaluation supports 34 actions.
- **Loss:** focal exponent 2, positive BCE weight 1.5, soft-F1 coefficient 0.20. The coefficient is a configured choice; the current experiments do not establish its individual optimality.
- **Best tested versus selected:** the action blend is strongest within the restricted deployment set. The diagnostic global blend reaches 0.4719. Equal blending reaches 0.4693, and each candidate has a separately fitted decoder.
- **Deployment:** the 45.13-second result concerns one public video. The 2.51-hour figure for 200 videos is a linear projection. Current evidence does not establish a completed release-acceptance audit for this model lineage.
- **Kaggle score and rank:** remain unresolved placeholders in the manuscript. This segment presents local development scores only.
- **Visual:** Slide 1 uses a labelled AI-generated concept illustration, not a recording or measured trajectory from the project.

## Sources

Project implementation and results: [temporal configuration](../../../configs/temporal-performance.yaml), [temporal model](../../../src/mabe/temporal_model.py), [ensemble report](../../../artifacts/phase11/ensemble-performance-v2/phase11_report.json), and [inference report](../../../artifacts/phase12/runtime-performance-v2/inference_report.json).

Methodological references: [Lin et al. 2017, focal loss](https://doi.org/10.1109/ICCV.2017.324); [Vaswani et al. 2017, self-attention](https://papers.nips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html); [Zhang et al. 2018, MixUp](https://arxiv.org/abs/1710.09412); [Yun et al. 2019, CutMix](https://arxiv.org/abs/1905.04899); [Cawley and Talbot 2010, model-selection bias](https://www.jmlr.org/papers/v11/cawley10a.html).
