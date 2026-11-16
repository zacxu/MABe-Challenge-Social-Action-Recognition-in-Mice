# Understanding the temporal model's loss

This guide explains the objective already implemented in the project. It does not introduce a new loss, change training settings, or claim that these settings are optimal. The paper-facing derivation is in Section 1.4.1 of [the research section](phase9_13_research_section.md); its settings are in Table 3.

## 1. Where it comes from

There are three separate sources of the formulation:

1. **Your implementation is the source of the exact joint objective.** [temporal_model.py](../src/mabe/temporal_model.py) defines `masked_bce_with_logits`, `masked_macro_soft_f1_loss`, and `combined_temporal_loss`. The last function returns `bce + soft_f1_weight * soft_f1`.
2. **The focal factor follows published work.** [Lin et al. (2017), Section 3.2, p. 3](https://arxiv.org/pdf/1708.02002v1#page=3) describes downweighting easy examples using a focusing exponent. This project adapts that idea, rather than copying their entire loss or prescribing this joint objective.
3. **Soft F1 follows the ordinary F1 identity.** We substitute probabilities for hard predictions to obtain differentiable counts. Adding its penalty to focal BCE is a design choice; the addition and the coefficient 0.20 do not follow necessarily from the F1 algebra.

The positive-weighted BCE formula is also documented in the [official PyTorch reference](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BCEWithLogitsLoss.html). That is a documentation source, not a claim that this version was used for the saved training run.

The four configured loss settings come from [temporal-performance.yaml](../configs/temporal-performance.yaml). Its SHA-256 is `6164798648263fd1bfba18d2a037d16c98f3cdc7ee239dc6352248f246f05d0e`, matching the saved [Phase 9 report](../artifacts/temporal/temporal-performance-v2/phase9_report.json). Thus, these settings are linked to the recorded performance-v2 run, not just to an unrelated example configuration.

## 2. Notation: what is a prediction here?

For batch example $b$, frame $t$, and action $a$:

- $z_{bta}$ is the model's raw output, or logit.
- $p_{bta}=\sigma(z_{bta})$ is its probability between zero and one.
- $y_{bta}$ is the target: normally zero or one, but potentially fractional after MixUp.
- $m_{bta}$ is the supervision mask.
- $s_{\ell(b)}$ is the weight of the example's laboratory.
- $w_{bta}=m_{bta}s_{\ell(b)}$ is the effective supervision weight.

The model's convolution, attention, embedding, and output-head weights are **learned parameters**. The coefficients below are **hyperparameters**, chosen outside gradient descent. Masks, targets, probabilities, and the active action set are data/model-dependent quantities, not hyperparameters.

## 3. Framewise classification: weighted binary cross-entropy

Ordinary BCE penalizes probability assigned away from the target. The implementation adds positive weight $\rho$:

$$
\ell^{\mathrm{BCE}}=-\left[\rho y\log p+(1-y)\log(1-p)\right].
$$

For a positive target, a low probability incurs a large penalty; for a negative target, a high probability incurs a large penalty. With $\rho=1.5$, only the positive contribution receives the extra factor. The negative contribution retains coefficient one. This gives positive examples more importance relative to background.

This is PyTorch's `pos_weight` convention, not the paper's parameter named alpha, and it does not weight the soft-F1 counts. For binary targets, weights (1.5, 1) equal 2.5 times (0.6, 0.4), so they are proportional to alpha balancing with alpha 0.6, rather than a different class-balancing principle. See [Lin et al. (2017), Section 3.1, Eq. (3), and Section 3.2, Eq. (5), p. 3](https://arxiv.org/pdf/1708.02002v1#page=3). The overall scale matters when adding soft F1 with a fixed coefficient. The code computes BCE from logits for numerical stability; the probability/log expression above explains its mathematical meaning.

## 4. Focal modulation: reduce easy-example contributions

The code defines

$$
p^{*}=py+(1-p)(1-y),\qquad
\ell^{\mathrm{focal}}=(1-p^{*})^{\gamma}\ell^{\mathrm{BCE}}.
$$

For binary labels, $p^{*}$ corresponds to the paper's correct-class probability $p_t$ in [Eq. (2), p. 3](https://arxiv.org/pdf/1708.02002v1#page=3). Section 3.2, Eq. (4), supplies the focal factor; its discussion explains easy-example suppression, the gamma-zero case, and the 100-fold reduction when $p_t=0.9$ and $\gamma=2$. Substitution gives $(1-0.9)^2=0.01$ and $(1-0.2)^2=0.64$. The 0.64 example is our arithmetic, not a reported experiment. With gamma zero, our implementation retains positive-weighted BCE.

For fractional MixUp targets, the same expression is evaluated directly. This is verified in [the project's masked_bce_with_logits function](../src/mabe/temporal_model.py), not claimed by the original focal-loss paper. It is not obtained by averaging two separately evaluated hard-label focal losses. In the actual training loop, MixUp pairs examples from the same laboratory and intersects their supervision masks; CutMix copies the partner's mask along with the replaced sequence interval.

## 5. Masks and laboratory weights: decide which entries count

The mask is created in [temporal_data.py](../src/mabe/temporal_data.py), in `materialize_window()`. It requires the frame to be within the video's resampled extent rather than window padding, the relevant mouse or pair to be valid, and the action to be annotated for that entity. An unannotated action must not be treated as a negative example. Setting its mask to zero removes its direct contribution to both loss terms.

Laboratory weights are computed in `_lab_loss_weights()` in [temporal_training.py](../src/mabe/temporal_training.py):

$$
s_{\ell}=\frac{n_{\ell}^{-\eta}}{|\mathcal{J}|^{-1}\sum_{j\in\mathcal{J}}n_j^{-\eta}}.
$$

Here $n_{\ell}$ counts training video/sequence artifacts, excluding the held-out fold, and $\mathcal J$ contains the represented training laboratories. With $\eta=0.5$, weights are inverse square roots of these counts, normalized to mean one across laboratories. This partly counterbalances unequal laboratory dataset sizes; it does not guarantee equal influence or count annotated frames/windows.

For example, laboratories with 100 and 25 training videos receive raw weights 0.1 and 0.2. Dividing by their mean, 0.15, gives weights approximately 0.667 and 1.333. With $\eta=0$, every laboratory instead receives weight one.

The reduced focal term is

$$
\mathcal L_{\mathrm{focal}}=\frac{\sum_{b,t,a}w_{bta}\ell_{bta}^{\mathrm{focal}}}{\max\left(1,\sum_{b,t,a}w_{bta}\right)}.
$$

The denominator normally makes this a weighted average. Its floor of one is a code-level guard for zero or subunit total weight; in that case, it is not division by the unmodified weight sum.

## 6. Soft F1: turn an evaluation idea into a differentiable penalty

Starting with precision and recall,

$$
P=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}},\qquad R=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}},\qquad F_1=\frac{2PR}{P+R},
$$

substitution and cancellation give

$$
F_1=\frac{2\mathrm{TP}}{2\mathrm{TP}+\mathrm{FP}+\mathrm{FN}}.
$$

Hard predictions, such as thresholding at 0.5, do not provide useful gradients. Your implementation instead forms continuous counts from probabilities:

$$
\mathrm{TP}_a=\sum_{b,t}w_{bta}p_{bta}y_{bta},\qquad
\mathrm{FP}_a=\sum_{b,t}w_{bta}p_{bta}(1-y_{bta}),\qquad
\mathrm{FN}_a=\sum_{b,t}w_{bta}(1-p_{bta})y_{bta}.
$$

A positive target predicted with probability 0.8 contributes 0.8 to soft TP and 0.2 to soft FN before weighting. These are fractional contributions, not hard counts.

The stabilized ratio is

$$
F_{1,a}^{\mathrm{soft}}=\frac{2\mathrm{TP}_a+\epsilon}{2\mathrm{TP}_a+\mathrm{FP}_a+\mathrm{FN}_a+\epsilon}.
$$

The fixed $\epsilon=10^{-6}$ prevents an undefined zero-denominator ratio. It is added to both numerator and denominator. Positive BCE weight $\rho$ does not appear here: the counts use only probabilities, targets, and effective supervision weights.

Let $\mathcal A$ contain the actions with at least one entry of positive effective supervision weight in the current batch. An action can be active even if all its valid targets are negative. Unsupervised actions are excluded. Because training alternates solo and pair batches, this set belongs to the current route, not all 37 actions simultaneously.

The auxiliary penalty is

$$
\mathcal L_{\mathrm{softF1}}=1-\frac{1}{|\mathcal A|}\sum_{a\in\mathcal A}F_{1,a}^{\mathrm{soft}}.
$$

The macro mean gives each supervised action one contribution. The subtraction converts “higher is better” into “lower is better.” If no actions have supervision, the code explicitly returns a differentiable zero rather than computing an empty mean.

## 7. Combining the terms: an additive design choice

$$
\mathcal L=\mathcal L_{\mathrm{focal}}+\lambda\mathcal L_{\mathrm{softF1}},\qquad\lambda=0.20.
$$

The first term penalizes framewise classification errors, emphasizing difficult examples. The second encourages action-level positive overlap. Lambda determines the scale of the added penalty relative to the first term. Setting it to zero disables the auxiliary term.

This is **not** $0.80\mathcal L_{\mathrm{focal}}+0.20\mathcal L_{\mathrm{softF1}}$. A coefficient of 0.20 also does not mean that 20% of the observed loss or gradient comes from soft F1: those proportions depend on the values and gradients of both components.

This surrogate is not identical to competition F1. It uses minibatch probabilities, laboratory weights, and current-route action averaging. The competition scorer instead evaluates decoded intervals and averages actions within laboratories before averaging laboratories.

## 8. Hyperparameters and numerical guards

| Quantity | Code setting | Value | What it controls |
|---|---|---:|---|
| $\rho$ | `positive_class_weight` | 1.5 | Relative weight of the positive BCE term |
| $\gamma$ | `focal_gamma` | 2.0 | Degree of easy-example suppression |
| $\lambda$ | `soft_f1_weight` | 0.20 | Scale of the added macro soft-F1 penalty |
| $\eta$ | `lab_loss_weight_power` | 0.5 | Strength of inverse-video-count laboratory weighting |
| $\epsilon$ | Function default, not a YAML setting | $10^{-6}$ | Numerical stabilization of soft F1 |
| Denominator floor | Fixed `.clamp_min(1.0)` | 1.0 | Protection against empty/subunit weighted reduction |

The first four are configured loss hyperparameters. The last two are numerical guards fixed in this training path. No controlled performance-v2 loss ablation establishes that these values are optimal or isolates their individual contributions to the reported F1. Historical Phase 10 soft-F1 screens used a different configuration and are not that evidence.

Learning rate, batch size, clipping, dropout, and AdamW weight decay affect training but are separate from this loss expression. In particular, AdamW's decoupled weight decay is not an additional regularization term written into `combined_temporal_loss()`.

## 9. Worked example

This is a synthetic two-frame example, not an experimental result. Use one action, targets $y=(1,0)$, probabilities $p=(0.8,0.6)$, and unit effective supervision weights.

| Quantity | Positive frame | Negative frame |
|---|---:|---:|
| Target | 1 | 0 |
| Predicted probability | 0.8 | 0.6 |
| Positive-weighted BCE | $-1.5\log(0.8)=0.33471533$ | $-\log(0.4)=0.91629073$ |
| Correct-class probability | 0.8 | 0.4 |
| Focal factor | $0.2^2=0.04$ | $0.6^2=0.36$ |
| Focal-weighted BCE | 0.01338861 | 0.32986466 |

The mean focal loss is 0.17162664. Soft TP, FP, and FN are 0.8, 0.6, and 0.2, respectively. Including epsilon gives soft F1 approximately 0.66666681, so the auxiliary penalty is 0.33333319. Therefore,

$$
\mathcal L\approx0.17162664+0.20(0.33333319)=0.23829328.
$$

Notice how the uncertain negative frame contributes more focal loss than the confidently positive frame. This explains the arithmetic, not a claim about performance on mouse videos.

## 10. Checking that the explanation matches the code

[check_temporal_loss_derivation.py](check_temporal_loss_derivation.py) independently evaluates these equations and compares their components and gradients with the implemented functions. Run it with:

```powershell
.venv\Scripts\python.exe report\check_temporal_loss_derivation.py
```

The [saved validation](temporal_loss_derivation_validation.json) records 10 passing float64 synthetic cases: binary targets, fractional targets, partial masks/inactive actions, unequal weights, empty supervision, subunit denominator clamping, lambda zero, gamma zero, valid negative-only actions, and the worked example. These checks verify mathematical transcription; they do not retrain the model or establish hyperparameter optimality.
