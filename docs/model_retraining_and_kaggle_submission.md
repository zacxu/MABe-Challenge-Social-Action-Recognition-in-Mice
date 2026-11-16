# Model parameter editing, retraining, and Kaggle submission

This document describes the complete performance-model workflow: configuring
an experiment, preparing data, training every model stage, checking local
cross-fitted scores, creating the offline inference bundle, and submitting it
through a Kaggle Notebook.

For the first performance run, use the current `v2` model configurations
unchanged. For later experiments, assign new version names so artifacts and
scores from different configurations cannot be mixed.

## 1. Copy the project to the GPU computer

Copy the project directory, including:

- `src/`
- `configs/`
- `scripts/`
- `notebooks/`
- `requirements.txt`
- `pyproject.toml`

Old `artifacts/`, `cache/`, and `.venv/` directories are optional. Copy them
only when you deliberately want to reuse previous results.

Place the Kaggle competition data on the GPU computer. Set
`paths.dataset_root` in [`configs/base.yaml`](../configs/base.yaml) to the
competition directory, for example:

```yaml
paths:
  dataset_root: MABe-mouse-behavior-detection
```

## 2. Install the environment

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-optional.txt
.\.venv\Scripts\python.exe -m pip install --no-deps --editable .
```

### Linux

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install --no-deps --editable .
```

Confirm that PyTorch can access the GPU:

```powershell
nvidia-smi
.\.venv\Scripts\python.exe -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

Do not begin the temporal training run unless this reports `CUDA: True` on the
GPU computer.

## 3. Change model parameters

### 3.1 XGBoost parameters

Edit [`configs/xgboost-performance.yaml`](../configs/xgboost-performance.yaml).

The main tree parameters are under `xgboost:`:

```yaml
xgboost:
  device: auto
  n_estimators: 1500
  early_stopping_rounds: 100
  learning_rate: 0.03
  max_depth: 6
  min_child_weight: 8.0
  subsample: 0.85
  colsample_bytree: 0.80
  reg_alpha: 0.20
  reg_lambda: 5.0
  positive_class_weight: 1.0
```

Training sample sizes are under `sampler:`:

```yaml
sampler:
  max_positive_rows_per_lab_action: 50000
  negative_to_positive_ratio: 5.0
  max_rows_per_lab_action: 150000
```

On a confirmed GPU computer, `device: cuda` can be used to make the run fail
immediately when CUDA is unavailable instead of silently using the CPU.

### 3.2 GBDT search and decoder parameters

Edit [`configs/phase8-performance.yaml`](../configs/phase8-performance.yaml).

This file controls:

- `feature_candidates`
- `hyperparameter_candidates`
- Probability threshold search
- Additional smoothing
- Hysteresis
- Short-gap filling
- Minimum predicted-run duration
- Exact-F-beta model and decoder promotion thresholds

The winning screened configuration is retrained across all five folds and is
only promoted if it clears the exact official-metric gate.

### 3.3 Temporal model parameters

Edit [`configs/temporal-performance.yaml`](../configs/temporal-performance.yaml).

The architecture is under `architecture:`:

```yaml
architecture:
  hidden_channels: 128
  stem_kernel_sizes: [3, 5, 7, 9]
  tcn_dilations: [1, 2, 4, 8, 16, 32]
  dropout: 0.25
  attention_heads: 8
  attention_layers: 4
```

Optimizer, training capacity, loss, and early stopping are under `training:`:

```yaml
training:
  epochs: 30
  steps_per_epoch: 1000
  batch_size: 16
  learning_rate: 0.0003
  weight_decay: 0.0001
  early_stopping_patience: 6
  early_stopping_metric: average_precision
  focal_gamma: 2.0
  soft_f1_weight: 0.20
```

Data augmentation is under `augmentation:` in the same file. It includes:

- Coordinate noise
- Time stretching
- Keypoint dropout
- Left/right landmark swapping
- Mouse permutation
- Rotation and reflection
- Scale jitter
- Input-frame shifting
- Same-lab MixUp
- Same-lab CutMix

If training runs out of GPU memory, reduce parameters in this order:

1. `batch_size`: `16` to `8`
2. `window_frames`: `512` to `384` or `256`
3. `hidden_channels`: `128` to `96`
4. `attention_layers`: `4` to `3` or `2`

The Python architecture is implemented in
[`src/mabe/temporal_model.py`](../src/mabe/temporal_model.py). Edit the Python
file only when adding a completely new layer or architecture family. Ordinary
architecture changes belong in the YAML file.

### 3.4 Ensemble and stacker parameters

Edit [`configs/ensemble-performance.yaml`](../configs/ensemble-performance.yaml).

This file controls:

- Cross-fitted blend-weight search
- Threshold and interval-decoder search
- Nonlinear probability stacker capacity
- Exact-F-beta promotion gates
- Six-hour inference budget and required headroom

### 3.5 Inference parameters

Edit [`configs/inference-performance.yaml`](../configs/inference-performance.yaml).

This file selects the trained artifacts used for inference, test-fold
aggregation, runtime budget, and the AdaptableSnail 25-fps policy.

The default policy is `drop`, because the existing local exact-OOF replay
scored:

- `drop`: `0.311348`
- `scale_30_over_25`: `0.307920`
- `none`: `0.307903`

Treat a different policy as a separate Kaggle A/B experiment and use a new
inference version.

## 4. Version every new experiment

For the first run, retain the current `v2` model names. For a new experiment,
use a consistent set of new versions. For example:

| Configuration | Example new value |
|---|---|
| XGBoost `model_version` | `xgb-performance-v3` |
| Phase 8 `phase8_version` | `gbdt-performance-v3` |
| Phase 8 `baseline_model_version` | `xgb-performance-v3` |
| Phase 8 `promoted_model_version` | `xgb-performance-v3-tuned` |
| Temporal `temporal_version` | `temporal-performance-v3` |
| Ensemble `phase11_version` | `ensemble-performance-v3` |
| Inference `inference_version` | `kaggle-inference-performance-v8` |

Update the GBDT and temporal input manifests in
`configs/ensemble-performance.yaml`:

```yaml
base_streams:
  - name: gbdt
    manifest: artifacts/phase8/gbdt-performance-v3/final_oof_manifest.json
    deployable: true
    eligible_for_optimized_blend: true

  - name: temporal_phase9
    manifest: artifacts/temporal/temporal-performance-v3/oof_manifest.json
    deployable: true
    eligible_for_optimized_blend: true
```

Update the corresponding routes in `configs/inference-performance.yaml`:

```yaml
assets:
  xgboost_registry: artifacts/phase8/gbdt-performance-v3/final_model_registry.parquet
  xgboost_config: artifacts/phase8/gbdt-performance-v3/selected_xgboost.yaml

  temporal_lab_registry: artifacts/temporal/temporal-performance-v3/lab_registry.json
  temporal_checkpoint_root: artifacts/temporal/temporal-performance-v3/checkpoints

  blend_weights: artifacts/phase11/ensemble-performance-v3/selected_blend_weight_registry.parquet
  decoder_parameters: artifacts/phase11/ensemble-performance-v3/selected_decoder_parameters.parquet
  probability_decoder_selection: artifacts/phase11/ensemble-performance-v3/selected_probability_decoder_selection.parquet
  stacker_registry: artifacts/phase11/ensemble-performance-v3/stacker/model_registry.parquet
  final_model_registry: artifacts/phase11/ensemble-performance-v3/final_model_registry.json

  deployment_weights: artifacts/phase12/kaggle-inference-performance-v8/deployment_weights.parquet
  deployment_decoder_parameters: artifacts/phase12/kaggle-inference-performance-v8/deployment_decoder_parameters.parquet
```

Unique versions prevent stale checkpoints, candidate reports, or deployment
parameters from being reused accidentally.

## 5. Prepare the data

Run these commands once on a new computer or dataset copy:

```powershell
.\.venv\Scripts\python.exe -m mabe audit-data --config configs\base.yaml
.\.venv\Scripts\python.exe -m mabe build-labels --config configs\base.yaml
.\.venv\Scripts\python.exe -m mabe build-folds --config configs\base.yaml
.\.venv\Scripts\python.exe -m mabe build-poses --config configs\base.yaml --pose-config configs\pose.yaml
.\.venv\Scripts\python.exe -m mabe build-features --config configs\base.yaml --pose-config configs\pose.yaml --feature-config configs\features.yaml
```

Rerun `build-poses` after modifying `configs/pose.yaml`. Rerun
`build-features` after modifying `configs/features.yaml`.

## 6. Train the high-capacity XGBoost baseline

```powershell
.\.venv\Scripts\python.exe -m mabe train-xgb-baseline --config configs\base.yaml --pose-config configs\pose.yaml --feature-config configs\features.yaml --xgboost-config configs\xgboost-performance.yaml
```

Print the exact cross-fitted score:

```powershell
.\.venv\Scripts\python.exe -c "import json; r=json.load(open(r'artifacts/xgb/xgb-performance-v2/cv_report.json')); print('GBDT exact OOF F-beta:', r['overall_fbeta'])"
```

The primary report is:

```text
artifacts/xgb/xgb-performance-v2/cv_report.json
```

When using a newer version, replace `xgb-performance-v2` in the report path.

## 7. Run GBDT tuning, full retraining, and calibration

```powershell
.\.venv\Scripts\python.exe -m mabe strengthen-gbdt --config configs\base.yaml --phase8-config configs\phase8-performance.yaml --pose-config configs\pose.yaml --feature-config configs\features.yaml --xgboost-config configs\xgboost-performance.yaml
```

Print the selected result:

```powershell
.\.venv\Scripts\python.exe -c "import json; r=json.load(open(r'artifacts/phase8/gbdt-performance-v2/phase8_report.json')); print('Final GBDT exact OOF F-beta:', r['final_overall_fbeta']); print('Gain:', r['exact_fbeta_gain']); print('Model promoted:', r['model_probability_change_promoted'])"
```

The selected deployment configuration is written automatically to:

```text
artifacts/phase8/gbdt-performance-v2/selected_xgboost.yaml
```

Do not manually select a screened candidate. The full cross-fitted exact-metric
gate performs that selection.

## 8. Train the temporal model

Use a CUDA GPU for this stage:

```powershell
.\.venv\Scripts\python.exe -m mabe train-temporal --config configs\base.yaml --temporal-config configs\temporal-performance.yaml --pose-config configs\pose.yaml --phase8-config configs\phase8-performance.yaml
```

The pipeline trains fold 0 first. If its validation average precision is below
the configured minimum, it stops before spending time on folds 1 through 4.

Print the temporal score and gate result:

```powershell
.\.venv\Scripts\python.exe -c "import json; r=json.load(open(r'artifacts/temporal/temporal-performance-v2/phase9_report.json')); print('Temporal exact OOF F-beta:', r['neural_overall_fbeta']); print('GBDT reference:', r['phase8_gbdt_fbeta']); print('Gate:', r['one_fold_gate'])"
```

Confirm that the model card reports a CUDA device. Checkpoints are stored at:

```text
artifacts/temporal/temporal-performance-v2/checkpoints/
```

If training is interrupted, rerun the same command. A checkpoint is reused
only when its configuration hash matches the current temporal configuration.

## 9. Evaluate and select the final ensemble

```powershell
.\.venv\Scripts\python.exe -m mabe ensemble-oof --config configs\base.yaml --phase11-config configs\ensemble-performance.yaml
```

Print the selected model and every candidate score:

```powershell
.\.venv\Scripts\python.exe -c "import json; r=json.load(open(r'artifacts/phase11/ensemble-performance-v2/phase11_report.json')); print('Final exact OOF F-beta:', r['final_exact_fbeta']); print('Selected model:', r['selected_model_id']); print('Runtime feasible:', r['runtime_feasible']); print('Candidates:', r['candidate_scores'])"
```

Before proceeding, confirm:

- `runtime_feasible` is `true`.
- `selected_model_id` is present.
- `final_exact_fbeta` is not below the GBDT baseline.
- The command ended without a traceback.

A temporal model, convex blend, or stacker that does not clear its exact-score
and runtime gates cannot replace the GBDT baseline.

## 10. Run local test inference

Freeze the selected cross-fitted parameters:

```powershell
.\.venv\Scripts\python.exe scripts\run_inference.py freeze --config configs\inference-performance.yaml
```

Run inference on the available test data:

```powershell
.\.venv\Scripts\python.exe scripts\run_inference.py run --config configs\inference-performance.yaml --dataset-root MABe-mouse-behavior-detection --work-root artifacts\phase12\runtime-performance-v2 --submission artifacts\phase12\submission-performance-v2.csv
```

Confirm:

- At least one video completed.
- No top-level exception occurred.
- A route such as `blend`, `stacker`, `gbdt`, or `temporal` was used.
- The submission exists at
  `artifacts/phase12/submission-performance-v2.csv`.

The hidden Kaggle score cannot be calculated during local inference.

## 11. Build the Kaggle archive

```powershell
.\.venv\Scripts\python.exe scripts\run_inference.py archive --config configs\inference-performance.yaml --bundle-root artifacts\phase12\kaggle-bundle-performance-v2 --archive-path artifacts\phase12\mabe-performance-v2.zip
```

The archive command automatically:

1. Freezes the selected deployment parameters.
2. Copies the selected GBDT, temporal, decoder, and stacker lineage.
3. Rewrites model paths to portable relative paths.
4. Creates `bundle_manifest.json`.
5. Verifies all bundle hashes.
6. Creates the uploadable zip.

The file to upload is:

```text
artifacts/phase12/mabe-performance-v2.zip
```

If version names were changed, also use matching version names in
`--bundle-root` and `--archive-path`.

## 12. Upload the model bundle to Kaggle

1. Open Kaggle and go to **Datasets**.
2. Click **New Dataset**.
3. Upload `mabe-performance-v2.zip`.
4. Keep the Dataset private.
5. Name it clearly, for example `mabe-performance-v2`.
6. Finish creating the Dataset.

Do not upload this zip directly as the competition submission. It is an input
Dataset for the offline inference notebook.

## 13. Create the Kaggle inference notebook

1. Open the MABe competition.
2. Open **Code** and create a new Notebook.
3. Attach the MABe competition data.
4. Attach the private `mabe-performance-v2` Dataset.
5. Remove every older MABe model-bundle Dataset from the Notebook inputs.
6. Ensure only one `bundle_manifest.json` will be discoverable.
7. Select a GPU accelerator.
8. Turn Internet off.
9. Import or copy
   [`notebooks/phase12_kaggle_inference.ipynb`](../notebooks/phase12_kaggle_inference.ipynb).

## 14. Test the Kaggle notebook

Run all cells interactively once. Confirm that the output contains:

```text
Verified offline bundle
Inference version: kaggle-inference-performance-v7
Discovered competition data
```

At the end, confirm:

- There is no traceback.
- `Failed videos` is zero, ideally.
- Inference did not use only the `skipped` route.
- `/kaggle/working/submission.csv` exists.
- The submission has the seven required columns.
- The submission has at least one row.

## 15. Commit and submit to the competition

1. Click **Save Version**.
2. Choose **Save & Run All**.
3. Wait for the committed version to finish.
4. Confirm that its status is **Succeeded**.
5. Open the committed version output.
6. Confirm that `submission.csv` is present.
7. Click **Submit to Competition**.
8. Select `submission.csv`.
9. Enter a description containing the experiment version, local exact OOF
   score, selected model ID, and AdaptableSnail policy.
10. Submit and wait for Kaggle to calculate the score.

Store the Kaggle score together with these fields from
`artifacts/phase11/<phase11-version>/phase11_report.json`:

- `final_exact_fbeta`
- `selected_model_id`
- `runtime_feasible`
- `candidate_scores`

This makes each leaderboard result traceable to the configuration and local
cross-fitted score that produced it.
