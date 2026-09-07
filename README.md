# MABe performance training and Kaggle submission

This is the versioned performance path. It does not overwrite the original
v1/v6 artifacts, and every promoted model is selected with cross-fitted OOF
predictions and the official F-beta metric.

## Where to change parameters

| File | Parameters |
|---|---|
| `configs/xgboost-performance.yaml` | Tree depth/count, learning rate, regularization, class weight, sample sizes, feature groups, early stopping |
| `configs/phase8-performance.yaml` | GBDT search candidates, hyperparameter_candidates, full-model promotion gate, probability threshold search, smoothing, hysteresis, short gap filling, minimum run duration, exact-F-beta model and decoder promotion thresholds |
| `configs/temporal-performance.yaml` | Temporal architecture, windows, epochs/steps, optimizer, AP early stopping, data augmentations, TTA, device |
| `configs/ensemble-performance.yaml` | Blend grid, nonlinear probability stacker, decoder search, exact-F-beta and runtime promotion gates, inference budget |
| `configs/inference-performance.yaml` | Selected artifact routes, test fold aggregation, six-hour runtime budget, test-fold averaging, AdaptableSnail policy |

The temporal architecture is implemented in `src/mabe/temporal_model.py`.
Normally, change its `architecture:` block in
`configs/temporal-performance.yaml` rather than editing Python. When starting a
new experiment, change every version field and its downstream paths together;
never reuse a version name for incompatible artifacts.

## Version every new experiment

For the first run, retain the current `v2` model names. For a new experiment,
use a consistent set of new versions. For example: use v3

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
`configs/ensemble-performance.yaml`
Update the corresponding routes in `configs/inference-performance.yaml`:

## Install venv

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-optional.txt
.\.venv\Scripts\python.exe -m pip install --no-deps --editable .
```

On Linux:
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
Do not begin training run unless this reports `CUDA: True` on the
GPU computer.

## Prepare data once

Run these commands once on a new dataset:
```powershell
.\.venv\Scripts\python.exe -m mabe audit-data --config configs\base.yaml
.\.venv\Scripts\python.exe -m mabe build-labels --config configs\base.yaml
.\.venv\Scripts\python.exe -m mabe build-folds --config configs\base.yaml
.\.venv\Scripts\python.exe -m mabe build-poses --config configs\base.yaml --pose-config configs\pose.yaml
.\.venv\Scripts\python.exe -m mabe build-features --config configs\base.yaml --pose-config configs\pose.yaml --feature-config configs\features.yaml
```
Rerun everything if you changed any model parameters or hyperparameters



## Run all of the following steps in order
Use a CUDA GPU for the temporal run. Each training command prints JSON with
its exact OOF score. Run them in this order.

### High-capacity GBDT

```powershell
.\.venv\Scripts\python.exe -m mabe train-xgb-baseline --config configs\base.yaml --pose-config configs\pose.yaml --feature-config configs\features.yaml --xgboost-config configs\xgboost-performance.yaml
.\.venv\Scripts\python.exe -c "import json; r=json.load(open(r'artifacts/xgb/xgb-performance-v2/cv_report.json')); print('GBDT exact OOF F-beta:', r['overall_fbeta'])"
```

### GBDT search, full retraining, and calibration

```powershell
.\.venv\Scripts\python.exe -m mabe strengthen-gbdt --config configs\base.yaml --phase8-config configs\phase8-performance.yaml --pose-config configs\pose.yaml --feature-config configs\features.yaml --xgboost-config configs\xgboost-performance.yaml
.\.venv\Scripts\python.exe -c "import json; r=json.load(open(r'artifacts/phase8/gbdt-performance-v2/phase8_report.json')); print('Selected GBDT exact OOF F-beta:', r['final_overall_fbeta']); print('Probability model promoted:', r['model_probability_change_promoted'])"
```

Phase 8 retrains the winning screen configuration over all five folds. It only
promotes that probability stream when it clears the exact OOF gate; otherwise
the baseline GBDT remains selected. The chosen deployment configuration is
written to `artifacts/phase8/gbdt-performance-v2/selected_xgboost.yaml`.

### Temporal model

```powershell
.\.venv\Scripts\python.exe -m mabe train-temporal --config configs\base.yaml --temporal-config configs\temporal-performance.yaml --pose-config configs\pose.yaml --phase8-config configs\phase8-performance.yaml
.\.venv\Scripts\python.exe -c "import json; r=json.load(open(r'artifacts/temporal/temporal-performance-v2/phase9_report.json')); print('Temporal exact OOF F-beta:', r['neural_overall_fbeta']); print('GBDT reference:', r['phase8_gbdt_fbeta']); print('Gate:', r['one_fold_gate'])"
```

Fold 0 is the gate. If its validation AP is below the configured minimum, the
run stops before spending time on folds 1-4. Matching checkpoints are reused
when the command is restarted.

Checkpoints are stored at:
```text
artifacts/temporal/temporal-performance-v2/checkpoints/
```

If training is interrupted, rerun the same command. A checkpoint is reused
only when its configuration hash matches the current temporal configuration.


### 4. Cross-fitted blend and stacker

```powershell
.\.venv\Scripts\python.exe -m mabe ensemble-oof --config configs\base.yaml --phase11-config configs\ensemble-performance.yaml
.\.venv\Scripts\python.exe -c "import json; r=json.load(open(r'artifacts/phase11/ensemble-performance-v2/phase11_report.json')); print('Final exact OOF F-beta:', r['final_exact_fbeta']); print('Selected model:', r['selected_model_id']); print('Runtime feasible:', r['runtime_feasible']); print('Candidates:', r['candidate_scores'])"
```

Before proceeding, confirm:

- `runtime_feasible` is `true`.
- `selected_model_id` is present.
- `final_exact_fbeta` is not below the GBDT baseline.
- The command ended without a traceback.

The final registry selects only a deployable candidate that passes the
six-hour runtime/headroom gate. A weaker temporal model, blend, or stacker is
evaluated but cannot replace the stronger GBDT.

## Local test inference and Kaggle archive

Freeze the selected cross-fitted parameters:
```powershell
.\.venv\Scripts\python.exe scripts\run_inference.py freeze --config configs\inference-performance.yaml

Run inference on the available test set:
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


## Build the Kaggle archive
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


## Create the Kaggle inference notebook

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

## Test the Kaggle notebook

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

## Commit and submit to the competition

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