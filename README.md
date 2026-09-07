# MABe performance training and Kaggle submission

This is the versioned performance path. It does not overwrite the original
v1/v6 artifacts, and every promoted model is selected with cross-fitted OOF
predictions and the official F-beta metric.

## Where to change parameters

| File | Parameters |
|---|---|
| `configs/xgboost-performance.yaml` | Tree depth/count, learning rate, regularization, class weight, sample sizes, feature groups, early stopping |
| `configs/phase8-performance.yaml` | GBDT search candidates, full-model promotion gate, probability thresholds, smoothing, hysteresis, gap fill, minimum duration |
| `configs/temporal-performance.yaml` | Temporal architecture, windows, epochs/steps, optimizer, AP early stopping, augmentations, TTA, device |
| `configs/ensemble-performance.yaml` | Blend grid, nonlinear stacker, decoder search, exact-F-beta and runtime promotion gates |
| `configs/inference-performance.yaml` | Selected artifact routes, six-hour runtime budget, test-fold averaging, AdaptableSnail policy |

The temporal architecture is implemented in `src/mabe/temporal_model.py`.
Normally, change its `architecture:` block in
`configs/temporal-performance.yaml` rather than editing Python. When starting a
new experiment, change every version field and its downstream paths together;
never reuse a version name for incompatible artifacts.

## Install

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install --no-deps --editable .
```

On Linux, replace `.\.venv\Scripts\python.exe` with
`.venv/bin/python`. Set `paths.dataset_root` in `configs/base.yaml` if the
competition data is not at `MABe-mouse-behavior-detection/`.

## Prepare data once

```powershell
.\.venv\Scripts\python.exe -m mabe audit-data --config configs\base.yaml
.\.venv\Scripts\python.exe -m mabe build-labels --config configs\base.yaml
.\.venv\Scripts\python.exe -m mabe build-folds --config configs\base.yaml
.\.venv\Scripts\python.exe -m mabe build-poses --config configs\base.yaml --pose-config configs\pose.yaml
.\.venv\Scripts\python.exe -m mabe build-features --config configs\base.yaml --pose-config configs\pose.yaml --feature-config configs\features.yaml
```

## Train and print the score

Use a CUDA GPU for the temporal run. Each training command prints JSON with
its exact OOF score. Run them in this order.

### 1. High-capacity GBDT

```powershell
.\.venv\Scripts\python.exe -m mabe train-xgb-baseline --config configs\base.yaml --pose-config configs\pose.yaml --feature-config configs\features.yaml --xgboost-config configs\xgboost-performance.yaml
.\.venv\Scripts\python.exe -c "import json; r=json.load(open(r'artifacts/xgb/xgb-performance-v2/cv_report.json')); print('GBDT exact OOF F-beta:', r['overall_fbeta'])"
```

### 2. GBDT search, full retraining, and calibration

```powershell
.\.venv\Scripts\python.exe -m mabe strengthen-gbdt --config configs\base.yaml --phase8-config configs\phase8-performance.yaml --pose-config configs\pose.yaml --feature-config configs\features.yaml --xgboost-config configs\xgboost-performance.yaml
.\.venv\Scripts\python.exe -c "import json; r=json.load(open(r'artifacts/phase8/gbdt-performance-v2/phase8_report.json')); print('Selected GBDT exact OOF F-beta:', r['final_overall_fbeta']); print('Probability model promoted:', r['model_probability_change_promoted'])"
```

Phase 8 retrains the winning screen configuration over all five folds. It only
promotes that probability stream when it clears the exact OOF gate; otherwise
the baseline GBDT remains selected. The chosen deployment configuration is
written to `artifacts/phase8/gbdt-performance-v2/selected_xgboost.yaml`.

### 3. Temporal model

```powershell
.\.venv\Scripts\python.exe -m mabe train-temporal --config configs\base.yaml --temporal-config configs\temporal-performance.yaml --pose-config configs\pose.yaml --phase8-config configs\phase8-performance.yaml
.\.venv\Scripts\python.exe -c "import json; r=json.load(open(r'artifacts/temporal/temporal-performance-v2/phase9_report.json')); print('Temporal exact OOF F-beta:', r['neural_overall_fbeta']); print('Selected GBDT reference:', r['phase8_gbdt_fbeta'])"
```

Fold 0 is the gate. If its validation AP is below the configured minimum, the
run stops before spending time on folds 1-4. Matching checkpoints are reused
when the command is restarted.

### 4. Cross-fitted blend and stacker

```powershell
.\.venv\Scripts\python.exe -m mabe ensemble-oof --config configs\base.yaml --phase11-config configs\ensemble-performance.yaml
.\.venv\Scripts\python.exe -c "import json; r=json.load(open(r'artifacts/phase11/ensemble-performance-v2/phase11_report.json')); print('Final exact OOF F-beta:', r['final_exact_fbeta']); print('Selected model:', r['selected_model_id']); print('Candidates:', r['candidate_scores'])"
```

The final registry selects only a deployable candidate that passes the
six-hour runtime/headroom gate. A weaker temporal model, blend, or stacker is
evaluated but cannot replace the stronger GBDT.

## Local test inference and Kaggle archive

```powershell
.\.venv\Scripts\python.exe scripts\run_inference.py freeze --config configs\inference-performance.yaml
.\.venv\Scripts\python.exe scripts\run_inference.py run --config configs\inference-performance.yaml --dataset-root MABe-mouse-behavior-detection --work-root artifacts\phase12\runtime-performance-v7 --submission artifacts\phase12\submission-performance-v7.csv
.\.venv\Scripts\python.exe scripts\run_inference.py archive --config configs\inference-performance.yaml --bundle-root artifacts\phase12\kaggle-bundle-performance-v7 --archive-path artifacts\phase12\mabe-performance-v7.zip
```

The archive command freezes deployment parameters, builds the self-contained
offline bundle, verifies every bundled hash, and creates
`artifacts/phase12/mabe-performance-v7.zip`. Local inference cannot print a
hidden-test score; Kaggle supplies that score after submission.

## Submit to Kaggle

1. On Kaggle, create a new private Dataset and upload
   `artifacts/phase12/mabe-performance-v7.zip`. Kaggle extracts it to a folder
   named `kaggle-bundle-performance-v7`.
2. Open the MABe competition, create a new Notebook, and attach both the
   competition data and your private bundle Dataset. Remove older MABe bundle
   datasets so only one `bundle_manifest.json` is discoverable.
3. Import or copy `notebooks/phase12_kaggle_inference.ipynb`, choose a GPU
   accelerator, turn Internet off, and run all cells.
4. Confirm the output reports `kaggle-inference-performance-v7`, full inference
   completion, a valid route summary, and
   `/kaggle/working/submission.csv`. Any traceback means the version must not be
   submitted.
5. Click **Save Version**, choose **Save & Run All**, wait for a green
   **Succeeded** status, then use **Submit to Competition** and select
   `submission.csv` from the notebook output.
6. Record the Kaggle score together with the selected model ID and local exact
   OOF F-beta from the Phase 11 report.

The performance profile drops predictions only for AdaptableSnail 25-fps
videos: this scored `0.311348` in local exact-OOF replay versus `0.307920` for
`scale_30_over_25` and `0.307903` for `none` on the existing v1 ensemble. To
run a controlled alternative, change
`special_cases.adaptable_snail_25fps_policy` in
`configs/inference-performance.yaml`, then repeat freeze, local inference, and
archive under a new inference version.
