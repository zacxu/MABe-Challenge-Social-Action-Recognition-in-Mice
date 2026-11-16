# Phase 12 performance bundle runbook

The authoritative commands and Kaggle submission steps are in the project
`README.md`. The release profile is `configs/inference-performance.yaml`.

Build one verified, self-contained upload archive:

```powershell
.\.venv\Scripts\python.exe scripts\run_inference.py archive --config configs\inference-performance.yaml --bundle-root artifacts\phase12\kaggle-bundle-performance-v7 --archive-path artifacts\phase12\mabe-performance-v7.zip
```

The command first freezes the cross-fitted deployment weights and decoder,
copies only the selected GBDT/temporal/stacker lineage, rewrites model paths to
portable bundle-relative paths, verifies SHA-256 hashes, and then creates the
zip with `kaggle-bundle-performance-v7` as its top-level folder.

Before uploading, the Phase 11 report must name the selected deployable model
and report a runtime-feasible exact OOF score. A full local raw inference run
should also finish without a fallback-only submission:

```powershell
.\.venv\Scripts\python.exe scripts\run_inference.py run --config configs\inference-performance.yaml --dataset-root MABe-mouse-behavior-detection --work-root artifacts\phase12\runtime-performance-v7 --submission artifacts\phase12\submission-performance-v7.csv
```

In Kaggle, attach the competition data and exactly one private bundle Dataset,
use `notebooks/phase12_kaggle_inference.ipynb`, select a GPU, disable Internet,
run all cells, and save a committed notebook version. Submit only when the
committed version succeeds and exposes `/kaggle/working/submission.csv`.

The hidden score is unavailable during inference. It appears only after Kaggle
accepts the notebook output as a competition submission.
