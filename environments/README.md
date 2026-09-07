# Environment records

`local.json` is generated from the active virtual environment with:

```powershell
python -m mabe environment --name local --output environments/local.json
```

`kaggle-v170-gpu.json` identifies the published Kaggle GPU image used as the
portability reference. Kaggle images change, so the exact notebook runtime must
also be captured at the start of every Kaggle run:

```bash
python -m mabe environment \
  --name kaggle-runtime \
  --output /kaggle/working/kaggle-runtime.json
```

The command imports every core dependency by default and fails if an offline
Kaggle notebook cannot import one. If that happens, attach a compatible wheel as
a Kaggle Dataset rather than enabling internet access or silently changing the
pipeline.
