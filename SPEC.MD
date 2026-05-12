# Paper 2 Spec

This is the working contract for the Paper 2 experiment runners.

## Goals

- Keep one active package under `src/paper_2/`.
- Make dataset paths configurable with environment variables.
- Keep model outputs reproducible and easy to compare.
- Avoid hardcoded machine-specific paths in active code.

## Active Package Layout

```text
src/paper_2/
  config.py       dataset registry and hyperparameters
  data_loader.py  dataset loading + preprocessing
  glade_v2.py      booleanizer
  run_ml.py       ML baselines
  run_tinyml.py   TinyML baselines
  run_ml_tmmatched.py  ML baselines matched to TM capacity
  run_tm.py       GLADE + TM via Julia
  run_any.py      run one model family on one dataset
  run_all.py      run all model families
  utils.py        metrics and report helpers
  train_tm.jl     Julia TM training script
```

## Path Rules

Use environment variables instead of hardcoding:

```text
PAPER2_DATA_ROOT   base dataset root (default: /IoT/Datasets)
PAPER2_NSLKDD_ROOT NSL-KDD folder with KDDTrain+.txt and KDDTest+.txt
PAPER2_TSETLIN_PATH path to Tsetlin.jl
PAPER2_TMP_DIR     temp folder for binarized TM data
```

## Dataset Usage

Paper_2 expects datasets in these locations by default:

```text
/IoT/Datasets/TON_IoT/ton_iot_network.csv
/IoT/Datasets/medsec/MedSec-25.csv
/IoT/Datasets/WUSTL/wustl-ehms-2020_with_attacks_categories.csv
```

NSL-KDD is loaded from `PAPER2_NSLKDD_ROOT` and must contain:

```text
KDDTrain+.txt
KDDTest+.txt
```

If your datasets live elsewhere, set the environment variables before running:

```bash
export PAPER2_DATA_ROOT=/path/to/Datasets
export PAPER2_NSLKDD_ROOT=/path/to/nslkdd
```

If NSL-KDD is missing, you can still run the other datasets by temporarily
removing `nslkdd` from `DATASETS` in `src/paper_2/config.py`.

### NSL-KDD Download Helper

Use this helper to download NSL-KDD via KaggleHub:

```bash
python3 Datasets/download_nslkdd.py
PAPER2_NSLKDD_ROOT=/some/path python3 Datasets/download_nslkdd.py
```

Script content:

```python
#!/usr/bin/env python3
"""
Download NSL-KDD (hassan06/nslkdd) from Kaggle via kagglehub.

Prereqs:
  pip install kagglehub
  Kaggle API credentials at ~/.kaggle/kaggle.json
    (download from kaggle.com -> account -> "Create New API Token")

Usage:
  python3 download_nslkdd.py            # default dest: /IoT/Datasets/NSL-KDD
  PAPER2_NSLKDD_ROOT=/some/path python3 download_nslkdd.py

Required files placed in dest:
  KDDTrain+.txt
  KDDTest+.txt
"""

import os
import shutil
import sys

DEST = os.environ.get("PAPER2_NSLKDD_ROOT", "/IoT/Datasets/NSL-KDD")
NEEDED = ("KDDTrain+.txt", "KDDTest+.txt")


def main():
  try:
    import kagglehub
  except ImportError:
    sys.exit("kagglehub not installed. Run: pip install kagglehub")

  print("Downloading hassan06/nslkdd from Kaggle...")
  src = kagglehub.dataset_download("hassan06/nslkdd")
  print(f"  Kaggle cache: {src}")

  os.makedirs(DEST, exist_ok=True)
  for fn in NEEDED:
    s = os.path.join(src, fn)
    d = os.path.join(DEST, fn)
    if not os.path.exists(s):
      sys.exit(f"Missing in download: {s}")
    if not os.path.exists(d):
      shutil.copy(s, d)
    print(f"  {fn:<18} -> {d}")

  print(f"\nDone. PAPER2_NSLKDD_ROOT={DEST}")


if __name__ == "__main__":
  main()
```

## Results Rules

All outputs live under `results/`:

```text
results/<dataset>/reports/*.txt|*.json
results/<dataset>/models/*.pkl
results/<dataset>/models/*.pkl.zst
```

Do not commit generated caches, `__pycache__`, or binaries.

## Validation

Use the smallest run that proves your change:

```bash
python -m paper_2.run_any --task ml --dataset nslkdd
python -m paper_2.run_any --task tm --dataset nslkdd
```
