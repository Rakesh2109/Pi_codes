# Fuzzy TM Local Inference

Python-first local inference package for the bundled Fuzzy TM models.

## Layout

```text
fuzzy_tm_infer/
  algorithms/           Python/Numba algorithm implementations
    py/                 Python/Numba algorithm implementations
    c/                  Optional C algorithm implementations
  booleanizers/         Feature-to-Boolean threshold transformers
  benchmarks/           Benchmark orchestration
  tm_infer.py           Python benchmark entry point
  benchmark.py          Compatibility entry point for benchmarks
  data.py               Dataset/model path loading helpers
  metrics/              Evaluation metrics
  assets.py             Asset extraction helper
  scripts/              Operational scripts and verification tools
  tools/                Non-core experiments/utilities
  docs/                 Notes for optional backends/tools
```

## Python Inference

```bash
python3 -m pip install -r src/fuzzy_tm_infer/requirements.txt
python3 src/fuzzy_tm_infer/tm_infer.py
```

Compare against the Decision Tree models from `src/ml_models.zip`:

```bash
python3 src/fuzzy_tm_infer/tm_infer.py --compare-dt
```

That command extracts only `DecisionTree.pkl` and `testset.npz` for each
dataset into `assets/ml_models/`.

Compare booleanizer fit/transform speed:

```bash
python3 src/fuzzy_tm_infer/tm_infer.py --compare-booleanizers
```

## Logging

Python CLIs use `loguru` for operational logs. Benchmark result tables stay on
stdout, while setup/deployment messages honor:

```bash
fuzzy-tm-infer --verbose
fuzzy-tm-infer --quiet
FUZZY_TM_LOG_LEVEL=DEBUG fuzzy-tm-infer
```

## Booleanization

GLADE v2 is packaged as a reusable Boolean feature transformer:

```python
from fuzzy_tm_infer.booleanizers import (
    GLADEBooleanizer,
    KBinsBooleanizer,
    StandardBinarizer,
)

booleanizer = GLADEBooleanizer(n_bins=15).fit(X_train)
X_bits = booleanizer.transform(X_test)
X_packed = booleanizer.transform(X_test, pack_bits=True)

standard = StandardBinarizer(max_bits_per_feature=25).fit(X_train)
X_standard_bits = standard.transform(X_test)

kbins = KBinsBooleanizer(n_bins=8, strategy="quantile").fit(X_train)
X_kbins_bits = kbins.transform(X_test)
```

`ThermometerBinarizer` is also available for fixed evenly spaced thresholds,
including CIFAR-style color tensors.

Fast GLADE v2 transform is available after fitting the normal Python model:

```python
from fuzzy_tm_infer.algorithms.c import FastGLADEBooleanizer
from fuzzy_tm_infer.booleanizers import GLADEBooleanizer

glade = GLADEBooleanizer(n_bins=15).fit(X_train)
fast_glade = FastGLADEBooleanizer.from_booleanizer(glade)
X_packed_fast = fast_glade.transform(X_test, pack_bits=True)
```

Verify the native transform:

```bash
make -C src/fuzzy_tm_infer verify-fast-glade
```

## Optional Native Backend

```python
from fuzzy_tm_infer.data import load_X, model_path
from fuzzy_tm_infer.algorithms.c import FuzzyTMModel

X = load_X("WUSTL")

with FuzzyTMModel.from_fbz(model_path("wustl"), version="v17") as model:
    model.calibrate(X[:3000])
    y_pred = model.predict_batch(X)
```

Build and verify native backends:

```bash
make -C src/fuzzy_tm_infer native-v17
make -C src/fuzzy_tm_infer native-v19
make -C src/fuzzy_tm_infer verify-native
```

More native notes: `docs/native_backends.md`.

Raspberry Pi deployment and native test automation:

```bash
cp ansible/rpi.env.example ansible/rpi.env
tm-rpi-deploy --install-collections
```

Project development contract: `SPEC.MD`.

## Tools

Compression experiments are intentionally outside the root inference surface:

```bash
python3 -m pip install -r src/fuzzy_tm_infer/tools/compression/requirements.txt
python3 src/fuzzy_tm_infer/tools/compression/compare.py
```
