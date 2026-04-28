# Native Fuzzy TM Backend Family

This directory contains exact native C implementations of Fuzzy TM inference.
The public Python API is:

```python
from fuzzy_tm_infer.algorithms.c import FuzzyTMModel

model = FuzzyTMModel.from_fbz("model.fbz", version="v17")
```

## Current Versions

| Version | Role | Main Idea | Status |
|---|---|---|---|
| `v17` | Primary fast backend | Byte/delta-table scorer with AVX2/NEON kernels, lane-aligned/AOT-shaped paths, calibration for scoring/binarization choices. | Promoted as fastest current native backend. |
| `v19` | Reference/alternative backend | Feature-state/metadata-oriented formulation. Exact and simpler mathematically, but slower on bundled datasets. | Kept as reference backend. |

## Versioning Advice

Keep the simple `vNN` directory scheme for experiments. It is boring in the
best way: easy to sort, easy to reference in result files, and stable for the
Python wrapper. Put descriptive names inside each version README.

Recommended path for a new backend:

```bash
make -C src/fuzzy_tm_infer scaffold-native VERSION=v20 BASE=v17
```

Then edit:

```text
algorithms/c/fuzzy_tm/v20/README.md
algorithms/c/fuzzy_tm/v20/tm_algorithm.c
algorithms/c/fuzzy_tm/v20/tm_adapter.c
```

Build and verify:

```bash
make -C src/fuzzy_tm_infer native-v20
make -C src/fuzzy_tm_infer verify-native NATIVE_VERSIONS="v17 v19 v20"
```

## Backend Contract

Every version must expose the same native ABI through `common/tm_c_api.c`:

```text
fuzzy_tm_model_load
fuzzy_tm_model_free
fuzzy_tm_n_literals
fuzzy_tm_n_classes
fuzzy_tm_h_words
fuzzy_tm_model_calibrate
fuzzy_tm_predict_row
fuzzy_tm_predict_batch
```

Every version must keep behavior exact against the Python reference:

```bash
make -C src/fuzzy_tm_infer verify-native
```

## Files Per Version

Expected version folder:

```text
vNN/
  README.md          version hypothesis, status, and notes
  Makefile           build flags and native library target
  tm_algorithm.h     version-specific layout/API
  tm_algorithm.c     prediction kernel/API glue
  tm_adapter.c       FBZ-to-layout conversion, calibration, benchmark hooks
  tm_infer_c.c       includes the shared CLI runner
  tm_kernel_*.c/h    optional architecture-specific kernels
```

Do not keep unused experiment files in a version folder. If a path is rejected,
remove it or archive it.

## Performance Tracking

Use:

```text
PERFORMANCE.md
```

for promoted/reference version summaries, and use `results/` for detailed run
logs and one-off experiments.

