# Native C Algorithms

Native code is optional acceleration for the Python-first package. Keep it
isolated by algorithm family.

## Layout

```text
algorithms/c/
  common/              shared FBZ loading, CLI runner, Python ABI
  booleanizers/        native booleanizer kernels
  fuzzy_native.py      ctypes wrapper for Fuzzy TM native libraries
  fast_glade.py        ctypes wrapper for fast GLADE v2 transform
  fuzzy_tm/            native Fuzzy TM backend family
    v17/               fastest current backend
    v19/               feature-state reference backend
```

Backend versions are deliberately nested under `fuzzy_tm/` so future native
families can sit beside it without mixing code or build contracts.

## Fuzzy TM

The active native Fuzzy TM family is documented in:

```text
algorithms/c/fuzzy_tm/README.md
algorithms/c/fuzzy_tm/PERFORMANCE.md
algorithms/c/fuzzy_tm/VERSION_TEMPLATE.md
```

Build from the package root:

```bash
make native-v17
make native-v19
make verify-native
```

Python usage:

```python
from fuzzy_tm_infer.algorithms.c import FuzzyTMModel

model = FuzzyTMModel.from_fbz("model.fbz", version="v17")
```

Available native versions are discovered from `algorithms/c/fuzzy_tm/v*/`.

## Algorithm Families

| Family | Path | Public Wrapper | Current Versions | Purpose |
|---|---|---|---|---|
| Fuzzy TM | `fuzzy_tm/` | `fuzzy_native.py` | `v17`, `v19` | Exact native acceleration for bundled Fuzzy TM models. |
| GLADE v2 transform | `booleanizers/glade_v2/` | `fast_glade.py` | unversioned kernel | Fast transform for fitted GLADE thresholds. |

New native algorithm families should get their own sibling directory beside
`fuzzy_tm/`. Shared code goes in `common/` only when it is genuinely reusable
across families.

## Profiling

Native profiling guidance lives in:

```text
PROFILE_TEMPLATE.md
```

Use macro-guarded profiling for hot kernels and expose counters through the
Python wrapper instead of printing from C hot paths.

## Native Contract

Every Fuzzy TM native version must:

- build a CLI binary named `tm_infer_c`;
- build a shared library named `libfuzzy_tm_infer.so`;
- expose the ABI implemented by `common/tm_c_api.c`;
- match the Python Fuzzy TM reference exactly on all bundled datasets;
- record speed claims in `fuzzy_tm/PERFORMANCE.md` or `results/`;
- keep rejected or obsolete experiment code out of the active version folder.

Create a new iteration with:

```bash
make -C src/fuzzy_tm_infer scaffold-native VERSION=v20 BASE=v17
make -C src/fuzzy_tm_infer native-v20
make -C src/fuzzy_tm_infer verify-native NATIVE_VERSIONS="v17 v19 v20"
```
