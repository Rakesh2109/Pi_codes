# Native Fuzzy TM Backends

The package root is Python-first. Native code is intentionally isolated:

- `algorithms/c/fuzzy_tm/v17/`: fastest current native backend for the bundled datasets.
- `algorithms/c/fuzzy_tm/v19/`: feature-state reference backend.
- `algorithms/c/common/`: shared native loading, CLI, and Python-wrapper ABI.

Old numbered C attempts were removed from the active tree to keep the package
small and readable. Detailed benchmark history lives in the legacy result
archive, with the compact active native ledger in
`algorithms/c/fuzzy_tm/PERFORMANCE.md`.

Build and verify:

```bash
make -C src/fuzzy_tm_infer native-v17
make -C src/fuzzy_tm_infer native-v19
make -C src/fuzzy_tm_infer verify-native
```

Python usage:

```python
from fuzzy_tm_infer.algorithms.c import FuzzyTMModel

model = FuzzyTMModel.from_fbz("model.fbz", version="v17")
```
