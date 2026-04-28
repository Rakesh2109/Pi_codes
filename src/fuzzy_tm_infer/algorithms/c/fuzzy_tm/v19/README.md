# Fuzzy TM v19 Native Backend

Feature-state native backend. It is exact and mathematically clean, but slower
than v17 on the bundled datasets. This folder is self-contained for code and
build rules. Dataset/model assets are discovered by the shared CLI runner or
passed explicitly as the first CLI argument.

Build:

```bash
make clean all lib
```

Run the native benchmark:

```bash
./tm_infer_c --profile --stats
```

Use from Python:

```python
from fuzzy_tm_infer.algorithms.c import FuzzyTMModel

model = FuzzyTMModel.from_fbz("path/to/model.fbz", version="v19")
```

Required build flags are enabled by default:

```text
TM_ENABLE_SCORE_BLOCKS
TM_ENABLE_FEATURE_STATE_TABLES
```
