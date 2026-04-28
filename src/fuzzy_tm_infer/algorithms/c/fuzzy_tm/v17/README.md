# Fuzzy TM v17 Native Backend

Fast byte/delta-table native backend with AVX2 and NEON kernels. This folder is
self-contained for code and build rules. Dataset/model assets are discovered by
the shared CLI runner or passed explicitly as the first CLI argument.

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

model = FuzzyTMModel.from_fbz("path/to/model.fbz", version="v17")
```

Default build flags:

```text
TM_ENABLE_LANE_ALIGNED_RANGES
TM_ENABLE_AOT_SHAPES
```
