# Native C Agent Contract

This file applies to `src/fuzzy_tm_infer/algorithms/c/`.

## Scope

Native C work must stay isolated by algorithm family. Current family:

```text
fuzzy_tm/
booleanizers/glade_v2/
```

Shared code belongs only in:

```text
common/
```

The Python wrapper belongs only in:

```text
fuzzy_native.py
fast_glade.py
```

## Version Rules

1. Use `fuzzy_tm/vNN/` for native Fuzzy TM backend iterations.
2. Keep the public version ID short and stable, for example `v17`.
3. Put descriptive names in the version README, not in the directory name.
4. Do not create loose files such as `tm_algorithm_20.c` in the active tree.
5. New versions must build with `make -C src/fuzzy_tm_infer native-vNN`.
6. New versions must be exact against the Python reference before benchmark
   claims are made.

## Iteration Loop

1. Create or select a version.
2. State the hypothesis in that version's README or a results file.
3. Implement the smallest exact change.
4. Run local exactness.
5. Run local benchmark.
6. Run Raspberry Pi benchmark when performance is the claim.
7. Record the result in `fuzzy_tm/PERFORMANCE.md` or `results/`.
8. Promote, keep as reference, or archive.

## Instrumentation Rules

Native optimization without measurement is not accepted.

When adding or changing a hot C kernel:

1. Add macro-guarded profiling if the bottleneck is not already obvious.
2. Keep normal builds zero-cost or effectively zero-cost.
3. Use stable counter names and expose them through the Python wrapper.
4. Measure work units as well as time.
5. Separate setup, validation, kernel, packing, and total time when practical.
6. Keep C hot paths quiet; no debug printing inside per-row/per-clause loops.
7. Validate profiled builds for exactness before using the measurements.
8. Record profile-derived decisions in `results/`.

Recommended field names:

```text
total_calls
rows
bits
clauses
output_bytes
validate_ns
setup_ns
kernel_ns
pack_ns
total_ns
errors
```

Use a component-specific macro, for example:

```text
GLADE_V2_PROFILE
TM_VNN_PROFILE
```

The public wrapper should expose:

```python
profile_enabled
profile_reset()
profile_snapshot()
```

## Required Commands

```bash
make -C src/fuzzy_tm_infer native-vNN
make -C src/fuzzy_tm_infer verify-native
uv run --with '.[dt]' fuzzy-tm-infer --compare-all
make -C src/fuzzy_tm_infer verify-fast-glade
```

Use `EXPERIMENT_COMMANDS.md` for the canonical command IDs.

## Cleanup

Before finishing, remove generated native outputs:

```bash
make -C src/fuzzy_tm_infer clean-native
```
