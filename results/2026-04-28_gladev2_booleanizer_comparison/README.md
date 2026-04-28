# GLADE v2 Booleanizer Comparison

## Summary

This result compares Fast GLADE v2 against the available Python booleanizers on
local AVX2 and Raspberry Pi 5 NEON.

Fast GLADE uses Python fitting plus the native C transform kernel. Outputs are
verified exact against the Python GLADE v2 transform; the benchmark compares
transform latency after fitting.

## Commands

Local AVX2:

```bash
make -C src/fuzzy_tm_infer clean-fast-glade native-glade-v2
uv run --with numpy --with numba --with zstandard \
  fuzzy-tm-infer --compare-booleanizers --repeats 3
```

Raspberry Pi 5 NEON:

```bash
PYTHONPATH=src uv run --with ansible python \
  -m fuzzy_tm_infer.scripts.rpi_ansible \
  --booleanizer-benchmark --no-native-benchmark
```

The RPI run used:

```text
Raspberry Pi 5 Model B Rev 1.1
ARCHFLAGS=-mcpu=cortex-a76
backend=neon
```

## Tables

```text
tables/local_booleanizers.csv   full local AVX2 comparison
tables/rpi_booleanizers.csv     full Raspberry Pi 5 NEON comparison
tables/gladev2_summary.csv      Fast GLADE vs Python GLADE and next-best booleanizer
```

LaTeX booktabs renderings are in `latex/`.

## Decision

Accepted. Fast GLADE v2 is the fastest transform path on every measured dataset
on both local AVX2 and Raspberry Pi 5 NEON.
