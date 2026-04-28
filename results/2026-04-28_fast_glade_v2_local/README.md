# Fast GLADE v2 Local Transform Check

## Summary

Fast GLADE v2 transform was added for fitted Python GLADE thresholds. The
local AVX2 benchmark shows exact output and lower transform latency than the
Python/NumPy transform path across the bundled datasets.

## Hypothesis

A native row-wise compare/pack kernel can reduce transform overhead after GLADE
thresholds have been fitted in Python.

## Command

Command ID:

```text
local.validate.fast_glade_profile
```

Validation command:

```bash
make -C src/fuzzy_tm_infer clean-fast-glade verify-fast-glade
```

Profile build and timing command:

```bash
make -C src/fuzzy_tm_infer profile-fast-glade
uv run --with numpy --with numba --with zstandard python <profile-all-datasets-snippet>
```

## Platform

```text
local x86_64 AVX2
```

## Correctness

- Exact: true
- Validation command: `make -C src/fuzzy_tm_infer clean-fast-glade verify-fast-glade`

The verifier compares unpacked and packed native outputs against Python GLADE
on random data and the bundled datasets.

## Optimization Notes

- Float32 matrices now use a direct float32 ABI, avoiding Python-side float64
  conversion.
- Packed output no longer clears bytes that the SIMD loop overwrites.
- The wrapper precomputes 4-lane and 8-lane chunk metadata. SIMD chunks whose
  literals share a feature use a same-feature broadcast path; mixed-feature
  chunks keep the generic indexed path.
- AVX2 unpacked output expands each 8-lane comparison mask into eight `uint8`
  Boolean values with one 64-bit store.
- Packed mask conversion now uses a 256-entry table. Unpacked AVX2 expansion
  uses BMI2 `pdep` on this machine. Both preserve Python bit order while
  removing shift/or trees from the hot path.
- A feature-block prefix formulation was tested and rejected: it was exact, but
  slower than the chunked SIMD literal path on this local AVX2 benchmark.
- A packed-output 64-bit unroll was tested and rejected: it was exact, but did
  not improve the profile build and regressed small/medium packed transforms.

The table in `tables/profile_all_datasets.csv` is the current measured local
AVX2 record for these changes.
The table in `tables/streaming_api.csv` records the new preallocated-output and
bounded-streaming API timings against normal `transform()`.

## Artifacts

```text
tables/wustl_transform.csv
tables/profile_all_datasets.csv
tables/streaming_api.csv
latex/wustl_transform.tex
latex/profile_all_datasets.tex
latex/streaming_api.tex
raw/
```

## Decision

- accepted

Why: the native transform is exact and provides a useful AVX2 speedup for the
hot transform step. RPI/NEON timing should be recorded in a separate RPI result.
