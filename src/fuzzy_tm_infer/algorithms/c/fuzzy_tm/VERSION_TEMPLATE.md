# Native Fuzzy TM vNN Template

Copy this template into a new version README when starting a backend iteration.

## Version

```text
vNN
```

## Hypothesis

What computation, data layout, or dispatch change should make this faster?

## Expected Impact

| Platform | Expected Effect | Why |
|---|---|---|
| Local AVX2/x86 | | |
| Raspberry Pi NEON | | |

## Behavioral Contract

- Exact predictions against Python reference.
- No public Python API change unless explicitly documented.
- No model-format change unless documented and gated.

## Implementation Scope

Files expected to change:

```text
tm_algorithm.c
tm_algorithm.h
tm_adapter.c
tm_kernel_*.c/h
Makefile
```

## Validation Plan

```bash
make -C src/fuzzy_tm_infer native-vNN
make -C src/fuzzy_tm_infer verify-native NATIVE_VERSIONS="v17 v19 vNN"
uv run --with '.[dt]' fuzzy-tm-infer --compare-all
tm-rpi-deploy --compare-all
```

## Results

| Dataset | Baseline us | vNN us | Speedup | Exact |
|---|---:|---:|---:|---|
| WUSTL | | | | |
| NSLKDD | | | | |
| TonIoT | | | | |
| MedSec | | | | |

## Decision

- promote / keep as reference / archive / revert

