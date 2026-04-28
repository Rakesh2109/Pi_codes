# Booleanizer Speed Benchmark

Command:

```bash
uv run --with numpy --with numba --with zstandard \
  fuzzy-tm-infer --compare-booleanizers --repeats 2
```

The benchmark measures fit time, unpacked transform time, and packed transform
time on the bundled test matrices.

Important note: `thermo8` is a fixed evenly spaced threshold encoder. It is a
useful speed baseline, but it is not data-adaptive like GLADE, Standard, or
KBins.

| Dataset | Booleanizer | Rows | Features | Bits | Fit ms | Transform us/sample | Packed us/sample |
|---|---:|---:|---:|---:|---:|---:|---:|
| WUSTL | glade15 | 3,264 | 32 | 236 | 7.29 | 0.330 | 0.315 |
| WUSTL | standard25 | 3,264 | 32 | 391 | 0.36 | 0.260 | 0.379 |
| WUSTL | kbins8_quantile | 3,264 | 32 | 107 | 2.08 | 0.240 | 0.262 |
| WUSTL | kbins8_uniform | 3,264 | 32 | 203 | 0.34 | 0.442 | 0.352 |
| WUSTL | thermo8 | 3,264 | 32 | 224 | 0.00 | 0.149 | 0.173 |
| NSLKDD | glade15 | 29,704 | 118 | 247 | 71.30 | 0.839 | 0.798 |
| NSLKDD | standard25 | 29,704 | 118 | 694 | 9.55 | 0.811 | 0.901 |
| NSLKDD | kbins8_quantile | 29,704 | 118 | 175 | 48.29 | 0.795 | 0.808 |
| NSLKDD | kbins8_uniform | 29,704 | 118 | 791 | 19.85 | 1.503 | 1.696 |
| NSLKDD | thermo8 | 29,704 | 118 | 826 | 0.00 | 0.715 | 0.969 |
| TonIoT | glade15 | 42,209 | 38 | 116 | 33.66 | 0.314 | 0.322 |
| TonIoT | standard25 | 42,209 | 38 | 268 | 4.07 | 0.317 | 0.384 |
| TonIoT | kbins8_quantile | 42,209 | 38 | 72 | 21.91 | 0.244 | 0.258 |
| TonIoT | kbins8_uniform | 42,209 | 38 | 266 | 7.17 | 0.468 | 0.531 |
| TonIoT | thermo8 | 42,209 | 38 | 266 | 0.00 | 0.198 | 0.242 |
| MedSec | glade15 | 110,907 | 54 | 353 | 128.69 | 1.385 | 1.403 |
| MedSec | standard25 | 110,907 | 54 | 1,147 | 20.71 | 1.048 | 1.322 |
| MedSec | kbins8_quantile | 110,907 | 54 | 128 | 92.09 | 0.556 | 0.568 |
| MedSec | kbins8_uniform | 110,907 | 54 | 378 | 35.22 | 0.870 | 0.991 |
| MedSec | thermo8 | 110,907 | 54 | 378 | 0.00 | 0.347 | 0.451 |

Fastest unpacked transform per dataset:

| Dataset | Fastest | us/sample | Bits |
|---|---:|---:|---:|
| WUSTL | thermo8 | 0.149 | 224 |
| NSLKDD | thermo8 | 0.715 | 826 |
| TonIoT | thermo8 | 0.198 | 266 |
| MedSec | thermo8 | 0.347 | 378 |

