# tm_local_infer Python vs C Benchmark

Date: 2026-04-27

## Setup

- Local platform: WSL2 on Intel Core i9-14900K, x86_64
- Raspberry Pi platform: Raspberry Pi 5, Cortex-A76, aarch64
- Python runtime: `tm_local_infer.benchmark` with Numba JIT
- C runtime: `tm_infer_c --profile`
- C local build: `make -C src/tm_local_infer clean all`
- C Raspberry Pi build: `make clean all ARCHFLAGS='-mcpu=cortex-a76'`

## Results

| Platform | Dataset | Python us/sample | C us/sample | C speedup | Accuracy | Macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| Local i9-14900K | WUSTL | 1.622 | 0.108 | 15.0x | 0.9470 | 0.8157 |
| Local i9-14900K | NSLKDD | 2.072 | 0.314 | 6.6x | 0.9946 | 0.9477 |
| Local i9-14900K | TonIoT | 2.641 | 0.284 | 9.3x | 0.9703 | 0.9427 |
| Local i9-14900K | MedSec | 2.286 | 0.359 | 6.4x | 0.9700 | 0.8692 |
| Raspberry Pi 5 | WUSTL | 3.785 | 1.550 | 2.4x | 0.9470 | 0.8157 |
| Raspberry Pi 5 | NSLKDD | 5.551 | 2.728 | 2.0x | 0.9946 | 0.9477 |
| Raspberry Pi 5 | TonIoT | 8.926 | 1.973 | 4.5x | 0.9703 | 0.9427 |
| Raspberry Pi 5 | MedSec | 7.095 | 2.264 | 3.1x | 0.9700 | 0.8692 |

## C Profile Detail

| Platform | Dataset | Binarize us | Score us | Profile total us |
|---|---:|---:|---:|---:|
| Local i9-14900K | WUSTL | 0.049 | 0.068 | 0.107 |
| Local i9-14900K | NSLKDD | 0.060 | 0.273 | 0.301 |
| Local i9-14900K | TonIoT | 0.018 | 0.253 | 0.278 |
| Local i9-14900K | MedSec | 0.053 | 0.275 | 0.346 |
| Raspberry Pi 5 | WUSTL | 0.735 | 0.855 | 1.603 |
| Raspberry Pi 5 | NSLKDD | 0.467 | 2.298 | 2.690 |
| Raspberry Pi 5 | TonIoT | 0.194 | 1.770 | 1.966 |
| Raspberry Pi 5 | MedSec | 0.536 | 1.695 | 2.254 |

## Commands Used

```bash
PYTHONPATH=src uv run --with numpy --with numba --with zstandard \
  python -m tm_local_infer.benchmark

make -C src/tm_local_infer clean all
./src/tm_local_infer/tm_infer_c src/tm_local_infer/assets --profile

SSHPASS='rpi' src/tm_local_infer/rsync_rpi5.sh --build

SSHPASS='rpi' sshpass -e ssh -o StrictHostKeyChecking=accept-new \
  rpi@100.98.236.19 \
  'cd /home/rpi/Pi_codes && . .venv_tm_local/bin/activate && \
   PYTHONPATH=/home/rpi/Pi_codes/src python -m tm_local_infer.benchmark'

SSHPASS='rpi' sshpass -e ssh -o StrictHostKeyChecking=accept-new \
  rpi@100.98.236.19 \
  'cd /home/rpi/Pi_codes/src/tm_local_infer && ./tm_infer_c assets --profile'
```

## Iteration 1: Remaining-Score Early Exit + Stats

Changes:

- Fixed scalar dense scoring to subtract from remaining score and break the real word loop.
- Changed AVX2/NEON byte-table scoring to saturating remaining-score subtract with all-zero early exit.
- Changed AVX2 plane scoring to remaining-score subtract with all-dead early exit.
- Added `tm_infer_c assets --profile --stats`.

Correctness:

- Default C equivalence: all datasets `same=True`, `diffs=0`.
- Scalar fallback equivalence with `-DTM_NO_AVX2_SCORE`: all datasets `same=True`, `diffs=0`.

### Iteration 1 Results

| Platform | Dataset | Python us/sample | C us/sample | C speedup | Accuracy | Macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| Local i9-14900K | WUSTL | 1.509 | 0.148 | 10.2x | 0.9470 | 0.8157 |
| Local i9-14900K | NSLKDD | 2.152 | 0.413 | 5.2x | 0.9946 | 0.9477 |
| Local i9-14900K | TonIoT | 2.506 | 0.394 | 6.4x | 0.9703 | 0.9427 |
| Local i9-14900K | MedSec | 2.276 | 0.437 | 5.2x | 0.9700 | 0.8692 |
| Raspberry Pi 5 | WUSTL | 3.929 | 1.476 | 2.7x | 0.9470 | 0.8157 |
| Raspberry Pi 5 | NSLKDD | 5.610 | 3.574 | 1.6x | 0.9946 | 0.9477 |
| Raspberry Pi 5 | TonIoT | 9.001 | 3.461 | 2.6x | 0.9703 | 0.9427 |
| Raspberry Pi 5 | MedSec | 7.305 | 3.617 | 2.0x | 0.9700 | 0.8692 |

### Iteration 1 C Profile Detail

| Platform | Dataset | Binarize us | Score us | Profile total us |
|---|---:|---:|---:|---:|
| Local i9-14900K | WUSTL | 0.046 | 0.117 | 0.147 |
| Local i9-14900K | NSLKDD | 0.051 | 0.400 | 0.408 |
| Local i9-14900K | TonIoT | 0.019 | 0.358 | 0.378 |
| Local i9-14900K | MedSec | 0.049 | 0.377 | 0.422 |
| Raspberry Pi 5 | WUSTL | 0.468 | 1.013 | 1.481 |
| Raspberry Pi 5 | NSLKDD | 0.460 | 3.132 | 3.582 |
| Raspberry Pi 5 | TonIoT | 0.201 | 3.249 | 3.458 |
| Raspberry Pi 5 | MedSec | 0.526 | 3.076 | 3.597 |

### Iteration 1 C Delta vs Baseline

Positive means slower than the baseline C result above.

| Platform | Dataset | Baseline C us/sample | Iteration 1 C us/sample | Change |
|---|---:|---:|---:|---:|
| Local i9-14900K | WUSTL | 0.108 | 0.148 | +37.0% |
| Local i9-14900K | NSLKDD | 0.314 | 0.413 | +31.5% |
| Local i9-14900K | TonIoT | 0.284 | 0.394 | +38.7% |
| Local i9-14900K | MedSec | 0.359 | 0.437 | +21.7% |
| Raspberry Pi 5 | WUSTL | 1.550 | 1.476 | -4.8% |
| Raspberry Pi 5 | NSLKDD | 2.728 | 3.574 | +31.0% |
| Raspberry Pi 5 | TonIoT | 1.973 | 3.461 | +75.4% |
| Raspberry Pi 5 | MedSec | 2.264 | 3.617 | +59.8% |

### Iteration 1 Scalar Fallback

Built locally with:

```bash
CFLAGS='-O3 -std=c11 -Wall -Wextra -Wpedantic -funroll-loops -march=native -DTM_NO_AVX2_SCORE'
```

| Dataset | Scalar C us/sample | Binarize us | Score us | Profile total us |
|---|---:|---:|---:|---:|
| WUSTL | 0.396 | 0.049 | 0.322 | 0.362 |
| NSLKDD | 1.263 | 0.054 | 1.180 | 1.252 |
| TonIoT | 0.802 | 0.021 | 0.768 | 0.823 |
| MedSec | 1.037 | 0.050 | 0.948 | 1.010 |

### Iteration 1 Model Stats

| Dataset | Active lits avg | Active words avg | Empty | Linearizable | Clamp1 | Unused literals | Total postings | Dense bytes | Byte-table bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| WUSTL | 50.05 | 4.00 | 0 | 0 | 0 | 0 | 9009 | 11520 | 1777664 |
| NSLKDD | 41.77 | 3.98 | 0 | 0 | 0 | 0 | 18795 | 28800 | 4194304 |
| TonIoT | 25.92 | 2.00 | 0 | 0 | 0 | 0 | 25923 | 32000 | 4325376 |
| MedSec | 30.72 | 5.74 | 0 | 0 | 0 | 0 | 12287 | 38400 | 5046272 |

Observation: the remaining-score table path is exact but slower on most current benchmark cases. The next iteration should add backend selection or a guarded table mode before pursuing larger layout work.

## Rollback After Iteration 1 Regression

Action:

- Reverted the AVX2/NEON byte-table scorer to mismatch accumulation plus final saturating subtract.
- Reverted the AVX2 plane scorer to mismatch accumulation plus final clamp.
- Kept `--stats` because it does not change prediction behavior.
- Kept the scalar fallback remaining-score fix; the default AVX2/NEON table backend does not use it.

Correctness:

- Default C equivalence after rollback: all datasets `same=True`, `diffs=0`.

### Rollback C Results

| Platform | Dataset | Python us/sample | C us/sample | C speedup | Accuracy | Macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| Local i9-14900K | WUSTL | 1.464 | 0.111 | 13.2x | 0.9470 | 0.8157 |
| Local i9-14900K | NSLKDD | 2.130 | 0.293 | 7.3x | 0.9946 | 0.9477 |
| Local i9-14900K | TonIoT | 2.425 | 0.293 | 8.3x | 0.9703 | 0.9427 |
| Local i9-14900K | MedSec | 2.337 | 0.334 | 7.0x | 0.9700 | 0.8692 |
| Raspberry Pi 5 | WUSTL | 3.929 | 0.958 | 4.1x | 0.9470 | 0.8157 |
| Raspberry Pi 5 | NSLKDD | 5.610 | 2.323 | 2.4x | 0.9946 | 0.9477 |
| Raspberry Pi 5 | TonIoT | 9.001 | 1.976 | 4.6x | 0.9703 | 0.9427 |
| Raspberry Pi 5 | MedSec | 7.305 | 2.612 | 2.8x | 0.9700 | 0.8692 |

### Rollback C Profile Detail

| Platform | Dataset | Binarize us | Score us | Profile total us |
|---|---:|---:|---:|---:|
| Local i9-14900K | WUSTL | 0.046 | 0.068 | 0.108 |
| Local i9-14900K | NSLKDD | 0.056 | 0.244 | 0.327 |
| Local i9-14900K | TonIoT | 0.018 | 0.268 | 0.286 |
| Local i9-14900K | MedSec | 0.049 | 0.266 | 0.324 |
| Raspberry Pi 5 | WUSTL | 0.476 | 0.527 | 0.965 |
| Raspberry Pi 5 | NSLKDD | 0.453 | 1.889 | 2.306 |
| Raspberry Pi 5 | TonIoT | 0.194 | 1.754 | 1.937 |
| Raspberry Pi 5 | MedSec | 0.528 | 2.100 | 2.609 |

## Iteration 2: Current-Byte Copy and Backend Sweep

Goal: find behavior-preserving speedups after rolling back the regressed remaining-score table path.

Kept:

- Replaced the table backend's per-byte extraction loop:
  `current_bytes[b] = current[b >> 3] >> (8 * (b & 7))`
  with a little-endian `memcpy` helper and retained the old loop for non-little-endian builds.

Rejected after profiling:

- `TM_GROUPED_AVX2_BINARIZE`: slower locally.
- `TM_GROUPED_BINARIZE`: slower locally and mostly slower on Raspberry Pi 5.
- `TM_NO_AVX2_TABLE_SCORE` / plane path: slower locally.
- `TM_NO_NEON_TABLE_SCORE`: slower on Raspberry Pi 5.
- Feature-run/range binarization: exact on these sorted feature runs, but slower on Raspberry Pi 5 because run discovery and range setting outweighed fewer threshold comparisons.
- Separate positive/negative table range scoring: mixed locally and slower on Raspberry Pi 5 NSLKDD; reverted to the original combined class traversal.

Correctness:

- Local default C equivalence: all datasets `same=True`, `diffs=0`.
- Raspberry Pi default C equivalence: all datasets `same=True`, `diffs=0`.

### Iteration 2 Results

| Platform | Dataset | Python us/sample | C us/sample | C speedup | Accuracy | Macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| Local i9-14900K | WUSTL | 1.467 | 0.106 | 13.8x | 0.9470 | 0.8157 |
| Local i9-14900K | NSLKDD | 2.069 | 0.299 | 6.9x | 0.9946 | 0.9477 |
| Local i9-14900K | TonIoT | 2.370 | 0.276 | 8.6x | 0.9703 | 0.9427 |
| Local i9-14900K | MedSec | 2.306 | 0.362 | 6.4x | 0.9700 | 0.8692 |
| Raspberry Pi 5 | WUSTL | 3.923 | 0.932 | 4.2x | 0.9470 | 0.8157 |
| Raspberry Pi 5 | NSLKDD | 5.622 | 2.265 | 2.5x | 0.9946 | 0.9477 |
| Raspberry Pi 5 | TonIoT | 8.960 | 1.990 | 4.5x | 0.9703 | 0.9427 |
| Raspberry Pi 5 | MedSec | 7.246 | 2.273 | 3.2x | 0.9700 | 0.8692 |

### Iteration 2 C Profile Detail

| Platform | Dataset | Binarize us | Score us | Profile total us |
|---|---:|---:|---:|---:|
| Local i9-14900K | WUSTL | 0.046 | 0.062 | 0.103 |
| Local i9-14900K | NSLKDD | 0.055 | 0.233 | 0.287 |
| Local i9-14900K | TonIoT | 0.018 | 0.239 | 0.260 |
| Local i9-14900K | MedSec | 0.050 | 0.276 | 0.318 |
| Raspberry Pi 5 | WUSTL | 0.455 | 0.500 | 0.940 |
| Raspberry Pi 5 | NSLKDD | 0.460 | 1.846 | 2.276 |
| Raspberry Pi 5 | TonIoT | 0.196 | 1.784 | 1.966 |
| Raspberry Pi 5 | MedSec | 0.529 | 1.728 | 2.226 |

Observation: the exact byte-copy optimization is small but stable. The larger algorithmic wins are unlikely to come from grouped binarization or dense/plane toggles for these assets; next promising work remains a new backend/layout, but it should be benchmark-gated immediately.

### Iteration 2 Follow-up: Blocked Table Layout Rejected

Tested a blocked byte-table layout intended to improve clause-block locality:

```text
tables_blocked[block][byte_index][value][lane]
```

Result: exact but much slower on both AVX2 and NEON for the bundled assets, so the code was removed and the kept implementation remains the simple little-endian current-byte copy plus the original table layout.

Final cleaned-build correctness:

- Local default C equivalence: all datasets `same=True`, `diffs=0`.
- Raspberry Pi default C equivalence: all datasets `same=True`, `diffs=0`.

Final cleaned-build C profile after removing the blocked-table experiment:

| Platform | Dataset | C us/sample | Binarize us | Score us | Profile total us |
|---|---:|---:|---:|---:|---:|
| Local i9-14900K | WUSTL | 0.105 | 0.048 | 0.061 | 0.101 |
| Local i9-14900K | NSLKDD | 0.280 | 0.051 | 0.229 | 0.272 |
| Local i9-14900K | TonIoT | 0.299 | 0.018 | 0.244 | 0.270 |
| Local i9-14900K | MedSec | 0.317 | 0.049 | 0.265 | 0.307 |
| Raspberry Pi 5 | WUSTL | 0.933 | 0.476 | 0.499 | 0.940 |
| Raspberry Pi 5 | NSLKDD | 2.256 | 0.459 | 1.834 | 2.265 |
| Raspberry Pi 5 | TonIoT | 1.970 | 0.197 | 1.748 | 1.928 |
| Raspberry Pi 5 | MedSec | 2.414 | 0.532 | 1.915 | 2.416 |

Next optimization direction: current data says runtime is dominated by byte-table scoring, while no bundled model has empty or linearizable clauses. More SIMD reshuffling has been fragile; the next worthwhile iteration should be a benchmark-gated alternate nonlinear backend, probably a compact posting-list prototype or a reduced/table-compressed representation that can be enabled per model only when its measured build-time stats predict a win.

## Iteration 3: Additional Optimization Triage

Rejected:

- Direct little-endian byte view of `current` instead of copying to `current_bytes`: exact, but slower locally. Reverted to the `memcpy` helper.

Posting-list feasibility estimate from the actual test rows:

| Dataset | Current density | Best posting touches avg | Dirty clauses avg | Table vector row loads | Dense word checks |
|---|---:|---:|---:|---:|---:|
| WUSTL | 0.432 | 3970.8 | 180.0 / 180 | 186 | 720 |
| NSLKDD | 0.222 | 4962.3 | 445.5 / 450 | 480 | 1800 |
| TonIoT | 0.201 | 5839.3 | 952.6 / 1000 | 512 | 2000 |
| MedSec | 0.256 | 3139.7 | 377.5 / 400 | 572 | 2400 |

Interpretation: despite sparse-ish active literals, the current models dirty almost every clause under a literal-posting scorer. A posting backend may still win if implemented very compactly, but this estimate explains why it should be a guarded prototype rather than the next default path.

## Numbered Algorithm Attempts

Workflow from here:

- `src/tm_local_infer/tm_algorithm.c` remains the active best implementation.
- Each experiment gets a numbered file: `tm_algorithm_1.c`, `tm_algorithm_2.c`, `tm_algorithm_3.c`, ...
- Build a specific attempt with:

```bash
make -C src/tm_local_infer clean all TM_ALGORITHM_SRC=tm_algorithm_2.c
```

- Only promote a numbered attempt back into `tm_algorithm.c` after exact equivalence and local/Pi benchmark results.

| Attempt file | Status | Summary | Promotion decision |
|---|---|---|---|
| `tm_algorithm_1.c` | Kept checkpoint | Current best cleaned implementation: scalar early-stop fix, stats support in harness, original table scorer, little-endian `memcpy` current-byte preparation. | Active baseline for new numbered attempts. |
| `tm_algorithm_2.c` | Rejected / mixed | Avoids the second masked horizontal sum for table-score blocks that are fully positive or fully negative; only mixed boundary blocks use the masked sum. Exact locally. | Not promoted: local results were noise-to-small win, but Raspberry Pi 5 regressed NSLKDD. |
| `tm_algorithm_3.c` | Useful candidate | Zero-current baseline plus signed byte-delta table scorer; skips table loads for zero current bytes. | Not promoted directly: strong Pi win for 3/4 datasets, but WUSTL regressed and local AVX2 was mixed. |
| `tm_algorithm_4.c` | Rejected | Per-row hybrid between original table and delta table based on nonzero current-byte count. | Not promoted: selection/count overhead made it slower than pure delta on Pi and did not recover WUSTL. |
| `tm_algorithm_5.c` | Superseded | Platform/model heuristic: AVX2 uses delta table for 5-class models; NEON uses delta table for non-WUSTL models; WUSTL remains original table path. | Promoted briefly, then replaced by generic calibrated attempt 6. |
| `tm_algorithm_6.c` | Promoted | Generic calibrated selector: dense-current gate plus measured table-vs-delta score timing on sampled rows. | Promoted into active `tm_algorithm.c`. |

### Attempt 2 A/B Results

Local median of three runs, `TM_ALGORITHM_SRC=...`:

| Dataset | Attempt 1 C us | Attempt 2 C us | Change |
|---|---:|---:|---:|
| WUSTL | 0.104 | 0.107 | +2.9% |
| NSLKDD | 0.288 | 0.290 | +0.7% |
| TonIoT | 0.279 | 0.275 | -1.4% |
| MedSec | 0.329 | 0.314 | -4.6% |

Raspberry Pi 5 single A/B run, `ARCHFLAGS=-mcpu=cortex-a76`:

| Dataset | Attempt 1 C us | Attempt 2 C us | Change |
|---|---:|---:|---:|
| WUSTL | 0.927 | 0.936 | +1.0% |
| NSLKDD | 2.194 | 2.441 | +11.3% |
| TonIoT | 1.924 | 1.953 | +1.5% |
| MedSec | 2.291 | 2.267 | -1.0% |

## C TM Versus DecisionTree

The DecisionTree artifacts (`ml_models/*/DecisionTree.pkl`) are not present in this checkout or on the Raspberry Pi, so the DT columns below use the existing repository results rather than a fresh rerun:

- DT Numba latency: `README.md`, Raspberry Pi 5 table.
- DT full-pipeline / numpy-style latency and F1: `results/predict_time_table.txt`.

| Dataset | C TM Pi us | DT Numba Pi us | DT Numba / C TM | DT full us | DT full / C TM | C TM F1 | DT F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| WUSTL | 0.933 | 1.37 | 1.47x | 62.77 | 67.3x | 0.8157 | 0.7608 |
| NSLKDD | 2.256 | 1.39 | 0.62x | 62.98 | 27.9x | 0.9477 | 0.7965 |
| TonIoT | 1.970 | 1.38 | 0.70x | 63.30 | 32.1x | 0.9427 | 0.9210 |
| MedSec | 2.414 | 1.37 | 0.57x | 69.45 | 28.8x | 0.8692 | 0.8861 |

Interpretation: current C TM beats the deploy-style DT full pipeline by roughly 28x-67x on Raspberry Pi 5. Against the tighter Numba DT kernel, C TM is faster on WUSTL and slower on the larger-class/larger-clause datasets, while keeping stronger macro-F1 on WUSTL, NSLKDD, and TonIoT.

## Iteration 4: Zero-Baseline Delta Table Backend

Goal: reduce table-scoring traffic by avoiding selected byte-table rows for current bytes that are zero.

Added layout data:

- `base_mismatch_zero[c]`: mismatch count for an all-zero current vector.
- `byte_delta_tables[b][value][clause]`: signed int8 delta from the zero-current mismatch contribution.

Exact scoring idea:

```text
mismatches = base_mismatch_zero
for each current byte b where value != 0:
    mismatches += byte_delta_tables[b][value]
score = max(clamp - mismatches, 0)
```

This is exact because the original per-byte contribution is decomposed into:

```text
contrib(value) = contrib(0) + delta(value)
```

### Attempt 3 Results: Pure Delta Table

Local i9-14900K, representative A/B:

| Dataset | Attempt 1 C us | Attempt 3 C us | Change |
|---|---:|---:|---:|
| WUSTL | 0.102 | 0.128 | +25.5% |
| NSLKDD | 0.289 | 0.253 | -12.5% |
| TonIoT | 0.261 | 0.280 | +7.3% |
| MedSec | 0.319 | 0.278 | -12.9% |

Raspberry Pi 5, A/B:

| Dataset | Attempt 1 C us | Attempt 3 C us | Change |
|---|---:|---:|---:|
| WUSTL | 0.931 | 0.960 | +3.1% |
| NSLKDD | 2.346 | 1.670 | -28.8% |
| TonIoT | 1.985 | 1.580 | -20.4% |
| MedSec | 2.254 | 1.687 | -25.2% |

Decision: useful but not promoted directly because WUSTL regressed and local AVX2 was mixed.

### Attempt 4 Results: Per-Row Hybrid

Attempt 4 counted nonzero current bytes per row and selected the delta path only when at most 75% of bytes were nonzero.

Raspberry Pi 5:

| Dataset | Attempt 1 C us | Attempt 4 C us | Change |
|---|---:|---:|---:|
| WUSTL | 0.928 | 0.966 | +4.1% |
| NSLKDD | 2.259 | 1.701 | -24.7% |
| TonIoT | 1.912 | 1.631 | -14.7% |
| MedSec | 2.296 | 1.777 | -22.6% |

Decision: rejected. The per-row selection cost did not recover WUSTL and was slower than pure delta on the Pi.

### Attempt 5 Results: Platform/Model Heuristic

Attempt 5 removes per-row selection overhead:

- AVX2: use delta table for 5-class models.
- NEON: use delta table for models with more than 3 classes.
- Otherwise use the original byte-table scorer.

Promoted into active `tm_algorithm.c`.

Correctness:

- Local equivalence: all datasets `same=True`, `diffs=0`.
- Raspberry Pi 5 equivalence: all datasets `same=True`, `diffs=0`.

Local i9-14900K active result after promotion:

| Dataset | Previous active C us | New active C us | Change | Binarize us | Score us |
|---|---:|---:|---:|---:|---:|
| WUSTL | 0.105 | 0.102 | -2.9% | 0.050 | 0.062 |
| NSLKDD | 0.280 | 0.253 | -9.6% | 0.049 | 0.235 |
| TonIoT | 0.299 | 0.273 | -8.7% | 0.020 | 0.245 |
| MedSec | 0.317 | 0.296 | -6.6% | 0.053 | 0.245 |

Raspberry Pi 5 active result after promotion:

| Dataset | Previous active C us | New active C us | Change | Binarize us | Score us |
|---|---:|---:|---:|---:|---:|
| WUSTL | 0.933 | 0.931 | -0.2% | 0.458 | 0.500 |
| NSLKDD | 2.256 | 1.721 | -23.7% | 0.444 | 1.232 |
| TonIoT | 1.970 | 1.643 | -16.6% | 0.198 | 1.414 |
| MedSec | 2.414 | 1.762 | -27.0% | 0.528 | 1.197 |

Observation: the delta scorer is clearly valuable on memory-bandwidth-limited NEON and selectively useful on AVX2. The next likely win is to reduce the extra memory footprint of keeping both full byte tables and delta tables, or to build only the backend selected for a given model/platform.

Memory note: `--stats` now also prints `delta_table_bytes` and `base_mismatch_bytes`. The promoted implementation currently keeps both the original byte table and the delta table, so table memory is approximately doubled for the C harness. This is intentional for benchmark iteration; a later packaging pass should build only the selected backend per model/platform.

## Iteration 5: Generic Delta Backend Selection

Goal: remove the heuristic backend rule from attempt 5.

Attempt 6 replaces the class-count/platform heuristic with a generic calibration pass in `tm_infer_c`:

1. Binarize up to 3000 sample rows.
2. Count nonzero current bytes.
3. If more than 75% of current bytes are nonzero, select the original table scorer immediately.
4. Otherwise time both exact score backends on the same sampled currents.
5. Alternate measurement order across trials and keep the best observed time.
6. Select delta only when it is at least 1.5% faster.

This means selection is data-driven and does not depend on dataset names or class-count special cases.

### Attempt 6 Local Selection

Local i9-14900K, active `tm_algorithm.c` after promotion:

| Dataset | Selected backend | C us/sample | Binarize us | Score us |
|---|---|---:|---:|---:|
| WUSTL | table | 0.103 | 0.041 | 0.063 |
| NSLKDD | delta | 0.276 | 0.052 | 0.223 |
| TonIoT | table | 0.268 | 0.019 | 0.253 |
| MedSec | delta | 0.295 | 0.050 | 0.244 |

### Attempt 6 Raspberry Pi 5 Selection

Raspberry Pi 5, active `tm_algorithm.c` after promotion:

| Dataset | Selected backend | C us/sample | Binarize us | Score us |
|---|---|---:|---:|---:|
| WUSTL | table | 0.927 | 0.451 | 0.499 |
| NSLKDD | delta | 1.697 | 0.449 | 1.208 |
| TonIoT | delta | 1.620 | 0.192 | 1.412 |
| MedSec | delta | 1.755 | 0.525 | 1.194 |

Correctness:

- Local equivalence: all datasets `same=True`, `diffs=0`.
- Raspberry Pi 5 equivalence: all datasets `same=True`, `diffs=0`.

Decision: promoted attempt 6 into active `tm_algorithm.c`. The optimization is now generic: the delta scorer remains exact, and backend selection is based on sampled input density plus measured scorer speed.

## Cleanup Pass

Tidied the iteration workspace after promoting attempt 6:

- Added `src/tm_local_infer/ALGORITHM_ATTEMPTS.md` as a short source-side map of `tm_algorithm_1.c` through `tm_algorithm_6.c`.
- Kept `tm_algorithm.c` identical to promoted `tm_algorithm_6.c`.
- Added Makefile helpers:
  - `make -C src/tm_local_infer profile`
  - `make -C src/tm_local_infer profile-detail`
  - `make -C src/tm_local_infer help`
- Added `tm_infer_c --help`.
- Restored plain `--profile` to compact output; calibration details now appear only under `--profile-detail`.
- Added small comments for the calibrated `prefer_delta_table` flag and optional delta-table fields in `tm_algorithm.h`.

Post-cleanup checks:

- Local equivalence: all datasets `same=True`, `diffs=0`.
- Raspberry Pi 5 equivalence: all datasets `same=True`, `diffs=0`.

Post-cleanup Raspberry Pi 5 compact profile:

| Dataset | C us/sample | Binarize us | Score us |
|---|---:|---:|---:|
| WUSTL | 0.945 | 0.446 | 0.504 |
| NSLKDD | 1.707 | 0.448 | 1.217 |
| TonIoT | 1.635 | 0.194 | 1.408 |
| MedSec | 1.755 | 0.522 | 1.188 |

## Fine-Grained Profiling Harness

Added `--profile-detail` to `tm_infer_c`.

Command:

```bash
./src/tm_local_infer/tm_infer_c src/tm_local_infer/assets --profile-detail --stats
```

What it prints:

- Selected score backend (`table_avx2`, `table_neon`, `planes_avx2`, or `dense_scalar`).
- Current density: active bits, nonzero bytes, full bytes.
- Current-byte preparation time.
- Score work shape: vector blocks, vector lanes, scored clauses, table rows.
- Estimated table/clamp traffic per row.
- Per-class positive/negative clause counts and table block counts.

### First Detail Readout

Local i9-14900K, active `tm_algorithm.c`:

| Dataset | Backend | Active bits avg | Nonzero bytes avg | Byteprep us | Blocks/row | Table bytes/row | Score core us |
|---|---|---:|---:|---:|---:|---:|---:|
| WUSTL | table_avx2 | 106.74 | 25.79 | 0.0012 | 6 | 5,952 | 0.061 |
| NSLKDD | table_avx2 | 56.11 | 15.57 | 0.0023 | 15 | 15,360 | 0.239 |
| TonIoT | table_avx2 | 24.41 | 7.37 | 0.0010 | 40 | 20,480 | 0.246 |
| MedSec | table_avx2 | 91.42 | 21.52 | 0.0030 | 15 | 21,120 | 0.290 |

Raspberry Pi 5, active `tm_algorithm.c`:

| Dataset | Backend | Active bits avg | Nonzero bytes avg | Byteprep us | Blocks/row | Table bytes/row | Score core us |
|---|---|---:|---:|---:|---:|---:|---:|
| WUSTL | table_neon | 106.74 | 25.79 | 0.0068 | 12 | 5,952 | 0.790 |
| NSLKDD | table_neon | 56.11 | 15.57 | 0.0041 | 30 | 15,360 | 1.825 |
| TonIoT | table_neon | 24.41 | 7.37 | 0.0041 | 70 | 17,920 | 1.749 |
| MedSec | table_neon | 91.42 | 21.52 | 0.0047 | 25 | 17,600 | 1.968 |

Interpretation:

- Current-byte prep is too small to chase further.
- Table scoring is the dominant target: roughly 6-21 KB of selected table reads per row, before cache/TLB effects.
- TonIoT and MedSec are traffic-heavy despite smaller literal density because class/block shape drives table rows.
- A useful `tm_algorithm_3.c` should attack score traffic or block count, not byte preparation.

## Attempt 7: Readability Cleanup

Purpose: make the active attempt easier to maintain without changing the inference
algorithm or backend decision policy.

Changes:

- Snapshotted the cleaned active implementation as `src/tm_local_infer/tm_algorithm_7.c`.
- Added shared constants for AVX2/NEON table lanes and AVX2 plane lanes.
- Added `TMClassRange`, block-count, positive-lane-count, delta-backend-ready, and vote-recording helpers.
- Replaced repeated class range extraction in AVX2/NEON/scalar paths with the shared helper.
- Replaced harness magic numbers for calibration/profile warmup with named constants.
- Grouped `TMLayout` fields with comments in `tm_algorithm.h`.

Correctness checks:

- Local default AVX2 build: all datasets `same=True`, `diffs=0`.
- Local scalar fallback build with `-DTM_NO_AVX2_SCORE`: all datasets `same=True`, `diffs=0`.
- Raspberry Pi 5 NEON build: all datasets `same=True`, `diffs=0`.

Local default timing from the equivalence run:

| Dataset | us/sample | Acc | F1 |
|---|---:|---:|---:|
| WUSTL | 0.102 | 0.9470 | 0.8157 |
| NSLKDD | 0.266 | 0.9946 | 0.9477 |
| TonIoT | 0.271 | 0.9703 | 0.9427 |
| MedSec | 0.298 | 0.9700 | 0.8692 |

Raspberry Pi 5 timing from the equivalence run:

| Dataset | us/sample | Acc | F1 |
|---|---:|---:|---:|
| WUSTL | 0.933 | 0.9470 | 0.8157 |
| NSLKDD | 1.680 | 0.9946 | 0.9477 |
| TonIoT | 1.645 | 0.9703 | 0.9427 |
| MedSec | 1.753 | 0.9700 | 0.8692 |

## Attempt 8: Nonzero-Byte Delta Loop

Goal: remove work and branching from the delta-table backend.

Change:

- Added a per-row nonzero current-byte list.
- Delta-table AVX2/NEON scoring now loops over that list instead of all bytes with `if (value != 0)`.
- The list builder is branchless: it writes the candidate byte index and increments only when the byte is nonzero.
- Updated `--profile-detail` traffic estimates so delta-table rows use average nonzero bytes, not total bytes.

Why this is generic:

- Delta tables are defined relative to the all-zero-current baseline.
- A zero current byte has zero delta by construction, so skipping it is exact for any model using this layout.

Local default equivalence run:

| Dataset | Attempt 7 local us/sample | Attempt 8 local us/sample | Result |
|---|---:|---:|---|
| WUSTL | 0.102 | 0.102 | unchanged, table backend |
| NSLKDD | 0.266 | 0.203 | faster |
| TonIoT | 0.271 | 0.221 | faster |
| MedSec | 0.298 | 0.223 | faster |

Raspberry Pi 5 equivalence run:

| Dataset | Attempt 7 Pi us/sample | Attempt 8 Pi us/sample | Result |
|---|---:|---:|---|
| WUSTL | 0.933 | 0.940 | effectively unchanged, table backend |
| NSLKDD | 1.680 | 1.404 | faster |
| TonIoT | 1.645 | 1.426 | faster |
| MedSec | 1.753 | 1.558 | faster |

Correctness:

- Local default AVX2 build: all datasets `same=True`, `diffs=0`.
- Local scalar fallback build with `-DTM_NO_AVX2_SCORE`: all datasets `same=True`, `diffs=0`.
- Raspberry Pi 5 NEON build: all datasets `same=True`, `diffs=0`.

## Attempt 9: Branchless Bit-Set Binarization

Question: are we storing boolean literal outputs as bits and using shifts?

Answer: yes for `current`; each literal threshold comparison becomes one bit in a
`uint64_t` word. Attempt 9 removes the scalar branch around that bit write:

```c
word |= (uint64_t)(value >= threshold) << bit;
```

instead of:

```c
if (value >= threshold) {
    word |= UINT64_C(1) << bit;
}
```

This affects scalar binarization and AVX2 tail/grouped-scalar paths. It is most
useful on Raspberry Pi, where binarization is scalar.

Local default equivalence run:

| Dataset | Attempt 8 local us/sample | Attempt 9 local us/sample | Result |
|---|---:|---:|---|
| WUSTL | 0.102 | 0.102 | unchanged |
| NSLKDD | 0.203 | 0.197 | similar/slightly faster |
| TonIoT | 0.221 | 0.224 | similar |
| MedSec | 0.223 | 0.236 | noisy/slightly slower |

Raspberry Pi 5 equivalence run:

| Dataset | Attempt 8 Pi us/sample | Attempt 9 Pi us/sample | Result |
|---|---:|---:|---|
| WUSTL | 0.940 | 0.796 | faster |
| NSLKDD | 1.404 | 1.302 | faster |
| TonIoT | 1.426 | 1.414 | similar/slightly faster |
| MedSec | 1.558 | 1.483 | faster |

Raspberry Pi 5 profile after Attempt 9:

| Dataset | us/sample | Binarize us | Score us |
|---|---:|---:|---:|
| WUSTL | 0.797 | 0.288 | 0.508 |
| NSLKDD | 1.309 | 0.317 | 0.962 |
| TonIoT | 1.401 | 0.146 | 1.207 |
| MedSec | 1.473 | 0.412 | 1.020 |

Correctness:

- Local default AVX2 build: all datasets `same=True`, `diffs=0`.
- Raspberry Pi 5 NEON build: all datasets `same=True`, `diffs=0`.

Remaining packing candidates:

- `current`, `inter`, and `planes` are already bit-packed.
- `clamp` and `base_mismatch_zero` are already 8-bit.
- `feat_idx` could be narrower for these assets, but AVX2 gather wants 32-bit indices; compressing it would likely add decode cost.
- Original byte-table entries are `0..8`, so 4-bit nibble packing is possible.
- Delta-table entries are `-8..+8`, so they do not fit cleanly in 4 bits; 5-bit packing would be awkward for SIMD and probably slower.

## Attempt 10: 16-Bit Feature Indices For Scalar Binarization

Question: can more layout fields be stored narrower?

Feature ids in the current assets are small:

| Dataset | Max feature id |
|---|---:|
| WUSTL | 31 |
| NSLKDD | 117 |
| TonIoT | 37 |
| MedSec | 53 |

Change:

- Added optional `feat_idx_u16` to `TMLayout`.
- Build it when all feature ids fit in `uint16_t`.
- Scalar binarization uses `feat_idx_u16`.
- AVX2 gather keeps using the original 32-bit `feat_idx`, because `_mm256_i32gather_ps` wants 32-bit indices.

Raspberry Pi 5 profile:

| Dataset | Attempt 9 Pi us/sample | Attempt 10 Pi us/sample | Binarize us | Score us |
|---|---:|---:|---:|---:|
| WUSTL | 0.796 | 0.807 | 0.294 | 0.517 |
| NSLKDD | 1.302 | 1.280 | 0.327 | 0.941 |
| TonIoT | 1.414 | 1.361 | 0.149 | 1.164 |
| MedSec | 1.483 | 1.446 | 0.418 | 0.997 |

Correctness:

- Raspberry Pi 5 NEON build: all datasets `same=True`, `diffs=0`.
- Local scalar fallback build with `-DTM_NO_AVX2_SCORE`: all datasets `same=True`, `diffs=0`.

Current bit-width audit:

- Literal/current clause state: already bit-packed into `uint64_t`.
- Threshold comparison result: now written directly as one shifted bit.
- Feature indices: scalar path now uses 16-bit sidecar when possible.
- Clamp/base mismatch: already 8-bit.
- Original table contribution: still 8-bit even though it fits in 4 bits. Nibble table remains the next plausible bit-packing experiment.
- Delta contribution: signed `-8..+8`, so not a clean 4-bit fit.

## Attempt 11: Validated Feature-Block Binarization

Question: can we avoid per-literal feature reloads and threshold checks when
literals are already grouped by feature?

Change:

- Added an optional `TMFeatureBlock` layout for contiguous runs of the same
  feature with sorted thresholds.
- Feature-block binarization loads each feature once, finds the active threshold
  count, and sets a contiguous literal-bit range.
- The layout builder rejects invalid/non-contiguous/unsorted blocks and keeps
  literal-order binarization as the fallback.
- Runtime calibration measures literal-order vs feature-block binarization on
  sample rows and selects feature blocks only when faster.

This is generic: there are no dataset-name checks, class checks, or hard-coded
threshold counts. It is a validated layout plus measured runtime selection.

Local calibration with `--profile-detail --stats`:

| Dataset | Literal binarize us | Feature-block us | Blocks | Selected |
|---|---:|---:|---:|---|
| WUSTL | 0.0418 | 0.1676 | 32 | literal |
| NSLKDD | 0.0440 | 0.1771 | 115 | literal |
| TonIoT | 0.0209 | 0.0699 | 38 | literal |
| MedSec | 0.0534 | 0.1651 | 53 | literal |

Local default profile:

| Dataset | Attempt 11 local us/sample | Binarize us | Score us | Backend | Binarize |
|---|---:|---:|---:|---|---|
| WUSTL | 0.103 | 0.044 | 0.068 | table_avx2 | literal |
| NSLKDD | 0.201 | 0.053 | 0.148 | delta_table_avx2 | literal |
| TonIoT | 0.229 | 0.024 | 0.204 | delta_table_avx2 | literal |
| MedSec | 0.245 | 0.057 | 0.174 | delta_table_avx2 | literal |

Raspberry Pi 5 calibration with `--profile-detail --stats`:

| Dataset | Literal binarize us | Feature-block us | Blocks | Selected |
|---|---:|---:|---:|---|
| WUSTL | 0.4419 | 0.5520 | 32 | literal |
| NSLKDD | 0.3047 | 0.4601 | 115 | literal |
| TonIoT | 0.1492 | 0.1805 | 38 | literal |
| MedSec | 0.4171 | 0.3898 | 53 | feature |

Raspberry Pi 5 same-session A/B against Attempt 10:

| Dataset | Attempt 10 Pi us/sample | Attempt 11 Pi us/sample | Result |
|---|---:|---:|---|
| WUSTL | 0.798 | 0.799 | unchanged |
| NSLKDD | 1.297 | 1.285 | similar/slightly faster |
| TonIoT | 1.302 | 1.257 | faster |
| MedSec | 1.444 | 1.403 | faster, feature-block binarizer selected |

Correctness:

- Local default AVX2 build: all datasets `same=True`, `diffs=0`.
- Local scalar fallback build with `-DTM_NO_AVX2_SCORE`: all datasets `same=True`, `diffs=0`.
- Raspberry Pi 5 NEON build: all datasets `same=True`, `diffs=0`.

Notes:

- One Raspberry Pi 5 profile-detail run had a noisy first WUSTL number
  (`1.206 us/sample`). Immediate repeated `--profile` and same-session A/B runs
  returned to `0.805` and `0.799 us/sample`, so the accepted comparison uses the
  A/B run above.
- Since local calibration rejected feature blocks for every bundled dataset,
  local runtime is expected to stay effectively the same.

## Attempt 12: Sparse Byte-Delta Posting Scorer

Question: can scoring be rewritten as a simpler baseline-plus-delta equation
that touches fewer clause updates?

Change:

- Added an experimental sparse byte-delta posting scorer.
- It starts from zero-current class votes, applies only nonzero byte/value
  mismatch deltas, and then repairs votes for dirty clauses.
- This is exact and dataset-agnostic, but it is kept behind
  `TM_ENABLE_BYTE_POSTINGS` because the measured backend was slower.
- Active/default builds remain on Attempt 11 behavior.

Build command for the experiment:

```bash
make -C src/tm_local_infer clean all TM_ALGORITHM_SRC=tm_algorithm_12.c CFLAGS='-O3 -std=c11 -Wall -Wextra -Wpedantic -funroll-loops -march=native -DTM_ENABLE_BYTE_POSTINGS'
```

Local AVX2 calibration with byte postings enabled:

| Dataset | Current selected backend | Current score us | Byte-posting score us | Selected |
|---|---:|---:|---:|---|
| WUSTL | table | 0.0638 | 1.3663 | table |
| NSLKDD | delta | 0.1369 | 2.8013 | delta |
| TonIoT | delta | 0.1903 | 4.5189 | delta |
| MedSec | delta | 0.1710 | 2.0815 | delta |

Raspberry Pi 5 NEON calibration with byte postings enabled:

| Dataset | Current selected backend | Current score us | Byte-posting score us | Selected |
|---|---:|---:|---:|---|
| WUSTL | table | 0.5127 | 4.6825 | table |
| NSLKDD | delta | 0.9325 | 8.1928 | delta |
| TonIoT | delta | 1.1750 | 12.2810 | delta |
| MedSec | delta | 1.0053 | 6.1194 | delta |

Forced byte-posting equivalence and timing:

| Dataset | Local forced us/sample | Pi forced us/sample | Exact? |
|---|---:|---:|---|
| WUSTL | 1.478 | 4.927 | yes |
| NSLKDD | 2.816 | 8.527 | yes |
| TonIoT | 4.515 | 12.471 | yes |
| MedSec | 2.127 | 6.664 | yes |

Correctness:

- Local default active build after restoring Attempt 11 behavior: all datasets
  `same=True`, `diffs=0`.
- Local forced byte-posting build: all datasets `same=True`, `diffs=0`.
- Raspberry Pi 5 default build with byte-posting calibration available: all
  datasets `same=True`, `diffs=0`.
- Raspberry Pi 5 forced byte-posting build: all datasets `same=True`, `diffs=0`.

Conclusion:

- Fewer mathematical updates did not translate to faster runtime here.
- The sparse scorer touches fewer conceptual contributions, but each update is a
  scalar random write into clause scratch plus a dirty-clause finalization pass.
- Dense table/delta scoring wins because AVX2/NEON performs predictable
  contiguous loads and vector reductions very efficiently.
- Keep this attempt as a rejected data point; the next structural attempt should
  avoid random per-clause writes or work at a coarser block/batch level.

## Inference Equation And Profiling Notes

Added [INFERENCE_MATH.md](/home/per/Pi_codes/src/tm_local_infer/INFERENCE_MATH.md)
to make the exact computation explicit:

```text
x_i    = 1 if row[feature_i] >= threshold_i else 0
s_c(x) = max(b_c + sum_i a_ci * x_i, 0)
vote_k = sum_c W_kc * s_c(x)
pred   = argmax_k vote_k
```

In matrix form:

```text
pred = argmax_k [ W * ReLU(b + A*x) ]_k
```

The current byte-table and delta-table scorers are vectorized ways of evaluating
this exact clipped sparse affine layer.

Fresh active-build profile, local AVX2:

| Dataset | Total us | Binarize us | Score us | Backend |
|---|---:|---:|---:|---|
| WUSTL | 0.108 | 0.041 | 0.069 | table_avx2 |
| NSLKDD | 0.201 | 0.052 | 0.155 | delta_table_avx2 |
| TonIoT | 0.216 | 0.021 | 0.198 | delta_table_avx2 |
| MedSec | 0.238 | 0.060 | 0.174 | delta_table_avx2 |

Fresh active-build profile, Raspberry Pi 5 NEON:

| Dataset | Total us | Binarize us | Score us | Backend |
|---|---:|---:|---:|---|
| WUSTL | 0.808 | 0.294 | 0.503 | table_neon |
| NSLKDD | 1.314 | 0.321 | 0.961 | delta_table_neon |
| TonIoT | 1.311 | 0.149 | 1.150 | delta_table_neon |
| MedSec | 1.465 | 0.391 | 1.038 | delta_table_neon |

Sample clause activation over the first 3000 rows:

| Dataset | Active clauses/row | Percent active | Clauses always zero on sample |
|---|---:|---:|---:|
| WUSTL | 24.4 / 180 | 13.6% | 25 |
| NSLKDD | 53.5 / 450 | 11.9% | 34 |
| TonIoT | 535.8 / 1000 | 53.6% | 11 |
| MedSec | 120.6 / 400 | 30.2% | 0 |

Optimization implication:

- Exact single-row collapse to one linear class equation is blocked by the
  `ReLU`/clamp, and model stats show `linearizable=0` for all bundled models.
- Sample-only always-zero clause pruning is not exact, so it should not be used
  as a default backend.
- The most promising exact next step is the batch equation:

```text
S = ReLU(ones*b^T + X*A^T)
Y = S*W^T
```

implemented as a small-batch scorer that reuses model data across multiple rows.

## Attempt 13: Four-Row Batch Table Scorer

Question: can we evaluate the batch equation directly and reuse model data
across four rows?

Change:

- Added an experimental `tm_predict_batch()` implementation in
  `tm_algorithm_13.c`.
- The batch scorer keeps four current rows in flight and evaluates the same
  byte-table/delta-table blocks for each row.
- The harness can profile it with `TM_ENABLE_BATCH_API`.

Build command:

```bash
make -C src/tm_local_infer clean all TM_ALGORITHM_SRC=tm_algorithm_13.c CFLAGS='-O3 -std=c11 -Wall -Wextra -Wpedantic -funroll-loops -march=native -DTM_ENABLE_BATCH_API'
```

Local AVX2 profile:

| Dataset | Single total us | Batch total us | Batch diffs |
|---|---:|---:|---:|
| WUSTL | 0.108 | 0.116 | 0 |
| NSLKDD | 0.205 | 0.313 | 0 |
| TonIoT | 0.237 | 0.320 | 0 |
| MedSec | 0.229 | 0.350 | 0 |

Raspberry Pi 5 NEON profile:

| Dataset | Single total us | Batch total us | Batch diffs |
|---|---:|---:|---:|
| WUSTL | 0.805 | 0.867 | 0 |
| NSLKDD | 1.255 | 1.383 | 0 |
| TonIoT | 1.320 | 1.548 | 0 |
| MedSec | 1.386 | 1.776 | 0 |

Conclusion:

- The exact batch equation works, but this four-row schedule is slower.
- It reuses some clamp/range work, but still performs different table-row loads
  for each input row and adds register/loop pressure.
- Keep as rejected. A future batch attempt would need a different layout, for
  example grouping rows by equal current byte values or using a true bit-sliced
  row layout.

## v2: Clean Equation-Form Implementation

Question: can we write a cleaner implementation around:

```text
pred = argmax_k [ W * ReLU(b + A*x) ]_k
```

Change:

- Added `tm_algorithm_v2.c`.
- Added `TMVoteBlock` metadata to encode `W` as contiguous positive/negative
  class vote blocks.
- v2 keeps the optimized byte-table/delta-table representation for `A*x`, so it
  does not materialize dense `A`.
- The hot scoring loop becomes:

```text
score block = ReLU(byte-table affine block)
vote[class] += sum positive scores - sum negative scores
```

Build command:

```bash
make -C src/tm_local_infer clean all TM_ALGORITHM_SRC=tm_algorithm_v2.c
```

Local AVX2 v2 profile:

| Dataset | v2 us/sample | Binarize us | Score us | Backend |
|---|---:|---:|---:|---|
| WUSTL | 0.104 | 0.041 | 0.063 | table_avx2 |
| NSLKDD | 0.206 | 0.053 | 0.153 | delta_table_avx2 |
| TonIoT | 0.223 | 0.021 | 0.207 | delta_table_avx2 |
| MedSec | 0.245 | 0.058 | 0.175 | delta_table_avx2 |

Raspberry Pi 5 NEON v2 profile:

| Dataset | v2 us/sample | Binarize us | Score us | Backend |
|---|---:|---:|---:|---|
| WUSTL | 0.805 | 0.319 | 0.507 | table_neon |
| NSLKDD | 1.305 | 0.328 | 0.932 | delta_table_neon |
| TonIoT | 1.319 | 0.151 | 1.108 | delta_table_neon |
| MedSec | 1.407 | 0.393 | 0.998 | delta_table_neon |

Correctness:

- Local v2 AVX2 build: all datasets `same=True`, `diffs=0`.
- Raspberry Pi 5 v2 NEON build: all datasets `same=True`, `diffs=0`.

Conclusion:

- v2 is a good clean reference/candidate implementation of the equation.
- Performance is roughly tied with active Attempt 11, not a decisive speedup.
- It is a better base for future equation-level work because `A` and `W` are
  separated in the layout: byte tables/delta tables for `A*x`, vote blocks for
  `W`.

## v3: Fused Score-Block Equation Layout

Question: can we make the equation implementation even more direct, with fewer
range loops and a fused block layout?

Implemented:

- Added `tm_algorithm_v3.c`.
- Added gated score-block metadata and blocked table storage, built only with
  `TM_ENABLE_SCORE_BLOCKS`.
- Scoring loop:

```text
for each score block:
    score_lanes = ReLU(clamp_lanes - table_sum_lanes)
    vote[class] += sign_lanes dot score_lanes
```

Build command:

```bash
make -C src/tm_local_infer clean all TM_ALGORITHM_SRC=tm_algorithm_v3.c CFLAGS='-O3 -std=c11 -Wall -Wextra -Wpedantic -funroll-loops -march=native -DTM_ENABLE_SCORE_BLOCKS'
```

Local AVX2 v3 profile:

| Dataset | v3 us/sample | Binarize us | Score us | Correct |
|---|---:|---:|---:|---|
| WUSTL | 0.117 | 0.041 | 0.071 | yes |
| NSLKDD | 0.216 | 0.064 | 0.166 | yes |
| TonIoT | 0.267 | 0.023 | 0.227 | yes |
| MedSec | 0.329 | 0.054 | 0.224 | yes |

Raspberry Pi 5 NEON v3 profile:

| Dataset | v3 us/sample | Binarize us | Score us | Correct |
|---|---:|---:|---:|---|
| WUSTL | 1.024 | 0.297 | 0.749 | yes |
| NSLKDD | 3.080 | 0.332 | 2.882 | yes |
| TonIoT | 3.729 | 0.152 | 3.589 | yes |
| MedSec | 3.751 | 0.379 | 3.383 | yes |

Correctness:

- Local v3 AVX2 build: all datasets `same=True`, `diffs=0`.
- Raspberry Pi 5 v3 NEON build: all datasets `same=True`, `diffs=0`.

Conclusion:

- v3 is mathematically clean and exact.
- It is not faster. It is slightly slower locally and much slower on RPi5.
- The existing active table/delta loops remain better because they keep wider,
  simpler contiguous clause blocks. v3’s cleaner fused abstraction creates more
  small blocks and extra per-block reduction/sign work, especially painful for
  16-lane NEON.
- Keep v3 as rejected/reference behind `TM_ENABLE_SCORE_BLOCKS`; do not promote
  it to `tm_algorithm.c`.

## Attempt 14: Selected Table Row Pointer Hoist

Question: can we make the active table/delta scorer faster without changing the
data model or behavior?

Change:

- Keep the active byte-table and delta-table layout.
- For each sample, precompute the selected row pointer for every current byte:

```text
selected_rows[b] = &table[b, current_byte_b, 0]
```

- For delta-table scoring, precompute selected row pointers only for nonzero
  current bytes.
- Inner AVX2/NEON clause-block loops now use `selected_row + clause_start`
  instead of recomputing `((byte * 256 + value) * table_stride + clause_start)`
  for every block.

This is generic and dataset-agnostic; it is just loop-invariant address
arithmetic hoisting.

Local AVX2 profile:

| Dataset | Previous active us | Attempt 14 us | Speedup | Correct |
|---|---:|---:|---:|---|
| WUSTL | 0.103 | 0.095 | 1.08x | yes |
| NSLKDD | 0.206 | 0.172 | 1.20x | yes |
| TonIoT | 0.222 | 0.170 | 1.31x | yes |
| MedSec | 0.325 | 0.288 | 1.13x | yes |

Raspberry Pi 5 NEON profile:

| Dataset | Previous active us | Attempt 14 us | Speedup | Correct |
|---|---:|---:|---:|---|
| WUSTL | 0.807 | 0.682 | 1.18x | yes |
| NSLKDD | 1.304 | 0.990 | 1.32x | yes |
| TonIoT | 1.361 | 0.944 | 1.44x | yes |
| MedSec | 1.446 | 1.037 | 1.39x | yes |

Correctness:

- Local AVX2 build: all datasets `same=True`, `diffs=0`.
- Raspberry Pi 5 NEON build: all datasets `same=True`, `diffs=0`.

Conclusion:

- Promote this change.
- This is the opposite lesson from v3: keep the mature table layout, but remove
  repeated address arithmetic in the hot loops.
- `tm_algorithm.c` and `tm_algorithm_14.c` now contain this active
  implementation.

## Attempt 15: Multi-Block Class Scoring

Question: after precomputing selected table rows, can the hot loop reuse those
row pointers across several clause blocks in the same class band?

Change:

- Keep Attempt 14 selected-row pointers.
- For contiguous positive/negative class bands, accumulate multiple SIMD clause
  blocks while walking the selected byte-row list once.
- AVX2 path handles 2-4 contiguous 32-lane blocks.
- NEON path handles 2-8 contiguous 16-lane blocks.
- Other range shapes fall back to the normal per-block scorer.

The math is unchanged:

```text
s_c(x) = ReLU(clamp_c - sum_b T[b, x_b, c])
votes = W * s(x)
```

Local AVX2 profile:

| Dataset | Attempt 14 us | Attempt 15 us | Speedup | Correct |
|---|---:|---:|---:|---|
| WUSTL | 0.095 | 0.096 | 0.99x | yes |
| NSLKDD | 0.172 | 0.148 | 1.16x | yes |
| TonIoT | 0.170 | 0.146 | 1.16x | yes |
| MedSec | 0.288 | 0.183 | 1.57x | yes |

Raspberry Pi 5 NEON profile:

| Dataset | Attempt 14 us | Attempt 15 us | Speedup | Correct |
|---|---:|---:|---:|---|
| WUSTL | 0.682 | 0.609 | 1.12x | yes |
| NSLKDD | 0.990 | 0.902 | 1.10x | yes |
| TonIoT | 0.944 | 0.813 | 1.16x | yes |
| MedSec | 1.037 | 0.964 | 1.08x | yes |

Correctness:

- Local AVX2 build: all datasets `same=True`, `diffs=0`.
- Raspberry Pi 5 NEON build: all datasets `same=True`, `diffs=0`.

Conclusion:

- Promote this change.
- It is generic and exact: it only changes loop order inside contiguous class
  bands.
- `tm_algorithm.c` and `tm_algorithm_15.c` now contain this active
  implementation.

## Attempt 16: Stripped Table/Delta-Only Reference

Question: can we strip `tm_algorithm_15.c` down to the code path we actually
use?

Change:

- Added `tm_algorithm_16.c`.
- Reduced the algorithm file from 1694 lines to 629 lines after the second
  cleanup pass.
- Kept literal/feature-block binarization.
- Kept selected table-row pointers.
- Kept AVX2 and NEON byte-table/delta-table scorers.
- Removed the scalar table fallback; this attempt is intentionally an
  AVX2/NEON-only stripped scorer.
- Removed dense scalar clause scanning, AVX2 plane scoring, fixed-H plane
  macros, split range helper scorers, and old experimental compile-time
  binarization branches.

Local AVX2 profile:

| Dataset | Attempt 15 us | Attempt 16 us | Correct |
|---|---:|---:|---|
| WUSTL | 0.093 | 0.095 | yes |
| NSLKDD | 0.161 | 0.159 | yes |
| TonIoT | 0.148 | 0.160 | yes |
| MedSec | 0.167 | 0.186 | yes |

Raspberry Pi 5 NEON profile after the second cleanup pass:

| Dataset | Attempt 15 us | Attempt 16 us | Change | Correct |
|---|---:|---:|---:|---|
| WUSTL | 0.609 | 0.613 | -0.7% | yes |
| NSLKDD | 0.902 | 0.896 | +0.7% | yes |
| TonIoT | 0.813 | 0.792 | +2.7% | yes |
| MedSec | 0.964 | 0.929 | +3.8% | yes |

Correctness:

- Local AVX2 build: all datasets `same=True`, `diffs=0`.
- Raspberry Pi 5 NEON build: all datasets `same=True`, `diffs=0`.

Conclusion:

- Keep `tm_algorithm_16.c` as the clean stripped reference.
- Do not promote it over Attempt 15 yet; it is exact and much shorter, but the
  local AVX2 run regresses on TonIoT/MedSec and the RPi5 result is close enough
  that Attempt 15 remains the safer active implementation.
- Active `tm_algorithm.c` stays aligned with `tm_algorithm_15.c`.

## Attempt 17: Split Fixed-Shape Kernel

Question: can we reduce runtime scorer logic by moving SIMD kernels into a
separate kernel file and specializing common class block shapes?

Change:

- Added `tm_algorithm_17.c`, `tm_kernel_17.c`, and `tm_kernel_17.h`.
- Added `TM_KERNEL_SRC` to the Makefile so split-kernel attempts can build
  without changing the default active implementation.
- Kept `tm_algorithm_17.c` as API glue plus binarization.
- Moved AVX2/NEON table and delta scoring into `tm_kernel_17.c`.
- Added fixed-shape AVX2 kernels for 2, 3, and 4 blocks per class.
- Added fixed-shape NEON kernels for 2 through 8 blocks per class.
- Kept generic fallbacks for unusual block counts.
- Defaulted reduction to a split full-positive/full-negative/mixed form, with
  `TM17_SIMPLE_REDUCE` available for the old reduction.

Build:

```text
make -C src/tm_local_infer clean all TM_ALGORITHM_SRC=tm_algorithm_17.c TM_KERNEL_SRC=tm_kernel_17.c
```

Local AVX2 profile, median of seven compact runs:

| Dataset | Attempt 15 us | Attempt 17 us | Speedup | Correct |
|---|---:|---:|---:|---|
| WUSTL | 0.093 | 0.090 | 1.03x | yes |
| NSLKDD | 0.145 | 0.148 | 0.98x | yes |
| TonIoT | 0.155 | 0.143 | 1.08x | yes |
| MedSec | 0.172 | 0.166 | 1.04x | yes |

Raspberry Pi 5 NEON profile, median of five compact runs:

| Dataset | Attempt 15 us | Attempt 17 us | Speedup | Correct |
|---|---:|---:|---:|---|
| WUSTL | 0.609 | 0.574 | 1.06x | yes |
| NSLKDD | 0.902 | 0.883 | 1.02x | yes |
| TonIoT | 0.813 | 0.708 | 1.15x | yes |
| MedSec | 0.964 | 0.855 | 1.13x | yes |

Correctness:

- Local AVX2 build: all datasets `same=True`, `diffs=0`.
- Raspberry Pi 5 NEON build: all datasets `same=True`, `diffs=0`.

Conclusion:

- Attempt 17 is exact and is the fastest tested variant on Raspberry Pi 5.
- Locally it is mixed but positive overall, with only NSLKDD slightly slower
  than the active Attempt 15 baseline.
- Do not promote automatically in this entry because it changes the build shape
  from one algorithm source file to `tm_algorithm_17.c + tm_kernel_17.c`.

## Attempts 18-19: Three CPU-Compatible Reformulation Tests

Question: can we reformulate the all-clauses/all-literals computation into a
more CPU-native primitive than the current dense byte-table scorer?

Tested three exact formulations:

1. **Blocked byte table**: existing `tm_algorithm_v3.c` with
   `TM_ENABLE_SCORE_BLOCKS`.
2. **Blocked nibble table**: new `tm_algorithm_18.c` with 4-bit input symbols
   and `TM_ENABLE_NIBBLE_TABLES`.
3. **Feature-state table**: new `tm_algorithm_19.c`, computing feature threshold
   states directly and using precomputed feature-state delta rows.

Builds:

```text
# blocked byte
make -C src/tm_local_infer clean all TM_ALGORITHM_SRC=tm_algorithm_v3.c \
  CFLAGS='-O3 -std=c11 -Wall -Wextra -Wpedantic -funroll-loops -march=native -DTM_ENABLE_SCORE_BLOCKS'

# blocked nibble
cc -O3 -std=c11 -Wall -Wextra -Wpedantic -funroll-loops -march=native \
  -DTM_ENABLE_SCORE_BLOCKS -DTM_SCORE_BLOCKS_METADATA_ONLY -DTM_ENABLE_NIBBLE_TABLES \
  src/tm_local_infer/tm_infer_c.c src/tm_local_infer/tm_algorithm_18.c -o /tmp/tm18_nibble -lzstd

# feature state
cc -O3 -std=c11 -Wall -Wextra -Wpedantic -funroll-loops -march=native \
  -DTM_ENABLE_SCORE_BLOCKS -DTM_SCORE_BLOCKS_METADATA_ONLY -DTM_ENABLE_FEATURE_STATE_TABLES \
  src/tm_local_infer/tm_infer_c.c src/tm_local_infer/tm_algorithm_19.c -o /tmp/tm19_feature_state -lzstd
```

Local AVX2 profile, median of compact runs:

| Dataset | Attempt 17 us | Blocked byte us | Nibble us | Feature-state us |
|---|---:|---:|---:|---:|
| WUSTL | 0.090 | 0.113 | 0.177 | 0.208 |
| NSLKDD | 0.148 | 0.221 | 0.302 | 0.977 |
| TonIoT | 0.143 | 0.272 | 0.312 | 0.784 |
| MedSec | 0.166 | 0.297 | 0.370 | 0.503 |

Raspberry Pi 5 NEON profile, median of compact runs:

| Dataset | Attempt 17 us | Blocked byte us | Nibble us | Feature-state us |
|---|---:|---:|---:|---:|
| WUSTL | 0.574 | 1.364 | 1.185 | 0.784 |
| NSLKDD | 0.883 | 3.280 | 1.802 | 4.765 |
| TonIoT | 0.708 | 3.732 | 2.434 | 3.929 |
| MedSec | 0.855 | 3.934 | 1.985 | 2.165 |

Correctness:

- Local AVX2: all three formulations matched Python exactly on all four
  datasets.
- Raspberry Pi 5 NEON: all three formulations matched Python exactly on all
  four datasets.

Conclusion:

- Blocked byte, blocked nibble, and feature-state tables are exact but slower
  than Attempt 17 on these models.
- The reason is that they reduce or reorganize memory, but increase loop count,
  score-block count, or feature-state table lookups. Attempt 17's class-shaped
  dense delta scorer does less actual runtime work for these dense-ish bundled
  models.
- Do not promote Attempts 18-19.

## Attempt 17 Micro-Check: Split Backend Branch

Question: can `tm_kernel_17.c` get faster by moving the table-vs-delta branch
outside the per-class loop?

Change tested:

- Split `tm17_predict_current_avx2()` and `tm17_predict_current_neon()` into
  separate table and delta class loops.
- This removes one runtime ternary per class, but duplicates the class loop
  shape.

Local AVX2 median of compact runs:

| Dataset | Attempt 17 us | Branch-split us | Result |
|---|---:|---:|---|
| WUSTL | 0.090 | 0.096 | slower |
| NSLKDD | 0.148 | 0.166 | slower |
| TonIoT | 0.143 | 0.185 | slower |
| MedSec | 0.166 | 0.196 | slower |

Raspberry Pi 5 NEON median of compact runs:

| Dataset | Attempt 17 us | Branch-split us | Result |
|---|---:|---:|---|
| WUSTL | 0.574 | 0.571 | slightly faster |
| NSLKDD | 0.883 | 0.887 | slightly slower |
| TonIoT | 0.708 | 0.748 | slower |
| MedSec | 0.855 | 0.879 | slower |

Conclusion:

- Rejected and reverted. The original Attempt 17 backend branch inside the
  class loop is faster overall.

## Attempt 17 Micro-Check: Delta Row Setup And Binarizer

Question: after profiling `tm_algorithm_17.c + tm_kernel_17.c`, can we remove
meaningful setup work before the class-shaped SIMD scorer?

Profile finding:

- The remaining hot path is score-core table/delta-row replay:
  `current -> selected byte rows -> class-shaped SIMD blocks -> vote reduce`.
- On RPi5, binarization is also material for WUSTL and MedSec, but the
  scorer still spends most of its time loading selected table rows.
- Byte preparation itself is tiny; changing it only matters if the change also
  removes loop work without adding a branchy second path.

Changes tested:

- **Kept:** fuse nonzero-byte collection and delta-row pointer selection into
  one pass over `current_bytes`. This removes the intermediate `nonzero_bytes`
  array and one setup loop while preserving the same delta scorer.
- **Rejected:** direct extraction from `uint64_t current` for all delta models.
  It avoids `memcpy`, but repeated shifts were not consistently better than
  the byte view.
- **Rejected:** direct extraction only for small `n_bytes <= 16`. It targeted
  TonIoT, but the extra path was not stable across repeated RPi5 runs.
- **Rejected:** scalar 8-literal binarization unroll. It improved MedSec on one
  RPi5 run but slowed WUSTL, NSLKDD, and TonIoT.
- **Rejected:** `-flto` on the split kernel. Local timing regressed heavily.

Local AVX2 profile, median of compact runs:

| Dataset | Attempt 17 us | Kept row-setup us | Result |
|---|---:|---:|---|
| WUSTL | 0.090 | 0.098 | noisy/slower |
| NSLKDD | 0.148 | 0.165 | noisy/slower |
| TonIoT | 0.143 | 0.149 | roughly tied |
| MedSec | 0.166 | 0.183 | noisy/slower |

Raspberry Pi 5 NEON profile, median of five compact runs:

| Dataset | Attempt 17 us | Kept row-setup us | Result |
|---|---:|---:|---|
| WUSTL | 0.574 | 0.577 | tied |
| NSLKDD | 0.883 | 0.884 | tied |
| TonIoT | 0.708 | 0.706 | tied/slightly faster |
| MedSec | 0.855 | 0.857 | tied |

Rejected scalar unroll on Raspberry Pi 5 NEON, median of five compact runs:

| Dataset | Attempt 17 us | Scalar-unroll us | Result |
|---|---:|---:|---|
| WUSTL | 0.574 | 0.596 | slower |
| NSLKDD | 0.883 | 0.900 | slower |
| TonIoT | 0.708 | 0.717 | slower |
| MedSec | 0.855 | 0.840 | faster |

Correctness:

- Local AVX2 kept row-setup build: all datasets `same=True`, `diffs=0`.
- Raspberry Pi 5 NEON kept row-setup build: all datasets `same=True`,
  `diffs=0`.

Conclusion:

- Attempt 17 is already close to the useful CPU formulation for these bundled
  dense-ish models.
- The only kept code change is a small generic simplification in
  `tm_kernel_17.c`; it is exact and roughly timing-neutral.
- No new promotion. The significant bottleneck remains selected table-row
  memory replay plus SIMD reduction; further big wins likely require a true
  layout change with a convincing benchmark, not another branch-level tweak.

## Attempt 17 Micro-Check: Dispatch, Exactness, And Generic Chunks

Question: do the cleaner kernel suggestions produce a stable win in Attempt 17?

Implemented and kept:

- Added `layout->delta_table_exact`, set during layout construction only when
  every clause's maximum possible mismatch count fits in `uint8_t`.
- Gated the signed `int8_t` delta-table kernel on `delta_table_exact` so larger
  future models do not wrap mismatches modulo 256.
- Added 4-block chunking to the generic AVX2 table/delta scorers.
- Added 8-block chunking to the generic NEON table/delta scorers.

Tested and rejected:

- Little-endian `current` byte aliasing plus per-row dynamic delta-vs-table
  dispatch using `n_nonzero + 1 < n_bytes`.
- It was exact, but RPi5 timing was worse on TonIoT and MedSec. The calibration
  already chooses the better default backend for these bundled models.

Local AVX2 dynamic-dispatch median of seven compact runs:

| Dataset | Attempt 17 us | Dynamic dispatch us | Result |
|---|---:|---:|---|
| WUSTL | 0.090 | 0.094 | slightly slower |
| NSLKDD | 0.148 | 0.151 | tied/slower |
| TonIoT | 0.143 | 0.136 | faster |
| MedSec | 0.166 | 0.179 | slower |

Raspberry Pi 5 NEON dynamic-dispatch median of seven compact runs:

| Dataset | Attempt 17 us | Dynamic dispatch us | Result |
|---|---:|---:|---|
| WUSTL | 0.574 | 0.580 | slower |
| NSLKDD | 0.883 | 0.871 | faster |
| TonIoT | 0.708 | 0.739 | slower |
| MedSec | 0.855 | 0.881 | slower |

Kept exactness/generic-chunk build, median of five compact runs:

| Dataset | Local AVX2 us | RPi5 NEON us |
|---|---:|---:|
| WUSTL | 0.092 | 0.585 |
| NSLKDD | 0.169 | 0.877 |
| TonIoT | 0.147 | 0.758 |
| MedSec | 0.171 | 0.880 |

Correctness:

- Local AVX2 kept build: all datasets `same=True`, `diffs=0`.
- Raspberry Pi 5 NEON kept build: all datasets `same=True`, `diffs=0`.

Conclusion:

- Dynamic per-row dispatch is rejected for now. It adds decision/setup work and
  only helps one dataset per machine.
- Generic chunking is retained because it is dataset-agnostic and only affects
  future models whose class ranges exceed the current fixed-shape kernels.
- The delta exactness guard is retained as a correctness fix for larger future
  models; current bundled models all satisfy the `uint8_t` mismatch bound.

## Attempt 17 Micro-Check: Adaptive Lane Alignment And AOT Shapes

Question: can we get a larger exact win by making class ranges friendlier to
SIMD reductions and by dispatching directly to bundled-model fixed shapes?

Implemented:

- Added opt-in `TM_ENABLE_LANE_ALIGNED_RANGES`.
- The layout pads positive and negative ranges separately with zero-scoring
  dummy clauses, but only when doing so does not increase that class's SIMD
  block count.
- Added opt-in `TM_ENABLE_AOT_SHAPES`.
- AOT shape dispatch recognizes the four bundled model shapes and calls their
  fixed-shape AVX2/NEON scorers directly, falling back to the normal Attempt 17
  path for every other model.

Builds:

```text
# adaptive lane alignment
make -C src/tm_local_infer clean all TM_ALGORITHM_SRC=tm_algorithm_17.c \
  TM_KERNEL_SRC=tm_kernel_17.c \
  CFLAGS='-O3 -std=c11 -Wall -Wextra -Wpedantic -funroll-loops -march=native -DTM_ENABLE_LANE_ALIGNED_RANGES'

# bundled-shape AOT dispatch
make -C src/tm_local_infer clean all TM_ALGORITHM_SRC=tm_algorithm_17.c \
  TM_KERNEL_SRC=tm_kernel_17.c \
  CFLAGS='-O3 -std=c11 -Wall -Wextra -Wpedantic -funroll-loops -march=native -DTM_ENABLE_AOT_SHAPES'

# combined
make -C src/tm_local_infer clean all TM_ALGORITHM_SRC=tm_algorithm_17.c \
  TM_KERNEL_SRC=tm_kernel_17.c \
  CFLAGS='-O3 -std=c11 -Wall -Wextra -Wpedantic -funroll-loops -march=native -DTM_ENABLE_LANE_ALIGNED_RANGES -DTM_ENABLE_AOT_SHAPES'
```

Local AVX2 profile, median of compact runs:

| Dataset | Attempt 17 us | Adaptive align us | AOT us | Combined us |
|---|---:|---:|---:|---:|
| WUSTL | 0.090 | 0.083 | 0.097 | 0.106 |
| NSLKDD | 0.148 | 0.154 | 0.160 | 0.150 |
| TonIoT | 0.143 | 0.131 | 0.144 | 0.147 |
| MedSec | 0.166 | 0.170 | 0.176 | 0.171 |

Raspberry Pi 5 NEON profile, median of compact runs:

| Dataset | Attempt 17 us | Adaptive align us | AOT us | Combined us |
|---|---:|---:|---:|---:|
| WUSTL | 0.574 | 0.540 | 0.570 | 0.539 |
| NSLKDD | 0.883 | 0.758 | 0.851 | 0.755 |
| TonIoT | 0.708 | 0.751 | 0.673 | 0.675 |
| MedSec | 0.855 | 0.879 | 0.840 | 0.823 |

Correctness:

- Local AVX2 adaptive-align build: all datasets `same=True`, `diffs=0`.
- Local AVX2 combined build: all datasets `same=True`, `diffs=0`.
- Raspberry Pi 5 NEON adaptive-align build: all datasets `same=True`,
  `diffs=0`.
- Raspberry Pi 5 NEON combined build: all datasets `same=True`, `diffs=0`.

Conclusion:

- Adaptive lane alignment is real, but model-dependent. It helps when it removes
  mixed positive/negative reduction without adding blocks; otherwise it is
  skipped.
- AOT shape dispatch is not useful locally, but helps the Raspberry Pi 5 NEON
  build.
- The combined opt-in build is the best RPi5 result in this round, improving
  all four datasets versus Attempt 17 baseline.
- Do not promote as the default local implementation because local AVX2 is
  mixed. Keep the knobs for target-specific RPi builds.

## Attempt 17 Assembly And Exact Reuse Check

Question: is there an assembly-level waste or exact reformulation that can
produce a large jump, ideally around 5x?

Assembly findings from the combined Attempt 17 build:

| Target | Symbol | Size |
|---|---:|---:|
| local AVX2 | `score_class_delta2_avx2` | 806 B |
| local AVX2 | `score_class_table2_avx2` | 861 B |
| local AVX2 | `score_class_delta4_avx2` | 1201 B |
| local AVX2 | `score_class_table4_avx2` | 1269 B |
| local AVX2 | `tm17_predict_current_avx2` | 13715 B |
| RPi5 NEON | `score_class_delta4_neon` | 952 B |
| RPi5 NEON | `score_class_table4_neon` | 956 B |
| RPi5 NEON | `score_class_delta7_neon` | 1600 B |
| RPi5 NEON | `score_class_table7_neon` | 1604 B |
| RPi5 NEON | `tm17_predict_current_neon` | 17716 B |

The score kernels are already mostly the expected vector loop:

```text
load selected byte/delta table row
vector add into clause-block accumulators
subtract from clamp / reduce lanes into class vote
```

So the remaining single-row cost is not a hidden scalar loop. The large AOT
dispatcher size also explains why AOT is mixed locally: it removes some
generic dispatch, but increases branch/code footprint.

Exact structure scan:

| Dataset | Clauses | Unique clauses | Rows | Unique current vectors | Current repeat rate |
|---|---:|---:|---:|---:|---:|
| WUSTL | 180 | 180 | 3264 | 2836 | 13.1% |
| NSLKDD | 450 | 450 | 29704 | 21517 | 27.6% |
| TonIoT | 1000 | 1000 | 42209 | 2345 | 94.4% |
| MedSec | 400 | 400 | 110907 | 12163 | 89.0% |

Clause deduplication gives nothing on the bundled models, but exact prediction
reuse can skip scoring when many rows binarize to the same current bit-vector.

Implemented a CLI-only exact cache:

```text
tm_infer_c assets pred_dir --cache
```

It hashes the full `current` bit-vector, confirms equality, and reuses the
class only after the exact compare. Public algorithm API is unchanged.

Local AVX2 combined build with `--cache`:

| Dataset | Raw total us | Cached timed us | Cached full-output us | Timed hit rate | Output hit rate | Correct |
|---|---:|---:|---:|---:|---:|---|
| WUSTL | 0.086 | 0.163 | 0.127 | 12.6% | 13.1% | yes |
| NSLKDD | 0.146 | 0.270 | 0.792 | 9.6% | 27.6% | yes |
| TonIoT | 0.132 | 0.101 | 0.067 | 82.1% | 94.4% | yes |
| MedSec | 0.164 | 0.178 | 0.121 | 72.6% | 89.0% | yes |

Raspberry Pi 5 NEON combined build with `--cache`:

| Dataset | Raw total us | Cached timed us | Cached full-output us | Timed hit rate | Output hit rate | Correct |
|---|---:|---:|---:|---:|---:|---|
| WUSTL | 0.534 | 0.573 | 0.563 | 12.6% | 13.1% | yes |
| NSLKDD | 0.731 | 0.876 | 2.049 | 9.6% | 27.6% | yes |
| TonIoT | 0.643 | 0.333 | 0.255 | 82.1% | 94.4% | yes |
| MedSec | 0.804 | 0.603 | 0.522 | 72.6% | 89.0% | yes |

Correctness:

- Local cached predictions match Python exactly on all four datasets.
- RPi5 cached predictions match the RPi5 non-cached predictions exactly.

Conclusion:

- A 5x single-row latency jump is not visible in the assembly; the hot kernels
  are already near the table-row formulation's natural lower bound.
- Exact reuse is the first real "do fewer computations" reformulation found.
  It is very useful for repeated-current batch workloads, especially TonIoT and
  MedSec, but it is not a universal default because WUSTL/NSLKDD do not repeat
  enough.
- Keep `--cache` as an explicit batch/benchmark option. A future adaptive cache
  can sample the first rows and enable itself only when the observed hit rate is
  high.

## Attempt 17 Reuse Iteration: Grouped Current Cache And Auto Cache

Question: can we make reuse more cache-local by grouping identical binarized
`current` vectors before scoring, and can we make exact caching safe as a
generic option?

Implemented:

- `--group-cache`: binarize rows, sort/group identical `current` bit-vectors,
  score each unique current once, then scatter classes back.
- `--cache-auto`: sample up to the first 3000 rows; enable row-by-row exact
  current caching only when sample hit rate is at least 50%.

Local AVX2 combined build:

| Dataset | Group timed us | Group full-output us | Auto selected | Auto timed us | Auto full-output us | Correct |
|---|---:|---:|---|---:|---:|---|
| WUSTL | 0.240 | 0.236 | none | 0.087 | raw | yes |
| NSLKDD | 0.292 | 0.320 | none | 0.151 | raw | yes |
| TonIoT | 0.256 | 0.154 | row-cache | 0.083 | 0.062 | yes |
| MedSec | 0.188 | 0.241 | row-cache | 0.162 | 0.116 | yes |

Raspberry Pi 5 NEON combined build:

| Dataset | Group timed us | Group full-output us | Auto selected | Auto timed us | Auto full-output us | Correct |
|---|---:|---:|---|---:|---:|---|
| WUSTL | 0.641 | 0.645 | none | 0.539 | raw | yes |
| NSLKDD | 0.827 | 0.818 | none | 0.746 | raw | yes |
| TonIoT | 0.393 | 0.364 | row-cache | 0.321 | 0.256 | yes |
| MedSec | 0.722 | 0.771 | row-cache | 0.628 | 0.521 | yes |

Correctness:

- Local `--group-cache` and `--cache-auto` predictions match Python exactly on
  all four datasets.
- RPi5 `--group-cache`, `--cache`, and `--cache-auto` predictions match the
  RPi5 non-cached predictions exactly.

Conclusion:

- Grouping by sorting is exact but not worth it here. The `qsort`/scatter cost
  is larger than the locality gain, even on high-repeat TonIoT/MedSec.
- The useful generic form is adaptive exact caching: sample, use cache only
  when repeats are high. This gives the high-repeat win without regressing
  low-repeat datasets.

## Codebase Tidy: Retired Attempts

After the reuse iteration, the top-level C implementation files were reduced to
the two best active variants:

- `src/tm_local_infer/tm_algorithm.c`: default stable single-file scorer.
- `src/tm_local_infer/tm_algorithm_17.c` +
  `src/tm_local_infer/tm_kernel_17.c`: best split-kernel candidate.

All other attempt sources were moved into:

```text
src/tm_local_infer/retired_attempts/
```

The archive has its own `README.md`, and the Makefile now includes:

```text
make -C src/tm_local_infer attempt17
```

for the best split-kernel build. Retired attempts can still be built via:

```text
make -C src/tm_local_infer clean all TM_ALGORITHM_SRC=retired_attempts/tm_algorithm_6.c
```

Validation:

- Default `tm_algorithm.c` build passed Python equivalence on all four datasets.
- `make attempt17` builds successfully after the move.
- A retired source (`retired_attempts/tm_algorithm_16.c`) builds successfully
  with the updated include path.

## Self-Contained Bundles: tm_v17 And tm_v19

Created two standalone C bundles:

```text
src/tm_local_infer/tm_v17/
src/tm_local_infer/tm_v19/
```

Each folder contains:

```text
tm_infer_c.c
tm_algorithm.h
tm_algorithm.c
Makefile
README.md
assets/
```

`tm_v17` also contains:

```text
tm_kernel_17.c
tm_kernel_17.h
```

The `assets/` trees are hardlinked local copies of the model/dataset assets, so
each bundle can be built and run from inside its own folder without referencing
the parent source tree.

Build commands:

```text
make -C src/tm_local_infer/tm_v17 clean all
make -C src/tm_local_infer/tm_v19 clean all
```

Run commands:

```text
./src/tm_local_infer/tm_v17/tm_infer_c src/tm_local_infer/tm_v17/assets /tmp/tm_v17_preds --cache-auto --profile
./src/tm_local_infer/tm_v19/tm_infer_c src/tm_local_infer/tm_v19/assets /tmp/tm_v19_preds --profile
```

Local validation:

- `tm_v17` builds and matches Python exactly on all four datasets.
- `tm_v19` builds and matches Python exactly on all four datasets.

Local timing snapshot:

| Bundle | WUSTL | NSLKDD | TonIoT | MedSec |
|---|---:|---:|---:|---:|
| `tm_v17` with `--cache-auto` | 0.088 | 0.160 | 0.084 | 0.177 |
| `tm_v19` | 0.209 | 0.936 | 0.785 | 0.500 |

## Bundle Harness Strip: tm_v17 And tm_v19

Stripped unused code from the self-contained bundle harnesses:

- `tm_v17/tm_infer_c.c`: removed leftover byte-posting calibration/stats paths
  and old plane profile branches. Cache/profile remain because v17 uses
  `tm_predict_current`.
- `tm_v19/tm_infer_c.c`: removed cache modes, byte-table/delta scoring
  calibration, old table/plane allocations, vote-block construction, and
  placeholder `tm_predict_current` profiling. The harness now reports the actual
  feature-state row scorer.

Line counts:

| File | Before strip | After strip |
|---|---:|---:|
| `tm_v17/tm_infer_c.c` | 2301 | 1823 |
| `tm_v19/tm_infer_c.c` | 2301 | 1043 |

Local validation after stripping:

- `make -C src/tm_local_infer/tm_v17 clean all`
- `make -C src/tm_local_infer/tm_v19 clean all`
- Both stripped bundles match Python exactly on WUSTL, NSLKDD, TonIoT, and
  MedSec.

Local timing snapshot after stripping:

| Bundle | WUSTL | NSLKDD | TonIoT | MedSec |
|---|---:|---:|---:|---:|
| `tm_v17` with `--cache-auto` | 0.085 | 0.152 | 0.083 | 0.163 |
| `tm_v19` | 0.211 | 0.956 | 0.765 | 0.500 |

## v19 API/Layout Strip

Removed another layer of v19-only leftovers:

- Removed unused `tm_binarize_row` and placeholder `tm_predict_current` from the
  v19 algorithm API.
- Removed the unused `current` argument from `tm_predict_row`.
- Removed old shared-layout fields from `tm_v19/tm_algorithm.h`: byte tables,
  postings, planes, vote blocks, calibration flags, nibble tables, and fallback
  literal-binarization arrays.
- Removed score-block `sign[]`, which v19 never read because it uses
  `pos_lanes`.

Line counts after this pass:

| File | Lines |
|---|---:|
| `tm_v19/tm_algorithm.h` | 64 |
| `tm_v19/tm_algorithm.c` | 155 |
| `tm_v19/tm_infer_c.c` | 1021 |

Validation:

- v19 builds cleanly with extra warnings:
  `-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wunused`.
- v19 still matches Python exactly on WUSTL, NSLKDD, TonIoT, and MedSec.

Local timing snapshot:

| Bundle | WUSTL | NSLKDD | TonIoT | MedSec |
|---|---:|---:|---:|---:|
| `tm_v19` | 0.208 | 1.140 | 0.767 | 0.558 |

## v19 Coverage-Guided Strip

Ran v19 with GCC coverage:

```text
CFLAGS='-O0 -g --coverage -std=c11 -Wall -Wextra -Wpedantic -march=native -DTM_ENABLE_SCORE_BLOCKS -DTM_ENABLE_FEATURE_STATE_TABLES -DTM_SCORE_BLOCKS_METADATA_ONLY'
LDLIBS='-lzstd --coverage'
```

Coverage runs included:

- full prediction output with `--profile-detail --stats`
- no prediction output with `--profile`
- `--help`

Useful uncovered code was limited to:

- scalar fallback backend, which is invalid for v19 because it returns a dummy
  class without AVX2/NEON scoring
- split positive/negative score-block fallback, which is not reached because the
  sorted v19 layout always forms one positive-then-negative class band
- error paths for malformed files/allocation failures

Removed:

- scalar backend reporting/fallback, replacing it with an explicit AVX2/NEON
  compile-time requirement
- split-range score-block construction and profile accounting

Validation:

- v19 normal optimized build passes.
- v19 extra-warning build passes.
- v19 matches Python exactly on WUSTL, NSLKDD, TonIoT, and MedSec.

Line counts after coverage-guided strip:

| File | Lines |
|---|---:|
| `tm_v19/tm_algorithm.h` | 64 |
| `tm_v19/tm_algorithm.c` | 160 |
| `tm_v19/tm_infer_c.c` | 1006 |

Local timing snapshot:

| Bundle | WUSTL | NSLKDD | TonIoT | MedSec |
|---|---:|---:|---:|---:|
| `tm_v19` | 0.205 | 0.973 | 0.861 | 0.491 |
