# FPTM Inference — Language & Technique Comparison
## Raspberry Pi Zero 2W · aarch64 · Single-threaded · Per-sample

---

## Abstract

We benchmark Fuzzy Pattern Tsetlin Machine (FPTM) inference across four IoT security datasets, comparing four implementation strategies — pure numpy, Numba JIT, Julia LLVM, and native C — against fifteen classical ML baselines. All benchmarks run single-threaded on a Raspberry Pi Zero 2W (aarch64), measuring realistic streaming latency: one sample arrives, the model predicts, repeat.

---

## 1. Datasets

| Dataset | Test samples | Features | GLADE bits (N) | Classes (K) | Clauses/class |
|---------|:-----------:|:--------:|:--------------:|:-----------:|:-------------:|
| WUSTL-EHMS-2020  |   3 264 | 16 | 247 | 3  | 60  |
| NSL-KDD          |  29 704 | 41 | 253 | 5  | 90  |
| TON\_IoT Network |  42 209 | 44 | 122 | 10 | 100 |
| MedSec-25        | 110 907 | 79 | 351 | 5  | 80  |

---

## 2. TM Inference — All Implementations

### 2.1 Per-sample Latency (us) — Raspberry Pi 5 (Cortex-A76, 2.4 GHz, measured)

| Implementation | WUSTL | NSLKDD | TonIoT | MedSec | Notes |
|---------------|------:|-------:|-------:|-------:|-------|
| Python numpy (FBZEngine) | 39.4 | 51.3 | 67.4 | 53.7 | Vectorized, zero heap, xor trick |
| Python Numba (JIT)       |  5.40 |  7.30 | 13.92 |  9.01 | SWAR popcount, compiled loop |
| Julia (FBZ native)       |  3.12 |  6.43 |  8.08 |  7.79 | LLVM, ARM cnt instruction |
| C (gcc -O3 -march=native)|  2.32 |  4.82 |  9.85 |  6.24 | __builtin_popcountll -> ARM cnt |

### 2.2 Speedup over Python numpy

| vs Python | WUSTL | NSLKDD | TonIoT | MedSec | Avg |
|-----------|------:|-------:|-------:|-------:|----:|
| Numba     |  7.3x |   7.0x |   4.8x |   6.0x | 6.3x |
| Julia     | 12.6x |   7.9x |   8.3x |   6.9x | 8.9x |
| C         | 17.0x |  10.6x |   6.8x |   8.6x |10.8x |

### 2.3 Accuracy and F1

| Dataset | F1 (macro) | Accuracy | FBZ model size | Total clauses |
|---------|:----------:|:--------:|:--------------:|:-------------:|
| WUSTL   | 0.8055 | 94.45% |  7.3 KB | 360   |
| NSLKDD  | 0.9380 | 99.47% | 14.0 KB | 900   |
| TonIoT  | 0.9437 | 97.02% | 17.0 KB | 2 000 |
| MedSec  | 0.8694 | 97.02% | 12.0 KB | 800   |

> All four implementations produce **identical predictions** -- mathematically equivalent.

---

## 3. Optimization Techniques

### 3.1 Clause Bitmask Evaluation -- XOR Trick

The naive formula to count mismatches between input `x` and a clause:

```
mismatch = popcount( (~x & lits) | (x & inv) )   <- 4 ops per chunk
```

Using the identity `A ^ ((A^B) & x) = (~x & A) | (x & B)`:

```
xor_pre  = lits XOR inv                            <- computed once at load
mismatch = popcount( lits ^ (xor_pre & x) )        <- 2 ops per chunk
```

Halves the number of bitwise operations in the inner loop.
`xor_pre` stored alongside `lits` in the model -- zero inference cost.

### 3.2 Flat Clause Matrix

All K x 2 polarity groups flattened into one `(total_clauses, H)` matrix at model load.
Eliminates all Python-level loops over clauses. The entire predict step becomes:

```python
tmp   = np.bitwise_and(xor_pre, x)
val   = np.bitwise_xor(all_lits, tmp)
mism  = np.bitwise_count(val).sum(axis=1)
out   = np.maximum(all_clamp - mism, 0)
votes = np.bincount(all_class, weights=out * all_sign, minlength=K)
```

### 3.3 Pre-allocated Inference Buffers

```python
self._bits_buf = np.zeros(n_chunks * 64, dtype=np.uint8)
self._val_buf  = np.empty((total, H),    dtype=np.uint64)
self._mism_buf = np.empty(total,         dtype=np.int32)
```

Zero malloc() per inference call. Reduces GC pressure on 512 MB Pi RAM.

### 3.4 Numba JIT -- SWAR Popcount

```python
@njit(cache=True, nogil=True)
def _popcount64(v):
    v = v - ((v >> uint64(1)) & uint64(0x5555555555555555))
    v = (v & uint64(0x3333333333333333)) + ((v >> uint64(2)) & uint64(0x3333333333333333))
    v = (v + (v >> uint64(4))) & uint64(0x0F0F0F0F0F0F0F0F)
    return int32((v * uint64(0x0101010101010101)) >> uint64(56))
```

SWAR (SIMD Within A Register) -- counts bits using integer arithmetic inside one register.
Numba compiles the entire binarize + predict loop to ARM machine code.
First-call JIT compile ~30s; result cached to __pycache__.

### 3.5 C -- ARM cnt via -march=native

```c
static inline int pc64(uint64_t x) { return __builtin_popcountll(x); }
```

gcc -O3 -march=native on aarch64 emits the ARM NEON cnt instruction: one cycle per 64-bit word.
The flat C binary (*_c_model.bin) embeds xor_pre directly:

```c
for (uint32_t c = 0; c < m->total; c++) {
    int mism = 0;
    for (uint32_t ch = 0; ch < m->H; ch++)
        mism += pc64(lc[ch] ^ (xp[ch] & x[ch]));
    int out = m->clamp[c] - mism;
    if (out > 0) votes[m->cls[c]] += m->sign[c] * out;
}
```

---

## 4. ML Baseline Comparison

### 4.1 Per-sample Latency (us)

| Model | WUSTL | NSLKDD | TonIoT | MedSec |
|-------|------:|-------:|-------:|-------:|
| C TM         |   2.32 |   4.82 |   9.85 |   6.24 |
| Julia TM     |   3.12 |   6.43 |   8.08 |   7.79 |
| Numba TM     |   5.40 |   7.30 |  13.92 |   9.01 |
| Python TM    |  39.40 |  51.30 |  67.40 |  53.70 |
| DecisionTree |  62.77 |  62.98 |  63.30 |  69.45 |
| DT_Lmatched  |  62.58 |  62.72 |  63.33 |  70.74 |
| MLP_tiny     |  75.00 |  75.58 |  75.21 |  80.19 |
| MLP_C        |  77.92 |  86.35 |  79.29 |  86.57 |
| MLP_small    |  81.80 |  85.20 |  82.51 |  91.12 |
| LogisticReg  |  99.55 | 102.03 | 101.17 | 108.92 |
| LinearSVM    |  99.52 | 102.27 | 101.24 | 110.19 |
| NaiveBayes   | 171.32 | 227.93 | 346.00 | 249.91 |
| XGBoost_Cm   | 378.66 | 595.98 | 962.52 |1648.29 |
| XGBoost      | 413.54 | 647.09 |1132.78 |1702.11 |
| RF_Cmatched  |2362.65 |2672.36 |2417.93 |3762.96 |
| RandomForest |2900.36 |2931.29 |2960.25 |4435.13 |
| kNN_5        |2400.74 |36649.59|19718.18|74808.98|

### 4.2 Macro F1

| Model | WUSTL | NSLKDD | TonIoT | MedSec |
|-------|------:|-------:|-------:|-------:|
| TM (all impl) | 0.8055 | 0.9380 | 0.9437 | 0.8694 |
| XGBoost       | 0.8207 | 0.9488 | 0.9516 | 0.9000 |
| XGBoost_Cm    | 0.7936 | 0.9434 | 0.9498 | 0.8973 |
| DT_Lmatched   | 0.7731 | 0.8956 | 0.9510 | 0.8929 |
| kNN_5         | 0.8165 | 0.8881 | 0.9443 | 0.8799 |
| RandomForest  | 0.7285 | 0.8445 | 0.9478 | 0.8967 |
| DecisionTree  | 0.7608 | 0.7965 | 0.9210 | 0.8861 |
| RF_Cmatched   | 0.7187 | 0.5798 | 0.8232 | 0.8027 |
| LogisticReg   | 0.7201 | 0.8709 | 0.5819 | 0.6013 |
| LinearSVM     | 0.7214 | 0.8508 | 0.5810 | 0.5216 |
| MLP_C         | 0.6383 | 0.7245 | 0.8474 | 0.7359 |
| MLP_tiny      | 0.6447 | 0.7369 | 0.6674 | 0.4555 |
| NaiveBayes    | 0.3853 | 0.4449 | 0.3272 | 0.1945 |

---

## 5. TM Binarize vs Predict Breakdown (Python numpy)

| Dataset | Binarize us | Predict us | Total us | F1 | Acc |
|---------|:-----------:|:----------:|:--------:|:--:|:---:|
| WUSTL   | 20.13 | 31.67 | 51.79  | 0.8055 | 94.45% |
| NSLKDD  | 20.19 | 47.15 | 67.35  | 0.9380 | 99.47% |
| TonIoT  | 18.96 | 69.56 | 88.52  | 0.9437 | 97.02% |
| MedSec  | 21.39 | 48.29 | 69.67  | 0.8694 | 97.02% |

> Binarize cost is ~constant across datasets (~20 us) -- dominated by numpy dispatch overhead,
> independent of N. Predict scales with total_clauses x H.

---

## 6. Summary

|                  | C TM   | Numba TM | Julia TM | Python TM | DecisionTree | XGBoost |
|------------------|:------:|:--------:|:--------:|:---------:|:------------:|:-------:|
| Avg latency (us) |  5.8   |    8.9   |   6.4    |   53.0    |    64.6      |   971   |
| Avg F1           | 0.886  |  0.886   |  0.886   |  0.886    |   0.851      |  0.894  |
| Model size       | 50 KB  |  50 KB   |  50 KB   |  50 KB    |  varies      | varies  |
| Pi RAM           | <1 MB  |  <1 MB   |  <1 MB   |  <1 MB    |  varies      | varies  |
| Extra deps       | gcc    | numba    | Julia    | numpy     | sklearn      | xgboost |

**Key result:** FPTM achieves XGBoost-level F1 while running 10-200x faster per sample
in C/Julia/Numba on constrained edge hardware. All 4 models together occupy 50 KB on disk.
