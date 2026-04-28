# Pi_codes — GLADE+FPTM Edge Inference Benchmark

End-to-end IoT intrusion-detection benchmark measuring per-sample inference latency,
model size, and classification accuracy for 16 models across 4 public datasets,
deployed on a **Raspberry Pi 5** (Cortex-A76, 2.4 GHz, single core).

The core model is the **GLADE + Fuzzy-Pattern Tsetlin Machine** (FPTM):
GLADE binarises continuous features into Boolean literals; the FPTM classifies
using bit-parallel clause evaluation. The trained artefact is a self-contained
`.fbz` file of 7–17 KB.

---


## Repository Layout

```
Pi_codes/
├── src/
│   ├── glade_v2.py              Python — GLADE binariser (training)
│   ├── fcm_bitmask_zstd.py      Python — FBZ writer + reader + FPTM trainer
│   ├── tm_infer.py              Python — TM Numba JIT inference benchmark
│   ├── tm_dt_bench.py           Python — TM vs DT Numba head-to-head
│   ├── ml_numpy_infer.py        Python — pure-numpy ML inference benchmark
│   ├── export_ml_to_npz.py      Python — sklearn .pkl → .npz exporter
│   ├── tm_fbz_infer.c           C      — portable FBZ reader (~150 lines)
│   ├── run_all.sh               Shell  — train all 4 datasets end-to-end
│   ├── run_ml_corr.sh           Shell  — ML correction pipeline
│   ├── train_all_datasets.jl    Julia  — FPTM training (legacy)
│   ├── inference_benchmark.jl   Julia  — inference benchmark (legacy)
│   └── *.jl                     Julia  — other legacy scripts
└── results/
    ├── GLADE_FPTM_TII_final.tex     IEEE TII journal paper
    ├── bare_jrnl_new_sample4.tex    Journal draft
    ├── full_comparison.txt          Full benchmark output
    └── tm_inference_all_datasets.txt  TM detailed breakdown
```

---

## Hardware

| Item | Detail |
|------|--------|
| Board | Raspberry Pi 5 |
| CPU | ARM Cortex-A76, 4 cores @ 2.4 GHz |
| Measurement | Single core, 1 sample at a time (streaming simulation) |
| Python | 3.13.5 |
| Numba | 0.65.0 |
| NumPy | Pure-numpy for ML baselines; Numba JIT for TM and DT kernels |

---

## Datasets

| Dataset | Records | Features | Classes | Domain |
|---------|--------:|--------:|-------:|--------|
| NSL-KDD | 148,517 | 41 | 5 | Network intrusion |
| TON-IoT | 211,043 | 44 | 10 | IoT multi-protocol |
| MedSec-25 | 554,534 | 78 | 5 | IoMT / medical network |
| WUSTL-EHMS-2020 | 16,318 | 44 | 3 | IoMT + biomedical |

All datasets use an 80/20 stratified split (seed=42), single-core evaluation.

---

## Per-Sample Inference Latency (µs) — Raspberry Pi 5

> ML baselines: pure-NumPy (no sklearn at inference time).  
> TM and DT Numba: `@njit(cache=True)` kernel with SWAR popcount.  
> 500-sample warm-up, 3000-sample timed loop. **Bold = best per dataset.**

| Model | Runtime | TON-IoT | MedSec-25 | WUSTL | NSL-KDD |
|-------|---------|--------:|----------:|------:|--------:|
| **GLADE+FPTM** | Numba JIT | 6.72 | 7.42 | **4.10** | 6.31 |
| **GLADE+FPTM** | Python numpy | 88.52 | 69.67 | 51.79 | 67.35 |
| **DecisionTree** | Numba JIT | **1.38** | **1.37** | 1.37 | **1.39** |
| DecisionTree | numpy | 6.96 | 6.63 | 7.16 | 8.29 |
| DecisionTree\_Lmatched | numpy | 8.60 | 8.91 | 10.96 | 15.06 |
| RandomForest | numpy | 847.97 | 891.24 | 840.31 | 887.45 |
| RandomForest\_Cmatched | numpy | 422.66 | 417.14 | 405.28 | 486.10 |
| LogisticRegression | numpy | 7.86 | 8.06 | 7.25 | 7.94 |
| LinearSVM | numpy | 7.86 | 8.22 | 7.31 | 8.05 |
| NaiveBayes | numpy | 24.30 | 23.96 | 20.54 | 26.91 |
| k-NN (k=5) | numpy | 21,675 | 96,873 | 1,018 | 45,862 |
| MLP\_tiny | numpy | 12.68 | 13.17 | 12.18 | 14.34 |
| MLP\_small | numpy | 18.81 | 19.45 | 18.10 | 21.16 |
| MLP\_med | numpy | 20.90 | 21.85 | 20.27 | 25.07 |
| MLP\_C | numpy | 15.48 | 16.53 | 14.27 | 21.26 |
| MLP\_2C | numpy | 18.23 | 21.56 | 17.14 | 30.12 |

**Numba JIT speed-up (FPTM):** 9.6×–12.6× over Python numpy on the same algorithm.  
**Throughput (Pi 5, single core):** 134,864–243,843 predictions/second (FPTM Numba).

---

## Model Size (KB)

> GLADE+FPTM: compressed `.fbz` (zstd-22 clause bitmasks, fully self-contained).  
> ML baselines: `.npz` arrays (no sklearn at inference time).

| Model | NSL-KDD | TON-IoT | MedSec-25 | WUSTL |
|-------|--------:|--------:|----------:|------:|
| **GLADE+FPTM (.fbz)** | **13.6** | **16.1** | **11.9** | **7.3** |
| DecisionTree | 44.8 | 85.5 | 70.6 | 19.7 |
| DecisionTree\_Lmatched | 11.4 | 9.9 | 7.3 | 7.1 |
| RandomForest | 8,490 | 14,735 | 15,735 | 3,562 |
| RandomForest\_Cmatched | 962 | 841 | 525 | 381 |
| LogisticRegression | 8.6 | 5.5 | 4.8 | 2.5 |
| LinearSVM | 8.6 | 5.5 | 4.7 | 2.4 |
| NaiveBayes | 13.2 | 8.5 | 6.8 | 3.0 |
| k-NN (k=5) | 110,463 | 51,444 | 190,624 | 2,857 |
| MLP\_tiny | 58.2 | 28.8 | 33.0 | 20.7 |
| MLP\_small | 115.4 | 55.6 | 65.8 | 43.5 |
| MLP\_med | 242.8 | 125.4 | 145.3 | 102.1 |
| MLP\_C | 360.9 | 166.2 | 161.7 | 70.3 |
| MLP\_2C | 1,205 | 915 | 702 | 350 |
| XGBoost | 1,266 | 2,508 | 1,701 | 615 |
| XGBoost\_Cmatched | 1,161 | 1,899 | 939 | 388 |

### FBZ internal size breakdown

| Dataset | N bits | Clauses | GLADE state | Uncompressed bitmask | Compressed |
|---------|-------:|--------:|:-----------:|:--------------------:|:----------:|
| NSL-KDD | 253 | 450 | 2.0 KB | 28 KB | 11 KB |
| TON-IoT | 122 | 1,000 | 1.0 KB | 32 KB | 15 KB |
| MedSec-25 | 351 | 400 | 2.7 KB | 34 KB | 9 KB |
| WUSTL | 247 | 180 | 1.9 KB | 11 KB | 5 KB |

### Can the .fbz size be reduced?

The clause bitmask (already at zstd level 22 = maximum) dominates the file.
Three practical approaches:

| Method | Saving | Tradeoff |
|--------|--------|---------|
| **Sparse literal indices** — store active literal indices (`int16`) instead of full bitmask. Saves ~40–60% when avg active literals < ⌈N/8⌉. | Large | Code change to reader/writer |
| **Reduce clause count (lower C)** — halving C roughly halves the compressed block | Medium | Slight accuracy drop |
| **float16 thresholds** — halves GLADE state (saves 0.5–1.4 KB) | Small | Negligible precision loss |

### What is inside a `.fbz` file?

```
[Header 20 B]     magic "FBZ1", N (GLADE bits), K (classes), total_clauses
[GLADE state]     feat_idx int32[N] + thresholds float32[N]  — the binariser
[String table]    feature names + class names (UTF-8, for rule decoding)
[zstd-22 block]   clause bitmasks (compressed)
                    per class × per polarity (positive / negative):
                      n_clauses  u16
                      per clause: clamp u8 | pos_mask uint8[⌈N/8⌉] | neg_mask uint8[⌈N/8⌉]
```

At inference: raw features → GLADE binarises to N bits → FPTM clause bitmasks
evaluated via bitwise AND + popcount → class vote → prediction.
Entire model fits in L1d cache (14–57 KB uncompressed on Cortex-A76).

---

## Classification Accuracy & Macro-F1

| Model | NSL Acc | NSL F1 | TON Acc | TON F1 | MedSec Acc | MedSec F1 | WUSTL Acc | WUSTL F1 |
|-------|:-------:|:------:|:-------:|:------:|:----------:|:---------:|:---------:|:--------:|
| **GLADE+FPTM** | **0.9951** | 0.9301 | 0.9703 | **0.9418** | 0.9707 | 0.8727 | 0.9442 | 0.7976 |
| XGBoost | 0.9956 | **0.9488** | 0.9775 | 0.9511 | 0.9768 | **0.8984** | 0.9406 | 0.7590 |
| RandomForest | 0.9926 | 0.8563 | 0.9765 | 0.9467 | **0.9767** | 0.8980 | 0.9320 | 0.6778 |
| DecisionTree | 0.9929 | 0.7965 | 0.9508 | 0.9210 | 0.9731 | 0.8824 | 0.9357 | 0.7216 |
| DecisionTree\_Lmatched | 0.9746 | 0.8956 | 0.7945 | 0.9510 | 0.9392 | 0.7203 | 0.9295 | 0.6936 |
| LogisticRegression | 0.9769 | 0.8580 | 0.7076 | 0.6347 | 0.9413 | 0.7763 | 0.9280 | 0.6430 |
| LinearSVM | 0.9655 | 0.8530 | 0.7114 | 0.6273 | 0.8538 | 0.5146 | 0.9286 | 0.6449 |
| NaiveBayes | 0.7567 | 0.4423 | 0.3875 | 0.3269 | 0.7673 | 0.4218 | 0.1293 | 0.3643 |
| k-NN (k=5) | 0.9894 | 0.8881 | 0.9742 | 0.9441 | 0.9733 | 0.8799 | 0.9375 | 0.7922 |
| MLP\_tiny | 0.9888 | 0.7369 | 0.7744 | 0.7088 | 0.9654 | 0.8440 | 0.9289 | 0.6507 |
| MLP\_small | 0.9901 | 0.7580 | 0.7983 | 0.7318 | 0.9646 | 0.8372 | 0.9280 | 0.6497 |
| MLP\_med | 0.9905 | 0.7173 | 0.8271 | 0.7734 | 0.9680 | 0.8590 | 0.9298 | 0.6603 |
| MLP\_C | 0.9923 | 0.7245 | 0.9097 | 0.8570 | 0.9661 | 0.8453 | 0.9375 | 0.7465 |
| MLP\_2C | 0.9924 | 0.7400 | 0.9352 | 0.8830 | 0.9660 | 0.8657 | **0.9485** | **0.8248** |

---

## How to Run

**1 — Export sklearn models to numpy (.npz):**
```bash
python src/export_ml_to_npz.py
```

**2 — Run ML pure-numpy inference benchmark:**
```bash
python src/ml_numpy_infer.py
```

**3 — Run TM Numba inference benchmark:**
```bash
python src/tm_infer.py
```

**4 — Run combined TM vs DT Numba head-to-head:**
```bash
python src/tm_dt_bench.py
```

Model files are expected at `/home/reddy/pi_zero2w_deploy/ml_models/{dataset}/`.

---

## Key Findings

| | GLADE+FPTM (Numba) | DecisionTree (Numba) |
|--|:-----------------:|:-------------------:|
| Latency (Pi 5) | 4.10–7.42 µs | 1.37–1.39 µs |
| Model size (.fbz) | 7.3–16.1 KB | 19.7–85.5 KB (.npz) |
| Best macro-F1 | 0.9418 (TON-IoT) | 0.9210 (TON-IoT) |
| Pi Zero 2W (projected) | 16–30 µs | ~5.5 µs |

GLADE+FPTM is 4–6× slower than DT Numba at inference but achieves higher macro-F1
on 3 of 4 datasets while storing the entire model in under 17 KB.
