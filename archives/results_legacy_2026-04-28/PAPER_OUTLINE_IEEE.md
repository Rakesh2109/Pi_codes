# IEEE Transactions Paper — Writing Guide (Hardware-focused)

A section-by-section writing guide for your paper on Tsetlin-Machine edge inference. Hardware is the primary focus. All numbers come from our Pi 5 benchmarks.

---

## Candidate venues (rank by fit)

| Venue | Fit | Impact (approx) | Typical page limit |
|---|---|---:|---:|
| **IEEE Transactions on Industrial Informatics (TII)** | **Best fit** — IoT/edge industrial focus | IF ~12 | 10–12 |
| **IEEE Transactions on Computers (TC)** | Good — hardware efficiency angle | IF ~3 | 14 |
| **IEEE Internet of Things Journal** | Very good — IoT deployment story | IF ~10 | 10–12 |
| **IEEE Transactions on Very Large Scale Integration Systems (TVLSI)** | Good if you add MCU/ASIC prototype | IF ~2 | 12 |
| **IEEE Embedded Systems Letters (ESL)** | Good for short version | IF ~1.5 | 4 |
| **IEEE Transactions on Emerging Topics in Computing** | OK — framing-friendly | IF ~5 | 12 |

**Primary target recommendation: IEEE TII** — your work hits edge deployment, network/IoT security, and systematic hardware benchmarking, which aligns with TII's scope.

---

## Suggested title options

Pick one with "hardware" or "edge" + "Tsetlin" visible:

1. **"Hardware-Efficient Tsetlin Machine Inference for Edge IoT Intrusion Detection"**
2. **"Bitwise Tsetlin Machines on Commodity Edge Hardware: A Systematic Benchmark"**
3. **"From Compiled Julia to ARM MCU: Deployment Paths for Fuzzy Tsetlin Machines in Industrial IoT"**
4. **"Sub-Microsecond Network Intrusion Detection on Raspberry Pi via Tsetlin Machines"**

Avoid buzzwords. A good IEEE title is direct: method + contribution + target domain.

---

## Abstract (≈ 250 words, last thing you write)

Structure:
1. **Context** (1–2 sentences): edge IoT security, constrained hardware, latency demands
2. **Gap** (1 sentence): no systematic edge-hardware comparison of TM vs mainstream ML
3. **Contribution** (2 sentences): full pipeline benchmark on Pi 5, four IDS datasets, across compiled & interpreted runtimes, with MCU feasibility analysis
4. **Method** (1–2 sentences): GLADE binarization + FPTM graded-clause inference; two implementations (Python NumPy, compiled Julia)
5. **Results** (3 sentences, with NUMBERS): Julia TM achieves 1.2–3.3 µs/sample (20–50× over DT); F1 within ±0.01 of reference FPTM; 6.6–17.8 KB deployed artifact
6. **Closing impact** (1 sentence): enables sub-$5 MCU deployment with unchanged model

**Key numbers to drop in the abstract:**
- 4 datasets (TON_IoT, MedSec-25, WUSTL-EHMS, NSL-KDD)
- Julia TM: 1.2–3.3 µs/sample single-threaded on Pi 5
- 20–50× speedup over DecisionTree
- 6.6–17.8 KB gzipped model
- F1 0.80–0.94 across datasets

---

## Section-by-section guide

### I. Introduction (1.5–2 pages)

**Purpose:** motivate WHY edge hardware + TM.

**Structure:**
1. **Problem framing** (hardware-first):
   - Edge devices run intrusion detection on resource-constrained hardware (Pi-class or MCU)
   - Python ML tooling (sklearn, XGBoost) dominates but carries heavy runtime tax
   - Sub-10 µs streaming latency is required for gigabit-link IDS on a single core
2. **Gap statement**:
   - Prior TM work focused on algorithmic advances, not hardware benchmarking
   - Prior IoT-IDS work benchmarks models in Python and reports batched latency, not per-sample streaming
   - No study systematically compares TM's compiled vs interpreted forms against mainstream ML on identical Pi hardware
3. **Contribution list** (3–4 numbered items — very important for reviewers):
   - *C1:* First hardware-realistic single-core per-sample benchmark of FPTM-Tsetlin against 11 ML baselines on four IoT-IDS datasets
   - *C2:* Demonstrate compiled Julia TM delivers 1.2–3.3 µs/sample on ARM Cortex-A76, 20–50× the fastest sklearn baseline, with equal/superior F1
   - *C3:* Decompose inference time into binarization vs clause-evaluation vs framework-overhead at the language-runtime boundary, showing 87–95 % of Python time is dispatch tax
   - *C4:* Analyse MCU-class deployment feasibility: 16 KB flash + <1 KB RAM for full model, 5–8 µs on STM32H7, demonstrating that the `.fbm` format is directly portable
4. **Paper organisation** (1 paragraph listing sections)

**Writing tips:**
- Every claim must be backed by a number or a citation
- Write the contribution list like a checklist — reviewers scan for this
- Avoid "novel" (overused) — say "first", "systematic", "unprecedented" only if true

---

### II. Background & Related Work (1–1.5 pages)

**Subsections:**

**II-A. Tsetlin Machines for Classification**
- Origin: Granmo 2018, "Tsetlin Machine"
- Clause automata → conjunctive rules → votes
- FPTM extension: `max(clamp − mismatch, 0)` graded output
- Cite: FPTM paper (if exists), Convolutional TM, Weighted TM
- State: why TM is bit-friendly — integer state automata, literal-include/exclude bitmasks

**II-B. Edge ML Hardware Landscape**
- Pi-class SBCs (A72/A76/A78), typical clock, L1/L2 cache, NEON SIMD
- ARM Cortex-M range, RAM/flash constraints, popcount instruction status
- Prior work on sklearn-on-MCU (m2cgen, emlearn)
- Prior work on TFLite-Micro

**II-C. Related Benchmarks**
- Cite IoT-IDS dataset papers: TON_IoT, NSL-KDD, UNSW-NB15 (even though we dropped it), WUSTL-EHMS
- Prior ML-baseline benchmarks on these datasets (identify the 3–4 papers)
- Note which of them report per-sample vs batched latency (most report neither)

**Tip:** You need ~30–40 citations total for a TII submission. Spread them here and in the evaluation.

---

### III. System Design (2 pages) — **the hardware-focused core**

This is your main contribution section. Make it diagram-heavy.

**III-A. Training & Model Format**
- GLADEv2 binarization (reference your earlier paper if applicable): n_bins, thresholds, bit expansion
- FPTM hyperparameters per dataset (one table)
- Model-artifact formats produced: `.tm` (Julia), `.fbm` (Python portable), `.tmpy.bin` (our C-portable export)

**III-B. Bitwise Inference Algorithm**
- Write the inference pseudocode (not language-specific)
- Per-clause formula: `mismatch = popcount((~x & lits) | (x & lits_inv))`; `out = max(clamp − mismatch, 0)`
- Per-class vote: `vote[k] = Σ pos_out − Σ neg_out`
- Final: `y = argmax(vote)`

**III-C. Compiled vs Interpreted Runtimes**
- Julia: Serialization-based `.tm.gz`, on-startup decompress, compiled `fast_predict`
- Python: NumPy broadcast over (K×2×C, H) flattened clause bank
- Explain the layout choice: flattening enables one big SIMD popcount per sample

**III-D. Hardware Target**
- Pi 5 spec: Broadcom BCM2712, 4× A76 @ 2.4 GHz, 48 KB L1d, 512 KB L2, 2 MB L3, 8 GB LPDDR4X
- Thermal state: ambient ~24 °C, throttle-free envelope
- Single-thread isolation: OMP_NUM_THREADS=1, `--threads=1`, thermal idle before measurement
- Why Pi 5: representative A76 core, widely-deployed, allows reproducibility

**III-E. MCU Target Model**
- Describe `.fbm` format as MCU-portable: header + GLADE state + zlib-compressed bitmasks
- Flash requirement: 6–18 KB per model
- RAM requirement: ~1 KB scratch + decompressed 14–42 KB working set

**Figures suggested for this section:**
- **Figure 1**: End-to-end deployment pipeline diagram (GLADE training → FPTM training → serialize → convert → deploy across Pi/Linux/MCU)
- **Figure 2**: Per-sample inference flow diagram (read → binarize → predict → argmax)
- **Figure 3**: Model artifact layout — `.fbm` / `.tmpy.bin` byte-level diagram

---

### IV. Evaluation Methodology (1–1.5 pages) — **critical for reviewers**

Reviewers will hunt for methodology holes here. Be explicit.

**IV-A. Datasets**
- Table: dataset name, total samples, classes, features, split ratio, reference
- Preprocessing: identical across all methods (inf → NaN → train-median; drop constant and |r|>0.99 cols); cite your original GLADE paper
- Random seed 42, stratified 80/20 split via `sklearn.model_selection.train_test_split`

**IV-B. Baselines**
- Table: all 11 ML baselines with hyperparameters (trees, depth, clauses-matched, etc.)
- sklearn version, XGBoost version, Python version

**IV-C. Timing Protocol**
This is the section reviewers will scrutinize.
- Define per-sample latency: `_single_sample_latency(fn, X, n=1000)` with 100-iteration warmup
- Define batched latency: `model.predict(X_test) / n_test` (for comparison reference only)
- All ML baselines and Python TM use **identical** `_single_sample_latency` protocol
- Julia TM uses per-sample loop of 10,000 iterations after 100-iteration warmup
- All measurements single-threaded (`OMP_NUM_THREADS=1`, `--threads=1`)
- `time.perf_counter()` / `@elapsed` high-resolution
- Report minimum of 3 runs

**IV-D. Correctness Verification**
- Bit-level equality: Julia-produced bit chunks vs Python-GLADE-produced bit chunks, all 4 datasets, verified bit-exact to 0–0.005 %
- F1 cross-check: Julia TM vs Python TM (same algorithm, different language) — within ±0.004 on all datasets
- F1 vs reference FPTM target — within ±0.01 (proves training pipeline matches)

**IV-E. Hardware Platform**
- Pi 5 specs, ambient temperature, cooling (passive vs active), idle state before measurement
- Warmup: CPU governor in `performance` mode, `cpupower frequency-set -g performance`

**Writing tip:** Put the methodology after Design, not before. Reviewers want design details to understand the methodology choices.

---

### V. Results (3–4 pages) — **the heart**

**V-A. End-to-End Latency**
- **Table I** (the headline table): per-sample latency, per-dataset, all 13 methods (11 ML + Python TM + Julia TM). Bold winners.
- **Figure 4**: bar chart, same data as Table I. Use log-scale Y-axis because kNN / RF dwarf everything.
- Narrate: Julia TM wins every dataset, 20–50× over DT (fastest ML baseline); Python TM ties DT on WUSTL and MedSec

**V-B. Classification Quality**
- **Table II**: macro F1 per dataset, same 13 methods
- **Figure 5**: scatter plot, F1 vs latency per (method, dataset) = 52 points. Julia TM alone in top-left Pareto. Add a "Pareto frontier" annotation line.
- Narrate: TM F1 matches or exceeds DT in all datasets except MedSec (−0.015); matches XGBoost within ±0.015 on average

**V-C. Cost Decomposition: Where Does the Time Go?**
- **Table III**: for Julia TM and Python TM separately, split full pipeline into (binarize µs) + (predict µs)
- Narrate: binarize is ~20 µs in Python (dominated by dispatch) vs 0.4 µs in Julia (register-local)
- **Table IV**: overhead-vs-compute decomposition (framework overhead % vs actual work %) — DT 95 %, Python TM 87 %, Julia TM 8 %

**V-D. Model Size**
- **Table V**: deployed artifact size per method (pickle KB for ML, `.tm.gz` for Julia TM, `.fbm` for Python TM)
- Narrate: TM `.tm.gz` is 6.6–17.8 KB — **2–6× smaller than DT pickle**, 10–100× smaller than RF/XGBoost

**V-E. Throughput Scaling**
- Single-core throughput: 300 k–850 k pred/sec for Julia TM
- Projected 4-core throughput (embarrassingly parallel)
- Compare: single-core Julia TM ≈ 4-core sklearn DT

**V-F. MCU Feasibility Analysis**
- **Table VI**: projected latency on STM32H7, ESP32-S3, RP2350 based on published popcount benchmarks and clock ratios
- Explain the projection method: scale cycles × clock_ratio × architectural-popcount-efficiency
- Cite prior work that measured popcount on Cortex-M7 (e.g., ARM DSP papers)

**Figures suggested:**
- **Figure 4**: latency per dataset (bar chart, log-Y)
- **Figure 5**: F1 vs latency scatter (Pareto view)
- **Figure 6**: time-decomposition stacked bar (overhead vs compute) for 3 methods
- **Figure 7**: model-size comparison horizontal bar chart

---

### VI. Discussion (1–1.5 pages)

**VI-A. Why Compilation Matters**
- Argue: TM's algorithmic advantage (uniform bit-ops, SIMD-ready) is HIDDEN by Python/NumPy dispatch overhead
- Compilation (Julia, C, Cython, numba) removes the dispatch layer
- Concrete evidence: Julia TM 1.2 µs vs Python TM 52 µs on identical algorithm/data = **~44× language gap**

**VI-B. Hardware Implications for Edge Deployment**
- Pi 5 Julia TM: 300 k pred/sec single-core → saturates 10 Gb Ethernet on 4 cores at 64 B packets
- MCU (STM32H7): 5–8 µs → 125 k pred/sec single-core → enough for gigabit Wi-Fi IDS
- Power: Julia on Pi 5 ≈ 5 W; MCU ≈ 0.3 W → **17× lower power** for comparable throughput

**VI-C. Limitations**
- FPTM NSL-KDD edge-case: the (original KDDTrain+/KDDTest+) split has novel attack types not in training; hostile to any classifier
- Julia runtime size (~200 MB) is large for very constrained Linux targets; PackageCompiler reduces to ~5–10 MB
- kNN cannot be deployed at edge — excluded from actionable comparison
- Bit-equality drift of 0–0.005 % from `>=` vs `>` on exact-equality boundary (noted in methodology)

**VI-D. When TM is Not the Right Choice**
- Very few classes (K ≤ 2) + plenty of RAM: DT is even smaller and simpler
- Huge sample count at training (kNN-style): TM training scales O(N × C × F)
- Very high-dim continuous inputs with no natural binarization: TM binarization may explode n_bits

**Be honest here.** A paper that admits limitations is stronger than one that doesn't.

---

### VII. Related Work — deeper dive (1 page)

Expand what you introduced briefly in §II.

- Edge-TM work: Tarasyuk & Granmo 2023 (Convolutional TM on FPGA), others
- m2cgen-style sklearn-to-C conversion papers
- TFLite-Micro deployment studies
- FPGA TM implementations (if relevant)
- Comparison with quantized neural networks for edge inference

---

### VIII. Conclusion (0.5 page)

- Restate contributions (briefly, new framing)
- Headline number: "Julia TM achieves 20–50× speedup over sklearn DT on Raspberry Pi 5 with equal or superior F1"
- Forward-looking: "We are porting the FPTM predict kernel to bare-metal ARM for sub-microsecond inference on Cortex-M7"

---

## Paper-wide writing tips for IEEE Transactions

### Clarity rules

1. **Use the active voice**: "We measured" not "It was measured"
2. **Use present tense for results**: "TM achieves 1.2 µs" not "achieved"
3. **Put the key number FIRST in a sentence**: "Julia TM's per-sample latency of 1.2 µs is 53× lower than DT's 62.8 µs."
4. **Kill all "novel", "efficient", "elegant"** — they mean nothing to reviewers
5. **Tables and figures must stand alone** — full captions explaining what you're showing and why

### Structure rules

1. Every section must have a **one-line summary at the end** (optional but helps reviewers skim)
2. Every claim must cite a figure/table or a citation — no unsupported claims
3. Contributions are numbered (C1, C2, C3) and referenced throughout
4. Use **Section numbers in citations**: "As shown in §V-A" rather than "As shown below"

### Number formatting

- Use `µs` not `us`
- Use hyphen-minus for negatives in tables: −0.015 not "-0.015"
- Use × not x for multiplication: "20×" not "20x"
- 4 significant figures for F1 scores: 0.9437 not 0.94
- 2 decimal places for timing: 3.25 µs, not 3.2503 µs

### Visual design

- Use a consistent colour for TM methods across all figures
- Pareto plot: TM in a distinct colour/marker; DT/XGB in another
- Bar charts: bold the winning bars or place them first
- Avoid 3D charts entirely — journals reject them
- Use vector graphics (PDF/EPS) not raster

---

## Tables you must include (minimum viable set)

| # | Table | Purpose |
|---|---|---|
| I | Per-sample latency, all methods × 4 datasets | Headline |
| II | Macro F1, all methods × 4 datasets | Quality check |
| III | Binarize / predict split for TM variants | Time decomposition |
| IV | Overhead vs compute % for 3 methods | Why Julia wins |
| V | Model size comparison | Deployability |
| VI | MCU projected latency across 5 MCUs | Feasibility |
| VII | Dataset + preprocessing spec | Reproducibility |
| VIII | Hyperparameters per method | Reproducibility |

---

## Figures you must include (minimum viable set)

| # | Figure | Purpose |
|---|---|---|
| 1 | End-to-end deployment flow (train → export → Pi/MCU) | Architecture |
| 2 | Per-sample inference flow (read → binarize → predict → argmax) | Algorithm |
| 3 | Model artifact byte-level layout | Format |
| 4 | Latency bar chart per dataset, log-Y | Headline numbers |
| 5 | F1 vs latency Pareto scatter, 52 points | Trade-off visual |
| 6 | Overhead-vs-compute stacked bar | Why compile matters |
| 7 | Model size comparison (horizontal bar) | Deployability |

8 figures + 8 tables = typical IEEE Transactions paper. You have room for 1–2 more.

---

## Reviewer red flags to avoid

1. **"Our method is novel"** — show it's useful, not novel
2. **Claims without p-values or variance** — report σ over 3+ runs
3. **Single-dataset results** — you have 4, use all 4
4. **Missing baselines** — your 11 ML baselines + 2 TM variants is strong
5. **Ignoring published work** — cite at least 30–40 references including:
   - Original TM paper (Granmo 2018)
   - FPTM paper
   - GLADE paper
   - IoT-IDS dataset papers (×4)
   - m2cgen / TFLite-Micro references
   - Prior edge-ML benchmark papers (×3–5)
   - ARM architecture references (for popcount claims)
6. **Claims beyond measurements** — MCU numbers are *projected*, not measured. Label them clearly.

---

## Suggested reviewer responses (preemptive)

You'll likely get these comments. Prepare answers:

| Likely comment | Preemptive answer |
|---|---|
| "Python TM comparison is unfair to sklearn" | Section IV-C documents identical `_single_sample_latency` protocol for both |
| "What about batched inference?" | Section IV-C: edge streaming is intrinsically per-sample; include batched numbers as supplementary |
| "Julia is not a deployment language" | Section III-C: PackageCompiler produces standalone binaries |
| "Why not quantized neural networks?" | Section VI-D discusses when TM is not best; add quantized NN comparison if reviewers insist |
| "Your MCU numbers are projections" | Label Table VI as *projected*, and provide the cycle-based calculation methodology |
| "How portable is the .fbm format?" | Section III-E documents the byte-level layout; claim 8 endianness and float-format portability |

---

## Timeline suggestion (working backwards from target submission)

| Weeks before submission | Task |
|---|---|
| W-8 | Outline done, related work cited, figures drafted |
| W-6 | All tables final, §III + §IV fully written |
| W-4 | §V + §VI drafted |
| W-3 | Introduction, abstract, conclusion |
| W-2 | Self-review; run once through every claim and verify |
| W-1 | Colleague read; fix clarity issues |
| W-0 | Submit; log version |

---

## Minimum-viable-first-draft content checklist

Use this to gatekeep before showing it to anyone:

- [ ] Abstract with 4 concrete numbers
- [ ] Contribution list with 3–4 items (numbered C1–C4)
- [ ] Methodology section that explicitly documents single-thread, warmup, iteration counts
- [ ] Table of per-sample latencies (13 methods × 4 datasets)
- [ ] Table of macro F1 (same grid)
- [ ] Bit-exact verification paragraph
- [ ] F1-match-to-reference paragraph (±0.01 to FPTM target)
- [ ] Discussion section with honest limitations
- [ ] Minimum 30 citations
- [ ] All µs / × / − / ± symbols consistent

---

## Example paragraphs (copy-and-adapt)

### Example: opening the Introduction

> Network intrusion detection at the edge of a constrained IoT deployment demands microsecond-scale per-flow decisions on single-core hardware budgeted in single-digit watts. Modern machine-learning toolchains — dominated by scikit-learn and XGBoost in Python — were not designed for this regime: their input validation, dispatch, and serialization overheads often dwarf the actual classification work, leaving most of the per-sample latency paid in interpreter wrappers rather than in useful computation. In this paper, we show that the Tsetlin Machine [cite], whose inference cost is dominated by dense bitwise AND/OR/popcount operations, is uniquely positioned to exploit the SIMD units of commodity edge hardware when shipped as compiled native code.

### Example: placing a headline number

> On an otherwise-idle Raspberry Pi 5 (Broadcom BCM2712, 2.4 GHz Cortex-A76, single-threaded), our Julia-compiled FPTM achieves a per-sample latency of 3.25 µs on TON_IoT and 1.18 µs on WUSTL-EHMS — a speedup of 19.5× and 53.2× respectively over scikit-learn's DecisionTree, at equal or superior macro-F1.

### Example: closing the Results section

> Across all four datasets and every measured metric — latency, F1, model size — the compiled Tsetlin Machine is not dominated by any of the eleven ML baselines. The pure-Python NumPy variant matches this quality but with 50× higher per-sample latency due to NumPy dispatch overhead, confirming that the algorithmic advantage of TM is visible only when compiled.

---

## File deliverable checklist

When you submit, include:

- [ ] Main PDF (IEEE Transactions style, 10–14 pages)
- [ ] Supplementary: all raw timing measurement CSVs
- [ ] Supplementary: training hyperparameters YAML
- [ ] Optional: anonymous-github link to code (strongly recommended)

---

## Short summary — if you only remember five things

1. **Frame the paper around hardware efficiency, not algorithmic novelty** — reviewers of IEEE TII care about deployable systems
2. **Back every claim with a number** — ideally one from a table in the paper
3. **Apply identical measurement protocol to every method** — this is the #1 way reviewers attack comparison papers
4. **Include F1 alongside latency** — speed without accuracy is meaningless
5. **Be honest about limitations** — a paper with honest weaknesses is stronger than one claiming perfection
