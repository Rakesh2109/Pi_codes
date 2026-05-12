# GLADE + FPTM — Lightweight IoT Intrusion Detection

Code for the paper *"GLADE: Gap-Aware Binarisation for Lightweight Tsetlin-Machine
Intrusion Detection at the IoT Edge"*. GLADE turns numerical traffic features into
compact Boolean literals; a Fuzzy Pattern Tsetlin Machine (FPTM) then classifies
with bitwise logic. Datasets: NSL-KDD, TON_IoT, MedSec-25, WUSTL-EHMS-2020.

## Layout

```text
src/paper_2/            Python package (run modules with `python -m paper_2.<name>` from src/)
  config.py             single source of truth: datasets, FPTM hyper-params, paths (env-overridable)
  data_loader.py        loaders + the unified preprocessing pipeline
  booleanizers/         binarisers: glade.py (GLADE / GLADEv2), standard.py, kbins.py, thermometer.py
  glade_v2.py           the GLADE binariser (3 stages: adaptive budget -> gap-aware placement -> local refinement)
  train_tm.jl           Julia FPTM trainer (called by the runners; hyper-params come from env vars)
  run_tm.py             train GLADE+FPTM on all datasets -> results/<ds>/models/model.pkl + reports
  run_ml.py             ML baselines (XGBoost, RandomForest, kNN, LogReg, LinearSVM, GaussianNB)
  run_ml_tmmatched.py   capacity-matched tree baselines (n_estimators = clause count C)
  run_tinyml.py         small MLP baselines
  run_all.py / run_any.py        orchestrators
  run_binarizer_comparison.py    FPTM trained on GLADE / StandardBinarizer / Thermometer (paper Table III)
  glade_ablation.py     turn each GLADE stage off, retrain (component ablation)
  tm_inference.py       numba-JIT FPTM predict path (shared by the inference benchmarks)
  fbz.py                the FBZ container (GLADE thresholds + FPTM clause bitmasks + zstd)
  bench_inference*.py   per-sample inference latency
  bench_codecs*.py / bench_compression.py   compression sweep over codecs
  verify_fbz.py / verify_tm_inference.py    correctness checks (byte-exact round-trips)
  plot_*.py             figure generation
  atlas/                TMAtlas-style structured export of a trained GLADE+FPTM model (tool)
  ieee_tii_v2.tex, references.bib, figures/   the paper sources (also mirrored in ../overleaf_upload/)
pi_bench/               Raspberry-Pi-5 binarisation/inference latency benchmark (`pi_bench/assets/` is gitignored)
results/                generated reports / tables / metrics JSON (model binaries are gitignored)
archives/               retired scripts and legacy outputs
```

## Requirements

```bash
pip install -r requirements.txt          # numpy, scikit-learn, xgboost, numba, zstandard, brotli
```

- **Julia** (>= 1.10) is required for FPTM training. `train_tm.jl` `include`s the
  FuzzyPatternTM source — default `/IoT/FuzzyPatternTM/src/Tsetlin.jl`; override with
  `export PAPER2_TSETLIN_PATH=/path/to/FuzzyPatternTM/src/Tsetlin.jl`. Julia needs the
  `JSON` package: `julia -e 'using Pkg; Pkg.add("JSON")'`.
- **Datasets** are not in this repo. Place them under `PAPER2_DATA_ROOT` (default `/IoT/Datasets`):
  `WUSTL/wustl-ehms-2020_with_attacks_categories.csv`, `TON_IoT/ton_iot_network.csv`,
  `medsec/MedSec-25.csv`, and NSL-KDD (`PAPER2_NSLKDD_ROOT`, defaults to the kagglehub cache path).

```bash
export PAPER2_DATA_ROOT=/path/to/datasets        # default /IoT/Datasets
export PAPER2_NSLKDD_ROOT=/path/to/nslkdd
export PAPER2_TSETLIN_PATH=/path/to/FuzzyPatternTM/src/Tsetlin.jl
# optional: export PAPER2_TMP_DIR=/tmp/glade_benchmark
```

All commands below are run from `src/`.

## Reproduction commands

### Main classification benchmark (paper Tables IV–V)

```bash
cd src

# GLADE + FPTM on all four datasets  ->  results/<ds>/models/model.pkl, model.fbz, tm_rules.json + reports
python -m paper_2.run_tm

# Conventional ML baselines (library-default hyper-params)
python -m paper_2.run_ml

# Capacity-matched tree baselines (n_estimators = clause count C)
python -m paper_2.run_ml_tmmatched

# Small-MLP "TinyML" baselines
python -m paper_2.run_tinyml

# Or run everything (long; needs Julia for the TM part)
python -m paper_2.run_all                       # add --skip-tm / --skip-ml / --skip-tinyml / --skip-ml-tmmatched
```

One dataset / a subset of models:

```bash
python -m paper_2.run_any --task tm  --dataset nslkdd
python -m paper_2.run_any --task ml  --dataset medsec --models XGBoost,RandomForest
python -m paper_2.run_any --task tinyml --dataset wustl --models MLP_tiny
```

### Binariser comparison — FPTM with 3 binarisers (paper Table III)

```bash
cd src
python -m paper_2.run_binarizer_comparison                  # GLADE / StandardBinarizer / Thermometer x 4 datasets
python -m paper_2.run_binarizer_comparison --dataset wustl  # one dataset, all 3 binarisers
python -m paper_2.run_binarizer_comparison --binariser glade
python -m paper_2.run_binarizer_comparison --render-latex   # print Table III from cached results
# results -> results/binarizer_comparison/<dataset>_<binariser>.json
```

### GLADE component ablation (which stage matters)

```bash
cd src
for ds in wustl nslkdd ton_iot medsec; do
  for v in full no_budget no_gap no_perturb; do
    python -m paper_2.glade_ablation --dataset $ds --variant $v --out ../results/ablation/${ds}_${v}.json
  done
done
# summary table -> results/ablation/ablation_table.tex ; see results/ablation/README.md
```

### Serialization & compression (paper §V-C, §V-D)

```bash
cd src
python -m paper_2.save_uncompressed_bundle      # build results/<ds>/models/model.pkl bundles
python -m paper_2.bench_compression             # pickle vs FBZ (zstd-1) sizes
python -m paper_2.bench_codecs_sweep            # full codec/level sweep -> results/SUMMARY_codecs_sweep.json
python -m paper_2.verify_fbz                    # byte-exact round-trip: model.fbz preds == tm_rules.json preds
python -m paper_2.fbz pack   ../results/wustl/models/tm_rules.json --glade ../results/wustl/models/glade.json -o out.fbz
python -m paper_2.fbz unpack out.fbz
```

### Inference latency (host)

```bash
cd src
python -m paper_2.bench_inference_all           # mean per-sample latency, all models/datasets -> results/SUMMARY_inference_all.json
python -m paper_2.verify_tm_inference           # FPTM predict path matches the Julia reference
```

### Raspberry Pi 5 binarisation/inference latency (edge-latency figure)

Run on the Pi itself (needs numpy / scikit-learn / xgboost / numba / zstandard there). See `pi_bench/`:

```bash
python pi_bench/bench_pi.py                      # uses pi_bench/assets/ (test sets + model bundles, gitignored)
```

### Figures

```bash
cd src
python -m paper_2.plot_storage_bar
python -m paper_2.plot_codec_sweep
python -m paper_2.plot_inference
```

### Inspect a trained model (atlas tool)

```bash
cd src
python -m paper_2.atlas ../results/medsec/models/model.pkl --dataset medsec --out medsec_atlas.json --html medsec_atlas.html
python -m paper_2.atlas serve                    # serves results/atlas/ -> http://localhost:8000/viewer.html
```

## Notes

- `config.py` holds the dataset list and the tuned per-dataset FPTM hyper-parameters
  (`TM_PARAMS_PER_DATASET`, paper Table II); all runners read it.
- All randomness uses `random_state = 42` (scikit-learn) / `TM_SEED = 42` (Julia FPTM).
- Generated model binaries (`*.pkl`, `*.fbz`, `*.zst`), zip archives, `pi_bench/assets/`,
  and JIT caches are git-ignored — regenerate them with the commands above. Text results
  (metrics JSON, CSV tables, LaTeX tables, reports) under `results/` are tracked.
- `archives/` holds retired code (e.g. the superseded, since-broken `compare_binarizers_chosen.py`).
