# Fuzzy TM Local Inference Spec

This document is the working contract for developing `fuzzy_tm_infer`.

## Goals

- Keep the Python package clean, importable, typed, and benchmarkable.
- Keep the active Fuzzy TM algorithms isolated under `fuzzy_tm_infer.algorithms`.
- Keep native C backends exact against the Python reference.
- Make every optimization iteration measurable on local x86 and Raspberry Pi 5.
- Record regressions honestly; correctness is the gate, speed is the target.

## Package Layout

```text
src/fuzzy_tm_infer/
  algorithms/
    py/
      fuzzy_tm_numba.py     Python/Numba reference Fuzzy TM algorithm
      decision_tree.py      Decision Tree comparison algorithm
    c/
      fuzzy_native.py       ctypes wrapper for native Fuzzy TM backends
      fast_glade.py       ctypes wrapper for fast GLADE v2 transform
      common/               shared C loading, CLI, ABI, metrics
      booleanizers/         native booleanizer kernels
      fuzzy_tm/
        v17/                fastest native backend
        v19/                feature-state/reference native backend
  booleanizers/
    glade.py                GLADE v2 numeric feature booleanizer
    kbins.py                KBins-style uniform/quantile booleanizer
    standard.py             TMU-style unique-value threshold booleanizer
    thermometer.py          fixed-threshold thermometer encoder
  benchmarks/
    runner.py               benchmark orchestration
  metrics/
    classification.py       evaluation metrics
  assets.py                 archive extraction and asset discovery
  data.py                   dataset/model path and loading helpers
  tm_infer.py               CLI entry point
```

Root project files:

```text
pyproject.toml              packaging, dependencies, CLI script, lint/type config
SPEC.MD                     this development contract
EXPERIMENT_COMMANDS.md      local/RPI command registry for experiments
ansible/                    Raspberry Pi deployment and test automation
results/                    structured experiment records and result spec
archives/                   retired code, legacy results, and source assets
```

## Public Imports

Preferred imports:

```python
from fuzzy_tm_infer.algorithms.py import TMModel, DecisionTreeModel
from fuzzy_tm_infer.algorithms.c import FuzzyTMModel
from fuzzy_tm_infer.booleanizers import (
    GLADEBooleanizer,
    KBinsBooleanizer,
    StandardBinarizer,
)
from fuzzy_tm_infer.metrics import macro_f1
```

Top-level convenience imports are allowed:

```python
from fuzzy_tm_infer import (
    GLADEBooleanizer,
    KBinsBooleanizer,
    StandardBinarizer,
    TMModel,
    FuzzyTMModel,
)
```

Avoid adding new compatibility shims at the package root unless there is a real
external caller to preserve.

## Dependency Policy

`pyproject.toml` is the source of truth.

Base dependencies are only for Fuzzy TM inference:

```text
numpy
numba
loguru
zstandard
```

Optional dependencies:

```text
.[dt]            Decision Tree comparison, includes scikit-learn
.[compression]   compression experiment tooling
.[dev]           development tools
.[all]           all optional groups
```

The package-local `requirements.txt` is only a convenience compatibility file.
Use `loguru` through `fuzzy_tm_infer._logging`; do not configure Python logging
independently in package modules.

## Local Development

Install editable package:

```bash
python3 -m pip install -e .
```

Install with Decision Tree comparison:

```bash
python3 -m pip install -e ".[dt]"
```

Run Python Fuzzy TM benchmark:

```bash
fuzzy-tm-infer
```

Run Python Fuzzy TM vs Decision Tree:

```bash
fuzzy-tm-infer --compare-dt
```

Run Python Fuzzy TM, native v17/v19, and Decision Tree:

```bash
fuzzy-tm-infer --compare-all
```

Run booleanizer fit/transform comparison:

```bash
fuzzy-tm-infer --compare-booleanizers
```

Direct source-tree commands are also supported:

```bash
uv run --with '.[dt]' fuzzy-tm-infer --compare-all
uv run --with '.[dt]' python -m compileall -q src/fuzzy_tm_infer
```

The full local/RPI command registry lives in `EXPERIMENT_COMMANDS.md`.

## Asset Rules

The package discovers source archives from `archives/source_assets/` first, with
repo-root and package-parent fallbacks:

```text
datasets.zip
tm_models.zip
ml_models.zip
```

`assets.py` extracts runtime assets on demand. Generated model/test assets must
not be committed unless that is a deliberate release decision.

To override archive discovery:

```bash
export FUZZY_TM_INFER_ARCHIVE_DIR=/path/to/archive_dir
```

## Results Rules

`results/` is no longer a flat dump folder. New durable results must follow:

```text
results/YYYY-MM-DD_<short_slug>/
  manifest.toml
  README.md
  tables/
  latex/
  raw/
```

Strict rules:

- Every publishable benchmark result must have paired CSV and LaTeX artifacts
  under `tables/` and `latex/`.
- Every result must declare command provenance in `manifest.toml`.
- Raw command output belongs in `raw/`, not in the result directory root.
- The root of `results/` is reserved for `README.md`, `SPEC.md`, `INDEX.md`,
  `templates/`, and structured result directories.
- Legacy result files belong in `archives/results_legacy_<date>/`.

Create a structured result directory with:

```bash
uv run fuzzy-tm-result \
  --slug native_v17_v19_local \
  --title "Native v17/v19 Local Comparison" \
  --format-id benchmark.table.v1 \
  --experiment-type full_comparison \
  --table-id latency \
  --command-id local.exp.full_compare
```

The detailed result contract lives in `results/SPEC.md`.

## Native C Backend Development

Active native backends:

```text
src/fuzzy_tm_infer/algorithms/c/fuzzy_tm/v17
src/fuzzy_tm_infer/algorithms/c/fuzzy_tm/v19
```

Shared C code:

```text
src/fuzzy_tm_infer/algorithms/c/common
```

Build locally:

```bash
make -C src/fuzzy_tm_infer clean-native native-v17 native-v19
```

Verify native predictions against the Python reference:

```bash
make -C src/fuzzy_tm_infer verify-native
```

Build and verify fast GLADE v2 transform:

```bash
make -C src/fuzzy_tm_infer clean-fast-glade verify-fast-glade
```

Build fast GLADE v2 with macro-guarded profiling:

```bash
make -C src/fuzzy_tm_infer clean-fast-glade profile-fast-glade
```

Run native v17 directly:

```bash
cd src/fuzzy_tm_infer/algorithms/c/fuzzy_tm/v17
make clean all lib
./tm_infer_c --profile --stats
```

Run native v19 directly:

```bash
cd src/fuzzy_tm_infer/algorithms/c/fuzzy_tm/v19
make clean all lib
./tm_infer_c --profile --stats
```

## C Optimization Iteration Loop

Use this loop for every native change:

1. State the hypothesis.

   Example: "Reorder selected byte rows to reduce cache misses in v17 scoring."

2. Make the smallest exact change.

   Keep write scope inside one backend unless shared code is required.

3. Build both backends.

   ```bash
   make -C src/fuzzy_tm_infer clean-native native-v17 native-v19
   ```

4. Verify exactness.

   ```bash
   make -C src/fuzzy_tm_infer verify-native
   ```

5. Benchmark locally.

   ```bash
   uv run --with '.[dt]' fuzzy-tm-infer --compare-all
   ```

6. Instrument before optimizing when the bottleneck is unclear.

   Use macro-guarded profiling counters for native kernels. Normal builds must
   keep instrumentation off by default; profile builds must expose enough
   counters to separate setup, validation, kernel work, and output packing.

   Fast GLADE example:

   ```bash
   make -C src/fuzzy_tm_infer clean-fast-glade profile-fast-glade
   ```

   ```python
   fast.profile_reset()
   fast.transform(X, pack_bits=True)
   print(fast.profile_snapshot())
   ```

7. Benchmark on Raspberry Pi 4/5.

   ```bash
   tm-rpi-deploy
   ```

8. Record outcome.

   Add timings, correctness status, CSV metrics, LaTeX table, and whether the
   attempt was promoted, retained as reference, or reverted.

## Native Profiling Contract

Native profiling must be explicit, reproducible, and cheap to disable.

Use this pattern for C kernels:

```text
normal build:    no timing overhead in hot loops
profile build:   compiled with a macro such as -DGLADE_V2_PROFILE
Python wrapper:  profile_enabled, profile_reset(), profile_snapshot()
```

Every profiled native component should expose stable field names. Prefer these
common fields when they apply:

```text
total_calls
rows
bits
output_bytes
validate_ns
setup_ns
kernel_ns
pack_ns
total_ns
errors
```

Instrumentation rules:

- Use monotonic time only in profile builds.
- Keep counters cumulative until `profile_reset()`.
- Count input work units, not only time. Examples: rows, literals, bytes,
  clauses, selected rows, output bytes.
- Separate setup/validation from the hot kernel when possible.
- Count error exits.
- Do not print from C hot paths. Expose counters through an ABI and let Python
  or the CLI format them.
- Validate profiled builds against the same Python reference as normal builds.
- Record profile runs in `results/` when they inform an optimization decision.

If a new profiler shape is needed, document it in the component README and add
a result format only if the existing `benchmark.table.v1` cannot represent it.

## Correctness Gates

These must pass before a C optimization can be considered:

```bash
uv run --with '.[dt]' python -m compileall -q src/fuzzy_tm_infer
make -C src/fuzzy_tm_infer verify-native
uv run --with '.[dt]' fuzzy-tm-infer --compare-all
```

Expected invariants:

- v17 and v19 predictions match Python Fuzzy TM exactly.
- Accuracy/F1 for Fuzzy TM are unchanged.
- Decision Tree metrics are comparison-only and do not gate Fuzzy TM changes.

## Raspberry Pi Development

Two supported paths exist.

Quick rsync path:

```bash
src/fuzzy_tm_infer/scripts/rsync_rpi5.sh --bench
```

Preferred reproducible path:

```bash
cp ansible/rpi.env.example ansible/rpi.env
tm-rpi-deploy --install-collections
```

Full comparison on the Pi:

```bash
tm-rpi-deploy --compare-all
```

The Ansible path:

- installs system build dependencies,
- syncs the repo to the Pi,
- creates a Python virtual environment,
- installs `fuzzy-tm-infer`,
- extracts assets,
- builds v17 and v19 with auto-detected Cortex-A72/A76 flags,
- verifies native predictions against Python,
- optionally runs native benchmarks and full Python/native/DT comparison.

Connection details live in `ansible/rpi.env`, which is ignored by git:

```text
TM_RPI_TARGET=rpi5
TM_RPI_HOST=
TM_RPI_USER=
TM_RPI_PASSWORD=
TM_RPI_ARCHFLAGS=auto
```

## Promotion Policy

Promote a native backend change only if:

- exactness passes on all bundled datasets,
- local x86 does not regress materially,
- Raspberry Pi 4/5 does not regress materially,
- code clarity remains acceptable.

If speed improves on one platform but regresses on another, keep it as an
explicit alternate backend or document the platform-specific tradeoff.
