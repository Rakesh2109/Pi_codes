# Booleanizer Speed Benchmark

## Summary

Legacy booleanizer speed measurements migrated into the structured results
layout. The benchmark compares GLADE, Standard, KBins, and Thermometer
booleanizers on the bundled dataset matrices.

## Hypothesis

Simpler threshold encoders should transform rows faster, while adaptive
booleanizers may produce fewer bits at higher fit cost.

## Command

Command ID:

```text
local.exp.booleanizers
```

Command:

```bash
uv run --with numpy --with numba --with zstandard fuzzy-tm-infer --compare-booleanizers --repeats 2
```

## Platform

```text
local
```

## Correctness

- Exact: false
- Validation command: `uv run fuzzy-tm-result-validate results/2026-04-28_booleanizer_speed`

This result is a speed-only booleanizer comparison; prediction exactness is not
applicable.

## Artifacts

```text
tables/booleanizer_speed.csv
tables/winners.csv
latex/booleanizer_speed.tex
latex/winners.tex
raw/booleanizer_speed_2026-04-28.md
```

## Decision

- accepted

Why: this is the first migrated benchmark record under the new results spec and
serves as a concrete example for future benchmark tables.
