# Model Family Comparison

## Summary

Migrated legacy result record from `archives/results_legacy_2026-04-28/`.

## Hypothesis

Legacy migration preserves the measured evidence and makes tabular data reusable as CSV and LaTeX.

## Command

Command ID:

```text
legacy.model_family
```

Command:

```bash

```

## Platform

```text
Raspberry Pi 5 / Pi-style single-threaded
```

## Correctness

- Exact: false
- Validation command: `uv run fuzzy-tm-result-validate results/2026-04-28_model_family_comparison`

## Artifacts

```text
tables/latency_us_per_sample.csv
tables/macro_f1.csv
tables/python_tm_stage_breakdown.csv
tables/python_tm_vs_decision_tree.csv
latex/latency_us_per_sample.tex
latex/macro_f1.tex
latex/python_tm_stage_breakdown.tex
latex/python_tm_vs_decision_tree.tex
raw/full_comparison.txt
raw/predict_time_table.txt
```

## Decision

- accepted

Why: migrated from the pre-spec result archive so the evidence is indexed and validated under the current result contract.
