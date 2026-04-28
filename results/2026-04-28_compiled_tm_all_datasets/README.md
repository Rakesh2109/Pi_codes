# Compiled TM All Datasets

## Summary

Migrated legacy result record from `archives/results_legacy_2026-04-28/`.

## Hypothesis

Legacy migration preserves the measured evidence and makes tabular data reusable as CSV and LaTeX.

## Command

Command ID:

```text
legacy.compiled_tm_all_datasets
```

Command:

```bash

```

## Platform

```text
Pi-style single-threaded
```

## Correctness

- Exact: false
- Validation command: `uv run fuzzy-tm-result-validate results/2026-04-28_compiled_tm_all_datasets`

## Artifacts

```text
tables/compiled_tm_summary.csv
tables/compiled_tm_detail.csv
latex/compiled_tm_summary.tex
latex/compiled_tm_detail.tex
raw/tm_inference_all_datasets.txt
```

## Decision

- accepted

Why: migrated from the pre-spec result archive so the evidence is indexed and validated under the current result contract.
