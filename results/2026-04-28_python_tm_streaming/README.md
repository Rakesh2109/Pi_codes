# Python TM Streaming

## Summary

Migrated legacy result record from `archives/results_legacy_2026-04-28/`.

## Hypothesis

Legacy migration preserves the measured evidence and makes tabular data reusable as CSV and LaTeX.

## Command

Command ID:

```text
legacy.python_tm_streaming
```

Command:

```bash

```

## Platform

```text
Pi-style single-threaded streaming
```

## Correctness

- Exact: false
- Validation command: `uv run fuzzy-tm-result-validate results/2026-04-28_python_tm_streaming`

## Artifacts

```text
tables/python_tm_streaming.csv
latex/python_tm_streaming.tex
raw/python_tm_pi_realistic.txt
```

## Decision

- accepted

Why: migrated from the pre-spec result archive so the evidence is indexed and validated under the current result contract.
