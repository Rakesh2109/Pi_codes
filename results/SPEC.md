# Results Spec

Every durable result must be reproducible, machine-readable, and paper-ready.
This spec is binding for new benchmark, analysis, and paper records.

The important change from the legacy layout is this:

```text
results/ is a registry of structured result records, not a dump folder.
```

## Directory Naming

Use:

```text
results/YYYY-MM-DD_<short_slug>/
```

Rules:

- Date is the local experiment date.
- Slug is lowercase ASCII with words separated by underscores.
- Do not overwrite an existing result directory.
- If a run is repeated, create a new result directory or put extra logs under
  `raw/`.

Examples:

```text
results/2026-04-28_native_v17_v19_local/
results/2026-04-28_booleanizer_speed/
results/2026-04-29_rpi5_native_profile/
results/2026-04-29_method_comparison_note/
```

## Common Layout

Every result directory must contain:

```text
manifest.toml
README.md
raw/
```

Most benchmark formats also contain:

```text
tables/
latex/
```

Optional artifact directories are allowed when the format declares them:

```text
figures/
models/
profiles/
```

Do not invent one-off top-level filenames. Add a format to
`results/formats/` when a new artifact shape is needed.

## Manifest

`manifest.toml` is the source of truth for provenance.

Required keys:

```toml
schema_version = "1"
format_id = "benchmark.table.v1"
date = "YYYY-MM-DD"
title = ""
command_id = ""
command = ""
platform = ""
status = "draft" # draft | accepted | rejected | archived
exact = false
raw_dir = "raw"
```

Benchmark-style records must also include:

```toml
experiment_type = "full_comparison"
tables = ["metrics"]
primary_csv = "tables/metrics.csv"
primary_latex = "latex/metrics.tex"
```

Use `exact = true` only when the result includes a correctness gate and it
passed.

## README

`README.md` must explain the result in human terms.

Required sections:

```text
# Title
## Summary
## Hypothesis
## Command
## Platform
## Correctness
## Artifacts
## Decision
```

## Table Artifacts

Benchmark and quantitative formats must provide paired CSV and LaTeX tables:

```text
tables/<table_id>.csv
latex/<table_id>.tex
```

Rules:

- Header row is required in CSV.
- Dataset-like columns should be named `dataset`.
- Timing units must be explicit in the column name, for example
  `latency_us_per_sample`.
- Ratios should be unitless and named with `_ratio` or `_speedup`.
- Boolean columns use `true` or `false`.
- Missing values are empty cells, not `n/a`.
- LaTeX tables must use `booktabs` commands: `\toprule`, `\midrule`,
  `\bottomrule`.
- LaTeX snippets must not include a full preamble.
- Escape underscores in LaTeX labels and table cells.

Minimal LaTeX table:

```tex
\begin{tabular}{lrrrr}
\toprule
Dataset & Baseline & Candidate & Speedup & Exact \\
\midrule
WUSTL &  &  &  &  \\
\bottomrule
\end{tabular}
```

## Raw Artifacts

Use `raw/` for command output and logs:

```text
raw/local_exp_full_compare.txt
raw/rpi_exp_native_profile.txt
raw/perf_stat.txt
```

Raw logs may be noisy. The top-level `README.md`, `tables/`, and `latex/`
must remain clean summaries.

## Format Registry

Supported result formats live in:

```text
results/formats/
```

Current format families:

| Format ID | Purpose |
|---|---|
| `benchmark.table.v1` | Quantitative benchmark tables with paired CSV and LaTeX artifacts. |
| `analysis.note.v1` | Non-executable analysis note with optional tables and supporting raw files. |

To add a new format:

1. Create `results/formats/<format_id_with_dots_replaced_by_underscores>.md`.
2. Define required files and artifact directories.
3. State whether CSV and LaTeX tables are mandatory.
4. Add the format to the table above.
5. Update `fuzzy-tm-result` if the format needs scaffold support.

Do not add a new benchmark format just because the columns differ. Use
`benchmark.table.v1` with a distinct `experiment_type` unless the artifact
layout itself changes.

## Result Lifecycle

Use one of these statuses:

- `draft`: incomplete or not yet validated.
- `accepted`: correctness and relevant benchmark gates passed.
- `rejected`: experiment ran but did not satisfy the hypothesis.
- `archived`: retained for history only.

Accepted or rejected experiments both remain useful evidence. Do not hide
regressions.

## Paper Compatibility

Every quantitative result intended for a paper, report, or slide must provide:

- a CSV table for scripted analysis;
- a LaTeX `booktabs` table;
- enough metadata in `manifest.toml` to cite the platform and command.

If a table appears in a paper draft, it must trace back to one structured result
directory or to `archives/results_legacy_2026-04-28/` while legacy material is
being migrated.
