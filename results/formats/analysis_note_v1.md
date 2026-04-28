# `analysis.note.v1`

Narrative analysis, interpretation, paper planning, slide notes, or archived
draft material.

Use this when the primary artifact is an explanation rather than a benchmark
table.

## Required Layout

```text
results/YYYY-MM-DD_<slug>/
  manifest.toml
  README.md
  raw/
```

## Optional Layout

Use these only when the note needs them:

```text
tables/
latex/
figures/
```

If the note makes quantitative claims, provide paired tables:

```text
tables/<table_id>.csv
latex/<table_id>.tex
```

## Example

```text
results/2026-04-29_method_comparison_note/
  manifest.toml
  README.md
  raw/
    legacy_METHOD_COMPARISON.md
  tables/
    cost_breakdown.csv
  latex/
    cost_breakdown.tex
```

Example `manifest.toml`:

```toml
schema_version = "1"
format_id = "analysis.note.v1"
date = "2026-04-29"
title = "Method Comparison Note"
command_id = ""
command = ""
platform = "Raspberry Pi 5"
status = "draft"
exact = false
raw_dir = "raw"
```
