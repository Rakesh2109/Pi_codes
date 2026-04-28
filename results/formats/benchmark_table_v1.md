# `benchmark.table.v1`

General quantitative benchmark result.

Use this for timing, accuracy, profile, booleanizer, native-backend, and model
family comparison tables. The exact columns are determined by the experiment,
but every table must be paired:

```text
tables/<table_id>.csv
latex/<table_id>.tex
```

Required files:

```text
manifest.toml
README.md
tables/
latex/
raw/
```

Required manifest keys:

```toml
format_id = "benchmark.table.v1"
experiment_type = "full_comparison" # free-form controlled slug
tables = ["latency"]
primary_csv = "tables/latency.csv"
primary_latex = "latex/latency.tex"
```

Rules:

- Use explicit units in column names, for example `latency_us_per_sample`.
- Use one CSV per logical table.
- Every CSV table used in a report or paper must have a matching LaTeX table.
- LaTeX snippets must use `booktabs` and must not include a full preamble.
- Put raw command output under `raw/`.

Suggested `experiment_type` values:

```text
full_comparison
booleanizer_speed
native_profile
model_family_matrix
python_tm
tm_vs_dt
rpi_full_comparison
rpi_native_profile
```

## Example

```text
results/2026-04-29_full_comparison/
  manifest.toml
  README.md
  tables/
    latency.csv
    accuracy.csv
  latex/
    latency.tex
    accuracy.tex
  raw/
    local_exp_full_compare.txt
```

Example `manifest.toml`:

```toml
schema_version = "1"
format_id = "benchmark.table.v1"
experiment_type = "full_comparison"
date = "2026-04-29"
title = "Full Comparison"
command_id = "local.exp.full_compare"
command = "uv run --with '.[dt]' fuzzy-tm-infer --compare-all"
platform = "local x86_64"
status = "accepted"
exact = true
raw_dir = "raw"
tables = ["latency", "accuracy"]
primary_csv = "tables/latency.csv"
primary_latex = "latex/latency.tex"
```

Example `tables/latency.csv`:

```csv
dataset,python_us_per_sample,v17_us_per_sample,v19_us_per_sample,dt_us_per_sample,best_backend
WUSTL,0.618,0.177,0.213,0.042,dt
NSLKDD,1.272,0.142,0.951,0.118,dt
```

Example `latex/latency.tex`:

```tex
\begin{tabular}{lrrrrl}
\toprule
Dataset & Python & v17 & v19 & DT & Best \\
\midrule
WUSTL & 0.618 & 0.177 & 0.213 & 0.042 & DT \\
NSLKDD & 1.272 & 0.142 & 0.951 & 0.118 & DT \\
\bottomrule
\end{tabular}
```
