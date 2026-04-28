# Results Index

This is the map of durable experiment evidence.

## Active Structured Results

| Result | Format | Status | Notes |
|---|---|---|---|
| `2026-04-28_gladev2_booleanizer_comparison/` | `benchmark.table.v1` | accepted | Fast GLADE v2 compared with Python GLADE, Standard, KBins, and Thermometer on local AVX2 and RPI5 NEON. |
| `2026-04-28_rpi_fast_glade_neon/` | `benchmark.table.v1` | accepted | RPI5 NEON Fast GLADE optimization history and final booleanizer timings. |
| `2026-04-28_booleanizer_speed/` | `benchmark.table.v1` | accepted | Migrated booleanizer speed benchmark with CSV and LaTeX tables. |
| `2026-04-28_fast_glade_v2_local/` | `benchmark.table.v1` | accepted | Fast GLADE v2 AVX2 transform exactness and WUSTL timing smoke check. |
| `2026-04-27_native_python_c_history/` | `benchmark.table.v1` | accepted | Migrated Python/C/native optimization history with extracted Markdown tables. |
| `2026-04-28_model_family_comparison/` | `benchmark.table.v1` | accepted | Migrated Pi-style model-family latency and macro-F1 matrices. |
| `2026-04-28_python_tm_inference/` | `benchmark.table.v1` | accepted | Migrated Python TM per-sample vs batched timing table. |
| `2026-04-28_python_tm_streaming/` | `benchmark.table.v1` | accepted | Migrated Python TM realistic streaming stage breakdown. |
| `2026-04-28_compiled_tm_all_datasets/` | `benchmark.table.v1` | accepted | Migrated compiled TM all-dataset summary and detail tables. |
| `2026-04-28_inference_comparison_note/` | `analysis.note.v1` | accepted | Migrated implementation comparison note. |
| `2026-04-28_method_comparison_note/` | `analysis.note.v1` | accepted | Migrated method comparison note. |
| `2026-04-28_slide_tables_note/` | `analysis.note.v1` | accepted | Migrated slide-table note. |
| `2026-04-28_learning_julia_note/` | `analysis.note.v1` | accepted | Migrated Julia learning note. |
| `2026-04-28_paper_outline_note/` | `analysis.note.v1` | accepted | Migrated IEEE paper outline note. |
| `2026-04-28_glade_fptm_tii_draft/` | `analysis.note.v1` | accepted | Migrated IEEE-style manuscript draft. |
| `2026-04-28_ieee_journal_draft/` | `analysis.note.v1` | accepted | Migrated older IEEE-style manuscript draft. |

Create new structured benchmark records with:

```bash
uv run fuzzy-tm-result --slug native_v17_v19_local \
  --title "Native v17/v19 Local Comparison" \
  --format-id benchmark.table.v1 \
  --experiment-type full_comparison \
  --table-id latency \
  --command-id local.exp.full_compare
```

The scaffold now creates benchmark-style artifacts under:

```text
tables/
latex/
raw/
```

## Legacy Archive

Flat pre-spec result files were moved to:

```text
archives/results_legacy_2026-04-28/
```

### Benchmark Logs And Tables

| File | Suggested Format | Notes |
|---|---|---|
| `booleanizer_speed_2026-04-28.md` | migrated | Migrated to `results/2026-04-28_booleanizer_speed/`. |
| legacy Python-vs-C native benchmark log | migrated | Migrated to `results/2026-04-27_native_python_c_history/`. |
| `full_comparison.txt` | migrated | Migrated to `results/2026-04-28_model_family_comparison/`. |
| `predict_time_table.txt` | migrated | Migrated to `results/2026-04-28_model_family_comparison/`. |
| `python_tm_inference.txt` | migrated | Migrated to `results/2026-04-28_python_tm_inference/`. |
| `python_tm_pi_realistic.txt` | migrated | Migrated to `results/2026-04-28_python_tm_streaming/`. |
| `tm_inference_all_datasets.txt` | migrated | Migrated to `results/2026-04-28_compiled_tm_all_datasets/`. |

### Analysis And Writing

| File | Suggested Format | Notes |
|---|---|---|
| `INFERENCE_COMPARISON.md` | migrated | Migrated to `results/2026-04-28_inference_comparison_note/`. |
| `METHOD_COMPARISON.md` | migrated | Migrated to `results/2026-04-28_method_comparison_note/`. |
| `SLIDES_TABLES.md` | migrated | Migrated to `results/2026-04-28_slide_tables_note/`. |
| `LEARNING_JULIA.md` | migrated | Migrated to `results/2026-04-28_learning_julia_note/`. |
| `PAPER_OUTLINE_IEEE.md` | migrated | Migrated to `results/2026-04-28_paper_outline_note/`. |
| `GLADE_FPTM_TII_final.tex` | migrated | Migrated to `results/2026-04-28_glade_fptm_tii_draft/`. |
| `bare_jrnl_new_sample4.tex` | migrated | Migrated to `results/2026-04-28_ieee_journal_draft/`. |

When a legacy result is still needed, migrate it into a structured result
directory instead of copying it back into the root of `results/`.
