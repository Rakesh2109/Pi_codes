# Results

This folder contains durable experiment records for `fuzzy_tm_infer`.

Use one directory per experiment run:

```text
results/
  YYYY-MM-DD_<short_slug>/
    manifest.toml
    README.md
    tables/
    latex/
    raw/
```

The contract is strict:

- `manifest.toml` records provenance and validation status.
- `README.md` explains the hypothesis, command, platform, and decision.
- Benchmark formats provide paired machine-readable CSV tables in `tables/`
  and LaTeX-compatible `booktabs` tables in `latex/`.
- `raw/` contains unedited command output, profiler dumps, or device logs.

Start here:

```text
SPEC.md       result artifact contract
INDEX.md      current result index and archived legacy result map
formats/      registry of supported result artifact formats
templates/    copyable result templates
```

Legacy flat result files from before this structure were moved to:

```text
archives/results_legacy_2026-04-28/
```
