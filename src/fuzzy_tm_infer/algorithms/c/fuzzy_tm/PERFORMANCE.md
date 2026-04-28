# Native Fuzzy TM Performance

This file is the compact performance ledger for active native Fuzzy TM backend
versions. Detailed logs and rejected attempts belong in structured
`results/YYYY-MM-DD_<slug>/` directories.

## Current Summary

Latest local full comparison command:

```bash
uv run --with '.[dt]' fuzzy-tm-infer --compare-all
```

Local timing from the latest recorded package comparison run after the
`algorithms/c/fuzzy_tm/vNN` layout move:

| Dataset | Python us | v17 us | v19 us | DT us | Python F1 | v17 F1 | v19 F1 | DT F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| WUSTL | 0.618 | 0.177 | 0.213 | 0.042 | 0.8157 | 0.8157 | 0.8157 | 0.7608 |
| NSLKDD | 1.272 | 0.142 | 0.951 | 0.118 | 0.9477 | 0.9477 | 0.9477 | 0.7965 |
| TonIoT | 1.583 | 0.137 | 0.755 | 0.047 | 0.9427 | 0.9427 | 0.9427 | 0.9210 |
| MedSec | 1.477 | 0.162 | 0.503 | 0.066 | 0.8692 | 0.8692 | 0.8692 | 0.8861 |

Direct native CLI timing from the source-tree version folders:

```bash
cd src/fuzzy_tm_infer/algorithms/c/fuzzy_tm/v17 && ./tm_infer_c --profile --stats
cd src/fuzzy_tm_infer/algorithms/c/fuzzy_tm/v19 && ./tm_infer_c --profile --stats
```

| Dataset | v17 us | v17 score/profile us | v19 us | v19 row/profile us |
|---|---:|---:|---:|---:|
| WUSTL | 0.088 | 0.044 | 0.253 | 0.249 |
| NSLKDD | 0.163 | 0.093 | 0.937 | 0.890 |
| TonIoT | 0.138 | 0.106 | 0.796 | 0.803 |
| MedSec | 0.191 | 0.104 | 0.512 | 0.517 |

Interpretation:

- `v17` is the current default native Fuzzy TM backend.
- `v19` remains exact but is a reference/backend-design comparison, not the
  default speed path.
- Decision Tree is often faster as a kernel, but Fuzzy TM has different
  accuracy/F1 tradeoffs and deployment characteristics.

## Historical RPI Reference

Historical Raspberry Pi 5 NEON compact-run medians from the old iteration log
show `v17` as the strongest promoted path:

| Dataset | v17 RPI5 us | Notes |
|---|---:|---|
| WUSTL | 0.574 | exact |
| NSLKDD | 0.883 | exact |
| TonIoT | 0.708 | exact |
| MedSec | 0.855 | exact |

Run fresh RPI numbers with:

```bash
tm-rpi-deploy --compare-all
```

## Record Template

```markdown
## YYYY-MM-DD - vNN <short hypothesis>

Command ID:

Command:

Platform:

Correctness:

| Dataset | Baseline us | Candidate us | Speedup | Exact |
|---|---:|---:|---:|---|
| WUSTL | | | | |
| NSLKDD | | | | |
| TonIoT | | | | |
| MedSec | | | | |

Decision:

- promote / keep as reference / archive / revert
```
