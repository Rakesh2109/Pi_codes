# RPI Fast GLADE NEON Booleanizer Benchmark

## Summary

Fast GLADE v2 was deployed and verified on Raspberry Pi 5. The playbook
auto-detected the board and built the native GLADE library with:

```text
ARCHFLAGS=-mcpu=cortex-a76
backend=neon
```

Fast GLADE was exact against the Python GLADE reference and was the fastest
transform path for all bundled datasets on the Pi.

After the first Pi run, the NEON mixed-feature fallback was changed from a
temporary 4-float gather buffer plus vector compare to direct scalar compares.
This preserves the exact threshold equation and improved the detailed Fast
GLADE kernel by roughly 20-30% on the measured datasets.

A second NEON optimization replaced four scalar lane extractions in same-feature
chunks with an in-vector mask reduction. This was also exact and improved the
headline RPI Fast GLADE transform timings further.

A third NEON optimization added a compact 4-lane chunk descriptor path. The
wrapper precomputes contiguous per-chunk feature indices and thresholds, and the
RPI NEON entry point iterates by chunk rather than recalculating bit offsets.
This was exact and improved every measured dataset again.

Two additional NEON experiments were rejected after measurement:

```text
1. Scalar comparisons for every 4-lane chunk regressed WUSTL and MedSec.
2. Fusing two same-feature 4-lane chunks for packed output regressed several packed runs.
```

## Command

```bash
PYTHONPATH=src uv run --with ansible python \
  -m fuzzy_tm_infer.scripts.rpi_ansible \
  --booleanizer-benchmark --no-native-benchmark
```

The run also exposed and fixed two RPI automation issues:

```text
1. The Ansible wrapper now passes derived remote root/venv paths explicitly.
2. The playbook uses shell for /proc/device-tree/model and explicit bool filters for run flags.
```

## Results

Primary table:

```text
tables/rpi_booleanizers.csv
```

Detailed Fast GLADE profile table:

```text
tables/fast_glade_detail.csv
```

LaTeX renderings are in `latex/`.

## Decision

Accepted. Fast GLADE NEON is exact and materially faster than the Python GLADE
transform path on Raspberry Pi 5.
