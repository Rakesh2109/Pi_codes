# Legacy Results Archive

This folder contains pre-spec result files that used to live directly in
`results/`.

They were archived on 2026-04-28 when the project adopted the structured result
contract in:

```text
results/SPEC.md
```

Do not add new files here unless preserving old material. New experiments must
use the current scalable result contract with artifact directories:

```text
results/YYYY-MM-DD_<short_slug>/
  manifest.toml
  README.md
  tables/
  latex/
  raw/
```

See `results/SPEC.md` and `results/formats/` for the active rules.
