# Pi Codes

Current active package:

```text
src/fuzzy_tm_infer/
```

Development contract:

```text
SPEC.MD
```

Experiment command registry:

```text
EXPERIMENT_COMMANDS.md
```

Result records:

```text
results/README.md
results/SPEC.md
results/INDEX.md
```

Raspberry Pi deployment:

```bash
cp ansible/rpi.env.example ansible/rpi.env
tm-rpi-deploy --install-collections
```

Local benchmark:

```bash
uv run --with '.[dt]' fuzzy-tm-infer --compare-all
```

Archived legacy/root files and source asset archives live under:

```text
archives/
```

Legacy top-level scripts that used to sit directly under `src/` are archived in:

```text
archives/src_legacy_2026-04-28/
```
