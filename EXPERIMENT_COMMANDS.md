# Experiment Command Registry

This registry is the canonical list of commands agents should use for local and
Raspberry Pi experiments. Use command IDs in result files so runs remain
traceable.

This belongs in the repository rather than a Codex skill because the commands
are project-specific and must evolve with `pyproject.toml`, `ansible/`,
`SPEC.MD`, and the active package layout.

## Rules

1. Run commands from the repository root unless a command says otherwise.
2. Record the command ID, date, platform, and output summary in a structured
   `results/YYYY-MM-DD_<slug>/` directory.
3. Use the smallest command that validates the claim.
4. Do not commit generated `.venv`, `__pycache__`, native binaries, shared
   libraries, `build/`, or `*.egg-info/`.
5. For RPI runs, keep connection details in `ansible/rpi.env`.
6. Benchmark result directories must include paired CSV and LaTeX tables under
   `tables/` and `latex/`, unless the registered result format says otherwise.

## Local Setup

| ID | Purpose | Command |
|---|---|---|
| `local.setup.editable` | Install editable package with all optional comparison tools. | `uv sync --all-extras` |
| `local.assets.ensure` | Extract bundled Fuzzy TM assets. | `uv run --with numpy --with numba --with zstandard python -c "from fuzzy_tm_infer.assets import ensure_assets; ensure_assets()"` |
| `local.assets.ensure_ml` | Extract Decision Tree assets. | `uv run --with '.[dt]' python -c "from fuzzy_tm_infer.assets import ensure_ml_models; ensure_ml_models()"` |
| `local.results.scaffold` | Create a strict result directory with manifest, CSV, LaTeX table, and raw log folder. | `uv run fuzzy-tm-result --slug native_v17_v19_local --title "Native v17/v19 Local Comparison" --format-id benchmark.table.v1 --experiment-type full_comparison --table-id latency --command-id local.exp.full_compare` |
| `local.results.render` | Render result CSV tables to LaTeX booktabs tables. | `uv run fuzzy-tm-result-render results/YYYY-MM-DD_<slug>` |
| `local.results.validate` | Validate structured result records and reject flat result files. | `uv run fuzzy-tm-result-validate` |

## Local Validation

| ID | Purpose | Command |
|---|---|---|
| `local.validate.compile` | Fast Python import/syntax validation. | `uv run --with '.[dt,rpi]' python -m compileall -q src/fuzzy_tm_infer` |
| `local.validate.native_exact` | Build v17/v19 and verify native predictions match Python. | `make -C src/fuzzy_tm_infer clean-native native-v17 native-v19 verify-native` |
| `local.validate.fast_glade` | Build fast GLADE v2 and verify transforms match Python. | `make -C src/fuzzy_tm_infer clean-fast-glade verify-fast-glade` |
| `local.validate.fast_glade_profile` | Build fast GLADE with macro-guarded profiling enabled. | `make -C src/fuzzy_tm_infer clean-fast-glade profile-fast-glade` |
| `local.validate.ansible_syntax` | Validate the RPI playbook syntax. | `uv run --with '.[rpi]' ansible-playbook -i ansible/inventory.rpi.example.ini ansible/playbooks/rpi_fuzzy_tm_infer.yml --syntax-check` |
| `local.validate.rpi_dry_run` | Check generated RPI Ansible command without connecting. | `PYTHONPATH=src uv run --with ansible python -m fuzzy_tm_infer.scripts.rpi_ansible --env-file ansible/rpi.env.example --host 127.0.0.1 --user rpi --dry-run` |
| `local.validate.package_build` | Build source and wheel distributions. | `uv run --with build python -m build --sdist --wheel --outdir /tmp/fuzzy_tm_infer_dist` |
| `local.validate.health` | Run compile, result validation, and stale-name checks. | `uv run fuzzy-tm-health` |

## Local Experiments

| ID | Purpose | Command | Record In |
|---|---|---|---|
| `local.exp.tm_python` | Python/Numba Fuzzy TM benchmark on bundled datasets. | `uv run --with numpy --with numba --with zstandard fuzzy-tm-infer` | `results/YYYY-MM-DD_tm_python/` |
| `local.exp.tm_vs_dt` | Python Fuzzy TM vs Decision Tree comparison. | `uv run --with '.[dt]' fuzzy-tm-infer --compare-dt` | `results/YYYY-MM-DD_tm_vs_dt/` |
| `local.exp.full_compare` | Python Fuzzy TM, native v17/v19, and Decision Tree comparison. | `uv run --with '.[dt]' fuzzy-tm-infer --compare-all` | `results/YYYY-MM-DD_full_compare/` |
| `local.exp.booleanizers` | Booleanizer fit/transform speed comparison. | `uv run --with numpy --with numba --with zstandard fuzzy-tm-infer --compare-booleanizers --repeats 3` | `results/YYYY-MM-DD_booleanizer_speed/` |
| `local.exp.native_scaffold` | Create a new native Fuzzy TM backend version from a promoted/reference base. | `make -C src/fuzzy_tm_infer scaffold-native VERSION=v20 BASE=v17` | `src/fuzzy_tm_infer/algorithms/c/fuzzy_tm/v20/README.md` |
| `local.exp.native_candidate` | Build and verify a candidate native backend against Python plus the retained references. | `make -C src/fuzzy_tm_infer native-v20 verify-native NATIVE_VERSIONS="v17 v19 v20"` | `src/fuzzy_tm_infer/algorithms/c/fuzzy_tm/PERFORMANCE.md` |
| `local.exp.native_v17_profile` | Direct v17 native profile/stat run. | `cd src/fuzzy_tm_infer/algorithms/c/fuzzy_tm/v17 && make clean all lib && ./tm_infer_c --profile --stats` | `results/YYYY-MM-DD_native_v17_profile/` |
| `local.exp.native_v19_profile` | Direct v19 native profile/stat run. | `cd src/fuzzy_tm_infer/algorithms/c/fuzzy_tm/v19 && make clean all lib && ./tm_infer_c --profile --stats` | `results/YYYY-MM-DD_native_v19_profile/` |

## Raspberry Pi Setup

Create the local env file:

```bash
cp ansible/rpi.env.example ansible/rpi.env
```

Fill in:

```text
TM_RPI_TARGET=rpi5
TM_RPI_HOST=<host-or-ip>
TM_RPI_USER=<ssh-user>
TM_RPI_PASSWORD=
TM_RPI_ARCHFLAGS=auto
```

The remote root and virtual environment are derived automatically.

## Raspberry Pi Validation

| ID | Purpose | Command |
|---|---|---|
| `rpi.validate.install_collections` | Install required Ansible collections locally. | `ansible-galaxy collection install -r ansible/requirements.yml` |
| `rpi.validate.deploy_native` | Sync repo, build native v17/v19, verify native exactness, run native profile. | `tm-rpi-deploy` |
| `rpi.validate.fast_glade` | Included in deploy: build and verify fast GLADE v2 on Pi. | `tm-rpi-deploy --no-native-benchmark` |
| `rpi.validate.full_compare` | Full Python/native/Decision Tree comparison on RPI. | `tm-rpi-deploy --compare-all` |
| `rpi.validate.python_only` | Run Python benchmark on RPI. | `tm-rpi-deploy --python-benchmark --no-native-benchmark` |
| `rpi.validate.no_assets` | Validate code/build while skipping asset sync. | `tm-rpi-deploy --no-assets --no-native-benchmark` |

## Raspberry Pi Experiments

| ID | Purpose | Command | Record In |
|---|---|---|---|
| `rpi.exp.native_profile` | Native v17 profile/stat run through Ansible. | `tm-rpi-deploy` | `results/YYYY-MM-DD_rpi_native_profile/` |
| `rpi.exp.full_compare` | RPI Python/native/Decision Tree comparison. | `tm-rpi-deploy --compare-all` | `results/YYYY-MM-DD_rpi_full_compare/` |
| `rpi.exp.booleanizers` | RPI booleanizer fit/transform speed comparison. | `tm-rpi-deploy --booleanizer-benchmark` | `results/YYYY-MM-DD_rpi_booleanizer_speed/` |
| `rpi.exp.arch_auto` | Validate automatic Pi 4/5 architecture flag selection. | `tm-rpi-deploy --no-native-benchmark` | `results/YYYY-MM-DD_rpi_arch_auto/` |
| `rpi.exp.arch_manual_a76` | Force Cortex-A76 build flags for comparison. | `tm-rpi-deploy --archflags "-mcpu=cortex-a76"` | `results/YYYY-MM-DD_rpi_arch_manual_a76/` |
| `rpi.exp.arch_manual_a72` | Force Cortex-A72 build flags for comparison. | `tm-rpi-deploy --archflags "-mcpu=cortex-a72"` | `results/YYYY-MM-DD_rpi_arch_manual_a72/` |

## Cleanup

| ID | Purpose | Command |
|---|---|---|
| `cleanup.generated` | Remove generated local artifacts. | `make -C src/fuzzy_tm_infer clean-native || true; rm -rf .venv build *.egg-info src/*.egg-info; find . -type d -name __pycache__ -prune -exec rm -rf {} +` |
| `cleanup.stale_names` | Search active files for stale pre-rename references. | `pattern="$(printf '%s|%s|%s|%s|%s' 'tm_''local_infer' 'tm-''local-infer' 'src/tm_''local' 'libtm_''local' '100\\.98')"; rg "$pattern" README.md SPEC.MD AGENTS.md EXPERIMENT_COMMANDS.md ansible src/fuzzy_tm_infer pyproject.toml .gitignore` |

## Result Record Template

Create the directory first:

```bash
uv run fuzzy-tm-result \
  --slug full_compare \
  --title "Full Comparison" \
  --format-id benchmark.table.v1 \
  --experiment-type full_comparison \
  --table-id latency \
  --command-id local.exp.full_compare \
  --command "uv run --with '.[dt]' fuzzy-tm-infer --compare-all"
```

Then fill:

```text
manifest.toml
README.md
tables/*.csv
latex/*.tex
raw/*.txt
```
