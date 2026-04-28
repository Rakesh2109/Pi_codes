# Agent Development Contract

This repository is developed by spec-driven, design-science iteration. Agents
must treat this file and `SPEC.MD` as binding instructions before changing code.

## Non-Negotiable Rules

1. Read `SPEC.MD` before making project changes.
2. Keep `src/fuzzy_tm_infer/` as the only active importable package under `src/`.
3. Keep experiments measurable. Every algorithmic change needs a hypothesis,
   a validation command, and an honest result.
4. Do not leave unused code in the active package. Retire, archive, or delete
   dead paths explicitly.
5. Do not hide regressions. Record speed regressions and correctness failures
   in `results/` or the relevant docs.
6. Do not hardcode machine-specific paths, Raspberry Pi addresses, users, or
   passwords. Use env files, examples, or automatic detection.
7. Do not introduce new top-level scripts in `src/`. Put package code inside
   `src/fuzzy_tm_infer/`, operational scripts in `src/fuzzy_tm_infer/scripts/`,
   and retired scripts in `archives/`.
8. Keep public imports deliberate. Avoid compatibility shims unless there is a
   real caller to preserve.
9. Prefer small exact changes over broad rewrites. If a rewrite is justified,
   preserve a working baseline until the replacement is validated.
10. Before finalizing a change, run the smallest validation suite that proves
    the claim being made.

## Design-Science Loop

Every substantial change must follow this loop:

1. Problem

   State the concrete problem or bottleneck. Example: "v17 scoring is exact but
   memory-bound on Raspberry Pi 5."

2. Artifact

   Identify the artifact being changed: Python booleanizer, native backend,
   benchmark, deployment automation, packaging, or documentation.

3. Hypothesis

   Make the claim testable. Example: "A quantile KBins booleanizer will reduce
   bit count and transform time relative to GLADE on bundled datasets."

4. Design

   Explain the intended structure before or while implementing. Keep the design
   aligned with `SPEC.MD` and the existing package layout.

5. Evaluation

   Run targeted tests and benchmarks. Correctness is the hard gate; performance
   is evidence, not belief.

6. Learning

   Record what happened. Promote, keep as reference, or deprecate the artifact.

## Active Repository Shape

The active package is:

```text
src/fuzzy_tm_infer/
```

Current package responsibilities:

```text
algorithms/py/           Python/Numba Fuzzy TM and Decision Tree implementations
algorithms/c/            Native Fuzzy TM backends and ctypes wrappers
algorithms/c/fuzzy_tm/v17/        Fastest current native backend
algorithms/c/fuzzy_tm/v19/        Alternative/reference native backend
booleanizers/            GLADE, KBins, TMU-style Standard, Thermometer encoders
benchmarks/              Reproducible benchmark entry points
metrics/                 Classification metrics
assets.py                Archive discovery and extraction
data.py                  Dataset/model loading helpers
tm_infer.py              CLI entry point
scripts/                 Operational scripts and verification utilities
tools/                   Non-core experiments/utilities
docs/                    Package-local design notes
```

Root responsibilities:

```text
pyproject.toml           Packaging, dependency groups, scripts, lint/type config
SPEC.MD                  Project specification and development workflow
AGENTS.md                This agent contract
README.md                Short orientation
ansible/                 Raspberry Pi deployment/test automation
results/                 Structured benchmark records and result spec
archives/                Retired source, legacy docs, source asset archives
EXPERIMENT_COMMANDS.md   Local/RPI command registry for validation and experiments
```

## Spec-Driven Change Protocol

Before changing files:

1. Locate the relevant spec/doc/code owner.
2. Identify whether the request changes API, behavior, speed, packaging, or
   only documentation.
3. Choose a narrow write scope.
4. Decide the validation command before implementing.
5. Prefer a registered command from `EXPERIMENT_COMMANDS.md`; add one when a
   new repeatable experiment workflow is created.

During implementation:

1. Follow existing local style.
2. Keep new files in the correct package area.
3. Avoid speculative abstractions.
4. Keep comments short and useful.
5. Update docs when public commands, imports, layout, or workflow change.

After implementation:

1. Run targeted validation.
2. Remove generated artifacts such as `.venv`, `__pycache__`, native binaries,
   shared libraries, `build/`, and `*.egg-info/`.
3. Check for stale names and hardcoded paths.
4. Summarize changed files, validation, and residual risks.

Useful stale-name sweep:

```bash
pattern="$(printf '%s|%s|%s|%s|%s' \
  'tm_''local_infer' \
  'tm-''local-infer' \
  'src/tm_''local' \
  'libtm_''local' \
  '100\\.98')"
rg "$pattern" \
  README.md SPEC.MD AGENTS.md ansible src/fuzzy_tm_infer pyproject.toml .gitignore
```

## Validation Gates

Use the smallest gate that proves the change.

The canonical local/RPI command list is `EXPERIMENT_COMMANDS.md`.

General Python/package check:

```bash
uv run --with '.[dt,rpi]' python -m compileall -q src/fuzzy_tm_infer
```

Full Python/native/Decision Tree comparison:

```bash
uv run --with '.[dt]' fuzzy-tm-infer --compare-all
```

Booleanizer speed comparison:

```bash
uv run --with numpy --with numba --with zstandard \
  fuzzy-tm-infer --compare-booleanizers
```

Native backend exactness:

```bash
make -C src/fuzzy_tm_infer clean-native native-v17 native-v19 verify-native
```

Ansible syntax check:

```bash
uv run --with '.[rpi]' ansible-playbook \
  -i ansible/inventory.rpi.example.ini \
  ansible/playbooks/rpi_fuzzy_tm_infer.yml --syntax-check
```

Raspberry Pi deployment dry run:

```bash
PYTHONPATH=src uv run --with ansible python \
  -m fuzzy_tm_infer.scripts.rpi_ansible \
  --env-file ansible/rpi.env.example \
  --host 127.0.0.1 \
  --user rpi \
  --dry-run
```

Raspberry Pi real validation:

```bash
tm-rpi-deploy --compare-all
```

## Algorithm Work

Native and Python algorithm work must be exact unless the user explicitly asks
for approximate behavior.

For native C changes:

1. Keep v17 and v19 self-contained under `algorithms/c/fuzzy_tm/`.
2. Shared ABI, loader, and CLI code belongs in `algorithms/c/common/`.
3. Do not add `tm_algorithm_20.c`, `tm_algorithm_21.c`, etc. to the active
   package. New attempts must live in an explicit version folder or an archive.
4. Verify native predictions against Python on all bundled datasets.
5. Benchmark local x86 and Raspberry Pi when the change is performance-related.
6. Record meaningful timing tables in `results/`.

For Python algorithm changes:

1. Put reusable implementations under `algorithms/py/`.
2. Keep benchmark orchestration under `benchmarks/`.
3. Keep metrics under `metrics/`.
4. Keep optional baselines explicit and documented.

For booleanizers:

1. Put reusable encoders under `booleanizers/`.
2. Provide `fit`, `transform`, and `fit_transform`.
3. If persistence is supported, provide `save_json`/`load_json` or document why
   not.
4. Prefer numpy-only implementations for core booleanizers.
5. Benchmark speed separately from downstream model accuracy.

## Deprecation Paths

Healthy deprecation is required. Do not let old experiments accumulate inside
the active package.

Use one of these paths:

1. Promote

   The artifact becomes the active implementation. Update imports, docs,
   benchmarks, and `SPEC.MD` if needed.

2. Keep As Reference

   The artifact stays only if it is intentionally useful. It must have a clear
   README or doc note stating why it exists.

3. Archive

   Move retired code to:

   ```text
   archives/<topic>_<YYYY-MM-DD>/
   ```

   Preserve enough context to recover it, but remove it from imports, build
   paths, package data, and active docs.

4. Delete Generated Artifacts

   Remove rebuildable outputs such as binaries, shared libraries, caches, build
   folders, extracted scratch files, and temporary virtual environments.

5. Remove Compatibility Shims

   When a compatibility file is no longer needed, remove it in a separate,
   documented cleanup step.

Deprecation checklist:

```text
[ ] No active import references remain.
[ ] No CLI/build path references remain.
[ ] Docs mention the new location or replacement.
[ ] Results/history are preserved if they matter.
[ ] Validation still passes.
```

## Cleanliness Rules

The latest version must remain clean.

Do not keep:

```text
unused functions
dead compile flags
abandoned algorithm attempt files
stale wrappers
generated binaries
copied source archives inside active code
machine-specific paths
large untracked scratch outputs
```

Allowed long-lived non-code artifacts:

```text
results/                   structured measured outputs and research notes
archives/source_assets/     original zip sources
archives/src_legacy_*/      retired source scripts
archives/root_legacy_*/     retired root docs
```

Generated cleanup command:

```bash
make -C src/fuzzy_tm_infer clean-native || true
rm -rf .venv build *.egg-info src/*.egg-info
find . -type d -name __pycache__ -prune -exec rm -rf {} +
```

## Documentation Rules

Update documentation when:

1. A public import changes.
2. A CLI flag changes.
3. A benchmark command changes.
4. A native backend is promoted, retired, or added.
5. RPI deployment behavior changes.
6. Assets or archive discovery changes.

Use these files intentionally:

```text
README.md                         short root orientation
SPEC.MD                           workflow and project contract
AGENTS.md                         agent behavior contract
src/fuzzy_tm_infer/README.md       package user guide
src/fuzzy_tm_infer/docs/           backend/tool notes
ansible/README.md                  Raspberry Pi deployment guide
results/YYYY-MM-DD_<slug>/        structured experiment records
EXPERIMENT_COMMANDS.md            local/RPI command registry
```

## Dependency Rules

`pyproject.toml` is the source of truth.

Core dependencies must stay small:

```text
numpy
numba
loguru
zstandard
```

Use extras for optional work:

```text
.[dt]             scikit-learn Decision Tree comparison
.[compression]    compression tooling
.[rpi]            Ansible deployment
.[dev]            development tools
.[all]            full optional stack
```

Do not add a core dependency for a single experiment unless it becomes part of
the active runtime.

## Raspberry Pi Rules

1. Never commit real hostnames, IPs, usernames, or passwords.
2. Keep secrets in `ansible/rpi.env`; keep examples in `ansible/rpi.env.example`.
3. Let the playbook derive the remote root and virtual environment from the
   remote SSH user's home directory and the local repository name.
4. Keep `TM_RPI_ARCHFLAGS=auto` as the default.
5. Prefer `tm-rpi-deploy` over ad hoc SSH commands for reproducible validation.

## Results Rules

Every benchmark record should include:

```text
date
command
machine/platform if relevant
dataset rows/features if relevant
timings
accuracy/F1 if model predictions are involved
whether the result promoted, retained, or deprecated an artifact
```

Use `results/` for durable evidence. Do not overwrite historical baselines
unless the file is explicitly a living report.

New result records must follow `results/SPEC.md` and include both:

```text
tables/*.csv
latex/*.tex
```

Use `fuzzy-tm-result` to create the directory skeleton. Do not add new flat
benchmark files directly under `results/`.

If a result does not fit an existing shape, add a format spec under
`results/formats/` instead of weakening the directory contract.

## Final Answer Expectations

When finishing a task, report:

1. What changed.
2. Where the important files are.
3. What validation ran.
4. Any known risks or follow-up decisions.

Keep the final answer concise. Do not paste full logs unless the user asked for
them.
