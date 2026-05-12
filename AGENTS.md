# Agent Development Contract

Agents must follow these rules when changing this repository:

1. Read `SPEC.MD` before making code changes.
2. Keep all active code inside `src/paper_2/`.
3. Do not hardcode machine-specific paths. Use env vars in `config.py`.
4. Avoid leaving unused files in the active package. Archive instead.
5. Keep results under `results/` and do not commit caches or binaries.
6. Prefer small, testable changes and note how to validate them.
