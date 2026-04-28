# Native Profiling Template

Use this template when adding profiling to a C backend or kernel.

## Purpose

```text
Component:
Kernel/function:
Question:
Hypothesis:
```

Example question:

```text
Is Fast GLADE v2 slow because of feature validation, gather/compare work, or
packing output bytes?
```

## Build Contract

Normal build:

```bash
make -C src/fuzzy_tm_infer <normal-target>
```

Profile build:

```bash
make -C src/fuzzy_tm_infer <profile-target>
```

Compile macro:

```text
COMPONENT_PROFILE_MACRO
```

Normal builds must not collect timings. Profile builds must remain exact.

## C ABI Pattern

Expose these functions from the component library:

```c
uint32_t component_profile_field_count(void);
const char *component_profile_field_name(uint32_t index);
int component_profile_enabled(void);
void component_profile_reset(void);
int component_profile_read(uint64_t *out, uint32_t capacity);
```

Rules:

- `profile_read()` returns the number of fields copied, or a negative error.
- `profile_read()` returns zero-valued counters when profiling is not compiled.
- Field order must remain stable once a result has been recorded.

## Python Wrapper Pattern

Expose:

```python
profile_enabled: bool
profile_reset() -> None
profile_snapshot() -> dict[str, int]
```

Do not expose raw ctypes details to callers.

## Recommended Counters

Use common names when possible:

```text
total_calls
rows
bits
clauses
selected_rows
output_bytes
validate_ns
setup_ns
kernel_ns
pack_ns
total_ns
errors
```

Component-specific counters are fine, but keep names stable and document them.

## Timing Rules

- Use monotonic clock in C.
- Only call the clock in profile builds.
- Do not time inside the innermost operation unless absolutely necessary.
- Prefer coarse regions: validation, setup, kernel, packing, total.
- Count work units so times can be normalized later.

## Validation

Run the normal exactness gate:

```bash
make -C src/fuzzy_tm_infer verify-native
make -C src/fuzzy_tm_infer verify-fast-glade
```

Run the profile build and verify exactness again:

```bash
make -C src/fuzzy_tm_infer <profile-target>
uv run <component-verifier>
```

## Result Recording

Record a profile result when it changes an optimization decision.

Use `benchmark.table.v1` and include at least:

```text
dataset
component
backend
profile_enabled
rows
work_units
validate_ns
setup_ns
kernel_ns
pack_ns
total_ns
errors
```

Keep raw command output in `raw/`.
