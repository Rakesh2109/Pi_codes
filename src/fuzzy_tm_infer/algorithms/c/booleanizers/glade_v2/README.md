# Fast GLADE v2 Transform

This folder contains the native transform kernel for fitted GLADE v2
booleanizers.

The Python implementation still owns fitting because it relies on NumPy
percentiles, unique values, and entropy cleanup. The C code owns the hot
operation after fitting:

```text
bit_j = X[row, feature_index_j] >= threshold_j
```

The ABI exposes unpacked `uint8` output and NumPy-compatible packed output
using `np.packbits(..., axis=1)` bit order.

Build:

```bash
make -C src/fuzzy_tm_infer/algorithms/c/booleanizers/glade_v2 lib
```

Build with profiling counters:

```bash
make -C src/fuzzy_tm_infer profile-fast-glade
```

Profiling is macro guarded by `GLADE_V2_PROFILE`. Normal `lib` builds include
the profile ABI but return zero counters and have no timing instrumentation.
Follow `algorithms/c/PROFILE_TEMPLATE.md` for new profiling counters.

Available profile fields:

```text
total_calls
u8_calls
packed_calls
rows
bits
output_bytes
validate_ns
kernel_ns
total_ns
errors
```

Python access:

```python
from fuzzy_tm_infer.algorithms.c import FastGLADEBooleanizer

model = FastGLADEBooleanizer.from_booleanizer(glade)
out = model.empty_output(len(X), pack_bits=True)
model.transform_into(X, out, pack_bits=True)

for bits in model.transform_stream(X, batch_size=4096, pack_bits=True):
    consume(bits)

model.profile_reset()
model.transform_into(X, out, pack_bits=True)
print(model.profile_enabled)
print(model.profile_snapshot())
```

Verify:

```bash
uv run fuzzy-tm-verify-fast-glade
```

Backends:

```text
AVX2   x86 gather/compare kernels with same-feature broadcast fast paths
NEON   aarch64 compare kernels with same-feature broadcast fast paths
scalar portable fallback
```

Implementation notes:

- Float32 input stays float32 end-to-end to avoid wrapper-side conversion.
- Streaming callers can reuse output buffers with `transform_into()` or consume
  bounded chunks with `transform_stream()`/`consume_stream()`.
- The chunked float32 wrapper validates the fitted model once and uses trusted C
  entry points that skip repeated feature-index validation per streaming batch.
- Packed output writes full bytes directly and only clears the final partial
  byte when needed.
- The Python wrapper precomputes 4-lane and 8-lane chunk metadata. SIMD chunks
  whose literals share one feature use one scalar feature load plus a vector
  threshold compare. Mixed-feature chunks fall back to the generic indexed path.
- On NEON, the wrapper also precomputes compact 4-lane chunk index/threshold
  arrays. The RPI path iterates by chunk using this layout, which reduces hot
  loop address arithmetic and improves locality while preserving the same
  threshold comparisons.
- AVX2 unpacked output expands an 8-lane comparison mask to eight `uint8`
  Boolean values with one 64-bit store. This is still exactly the same layout
  as Python's unpacked `uint8` output.
- Packed mask conversion uses a 256-entry table. Unpacked AVX2 expansion uses
  BMI2 `pdep` when compiled with BMI2 support and a small nibble-table fallback
  otherwise. Both paths preserve Python bit order while removing shift/or trees
  from the hot path.
- A feature-block prefix formulation was tested and rejected for this backend:
  it was exact, but slower than the chunked SIMD literal path on the bundled
  datasets.
