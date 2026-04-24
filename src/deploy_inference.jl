#!/usr/bin/env julia
#=
Pi deployment — single-language (Julia-only) inference.

No Python at runtime. Loads:
  • a compiled Tsetlin model (.tm or .tm.gz — auto-detects gzip)
  • a GLADE threshold table (.tsv: "feat_idx\tthreshold" per line, 0-based)
  • a raw Float32 test blob (header: <UInt32 n_rows><UInt32 n_cols>, then row-major data)

Binarises each sample in place (bitwise pack into a reused TMInput buffer)
and predicts via a Dict-iteration-free fast_predict. Reports per-sample
latency and throughput.

Usage:
  julia --threads=1 deploy_inference.jl <model> <binarizer.tsv> <raw.bin>

Example:
  julia --threads=1 deploy_inference.jl \
        /tmp/medsec_compiled.tm.gz \
        /tmp/MedSec_binarizer.tsv \
        /tmp/MedSec_X_test_raw.bin
=#

include("../src/Tsetlin.jl")
using .Tsetlin: TMInput, TMClassifier, predict, vote, load
using Printf
using Serialization

# ── CLI ──────────────────────────────────────────────────────
if length(ARGS) != 3
    println(stderr, "usage: julia deploy_inference.jl <model> <binarizer.tsv> <raw.bin>")
    exit(2)
end
model_path, bin_path, raw_path = ARGS

# ── Load model (gzip-aware) ──────────────────────────────────
function load_model(path::AbstractString)
    if endswith(path, ".gz")
        # Stream-decompress via gunzip; deserialize directly off the pipe —
        # no temp file, one copy of the bytes in memory.
        return open(`gunzip -c $path`, "r") do io
            Serialization.deserialize(io)
        end
    else
        return load(replace(path, r"\.tm$" => ""))  # Tsetlin.load adds .tm
    end
end

print("Loading model from $(model_path)... ")
compiled_tm = load_model(model_path)
println("done.")

# ── Load GLADE binariser (feat_idx + thresholds) ─────────────
feat_idx = Int[]; thresh = Float32[]
for line in eachline(bin_path)
    parts = split(line, '\t')
    push!(feat_idx, parse(Int, parts[1]) + 1)   # 0-based → 1-based
    push!(thresh,   parse(Float32, parts[2]))
end
n_bits   = length(thresh)
n_chunks = cld(n_bits, 64)
println("Binariser: $(n_bits) bits, $(n_chunks) UInt64 chunks.")

# ── Load raw test blob as column-major [feat, sample] ────────
local n_samples::Int, n_cols::Int
open(raw_path) do io
    global X_raw
    n_rows_disk = Int(read(io, UInt32))
    n_cols      = Int(read(io, UInt32))
    flat = Vector{Float32}(undef, n_rows_disk * n_cols)
    read!(io, flat)
    # Disk is row-major (sample-major); reshape to (n_cols, n_rows) column-
    # major so column i == sample i's contiguous feature row.
    X_raw = reshape(flat, n_cols, n_rows_disk)
    n_samples = n_rows_disk
end
println("Raw test data: $(n_samples) samples × $(n_cols) features (Float32).")

# ── Flatten class Dict to Vector → skip Dict iteration cost ──
const CLASS_TEAMS = collect(pairs(compiled_tm.clauses))

# ── Reusable TMInput buffer ──────────────────────────────────
xi     = TMInput(zeros(UInt64, n_chunks), n_bits)
chunks = xi.chunks

# ── Hot path: threshold → bit-pack → predict ─────────────────
@inline function pack_sample!(chunks, X_raw, feat_idx, thresh, i, n_bits, n_chunks)
    @inbounds for c in 1:n_chunks
        lo = (c - 1) << 6 + 1
        hi = min(c << 6, n_bits)
        w  = zero(UInt64)
        @inbounds @simd for k in lo:hi
            w |= UInt64(X_raw[feat_idx[k], i] >= thresh[k]) << (k - lo)
        end
        chunks[c] = w
    end
end

@inline function fast_predict(tm, class_teams, x)
    best_v::Int64 = typemin(Int64)
    local best_c = class_teams[1].first
    @inbounds for ct in class_teams
        p, n = vote(tm, ct.second, x)
        v = p - n
        if v > best_v
            best_v = v
            best_c = ct.first
        end
    end
    return best_c
end

# ── Warmup (JIT + cache) ─────────────────────────────────────
warmup_n = min(200, n_samples)
for i in 1:warmup_n
    pack_sample!(chunks, X_raw, feat_idx, thresh, i, n_bits, n_chunks)
    fast_predict(compiled_tm, CLASS_TEAMS, xi)
end

# ── Timed run over all samples ───────────────────────────────
preds = Vector{eltype(CLASS_TEAMS[1].first)}(undef, n_samples)
t = @elapsed begin
    for i in 1:n_samples
        pack_sample!(chunks, X_raw, feat_idx, thresh, i, n_bits, n_chunks)
        preds[i] = fast_predict(compiled_tm, CLASS_TEAMS, xi)
    end
end

latency_us  = t / n_samples * 1e6
throughput  = n_samples / t

println("\n", "="^52)
println("  DEPLOY INFERENCE (single-threaded, Julia-only)")
println("="^52)
@printf("  Samples:       %d\n", n_samples)
@printf("  Latency:       %.2f μs/sample (%.4f ms)\n", latency_us, latency_us/1000)
@printf("  Throughput:    %d predictions/sec\n", round(Int, throughput))
@printf("  Total time:    %.3f s\n", t)
println("="^52)

# First few predictions for a sanity print
show_n = min(10, n_samples)
println("\nFirst $(show_n) predictions: $(preds[1:show_n])")
