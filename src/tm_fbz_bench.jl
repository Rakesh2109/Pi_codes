#!/usr/bin/env julia
#=
FBZ inference benchmark — Julia, per-sample, single-threaded.
Data: X_test_raw.bin (preprocessed float32 test features) → binarize → TM predict.

Usage:  julia tm_fbz_bench.jl [wustl|nslkdd|toniot|medsec]
=#

using Printf
using DelimitedFiles

# ── FBZ reader ────────────────────────────────────────────────────────────────
struct FBZModel
    N         ::Int
    K         ::Int
    clamp_max ::Int
    feat_idx  ::Vector{Int32}
    thresholds::Vector{Float32}
    class_names::Vector{String}
    pos_masks ::Vector{Matrix{UInt64}}   # K*2 elements, each (n_chunks, n_clauses)
    neg_masks ::Vector{Matrix{UInt64}}
    clamps    ::Vector{Vector{Int32}}
    n_chunks  ::Int
end

function _read_u16(blob, off)
    Int(reinterpret(UInt16, blob[off:off+1])[1]), off + 2
end
function _read_u32(blob, off)
    Int(reinterpret(UInt32, blob[off:off+3])[1]), off + 4
end

function read_fbz(path::String)
    blob = read(path)
    off  = 1

    magic      = blob[1:4]
    @assert magic == b"FBZ1" "not an FBZ file"
    ver        = blob[5];   off = 6
    N,  off    = _read_u16(blob, off)
    K,  off    = _read_u16(blob, off)
    clamp_max  = Int(blob[off]); off += 1
    total_cl, off  = _read_u32(blob, off)
    comp_size, off = _read_u32(blob, off)
    uncomp_size, off = _read_u32(blob, off)

    feat_idx   = reinterpret(Int32,   blob[off : off + 4*N - 1]); off += 4*N
    thresholds = reinterpret(Float32, blob[off : off + 4*N - 1]); off += 4*N

    # string table — skip feat_names, read class_names
    n_feat, off = _read_u16(blob, off)
    for _ in 1:n_feat
        ln, off = _read_u16(blob, off); off += ln
    end
    n_cls, off = _read_u16(blob, off)
    class_names = String[]
    for _ in 1:n_cls
        ln, off = _read_u16(blob, off)
        push!(class_names, String(blob[off:off+ln-1])); off += ln
    end

    # decompress clause block with system zstd
    tmp_in  = tempname() * ".zst"
    tmp_out = tempname()
    write(tmp_in, blob[off : off + comp_size - 1])
    run(pipeline(`zstd -d -q -f -o $tmp_out $tmp_in`, stderr=devnull))
    bm = read(tmp_out)
    rm(tmp_in); rm(tmp_out)
    @assert length(bm) == uncomp_size

    n_chunks    = (N + 63) ÷ 64
    chunk_bytes = (N + 7)  ÷ 8

    # parse bitmask block
    pos_masks = Vector{Matrix{UInt64}}()
    neg_masks = Vector{Matrix{UInt64}}()
    clamps    = Vector{Vector{Int32}}()
    boff = 1
    for _k in 1:K, _pol in 1:2
        nc = Int(reinterpret(UInt16, bm[boff:boff+1])[1]); boff += 2
        pm = zeros(UInt64, n_chunks, nc)
        nm = zeros(UInt64, n_chunks, nc)
        cl = zeros(Int32, nc)
        for c in 1:nc
            cl[c] = Int32(bm[boff]); boff += 1
            for ch in 1:n_chunks
                w = UInt64(0)
                for b in 0:7
                    byte_pos = (ch-1)*8 + b
                    byte_pos >= chunk_bytes && break
                    w |= UInt64(bm[boff + byte_pos]) << (b * 8)
                end
                pm[ch, c] = w
            end
            boff += chunk_bytes
            for ch in 1:n_chunks
                w = UInt64(0)
                for b in 0:7
                    byte_pos = (ch-1)*8 + b
                    byte_pos >= chunk_bytes && break
                    w |= UInt64(bm[boff + byte_pos]) << (b * 8)
                end
                nm[ch, c] = w
            end
            boff += chunk_bytes
        end
        push!(pos_masks, pm)
        push!(neg_masks, nm)
        push!(clamps,    cl)
    end

    return FBZModel(N, K, clamp_max,
                    Vector{Int32}(feat_idx), Vector{Float32}(thresholds),
                    class_names, pos_masks, neg_masks, clamps, n_chunks)
end


# ── load test data ─────────────────────────────────────────────────────────────
function load_raw_bin(prefix)
    path = "/tmp/$(prefix)_X_test_raw.bin"
    buf  = read(path)
    n, d = reinterpret(UInt32, buf[1:8])
    X    = reinterpret(Float32, buf[9:end])
    return reshape(copy(X), Int(d), Int(n))', Int(n), Int(d)
end

function load_y(prefix)
    Int32.(readdlm("/tmp/$(prefix)_Y_test.txt", Int32))[:, 1]
end


# ── per-sample inference ──────────────────────────────────────────────────────
@inline function binarize_one(row::AbstractVector{Float32},
                               feat_idx::Vector{Int32},
                               thresh::Vector{Float32},
                               n_chunks::Int, N::Int)
    chunks = zeros(UInt64, n_chunks)
    for i in 1:N
        fi = Int(feat_idx[i]) + 1   # 0-indexed → 1-indexed
        if row[fi] >= thresh[i]
            ch  = (i-1) ÷ 64 + 1
            bit = (i-1) % 64
            chunks[ch] |= UInt64(1) << bit
        end
    end
    return chunks
end

@inline function predict_one(chunks::Vector{UInt64}, m::FBZModel)
    votes = zeros(Int32, m.K)
    not_x = .~chunks
    idx   = 1
    for k in 1:m.K
        for sign in (+1, -1)
            pm = m.pos_masks[idx]
            nm = m.neg_masks[idx]
            cl = m.clamps[idx]
            nc = size(pm, 2)
            for c in 1:nc
                miss = Int32(0)
                @inbounds for ch in 1:m.n_chunks
                    miss += count_ones((pm[ch, c] & not_x[ch]) |
                                       (nm[ch, c] & chunks[ch]))
                end
                votes[k] += sign * max(cl[c] - miss, 0)
            end
            idx += 1
        end
    end
    return argmax(votes) - 1   # 0-indexed class
end


# ── macro F1 ──────────────────────────────────────────────────────────────────
function macro_f1(y_pred, y_true)
    classes = sort(unique(vcat(y_pred, y_true)))
    mf1 = 0.0
    for c in classes
        tp = sum((y_pred .== c) .& (y_true .== c))
        fp = sum((y_pred .== c) .& (y_true .!= c))
        fn = sum((y_pred .!= c) .& (y_true .== c))
        p  = (tp + fp == 0) ? 0.0 : tp / (tp + fp)
        r  = (tp + fn == 0) ? 0.0 : tp / (tp + fn)
        mf1 += (p + r == 0.0) ? 0.0 : 2p*r/(p+r)
    end
    return mf1 / length(classes)
end


# ── benchmark ─────────────────────────────────────────────────────────────────
function bench(name, stem, prefix)
    fbz_path = "/tmp/$(stem)_model.fbz"
    if !isfile(fbz_path)
        println("  $name  SKIP — missing $fbz_path"); return
    end

    print("  Loading $name ... ")
    m     = read_fbz(fbz_path)
    X, n, d = load_raw_bin(prefix)
    y     = load_y(prefix)
    println("done  (N=$(m.N) K=$(m.K) n_test=$n)")

    N_WARM = 200
    N_TIME = min(2000, n)

    # warmup
    for i in 1:N_WARM
        xb = binarize_one(X[i, :], m.feat_idx, m.thresholds, m.n_chunks, m.N)
        predict_one(xb, m)
    end

    # time binarize only
    t0 = time_ns()
    for i in 1:N_TIME
        binarize_one(X[mod1(i,n), :], m.feat_idx, m.thresholds, m.n_chunks, m.N)
    end
    bin_us = (time_ns() - t0) / N_TIME / 1000.0

    # time binarize + predict
    t0 = time_ns()
    for i in 1:N_TIME
        xb = binarize_one(X[mod1(i,n), :], m.feat_idx, m.thresholds, m.n_chunks, m.N)
        predict_one(xb, m)
    end
    full_us = (time_ns() - t0) / N_TIME / 1000.0
    pred_us = full_us - bin_us

    # correctness on full test set
    y_pred = Int32[predict_one(
        binarize_one(X[i,:], m.feat_idx, m.thresholds, m.n_chunks, m.N), m)
        for i in 1:n]
    acc = sum(y_pred .== y) / n
    f1  = macro_f1(y_pred, y)

    @printf("  %-10s  binarize=%7.2f µs  predict=%7.2f µs  total=%7.2f µs  F1=%.4f  acc=%.4f\n",
            name, bin_us, pred_us, full_us, f1, acc)
    return (name=name, bin_us=bin_us, pred_us=pred_us, full_us=full_us, f1=f1, acc=acc)
end


# ── main ──────────────────────────────────────────────────────────────────────
CASES = [
    ("WUSTL",   "wustl",  "WUSTL"),
    ("NSLKDD",  "nslkdd", "NSLKDD"),
    ("TonIoT",  "toniot", "TonIoT"),
    ("MedSec",  "medsec", "MedSec"),
]

selected = isempty(ARGS) ? nothing : lowercase(ARGS[1])
cases = selected === nothing ? CASES :
        filter(c -> lowercase(c[2]) == selected, CASES)

println("=" ^ 90)
println("  FBZ INFERENCE BENCHMARK — Julia, per-sample, single-threaded")
println("  Pipeline: float32 row → binarize (GLADE) → TM predict (FBZ clause masks)")
println("=" ^ 90)
results = []
for (name, stem, prefix) in cases
    r = bench(name, stem, prefix)
    r !== nothing && push!(results, r)
end

println()
@printf("  %-10s  %9s  %9s  %9s  %7s  %7s\n",
        "Dataset","bin µs","pred µs","total µs","F1","acc")
println("  " * "-"^62)
for r in results
    @printf("  %-10s  %9.2f  %9.2f  %9.2f  %7.4f  %7.4f\n",
            r.name, r.bin_us, r.pred_us, r.full_us, r.f1, r.acc)
end
