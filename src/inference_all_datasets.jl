#!/usr/bin/env julia
#=
TM inference for ALL datasets (MedSec, TON_IoT, UNSW, WUSTL).
Loads pre-trained compiled models from /tmp — no retraining.
Runs 3 pipelines per dataset:
  [predict only]       vote on pre-built TMInput
  [register-local]     raw Float32 row → chunk-local binarize → predict
  [sorted gather]      monotone X_raw reads + scatter-write bits → predict
Also loads model from .tm.gz to prove the compressed artifact is usable.

Usage:  julia --threads=1 inference_all_datasets.jl
=#

include("../src/Tsetlin.jl")
using .Tsetlin: TMInput, TMClassifier, predict, vote, compile, save, load
using Printf
using Serialization

function load_gz(gz_path::AbstractString)
    open(`gunzip -c $gz_path`) do io
        Serialization.deserialize(io)
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

@inline function pack_sample_sorted!(chunks, X_raw, feat_idx_s, thresh_s,
                                     chunk_of, off_of, i, n_bits, n_chunks)
    @inbounds for c in 1:n_chunks
        chunks[c] = zero(UInt64)
    end
    @inbounds @simd for k in 1:n_bits
        bit = UInt64(X_raw[feat_idx_s[k], i] >= thresh_s[k])
        chunks[chunk_of[k]] |= bit << off_of[k]
    end
end

# ══════════════════════════════════════════════════════════════════════
# Dataset registry: (label, model_stem, data_prefix)
# ══════════════════════════════════════════════════════════════════════
DATASETS = [
    (name="TON_IoT",   model="/tmp/toniot_compiled",  prefix="TonIoT"),
    (name="MedSec-25", model="/tmp/medsec_compiled",  prefix="MedSec"),
    (name="WUSTL-IIoT",model="/tmp/wustl_compiled",   prefix="WUSTL"),
    (name="NSL-KDD",   model="/tmp/nslkdd_compiled",  prefix="NSLKDD"),
]

struct DatasetResult
    name::String
    n_test::Int
    n_features::Int
    n_classes::Int
    n_clauses::Int
    accuracy::Float64
    f1_macro::Float64
    precision_macro::Float64
    recall_macro::Float64
    model_kb::Float64
    model_gz_kb::Float64
    mem_kb::Float64
    gz_load_ms::Float64
    predict_us::Float64
    predict_tps::Float64
    full_us::Float64
    full_tps::Float64
    sorted_us::Float64
    sorted_tps::Float64
end

function run_dataset(ds)
    println("\n", "="^60)
    println("  ", ds.name)
    println("="^60)

    # ── Load pre-compiled model ──────────────────────────────────
    tm_path  = string(ds.model, ".tm")
    gz_path  = string(ds.model, ".tm.gz")
    if !isfile(tm_path)
        println("  SKIP: $tm_path not found")
        return nothing
    end
    # (re)create .tm.gz so the 17 KB-style artifact exists
    isfile(gz_path) && rm(gz_path)
    run(pipeline(`gzip -9 -n -k $tm_path`))

    model_size_bytes   = filesize(tm_path)
    model_size_gz_bytes = filesize(gz_path)

    print("  Loading from $gz_path ... ")
    t_gz = @elapsed begin
        compiled_tm = load_gz(gz_path)
    end
    println("done in $(round(t_gz * 1000, digits=1)) ms.")

    # ── Load test set (binarized) ────────────────────────────────
    X_test_bin = readlines("/tmp/$(ds.prefix)_X_test.txt")
    Y_test     = readlines("/tmp/$(ds.prefix)_Y_test.txt")
    x_test = [TMInput([parse(Bool, x) for x in split(l, " ")]) for l in X_test_bin]
    y_test = [parse(Int8, l) for l in Y_test]
    n_test = length(x_test)
    n_features = length(x_test[1])

    class_teams = collect(pairs(compiled_tm.clauses))
    n_classes = length(class_teams)
    n_clauses = compiled_tm.clauses_num

    println("  Test samples: $n_test, features: $n_features bits, classes: $n_classes")

    # Warmup
    for i in 1:min(100, n_test)
        fast_predict(compiled_tm, class_teams, x_test[i])
    end

    # ── TM predict only ──────────────────────────────────────────
    n_single = min(10000, n_test)
    t_single = @elapsed begin
        for i in 1:n_single
            fast_predict(compiled_tm, class_teams, x_test[i])
        end
    end
    predict_us = t_single / n_single * 1e6

    # batch throughput over full test set
    t_batch = @elapsed begin
        for i in 1:n_test
            fast_predict(compiled_tm, class_teams, x_test[i])
        end
    end
    predict_tps = n_test / t_batch

    # ── Full pipeline prep ───────────────────────────────────────
    raw_path = "/tmp/$(ds.prefix)_X_test_raw.bin"
    bin_path = "/tmp/$(ds.prefix)_binarizer.tsv"
    local X_raw
    open(raw_path) do io
        n_rows = Int(read(io, UInt32))
        n_cols = Int(read(io, UInt32))
        flat = Vector{Float32}(undef, n_rows * n_cols)
        read!(io, flat)
        # Feature-major: each sample's feature row = one contiguous column
        X_raw = reshape(flat, n_cols, n_rows)
    end
    feat_idx = Int[]; thresh = Float32[]
    for line in eachline(bin_path)
        parts = split(line, '\t')
        push!(feat_idx, parse(Int, parts[1]) + 1)
        push!(thresh,   parse(Float32, parts[2]))
    end
    n_bits   = length(thresh)
    n_chunks = cld(n_bits, 64)

    xi       = TMInput(zeros(UInt64, n_chunks), n_bits)
    chunks   = xi.chunks

    # Warmup
    for i in 1:min(100, n_single)
        pack_sample!(chunks, X_raw, feat_idx, thresh, i, n_bits, n_chunks)
        fast_predict(compiled_tm, class_teams, xi)
    end

    # ── [register-local] full pipeline ───────────────────────────
    t_full = @elapsed begin
        for i in 1:n_single
            pack_sample!(chunks, X_raw, feat_idx, thresh, i, n_bits, n_chunks)
            fast_predict(compiled_tm, class_teams, xi)
        end
    end
    full_us  = t_full / n_single * 1e6
    full_tps = n_single / t_full

    # ── [sorted gather] variant ──────────────────────────────────
    perm          = sortperm(feat_idx)
    feat_idx_sort = feat_idx[perm]
    thresh_sort   = thresh[perm]
    chunk_of      = [((perm[k] - 1) >> 6) + 1     for k in 1:n_bits]
    off_of        = [UInt64((perm[k] - 1) & 63)   for k in 1:n_bits]
    xi2     = TMInput(zeros(UInt64, n_chunks), n_bits)
    chunks2 = xi2.chunks

    for i in 1:min(100, n_single)
        pack_sample_sorted!(chunks2, X_raw, feat_idx_sort, thresh_sort,
                            chunk_of, off_of, i, n_bits, n_chunks)
        fast_predict(compiled_tm, class_teams, xi2)
    end
    t_sorted = @elapsed begin
        for i in 1:n_single
            pack_sample_sorted!(chunks2, X_raw, feat_idx_sort, thresh_sort,
                                chunk_of, off_of, i, n_bits, n_chunks)
            fast_predict(compiled_tm, class_teams, xi2)
        end
    end
    sorted_us  = t_sorted / n_single * 1e6
    sorted_tps = n_single / t_sorted

    # ── Full-test metrics (accuracy + macro F1/P/R) ──────────────
    y_pred = [fast_predict(compiled_tm, class_teams, x) for x in x_test]
    acc = sum(y_pred .== y_test) / n_test
    classes = sort(unique(y_test))
    mp = mr = mf1 = 0.0
    for c in classes
        tp = sum((y_pred .== c) .& (y_test .== c))
        fp = sum((y_pred .== c) .& (y_test .!= c))
        fn = sum((y_pred .!= c) .& (y_test .== c))
        p = tp + fp == 0 ? 0.0 : tp / (tp + fp)
        r = tp + fn == 0 ? 0.0 : tp / (tp + fn)
        f = p + r == 0.0 ? 0.0 : 2p * r / (p + r)
        mp += p; mr += r; mf1 += f
    end
    nc = length(classes)
    mp /= nc; mr /= nc; mf1 /= nc

    mem_model = Base.summarysize(compiled_tm)

    @printf("  predict-only:  %.2f μs   %.0f pred/sec\n", predict_us, predict_tps)
    @printf("  register-local: %.2f μs   %.0f pred/sec\n", full_us, full_tps)
    @printf("  sorted gather:  %.2f μs   %.0f pred/sec\n", sorted_us, sorted_tps)
    @printf("  model on disk:  %.1f KB   gzip-9: %.1f KB (%.2f×)\n",
            model_size_bytes/1024, model_size_gz_bytes/1024,
            model_size_bytes/model_size_gz_bytes)
    @printf("  accuracy: %.4f   F1_macro: %.4f\n", acc, mf1)

    return DatasetResult(ds.name, n_test, n_features, n_classes, n_clauses,
                         acc, mf1, mp, mr,
                         model_size_bytes/1024, model_size_gz_bytes/1024,
                         mem_model/1024, t_gz*1000,
                         predict_us, predict_tps,
                         full_us, full_tps,
                         sorted_us, sorted_tps)
end

# ══════════════════════════════════════════════════════════════════════
# Run all + write combined results
# ══════════════════════════════════════════════════════════════════════

results = DatasetResult[]
for ds in DATASETS
    r = run_dataset(ds)
    r === nothing || push!(results, r)
end

results_dir  = joinpath(@__DIR__, "results")
mkpath(results_dir)
out_path = joinpath(results_dir, "tm_inference_all_datasets.txt")

open(out_path, "w") do f
    println(f, "="^70)
    println(f, "  TM INFERENCE — ALL DATASETS (single-threaded, Pi-style)")
    println(f, "  Model loaded from gzipped .tm.gz artifact (17–80 KB range)")
    println(f, "="^70)
    println(f)
    for r in results
        println(f, "-"^70)
        println(f, "  ", r.name)
        println(f, "-"^70)
        @printf(f, "    Test samples:           %d\n",  r.n_test)
        @printf(f, "    Features (bits):        %d\n",  r.n_features)
        @printf(f, "    Classes:                %d\n",  r.n_classes)
        @printf(f, "    Clauses/class:          %d\n",  r.n_clauses)
        println(f)
        @printf(f, "    Accuracy:               %.4f\n", r.accuracy)
        @printf(f, "    F1 macro:               %.4f\n", r.f1_macro)
        @printf(f, "    Precision macro:        %.4f\n", r.precision_macro)
        @printf(f, "    Recall macro:           %.4f\n", r.recall_macro)
        println(f)
        @printf(f, "    TM predict only:        %7.2f μs   %8.0f pred/sec\n",
                r.predict_us, r.predict_tps)
        @printf(f, "    Full (register-local):  %7.2f μs   %8.0f pred/sec\n",
                r.full_us, r.full_tps)
        @printf(f, "    Full (sorted gather):   %7.2f μs   %8.0f pred/sec\n",
                r.sorted_us, r.sorted_tps)
        println(f)
        @printf(f, "    Model .tm:              %7.1f KB\n", r.model_kb)
        @printf(f, "    Model .tm.gz (deploy):  %7.1f KB  (%.2f× smaller)\n",
                r.model_gz_kb, r.model_kb / r.model_gz_kb)
        @printf(f, "    In-memory (RAM):        %7.1f KB\n", r.mem_kb)
        @printf(f, "    gzip decompress+load:   %7.1f ms  (one-time)\n", r.gz_load_ms)
        println(f)
    end
    println(f, "="^70)
    println(f, "  SUMMARY TABLE")
    println(f, "="^70)
    @printf(f, "  %-12s %8s %8s %10s %10s %9s %9s\n",
            "Dataset", "Acc", "F1", "Predict μs", "Full μs", "gz KB", "Pred/sec")
    for r in results
        @printf(f, "  %-12s %8.4f %8.4f %10.2f %10.2f %9.1f %9.0f\n",
                r.name, r.accuracy, r.f1_macro,
                r.predict_us, r.full_us, r.model_gz_kb, r.full_tps)
    end
end

println("\n", "="^60)
println("  All-dataset results written to:")
println("    ", out_path)
println("="^60)
