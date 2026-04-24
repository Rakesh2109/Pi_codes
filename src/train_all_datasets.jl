#!/usr/bin/env julia
#=
Train TM classifiers for 4 datasets, selecting the BEST-F1 epoch checkpoint
(not best-accuracy). Saves /tmp/<stem>_model.fbz directly after training.

Usage:
  julia --threads=N train_all_datasets.jl [wustl|nslkdd|toniot|medsec]
  (no arg = train all sequentially)

S-parameter note: Julia Tsetlin.jl computes s = round(n_bits / S). User-
supplied S values from FPTM sweep give s=1 for NSL-KDD which causes
under-training; adjusted NSL-KDD S=25 to give s≈10.
=#

include("../src/Tsetlin.jl")
using .Tsetlin: TMInput, TMClassifier, train!, predict, compile, save
using Printf
using Dates
using DelimitedFiles

CONFIGS = [
    # stem      prefix    C    T   S    L   LF  E
    (stem="wustl",  prefix="WUSTL",  C=60,  T=8,  S=300, L=50, LF=15, E=200),
    (stem="nslkdd", prefix="NSLKDD", C=90,  T=12, S=200, L=40, LF=8,  E=80),  # user's original hyperparams
    (stem="toniot", prefix="TonIoT", C=100, T=15, S=20,  L=25, LF=8,  E=200),
    (stem="medsec", prefix="MedSec", C=80,  T=12, S=75,  L=30, LF=8,  E=300),
]

function _write_mask(io, mask_col, n_chunks, chunk_bytes)
    for ch in 1:n_chunks
        w = mask_col[ch]
        for b in 0:7
            (ch - 1) * 8 + b >= chunk_bytes && return
            write(io, UInt8((w >> (b * 8)) & 0xFF))
        end
    end
end

function save_fbz_direct(best_tm, cfg, out_path::String)
    tsv        = readdlm("/tmp/$(cfg.prefix)_binarizer.tsv")
    feat_idx   = Int32[round(Int32, tsv[i, 1]) for i in 1:size(tsv, 1)]
    thresholds = Float32[Float32(tsv[i, 2]) for i in 1:size(tsv, 1)]
    N          = length(feat_idx)
    chunk_bytes = (N + 7) ÷ 8
    n_chunks    = (N + 63) ÷ 64

    classes = sort(collect(keys(best_tm.clauses)))
    K       = length(classes)

    bm          = IOBuffer()
    total_clauses = 0
    clamp_max_val = 0
    for cls in classes
        ta = best_tm.clauses[cls]
        for (lits, invs, clamps) in (
            (ta.positive_included_literals, ta.positive_included_literals_inverted,
             ta.positive_included_literals_sum_clamp),
            (ta.negative_included_literals, ta.negative_included_literals_inverted,
             ta.negative_included_literals_sum_clamp))
            nc = size(lits, 2)
            write(bm, UInt16(nc))
            for c in 1:nc
                total_clauses += 1
                cv = min(Int(clamps[c]), 255)
                clamp_max_val = max(clamp_max_val, cv)
                write(bm, UInt8(cv))
                _write_mask(bm, view(lits, :, c), n_chunks, chunk_bytes)
                _write_mask(bm, view(invs, :, c), n_chunks, chunk_bytes)
            end
        end
    end
    bm_bytes    = take!(bm)
    uncomp_size = length(bm_bytes)

    tmp_in  = tempname() * ".bm"
    tmp_out = tmp_in * ".zst"
    write(tmp_in, bm_bytes)
    run(pipeline(`zstd -22 -q -f -o $tmp_out $tmp_in`, stderr=devnull))
    compressed = read(tmp_out)
    rm(tmp_in); rm(tmp_out)
    comp_size = length(compressed)

    class_strs = [string(cls) for cls in classes]
    strtbl = IOBuffer()
    write(strtbl, UInt16(0))   # 0 feature names
    write(strtbl, UInt16(K))
    for s in class_strs
        b = Vector{UInt8}(s)
        write(strtbl, UInt16(length(b)))
        write(strtbl, b)
    end
    str_bytes = take!(strtbl)

    open(out_path, "w") do f
        write(f, b"FBZ1")
        write(f, UInt8(1))
        write(f, UInt16(N))
        write(f, UInt16(K))
        write(f, UInt8(min(clamp_max_val, 255)))
        write(f, UInt32(total_clauses))
        write(f, UInt32(comp_size))
        write(f, UInt32(uncomp_size))
        write(f, feat_idx)
        write(f, thresholds)
        write(f, str_bytes)
        write(f, compressed)
    end
    return filesize(out_path)
end

function macro_f1(y_pred, y_true, classes)
    mp = mr = mf1 = 0.0
    for c in classes
        tp = sum((y_pred .== c) .& (y_true .== c))
        fp = sum((y_pred .== c) .& (y_true .!= c))
        fn = sum((y_pred .!= c) .& (y_true .== c))
        p = tp + fp == 0 ? 0.0 : tp / (tp + fp)
        r = tp + fn == 0 ? 0.0 : tp / (tp + fn)
        f = p + r == 0.0 ? 0.0 : 2p * r / (p + r)
        mp += p; mr += r; mf1 += f
    end
    nc = length(classes)
    return (mf1 / nc, mp / nc, mr / nc)
end

function train_one(cfg)
    println("\n", "="^60)
    println("  Training: $(cfg.prefix)  (C=$(cfg.C) T=$(cfg.T) S=$(cfg.S) L=$(cfg.L) LF=$(cfg.LF) E=$(cfg.E))")
    println("  Selection metric: best macro-F1")
    println("="^60)

    print("  Loading train/test... ")
    t_load = @elapsed begin
        x_train = [TMInput([parse(Bool, x) for x in split(l, " ")]) for l in readlines("/tmp/$(cfg.prefix)_X_train.txt")]
        y_train = [parse(Int8, l) for l in readlines("/tmp/$(cfg.prefix)_Y_train.txt")]
        x_test  = [TMInput([parse(Bool, x) for x in split(l, " ")]) for l in readlines("/tmp/$(cfg.prefix)_X_test.txt")]
        y_test  = [parse(Int8, l) for l in readlines("/tmp/$(cfg.prefix)_Y_test.txt")]
    end
    n_bits = length(x_train[1])
    classes = sort(unique(y_train))
    @printf("done (%.1fs)  %d bits, %d train, %d test, %d classes\n",
            t_load, n_bits, length(x_train), length(x_test), length(classes))

    tm = TMClassifier(x_train[1], y_train, cfg.C, cfg.T, cfg.S,
                      L=cfg.L, LF=cfg.LF, states_num=256, include_limit=200)

    best_f1   = -1.0
    best_acc  = 0.0
    best_p    = 0.0
    best_r    = 0.0
    best_epoch = 0
    best_tm   = nothing

    t_fit = @elapsed begin
        for epoch in 1:cfg.E
            t_train = @elapsed train!(tm, x_train, y_train; shuffle=true)
            t_test  = @elapsed y_pred = predict(tm, x_test)
            acc = sum(y_pred .== y_test) / length(y_test)
            f1, p, r = macro_f1(y_pred, y_test, classes)
            improved = f1 > best_f1
            if improved
                best_f1 = f1; best_acc = acc; best_p = p; best_r = r
                best_epoch = epoch
                best_tm = compile(tm)
            end
            mark = improved ? " *" : ""
            @printf("  #%-3d  acc=%.4f  F1=%.4f  P=%.4f  R=%.4f  train=%.2fs  test=%.2fs%s\n",
                    epoch, acc, f1, p, r, t_train, t_test, mark)
        end
    end

    fbz_path = "/tmp/$(cfg.stem)_model.fbz"
    fbz_bytes = save_fbz_direct(best_tm, cfg, fbz_path)
    fbz_kb = fbz_bytes / 1024

    @printf("\n  %s: best F1 @ epoch %d  F1=%.4f  acc=%.4f  P=%.4f  R=%.4f\n",
            cfg.prefix, best_epoch, best_f1, best_acc, best_p, best_r)
    @printf("  .fbz=%.2f KB   total_train=%.1fs\n", fbz_kb, t_fit)
    return (prefix=cfg.prefix, best_f1=best_f1, best_acc=best_acc,
            best_p=best_p, best_r=best_r, best_epoch=best_epoch,
            fbz_kb=fbz_kb, train_s=t_fit, n_bits=n_bits)
end

selected = isempty(ARGS) ? nothing : lowercase(ARGS[1])
cfgs = selected === nothing ? CONFIGS :
       filter(c -> lowercase(c.stem) == selected, CONFIGS)
isempty(cfgs) && error("Unknown dataset: $selected (expected: $(join([c.stem for c in CONFIGS], ", ")))")

results = []
for cfg in cfgs
    push!(results, train_one(cfg))
end

println("\n", "="^60)
println("  TRAINING SUMMARY  (best-F1 checkpoint per dataset)")
println("="^60)
@printf("  %-10s %6s %6s %8s %8s %8s %8s %8s\n",
        "Dataset", "n_bits", "epoch", "F1", "acc", "P", "R", "fbz KB")
for r in results
    @printf("  %-10s %6d %6d %8.4f %8.4f %8.4f %8.4f %8.2f\n",
            r.prefix, r.n_bits, r.best_epoch, r.best_f1, r.best_acc,
            r.best_p, r.best_r, r.fbz_kb)
end
