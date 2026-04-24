#!/usr/bin/env julia
#=
Inference benchmark — WUSTL-EHMS (copy of inference_benchmark.jl, different /tmp prefix).
Single-threaded by default to simulate a Raspberry Pi core.
=#

include("../src/Tsetlin.jl")
using .Tsetlin: TMInput, TMClassifier, train!, predict, vote, accuracy, compile, save, load
using Printf

println("Loading test data...")
X_test_bin = readlines("/tmp/WUSTL_X_test.txt")
Y_test     = readlines("/tmp/WUSTL_Y_test.txt")

x_test = [TMInput([parse(Bool, x) for x in split(l, " ")]) for l in X_test_bin]
y_test = [parse(Int8, l) for l in Y_test]

n_test = length(x_test)
n_features = length(x_test[1])
println("  Test samples: $n_test, Features: $n_features bits")

println("\nTraining model for benchmark...")
X_train_lines = readlines("/tmp/WUSTL_X_train.txt")
x_train = [TMInput([parse(Bool, x) for x in split(l, " ")]) for l in X_train_lines]
y_train = [parse(Int8, l) for l in readlines("/tmp/WUSTL_Y_train.txt")]

CLAUSES = 80
T       = 10
S       = 61
L       = 60
LF      = 10
EPOCHS  = 30

tm = TMClassifier(x_train[1], y_train, CLAUSES, T, S,
                  L=L, LF=LF, states_num=256, include_limit=200)

t_fit = @elapsed begin
    tms = train!(tm, x_train, y_train, x_test, y_test, EPOCHS,
                 best_tms_size=1, shuffle=true, verbose=1)
end

best_tm, best_acc = tms[1]
compiled_tm = compile(best_tm)

save(compiled_tm, "/tmp/wustl_compiled")
model_size_bytes = filesize("/tmp/wustl_compiled.tm")
model_size_kb = model_size_bytes / 1024

println("\nModel compiled. Size: $(round(model_size_kb, digits=1)) KB")

println("\n", "="^60)
println("  INFERENCE BENCHMARK — WUSTL-EHMS (single-threaded)")
println("="^60)

for i in 1:min(100, n_test)
    predict(compiled_tm, x_test[i])
end

n_single = min(10000, n_test)
t_single = @elapsed begin
    for i in 1:n_single
        predict(compiled_tm, x_test[i])
    end
end
latency_us = t_single / n_single * 1e6
latency_ms = t_single / n_single * 1e3

n_loops = 3
batch_times = Float64[]
for loop in 1:n_loops
    t = @elapsed begin
        for i in 1:n_test
            predict(compiled_tm, x_test[i])
        end
    end
    push!(batch_times, t)
end
avg_batch = sum(batch_times) / n_loops
throughput = n_test / avg_batch

sample_bools = [parse(Bool, x) for x in split(X_test_bin[1], " ")]
n_input = 10000
t_input = @elapsed begin
    for i in 1:n_input
        TMInput(sample_bools)
    end
end
input_latency_us = t_input / n_input * 1e6

raw_path  = "/tmp/WUSTL_X_test_raw.bin"
bin_path  = "/tmp/WUSTL_binarizer.tsv"
open(raw_path) do io
    global X_raw
    n_rows = Int(read(io, UInt32))
    n_cols = Int(read(io, UInt32))
    flat = Vector{Float32}(undef, n_rows * n_cols)
    read!(io, flat)
    X_raw = permutedims(reshape(flat, n_cols, n_rows))
end
feat_idx = Int[]; thresh = Float32[]
for line in eachline(bin_path)
    parts = split(line, '\t')
    push!(feat_idx, parse(Int, parts[1]) + 1)
    push!(thresh,   parse(Float32, parts[2]))
end
n_bits = length(thresh)

t_full = @elapsed begin
    for i in 1:n_single
        bools = Vector{Bool}(undef, n_bits)
        @inbounds for j in 1:n_bits
            bools[j] = X_raw[i, feat_idx[j]] >= thresh[j]
        end
        xi = TMInput(bools)
        predict(compiled_tm, xi)
    end
end
full_latency_us = t_full / n_single * 1e6
full_throughput = n_single / t_full

mem_model = Base.summarysize(compiled_tm)
mem_input = Base.summarysize(x_test[1])

println("\n  RESULTS:")
println("  " * "-"^50)
@printf("  TM predict only:\n")
@printf("    Latency:    %.2f μs/sample (%.4f ms)\n", latency_us, latency_ms)
@printf("    Throughput: %d predictions/sec\n", round(Int, n_test / avg_batch))
@printf("  TMInput creation latency: %.2f μs/sample\n", input_latency_us)
@printf("  Full pipeline latency:    %.2f μs/sample\n", full_latency_us)
@printf("  Full pipeline throughput: %d predictions/sec\n", round(Int, full_throughput))
@printf("  Compiled model size:      %.1f KB\n", model_size_kb)
@printf("  In-memory model size:     %.1f KB\n", mem_model / 1024)
@printf("  Input vector size:        %d bytes\n", mem_input)
@printf("  Features: %d bits   Clauses: %d\n", n_features, CLAUSES)
@printf("  Accuracy: %.2f%%\n", best_acc * 100)
println("  " * "-"^50)

open("/tmp/inference_benchmark_wustl.txt", "w") do f
    println(f, "="^60)
    println(f, "  INFERENCE BENCHMARK — WUSTL-EHMS (single-threaded)")
    println(f, "="^60)
    @printf(f, "  TM predict latency:       %.2f μs/sample\n", latency_us)
    @printf(f, "  TM predict throughput:    %d pred/sec\n", round(Int, throughput))
    @printf(f, "  Full pipeline latency:    %.2f μs/sample\n", full_latency_us)
    @printf(f, "  Full pipeline throughput: %d pred/sec\n", round(Int, full_throughput))
    @printf(f, "  Compiled model size:      %.1f KB\n", model_size_kb)
    @printf(f, "  In-memory model size:     %.1f KB\n", mem_model / 1024)
    @printf(f, "  Features: %d bits, Clauses: %d\n", n_features, CLAUSES)
    @printf(f, "  Accuracy: %.2f%%\n", best_acc * 100)
end

println("\nSaved: /tmp/inference_benchmark_wustl.txt")

# ── Append GLADE_FPTM row to the ML summary.tsv ─────────────────────
y_pred_full = [predict(compiled_tm, x) for x in x_test]
acc_full = sum(y_pred_full .== y_test) / n_test
classes = sort(unique(y_train))
macro_p = 0.0; macro_r = 0.0; macro_f1 = 0.0
for c in classes
    global macro_p, macro_r, macro_f1
    tp = sum((y_pred_full .== c) .& (y_test .== c))
    fp = sum((y_pred_full .== c) .& (y_test .!= c))
    fn = sum((y_pred_full .!= c) .& (y_test .== c))
    p  = tp + fp == 0 ? 0.0 : tp / (tp + fp)
    r  = tp + fn == 0 ? 0.0 : tp / (tp + fn)
    f1 = p + r == 0.0 ? 0.0 : 2p * r / (p + r)
    macro_p += p; macro_r += r; macro_f1 += f1
end
nc = length(classes); macro_p /= nc; macro_r /= nc; macro_f1 /= nc

summary_path = joinpath(@__DIR__, "results", "wustl_ml_baselines", "summary.tsv")
mkpath(dirname(summary_path))
if !isfile(summary_path)
    open(summary_path, "w") do f
        println(f, "Model\tAccuracy\tF1_macro\tPrecision_macro\tRecall_macro\tTrainTime_s\tTestTime_s\tMemory_KB")
    end
end
if isfile(summary_path)
    lines = filter(l -> !startswith(l, "GLADE_FPTM\t"), readlines(summary_path))
    open(summary_path, "w") do f
        for l in lines; println(f, l); end
    end
end
open(summary_path, "a") do f
    @printf(f, "GLADE_FPTM\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.1f\n",
            acc_full, macro_f1, macro_p, macro_r, t_fit, avg_batch, model_size_kb)
end
println("Appended GLADE_FPTM row to: $summary_path")
