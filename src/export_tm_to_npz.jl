#!/usr/bin/env julia
#=
Export a Julia-trained compiled TM + GLADE binarizer state to a .npz-
compatible binary layout that Python can load directly.

Output file format (little-endian, single binary blob):

  header (36 bytes):
    magic        4 bytes    "TMPY"
    version      u32        = 1
    n_classes    u32
    n_clauses    u32        clauses per class per polarity
    n_chunks     u32
    n_bits       u32
    placeholder  u32 × 3    (reserved)

  For each class 0..n_classes-1:
    class_label      i32
    for each polarity (0=pos, 1=neg):
      for each clause 0..n_clauses-1:
        clamp         i32
        lits          u64[n_chunks]
        lits_inv      u64[n_chunks]

  feat_idx (0-based)        i32[n_bits]
  thresh                    f32[n_bits]
=#

include("../src/Tsetlin.jl")
using .Tsetlin: load
using Serialization

const MAGIC = UInt8[0x54, 0x4D, 0x50, 0x59]  # "TMPY"
const VERSION = UInt32(1)

function export_one(tm_path, bin_tsv, out_path)
    compiled_tm = open(`gunzip -c $(tm_path * ".gz")`) do io
        Serialization.deserialize(io)
    end

    classes = sort(collect(keys(compiled_tm.clauses)))
    n_classes = length(classes)

    first_team = compiled_tm.clauses[classes[1]]
    n_clauses = length(first_team.positive_included_literals_sum_clamp)
    n_chunks  = size(first_team.positive_included_literals, 1)

    # GLADE state
    feat_idx = Int[]; thresh = Float32[]
    for line in eachline(bin_tsv)
        parts = split(line, '\t')
        push!(feat_idx, parse(Int, parts[1]))        # keep 0-based for Python
        push!(thresh,   parse(Float32, parts[2]))
    end
    n_bits = length(thresh)

    open(out_path, "w") do f
        write(f, MAGIC)
        write(f, VERSION)
        write(f, UInt32(n_classes))
        write(f, UInt32(n_clauses))
        write(f, UInt32(n_chunks))
        write(f, UInt32(n_bits))
        write(f, UInt32(0)); write(f, UInt32(0)); write(f, UInt32(0))

        for cls in classes
            team = compiled_tm.clauses[cls]
            write(f, Int32(cls))
            for (lits_mat, inv_mat, clamps) in (
                (team.positive_included_literals,
                 team.positive_included_literals_inverted,
                 team.positive_included_literals_sum_clamp),
                (team.negative_included_literals,
                 team.negative_included_literals_inverted,
                 team.negative_included_literals_sum_clamp),
            )
                for c in 1:n_clauses
                    write(f, Int32(clamps[c]))
                    write(f, UInt64.(lits_mat[:, c]))
                    write(f, UInt64.(inv_mat[:, c]))
                end
            end
        end

        write(f, Int32.(feat_idx))
        write(f, thresh)
    end

    sz = filesize(out_path)
    println("  $(out_path)  classes=$n_classes  clauses=$n_clauses  chunks=$n_chunks  bits=$n_bits  size=$(round(sz/1024, digits=1)) KB")
end

DATASETS = [
    ("toniot", "TonIoT"),
    ("medsec", "MedSec"),
    ("wustl",  "WUSTL"),
    ("nslkdd", "NSLKDD"),
]

for (stem, prefix) in DATASETS
    tm_path  = "/tmp/$(stem)_compiled.tm"
    bin_tsv  = "/tmp/$(prefix)_binarizer.tsv"
    out_path = "/tmp/$(stem)_tmpy.bin"
    if !isfile(tm_path) || !isfile(bin_tsv)
        println("  skip $stem (missing $tm_path or $bin_tsv)")
        continue
    end
    export_one(tm_path, bin_tsv, out_path)
end
