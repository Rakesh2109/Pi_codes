# Learning Julia From Zero — Through the TM Inference Code

A beginner-friendly guide to Julia, taught through the actual code that runs our Tsetlin Machine inference. Zero prior Julia knowledge required.

---

## 1. What is Julia and why did we use it?

**Julia** is a modern programming language designed to look like Python but run as fast as C.

**The problem with Python for our use case:**
- Python is slow because every operation goes through an interpreter
- Even with NumPy, each call has 1–5 microseconds of overhead
- Our TM predict does ~8 operations per sample → ~40 µs wasted on overhead

**Julia's solution:**
- Write code that looks almost like Python
- Julia compiles it to fast machine code the first time you run a function
- After that, it runs as fast as C (no overhead per operation)

**Result for our TM inference:**
- Same algorithm in Python: ~50 µs per sample
- Same algorithm in Julia: ~1.2 µs per sample (42× faster)

**One sentence summary:** *Julia gives you Python's readability and C's speed.*

---

## 2. How to install and run Julia

### Install
```bash
# On Linux (what we use on the Pi):
curl -fsSL https://install.julialang.org | sh
# Then restart your shell, or:
source ~/.bashrc
# Verify:
julia --version
```

### Run a Julia file
```bash
julia my_file.jl                 # runs a script
julia --threads=1 my_file.jl     # single-threaded (what we use for benchmarks)
julia --threads=4 my_file.jl     # multi-threaded (for training)
```

### Start an interactive session (like Python REPL)
```bash
julia
# Now you get a prompt:
julia> 1 + 2
3
julia> println("hello")
hello
julia> exit()
```

---

## 3. Julia syntax you need — side by side with Python

If you know Python, Julia feels familiar. Here's a cheat-sheet.

| Task | Python | Julia |
|---|---|---|
| Print | `print("hi")` | `println("hi")` |
| Variable | `x = 5` | `x = 5` |
| Typed variable | `x: int = 5` | `x::Int = 5` |
| Function | `def f(x):` | `function f(x)` ... `end` |
| Short function | `f = lambda x: x+1` | `f(x) = x+1` |
| If / else | `if x > 0: ... else: ...` | `if x > 0 ... else ... end` |
| For loop | `for i in range(10):` | `for i in 1:10` ... `end` |
| List | `[1, 2, 3]` | `[1, 2, 3]` (Vector) |
| Array access | `x[0]` | `x[1]` (**1-indexed!**) |
| Array slice | `x[1:3]` | `x[2:3]` (**inclusive**) |
| Dict | `{"a": 1}` | `Dict("a" => 1)` |
| String format | `f"{x}"` | `"$(x)"` |
| Comment | `# hi` | `# hi` |

### The BIG differences from Python

1. **Julia is 1-indexed** — `x[1]` is the first element, not `x[0]`
2. **Blocks end with `end`** — not indentation
3. **Everything is typed under the hood** — but you don't have to declare types
4. **Functions get compiled** — the first call is slower (compiling), every call after is fast

### Simple example

```julia
function greet(name)
    println("Hello, $(name)")
end

greet("Pi")   # prints: Hello, Pi
```

---

## 4. Key Julia concepts used in our code

### 4.1. Types (optional but powerful)

In our benchmarks we write:
```julia
x::Vector{Float32}
```
This says "x is a vector (list) of 32-bit floats."

Why it matters: Julia compiles faster code when it knows the types. In hot loops, we always write types.

Common types in our code:
- `Int` — a 64-bit integer (CPU word size)
- `Int32`, `Int64` — fixed-width integers
- `Float32`, `Float64` — single/double precision floats
- `UInt64` — 64-bit unsigned integer (we use this for bit-packing)
- `Bool` — true/false
- `Vector{T}` — a 1-D array of type T (like a Python list with one type)
- `Matrix{T}` — a 2-D array

Example:
```julia
x::UInt64 = 0               # x is a 64-bit unsigned integer, starts at 0
y::Vector{Float32} = [1.0, 2.0, 3.0]   # a vector of 3 floats
```

### 4.2. Functions

```julia
function add(a, b)
    return a + b
end
```
Or the short form (one-line):
```julia
add(a, b) = a + b
```

In our TM code we see:
```julia
@inline function vote(tm, ta, x)
    ...
end
```
The `@inline` is a hint to the compiler: "paste this function's body directly where it's called, don't make a real function call." Makes hot loops faster.

### 4.3. Broadcasting (Julia's equivalent of NumPy vectorization)

Python NumPy:
```python
c = a + b    # element-wise if a, b are arrays
```

Julia:
```julia
c = a .+ b   # the dot means "broadcast" (element-wise)
```

The `.` before operators/functions = element-wise on arrays:
- `a .+ b` — element-wise add
- `a .* b` — element-wise multiply
- `count_ones.(a)` — apply `count_ones` to each element of `a`

### 4.4. Arrays and indexing

```julia
a = [10, 20, 30]
a[1]        # 10  (first element — 1-indexed!)
a[end]      # 30  (last element)
a[1:2]      # [10, 20] (slice, INCLUSIVE on both ends)
length(a)   # 3
```

2D arrays:
```julia
A = [1 2 3; 4 5 6]    # semicolon = new row
A[1, 2]               # 2  (row 1, column 2)
A[:, 1]               # [1, 4]  (whole first column)
```

### 4.5. For loops

```julia
for i in 1:10
    println(i)
end
# prints 1, 2, ..., 10 (inclusive)
```

Range syntax `1:10` is like Python's `range(1, 11)` — inclusive of the end.

### 4.6. `@inbounds` and `@simd` — performance hints

```julia
@inbounds for i in 1:n
    a[i] += 1
end
```

- **`@inbounds`**: "Skip array bounds checking" — saves ~1 cycle per array access
- **`@simd`**: "You may vectorize this loop" — helps the compiler use SIMD instructions

**Rule:** Use these ONLY after you're sure your code is correct (no bounds violations), because they skip safety checks.

### 4.7. Module and `using`

```julia
include("../src/Tsetlin.jl")   # load another file's code
using .Tsetlin: TMInput, vote  # import specific names from that module
```

`.Tsetlin` means "the Tsetlin module defined in the file I just included." Then we pick which functions/types to import.

This is like Python's:
```python
from Tsetlin import TMInput, vote
```

### 4.8. Structs (custom types)

```julia
struct Point
    x::Float64
    y::Float64
end

p = Point(1.0, 2.0)
println(p.x)   # 1.0
```

`mutable struct` lets you change fields after creation:
```julia
mutable struct Counter
    count::Int
end

c = Counter(0)
c.count = 5    # works because mutable
```

In our code, `TMInput` and `TATeam` are structs that hold TM model data.

### 4.9. Symbols — `:name`

You'll see `:something` in code. This is a "Symbol" — like an interned string label. In our code, `ct.first` is a class label (a Symbol or Int), `ct.second` is a team object.

---

## 5. Walkthrough of our TM inference code

Let me walk through the real inference code line-by-line.

### 5.1. Top of the file (imports and setup)

```julia
include("../src/Tsetlin.jl")
using .Tsetlin: TMInput, TMClassifier, train!, predict, vote, compile, save, load
using Printf
using Serialization
```

**Explanation:**
- `include(...)` — load the Tsetlin library from a file
- `using .Tsetlin: ...` — import specific names from it
- `using Printf` — Julia's formatted print library (like Python's `%` formatting)
- `using Serialization` — Julia's built-in object save/load

### 5.2. Load a gzipped model

```julia
compiled_tm = open(`gunzip -c /tmp/wustl_compiled.tm.gz`) do io
    Serialization.deserialize(io)
end
```

**Explanation:**
- `` `gunzip -c ...` `` — backticks make a shell command (like Python `subprocess`)
- `open(cmd) do io ... end` — run the command, pipe its output into `io`
- `Serialization.deserialize(io)` — read the binary data from the pipe and reconstruct a Julia object

This is equivalent to Python:
```python
with gzip.open("/tmp/wustl_compiled.tm.gz") as f:
    compiled_tm = pickle.load(f)
```

### 5.3. Prepare GLADE binarizer

```julia
feat_idx = Int[]        # empty Int array (will grow)
thresh   = Float32[]    # empty Float32 array
for line in eachline("/tmp/WUSTL_binarizer.tsv")
    parts = split(line, '\t')
    push!(feat_idx, parse(Int, parts[1]) + 1)   # 0-based → 1-based
    push!(thresh,   parse(Float32, parts[2]))
end
```

**Explanation:**
- `Int[]` — empty array of Ints (equivalent to Python `[]` but typed)
- `eachline(path)` — iterate over lines of a file (lazy, doesn't load all at once)
- `split(line, '\t')` — split on tab
- `parse(Int, "5")` — convert string "5" to integer 5
- `push!(arr, x)` — append x to arr (the `!` means "modifies arr")

**`!` convention in Julia:** Functions ending in `!` modify their arguments in place. `push!(arr, x)` adds to arr; `sort(x)` returns a sorted copy; `sort!(x)` sorts x in place.

### 5.4. Pre-allocate buffers for the hot loop

```julia
n_bits   = length(thresh)
n_chunks = cld(n_bits, 64)            # ceil(n_bits / 64)
xi       = TMInput(zeros(UInt64, n_chunks), n_bits)
chunks   = xi.chunks
```

**Explanation:**
- `cld(a, b)` — ceiling division (a/b rounded up)
- `zeros(UInt64, n_chunks)` — make an array of UInt64 zeros, length n_chunks
- `TMInput(...)` — create a TMInput struct (our custom type)
- `chunks = xi.chunks` — reference to xi's internal chunks array (so we can write to it)

**Why pre-allocate?** Every loop iteration, we REUSE this buffer instead of allocating a new one. Saves ~1 µs per iteration.

### 5.5. The binarize function — register-local packing

```julia
@inline function pack_sample!(chunks, X_raw, feat_idx, thresh, i, n_bits, n_chunks)
    @inbounds for c in 1:n_chunks
        lo = (c - 1) << 6 + 1           # first bit index of this chunk
        hi = min(c << 6, n_bits)        # last bit index
        w  = zero(UInt64)               # accumulator starts at 0
        @inbounds @simd for k in lo:hi
            w |= UInt64(X_raw[feat_idx[k], i] >= thresh[k]) << (k - lo)
        end
        chunks[c] = w
    end
end
```

Let me break this down completely:

**The function signature:**
- `@inline` — inline this function at call sites (faster)
- `function pack_sample!(...)` — `!` means it modifies `chunks`
- Arguments: chunks to write into, X_raw input, thresholds, sample index, sizes

**What it does:** For each UInt64 chunk (there are 4 for WUSTL), pack 64 bits into one UInt64.

**Line-by-line inside:**
1. `lo = (c - 1) << 6 + 1` — `<< 6` is left-shift by 6 = multiply by 64. So for c=1, lo=1; for c=2, lo=65.
2. `hi = min(c << 6, n_bits)` — last bit index, but don't exceed n_bits.
3. `w = zero(UInt64)` — initialize a UInt64 zero. This variable stays in a CPU register.
4. Inner loop: for each bit k in this chunk:
   - `X_raw[feat_idx[k], i]` — the feature value for this sample
   - `>= thresh[k]` — compare to threshold → Bool
   - `UInt64(...)` — convert Bool to UInt64 (0 or 1)
   - `<< (k - lo)` — shift to the correct bit position within the chunk
   - `w |= ...` — OR into the accumulator
5. `chunks[c] = w` — ONE memory write per 64 bits (this is the big optimization)

**Why is this fast?** Because `w` never leaves a CPU register. 64 OR-ops happen in the register, then one write to memory. Compared to naive `chunks[k>>6] |= bit << (k&63)` which does 3 memory ops per bit.

### 5.6. The predict function

```julia
@inline function fast_predict(tm, class_teams, x)
    best_v::Int64 = typemin(Int64)      # start with worst possible vote
    local best_c = class_teams[1].first # default class label
    @inbounds for ct in class_teams
        p, n = vote(tm, ct.second, x)   # vote returns (positive_sum, negative_sum)
        v = p - n
        if v > best_v
            best_v = v
            best_c = ct.first
        end
    end
    return best_c
end
```

**Explanation:**
- `typemin(Int64)` — smallest possible Int64 (start point for finding max)
- `class_teams` — a vector of (class_label, TATeam) pairs
- `ct.first` — the class label (first element of the pair)
- `ct.second` — the TATeam object (second element)
- `vote(...)` returns two values (p, n) — you unpack them with tuple assignment
- The loop finds the class with the highest `p - n` vote

Equivalent Python:
```python
def fast_predict(tm, class_teams, x):
    best_v = float('-inf')
    best_c = class_teams[0][0]
    for label, team in class_teams:
        p, n = vote(tm, team, x)
        v = p - n
        if v > best_v:
            best_v = v
            best_c = label
    return best_c
```

### 5.7. The inner kernel — `check_clause`

This is the hottest code in the system. Don't skip it:

```julia
@inline function check_clause(tm, x, literals, literals_inverted, clamp)
    c::Int64 = 0
    @inbounds for i in 1:n_chunks
        val = (~x.chunks[i] & literals[i]) | (x.chunks[i] & literals_inverted[i])
        c += count_ones(val)
    end
    return max(clamp - c, 0)
end
```

**What it computes per clause:**
1. For each 64-bit chunk:
   - `~x.chunks[i]` — bitwise NOT of the sample chunk
   - `& literals[i]` — AND with the clause's "positive literals" mask → bits where we want x=1 but got x=0 (mismatch type 1)
   - `x.chunks[i] & literals_inverted[i]` — bits where we want x=0 but got x=1 (mismatch type 2)
   - OR together → all mismatches
   - `count_ones(val)` — count set bits (popcount). Compiles to one ARM `CNT` instruction.
   - Add to running total `c`
2. `max(clamp - c, 0)` — graded FPTM output: full clamp if zero mismatches, decreasing linearly, 0 if too many mismatches

**This is where the CPU spends most of its time.** Julia compiles this to 3 instructions per chunk: AND, OR, CNT. Very close to hardware limit.

### 5.8. Timing code

```julia
t = @elapsed begin
    for i in 1:n_single
        pack_sample!(chunks, X_raw, feat_idx, thresh, i, n_bits, n_chunks)
        fast_predict(compiled_tm, class_teams, xi)
    end
end
latency_us = t / n_single * 1e6
```

**Explanation:**
- `@elapsed` — macro that returns seconds taken to execute a block
- `begin ... end` — a block of multiple statements
- We loop n_single times, each iteration calls binarize + predict
- Divide by n_single → per-sample seconds
- Multiply by 1e6 → per-sample microseconds

This is how we measure Julia TM's 1.2 µs/sample.

---

## 6. The Julia mental model

### Key difference from Python

In Python, every line is interpreted at runtime. In Julia, the FIRST time you call a function, it gets compiled to machine code. Every call after that runs as fast as C.

```julia
f(x) = x * 2

f(5)    # first call — compiles, then runs. Maybe 100 ms.
f(5)    # second call — already compiled. < 1 µs.
f(5)    # third call — same.
```

**Implication for benchmarks:** We always run a **warmup** (100 calls) before timing. The first calls include compile time; we don't want to measure that.

```julia
for i in 1:100
    fast_predict(tm, ct, xi)   # warmup — triggers compilation
end

t = @elapsed for i in 1:10000
    fast_predict(tm, ct, xi)   # actual timing
end
```

### Why the first call is slow

Julia compiles functions for the specific types they're called with. If you call `f(5)` (Int) and `f(5.0)` (Float64), you get TWO compiled versions — one for each type. This is called **multiple dispatch** and it's Julia's killer feature.

**Consequence:** Changing types in a hot loop forces recompilation. Always use consistent types.

### When Julia is much faster than Python

- Tight numerical loops (our TM inference)
- Functions called millions of times
- Code where type annotations help the compiler

### When Julia is NOT faster

- Short scripts with one big function call (Python's overhead is only paid once)
- String processing (Python is actually quite good here)
- Code dominated by I/O (both languages wait for disk)

---

## 7. The file structure of our project

```
pi_codes/
├── train_all_datasets.jl           # trains TM models
├── inference_all_datasets.jl       # benchmarks Julia TM inference
├── export_tm_to_npz.jl             # exports Julia model → binary for Python
├── python_tm_inference.py          # Python TM that reads the exported binary
├── python_tm_pi_realistic.py       # per-sample streaming benchmark
├── prepare_dataset_fastkbin_*.py   # Python GLADE preprocessing (per dataset)
├── ml_baselines_*.py               # sklearn baselines (per dataset)
src/
└── Tsetlin.jl                       # the TM library (from Granmo et al)
```

### The data flow

```
1. Python script: load CSV → preprocess → GLADE binarize → save
     /tmp/<prefix>_X_train.txt (bits)
     /tmp/<prefix>_X_test.txt (bits)
     /tmp/<prefix>_X_test_raw.bin (original float32 for streaming)
     /tmp/<prefix>_binarizer.tsv (GLADE thresholds)

2. Julia: read bits → train TM → save
     /tmp/<stem>_compiled.tm.gz (compressed Julia model)

3. Julia (inference): load .tm.gz → run predict on test data → F1 + timing

4. Julia (export): convert .tm.gz → .tmpy.bin (for Python consumption)

5. Python (inference): read .tmpy.bin + raw test → predict → F1 + timing
```

---

## 8. Hands-on mini-exercises

Try these at the Julia REPL (`julia` in your shell):

### Exercise 1 — First program
```julia
println("Hello from Julia!")
x = 1:10                        # a range 1 to 10
sum(x)                          # 55 — Julia has a `sum` function
sum(x .^ 2)                     # 385 — element-wise square, then sum
```

### Exercise 2 — A function
```julia
function fib(n)
    a, b = 0, 1
    for _ in 1:n
        a, b = b, a + b
    end
    return a
end

fib(10)    # 55
fib(20)    # 6765
```

### Exercise 3 — Bit operations (relevant to TM)
```julia
x = UInt64(0b10110101)              # binary literal — `0b` prefix
count_ones(x)                       # 5  — how many set bits
y = UInt64(0b11110000)
x & y                               # 0b10110000 = 176
x | y                               # 0b11110101 = 245
~x                                  # bitwise NOT (all 64 bits)
x << 3                              # shift left by 3
```

### Exercise 4 — Timing code
```julia
function sum_squares(n)
    s = 0
    for i in 1:n
        s += i * i
    end
    return s
end

@time sum_squares(1_000_000)     # first call: compile + run
@time sum_squares(1_000_000)     # second call: just run (fast)
```

### Exercise 5 — Your own register-local bit pack
```julia
function pack_one_chunk(values, thresholds)
    w::UInt64 = 0
    for k in 1:64
        bit = UInt64(values[k] >= thresholds[k])
        w |= bit << (k - 1)
    end
    return w
end

vals   = rand(Float32, 64)        # random floats
thr    = fill(Float32(0.5), 64)   # all thresholds = 0.5
chunk  = pack_one_chunk(vals, thr)
println(chunk)                     # some UInt64 with ~half bits set
count_ones(chunk)                  # roughly 32
```

This is a simplified version of what our `pack_sample!` does.

---

## 9. Common Julia commands we use

| Command | What it does |
|---|---|
| `julia script.jl` | Run a script |
| `julia --threads=N script.jl` | With N threads |
| `julia` | Interactive REPL |
| `include("file.jl")` | Load a file into REPL |
| `@time expr` | Time one expression |
| `@elapsed expr` | Return time in seconds |
| `@benchmark expr` (requires BenchmarkTools.jl) | Proper statistical benchmarking |
| `?function_name` | Help on a function |
| `methods(sum)` | List all method signatures |
| `Pkg.add("PackageName")` | Install a package |

### Installing a package

```julia
julia> using Pkg
julia> Pkg.add("BenchmarkTools")
```

Or in the REPL, press `]` to enter package mode:
```
pkg> add BenchmarkTools
```

---

## 10. Debugging tips

**Error: "UndefVarError: x not defined"**
→ You used `x` before assigning it. Check the scope — in Julia, `for` loops have their own scope, so `x` inside a loop is local unless you use `global x`.

**Error: "MethodError: no method matching f(::Int64)"**
→ You called `f` with a type it doesn't handle. Julia's dispatch is strict.

**Slow first call**
→ That's JIT compilation. Run a warmup before timing.

**Unexpected results from array indexing**
→ Remember Julia is 1-indexed. `x[1]` is first, not `x[0]`.

**Variable is the wrong type in a hot loop**
→ Julia can't specialize if types vary. Use type annotations `x::Float64`.

---

## 11. What to read next

After this guide, you should be able to read our `inference_all_datasets.jl` end-to-end. When you want to go deeper:

1. **Official Julia Docs** (docs.julialang.org) — start with "Manual" → "Integers and Floating-Point Numbers"
2. **JuliaByExample.com** — short, practical examples
3. **"Think Julia" book** (free online) — comprehensive intro
4. **BenchmarkTools.jl** — proper benchmarking library
5. **PackageCompiler.jl** — compile your Julia code to a standalone binary for deployment

For our specific use case (TM, bitwise, SIMD):
- Look at `@code_native function_call()` — it shows the machine code Julia generates
- Look at `@code_warntype function_call()` — shows type-stability (critical for speed)

---

## 12. Summary — the 10 most important takeaways

1. **Julia looks like Python but compiles to machine code** — fast numerical code
2. **1-indexed arrays** — `x[1]` is first, not `x[0]`
3. **Blocks end with `end`** — no indentation requirement
4. **`!` at end of function name = modifies arguments** — `push!`, `sort!`, `pack_sample!`
5. **`@inline`, `@inbounds`, `@simd`** — performance hints for hot loops
6. **Type annotations help the compiler** — `x::UInt64`, `::Float32`
7. **First call compiles, later calls are fast** — always warmup before benchmarking
8. **`count_ones` compiles to one CPU instruction** — this is why TM is fast in Julia
9. **Broadcasting with `.`** — `a .+ b` is element-wise on arrays
10. **Register-local accumulation** — keep a variable in a register (like `w = 0` in pack_sample!), update it in a loop, one memory write at the end. This is the single biggest perf trick in our code.

---

## 13. The one-paragraph takeaway

Julia lets us write code that looks like Python but runs like C. Our TM inference uses four Julia features heavily: **typed variables** (for compiler speed), **`@inbounds` and `@simd`** (for tight loops), **built-in `count_ones`** (compiles to NEON popcount), and **pre-allocated buffers** (zero allocations per sample). The result is per-sample inference at 1.2 µs — 50× faster than the same algorithm in Python. If you understand the `pack_sample!` and `check_clause` functions line-by-line, you understand 90% of the performance story.
