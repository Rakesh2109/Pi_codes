# Final GLADE (conditional Steps 2 & 3)

Same code on every dataset. Steps apply **only when per-threshold gates fire**.

## Step 2 — gap snap (optional per quantile)
Replace τ by gap midpoint when:
1. `g_i > median(g)` on that feature
2. `u_i < tau < u_{i+1}`

Otherwise keep the quantile.

## Step 3 — local refinement (optional per threshold)
Skip when `k >= s(n)` and `(n-k) >= s(n)` with `s(n) = max(5, ceil(log2(n)) + 2)`.

Otherwise try `tau ± 1/4 * adjacent_span`; keep move only if `p(1-p)` strictly increases.

## Observed trigger rates (automatic)
Run: `python3 -m paper_2.glade_gate_analysis`

Typical: high snap on TON_IoT/NSL-KDD; Step 3 mostly inactive on MedSec/WUSTL.
